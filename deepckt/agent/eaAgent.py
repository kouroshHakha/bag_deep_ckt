from .agent import Agent
import time
import os.path as osp


class EvoAgent(Agent):

    def __init__(self, y_fname):
        Agent.__init__(self, y_fname)
        self.k = self.agent_params['k']
        self.max_n_gen = self.agent_params['max_n_gen']

        self.n_sims = 0
        self.sim_time = 0
        # self.total_time = 0
        self.n_sims_list, self.sim_time_list, self.total_time_list = [], [], []

    def run(self):
        self.decision_box.update_heuristics(self.db, self.k)
        if self.decision_box.has_converged():
            return [], True

        parent1, parent2, _ = self.decision_box.get_parents_and_ref_design(self.db, self.k)

        offsprings = []
        n_iter = 0

        self.logger.log_text(30*"-")
        self.logger.log_text("[INFO] running model ... ")

        self.ea.prepare_for_generation(self.db, self.k)
        while len(offsprings) < self.n_new_samples:
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)

            for new_design in new_designs:
                if any([(new_design == row) for row in self.db]) or \
                        any([(new_design == row) for row in offsprings]):
                    # if design is already in the design pool skip ...
                    self.logger.log_text("[debug] design {} already exists".format(new_design))
                    continue

                n_iter += 1
                offsprings.append(new_design)

        self.logger.log_text(30*"-")
        for offspring in offsprings:
            self.logger.log_text(offspring)

        s = time.time()
        design_results = self.eval_core.evaluate(offsprings)
        e = time.time()
        self.n_sims += len(offsprings)
        self.sim_time += e - s

        self.logger.log_text('[INFO] design evaluation time: {:.2f}'.format(e-s))
        list_to_be_removed = []
        for i, design in enumerate(offsprings):
            design_result = design_results[i]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]

                self.logger.log_text("[debug] design {} with cost {} was added".format(design,
                                                                                       design.cost))
                self.logger.log_text("[debug] {}".format(design.specs))

            else:
                self.logger.log_text("[debug] design {} did not produce valid results".format(
                    design))
                list_to_be_removed.append(design)

        for design in list_to_be_removed:
            offsprings.remove(design)

        self.logger.log_text("[INFO] new designs tried: %d" % n_iter)

        return offsprings, False

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(self.db)
        self.n_sims_list.append(self.n_sims)
        self.sim_time_list.append(self.sim_time)
        self.total_time_list.append(0)

        for i in range(self.max_n_gen):
            offsprings, is_converged = self.run()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db = self.db + offsprings
            self.data_set_list.append(offsprings)
            self.n_sims_list.append(self.n_sims)
            self.sim_time_list.append(self.sim_time)
            self.total_time_list.append(time.time()-start)
            if i % 10 == 0:
                self.logger.store_db(self.data_set_list)
                self.store_database_and_times()

            # self.db = sorted(self.db, key=lambda x: x.cost)
            self.logger.log_text("[INFO] n_iter = {}".format(i))
            self.logger.log_text("[INFO] n_simulations = {}".format(self.n_sims))
            self.logger.log_text("[INFO] sim_time = {}".format(self.sim_time))
            self.logger.log_text("[INFO] total_time = {}".format(time.time()-start))

        self.logger.store_db(self.data_set_list)
        self.store_database_and_times()

        sorted_db = sorted(self.db, key=lambda x: x.cost)

        self.logger.log_text("[finished] n_simulations = {}".format(self.n_sims))
        self.logger.log_text("[finished] sim_time = {}".format(self.sim_time))
        self.logger.log_text("[finished] total_time = {}".format(time.time()-start))

        for ind in sorted_db[:10]:
            self.logger.log_text("{} -> {} -> {}".format(ind, ind.cost, ind.specs))

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_query=self.n_sims_list,
            query_time=self.sim_time_list,
            total_time=self.total_time_list,
        )
        self.logger.store_db(dict_to_save, fpath=osp.join(self.logger.log_path, 'db_time.pkl'))
