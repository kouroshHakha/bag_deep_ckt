from .agent import Agent
import time
from util import is_x_better_than_y
from copy import deepcopy
import os.path as osp


class OracleAgent(Agent):

    def __init__(self, y_fname):
        Agent.__init__(self, y_fname)
        # get the parameters in agent_params
        self.max_n_retraining = self.agent_params['max_n_steps']

        # from db designs with indices less than this number are considered during training
        self.k = self.agent_params['k']
        self.max_iter = self.agent_params['max_iter']

        # paper stuff variables
        self.n_queries = 0
        self.query_time = 0
        self.n_queries_list, self.query_time_list, self.total_time_list  = [], [], []
        self.oracle_save_to = self.agent_params.get("oracle_save_to", None)
        if self.oracle_save_to:
            self.oracle_queries = dict(
                inputs1=[],
                inputs2=[],
                critical_specs=[]
            )
            for kwrd in self.eval_core.spec_range.keys():
                self.oracle_queries[kwrd] = []

    def run(self):

        self.decision_box.update_heuristics(self.db, self.k)

        if self.decision_box.has_converged():
            return [], True

        parent1, parent2, ref_design = self.decision_box.get_parents_and_ref_design(self.db, self.k)

        offsprings = []
        n_iter = 0

        self.logger.log_text(30*"-")
        self.logger.log_text("[INFO] running model ... ")
        q_start = time.time()

        self.ea.prepare_for_generation(self.db, self.k)
        while len(offsprings) < self.n_new_samples and n_iter < self.max_iter:
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)

            for new_design in new_designs:
                if any([(new_design == row) for row in self.db]) or \
                        any([(new_design == row) for row in offsprings]):
                    # if design is already in the design pool skip ...
                    self.logger.log_text("[debug] design {} already exists".format(new_design))
                    continue

                n_iter += 1
                design_result = self.eval_core.evaluate([new_design])[0]
                if design_result['valid']:
                    new_design.cost = design_result['cost']
                    for key in new_design.specs.keys():
                        new_design.specs[key] = design_result[key]
                    is_new_design_better_oracle = [is_x_better_than_y(self.eval_core,
                                                                      new_design.specs[kwrd],
                                                                      ref_design.specs[kwrd],
                                                                      kwrd) for kwrd in
                                                   self.decision_box.critical_specs]

                    if all(is_new_design_better_oracle):
                        offsprings.append(new_design)
                        self.logger.log_text("[debug] design {} with cost {} was added".format(
                            new_design, new_design.cost))
                        self.logger.log_text("{}".format(new_design.specs))

                    if self.oracle_save_to:
                        self.store_oracle_query(new_design, ref_design,
                                                self.decision_box.critical_specs)

                self.n_queries += 1

        self.query_time += time.time() - q_start

        if len(offsprings) < self.n_new_samples:
            return offsprings, True

        self.logger.log_text(30*"-")

        self.logger.log_text("[INFO] new designs tried: %d" % n_iter)
        self.logger.log_text("[INFO] new candidates size: %d " % len(offsprings))

        return offsprings, False

    def store_oracle_query(self, input1, input2, critical_specs):
        self.oracle_queries['inputs1'].append(deepcopy(input1))
        self.oracle_queries['inputs2'].append(deepcopy(input2))
        self.oracle_queries['critical_specs'].append(critical_specs.copy())
        for kwrd in self.eval_core.spec_range.keys():
            self.oracle_queries[kwrd].append(is_x_better_than_y(self.eval_core,
                                                                input1.specs[kwrd],
                                                                input2.specs[kwrd],
                                                                kwrd))
        self.logger.store_db(self.oracle_queries, fpath=self.oracle_save_to)

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(self.db)
        self.query_time_list.append(self.query_time)
        self.n_queries_list.append(self.n_queries)
        self.total_time_list.append(0)

        for i in range(self.max_n_retraining):
            offsprings, is_converged = self.run()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db = self.db + offsprings
            self.data_set_list.append(offsprings)
            self.query_time_list.append(self.query_time)
            self.n_queries_list.append(self.n_queries)
            self.total_time_list.append((time.time()-start))
            self.logger.store_db(self.data_set_list)
            self.store_database_and_times()

            # adjust dataset size for training, if not desired, comment the agent.k_top= ... line
            self.db = sorted(self.db, key=lambda x: x.cost)
            worst_offspring = max(offsprings, key=lambda x: x.cost)
            self.logger.log_text('[INFO] k_top alternative: {}'.format(self.db.index(
                worst_offspring)))
            self.k = max(self.n_init_samples, self.db.index(worst_offspring))

            self.logger.log_text("[INFO] n_queries = {}".format(self.n_queries))
            self.logger.log_text("[INFO] query_time = {}".format(self.query_time))
            self.logger.log_text("[INFO] total_time = {}".format(time.time()-start))

        self.logger.store_db(self.data_set_list)

        sorted_db = sorted(self.db, key=lambda x: x.cost)
        # paper stuff
        self.logger.log_text("[finished] n_queries = {}".format(self.n_queries))
        self.logger.log_text("[finished] query_time = {}".format(self.query_time))
        self.logger.log_text("[finished] total_time = {}".format(time.time()-start))

        self.logger.log_text("[finished] total_n_evals = {}".format(len(self.db)))
        self.logger.log_text("[finished] best_solution = {}".format(sorted_db[0]))
        self.logger.log_text("[finished] id = {}".format(sorted_db[0].id))
        self.logger.log_text("[finished] cost = {}".format(sorted_db[0].cost))
        self.logger.log_text("[finished] performance \n{} ".format(sorted_db[0].specs))
        for ind in sorted_db[:self.decision_box.ref_index]:
            self.logger.log_text("{} -> {} -> {}".format(ind, ind.cost, ind.specs))

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_query=self.n_queries_list,
            query_time=self.query_time_list,
            total_time=self.total_time_list,
        )
        self.logger.store_db(dict_to_save, fpath=osp.join(self.logger.log_path, 'db_time.pkl'))