import importlib
from .agent import Agent
import time
import os.path as osp

class DeepCKTAgent(Agent):

    def __init__(self, y_fname):
        Agent.__init__(self, y_fname)

        model_module = importlib.import_module(self.specs['model_module_name'])
        model_cls = getattr(model_module, self.specs['model_class_name'])
        self.model = model_cls(num_params_per_design=self.eval_core.num_params,
                               spec_kwrd_list=list(self.eval_core.spec_range.keys()),
                               logger=self.logger,
                               **self.specs['model_params'])

        # get the parameters in agent_params that are related to deep learning
        self.valid_frac = self.agent_params['valid_frac']
        self.max_n_retraining = self.agent_params['max_n_retraining']

        # from db designs with indices less than this number are considered during training
        self.k = self.agent_params['k']

        # training settings
        self.num_epochs = self.agent_params['num_epochs']
        self.batch_size = self.agent_params['batch_size']
        self.display_step = self.agent_params['display_step']
        self.ckpt_step = self.agent_params['ckpt_step']
        self.max_iter = self.agent_params['max_iter']

        # paper stuff variables
        self.n_sims = 0
        self.sim_time = 0
        self.n_queries = 0
        self.query_time = 0
        self.n_training = 0
        self.training_time = 0
        self.n_sims_list, self.sims_time_list = [], []
        self.n_nn_queries_list, self.nn_query_time_list = [], []
        self.n_training_list, self.training_time_list = [], []
        self.total_time_list = []

    def train(self):
        t_minus = time.time()
        ds = self.model.get_train_valid_ds(self.db, self.k, self.eval_core, self.valid_frac)
        self.model.train(ds, self.batch_size, self.num_epochs, self.ckpt_step, self.display_step)
        t_plus = time.time()
        self.n_training += 1
        self.training_time += (t_plus - t_minus)
        self.logger.log_text("[INFO] training done %.2fSec" % (t_plus - t_minus))

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

        while_time = 0
        gen_time = 0
        check_time = 0
        q_time = 0
        decision_time = 0
        deletion_time = 0

        self.ea.prepare_for_generation(self.db, self.k)
        while_s = time.time()
        while len(offsprings) < self.n_new_samples and n_iter < self.max_iter:
            gen_s = time.time()
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)
            gen_e = time.time()
            gen_time += gen_e - gen_s

            for new_design in new_designs:
                check_s = time.time()
                if any([(new_design == row) for row in self.db]) or \
                        any([(new_design == row) for row in offsprings]):
                    # if design is already in the design pool skip ...
                    self.logger.log_text("[debug] design {} already exists".format(new_design))
                    continue
                check_e = time.time()
                check_time += check_e - check_s

                n_iter += 1
                q_s = time.time()
                prediction = self.model.query(input1=new_design, input2=ref_design)
                q_e = time.time()
                q_time += q_e - q_s
                self.n_queries += 1

                decision_s = time.time()
                is_new_design_better = self.decision_box.accept_new_design(prediction)
                decision_e = time.time()
                decision_time += decision_e - decision_s

                deletion_s = time.time()
                if is_new_design_better:
                    offsprings.append(new_design)
                deletion_e = time.time()
                deletion_time += decision_e - deletion_s

        while_e = time.time()
        while_time = while_e - while_s
        self.query_time += time.time() - q_start

        self.logger.log_text("avg_gen_time = {}".format(gen_time/n_iter))
        self.logger.log_text("avg_check_time = {}".format(check_time/n_iter))
        self.logger.log_text("avg_q_time = {}".format(q_time/n_iter))
        self.logger.log_text("avg_decision_time = {}".format(decision_time/n_iter))
        self.logger.log_text("avg_deletion_time = {}".format(deletion_time/n_iter))
        self.logger.log_text("avg_while_time = {}".format(while_time/n_iter))

        if len(offsprings) < self.n_new_samples:
            return offsprings, True

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
        self.logger.log_text("[INFO] new candidates size: %d " % len(offsprings))

        return offsprings, False

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(self.db)
        self.update_time_info(0)

        self.model.init()
        self.train()

        for i in range(self.max_n_retraining):

            offsprings, is_converged = self.run()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db = self.db + offsprings
            self.data_set_list.append(offsprings)
            self.logger.store_db(self.data_set_list)
            self.update_time_info(time.time()-start)
            self.store_database_and_times()

            # adjust dataset size for training, if not desired, comment the agent.k_top= ... line
            self.db = sorted(self.db, key=lambda x: x.cost)
            worst_offspring = max(offsprings, key=lambda x: x.cost)
            self.logger.log_text('[INFO] k_top alternative: {}'.format(self.db.index(
                worst_offspring)))
            self.k = max(self.n_init_samples, self.db.index(worst_offspring))

            self.train()
            self.logger.log_text("[INFO] n_queries = {}".format(self.n_queries))
            self.logger.log_text("[INFO] query_time = {}".format(self.query_time))
            self.logger.log_text("[INFO] n_simulations = {}".format(self.n_sims))
            self.logger.log_text("[INFO] sim_time = {}".format(self.sim_time))
            self.logger.log_text("[INFO] n_training = {}".format(self.n_training))
            self.logger.log_text("[INFO] training_time = {}".format(self.training_time))
            self.logger.log_text("[INFO] total_time = {}".format(time.time()-start))

        self.logger.store_db(self.data_set_list)
        self.store_database_and_times()

        sorted_db = sorted(self.db, key=lambda x: x.cost)
        # paper stuff
        self.logger.log_text("[finished] n_queries = {}".format(self.n_queries))
        self.logger.log_text("[finished] query_time = {}".format(self.query_time))
        self.logger.log_text("[finished] n_simulations = {}".format(self.n_sims))
        self.logger.log_text("[finished] sim_time = {}".format(self.sim_time))
        self.logger.log_text("[finished] n_training = {}".format(self.n_training))
        self.logger.log_text("[finished] training_time = {}".format(self.training_time))
        self.logger.log_text("[finished] total_time = {}".format(time.time()-start))

        self.logger.log_text("[finished] total_n_evals = {}".format(len(self.db)))
        self.logger.log_text("[finished] best_solution = {}".format(sorted_db[0]))
        self.logger.log_text("[finished] id = {}".format(sorted_db[0].id))
        self.logger.log_text("[finished] cost = {}".format(sorted_db[0].cost))
        self.logger.log_text("[finished] performance \n{} ".format(sorted_db[0].specs))
        for ind in sorted_db[:self.decision_box.ref_index]:
            self.logger.log_text("{} -> {} -> {}".format(ind, ind.cost, ind.specs))

    def update_time_info(self, total_time):
        self.n_sims_list.append(self.n_sims)
        self.sims_time_list.append(self.sim_time)
        self.n_nn_queries_list.append(self.n_queries)
        self.nn_query_time_list.append(self.query_time)
        self.n_training_list.append(self.n_training)
        self.training_time_list.append(self.training_time)
        self.total_time_list.append(total_time)

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_nn_query=self.n_nn_queries_list,
            nn_query_time=self.nn_query_time_list,
            n_sims=self.n_sims_list,
            sims_time=self.sims_time_list,
            n_training=self.n_training_list,
            training_time=self.training_time_list,
            total_time=self.total_time_list,
        )
        self.logger.store_db(dict_to_save, fpath=osp.join(self.logger.log_path, 'db_time.pkl'))