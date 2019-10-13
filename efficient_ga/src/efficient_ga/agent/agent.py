import yaml
import importlib
from efficient_ga.src.efficient_ga.logger import Logger
import pickle
import os
from util import clean, relable
from efficient_ga.src.efficient_ga.helper_module import DecisionBox
import random


def read_yaml(fname):
    with open(fname, 'r') as f:
        return yaml.load(f)


class Agent:

    def __init__(self, y_fname):

        self.specs = read_yaml(y_fname)
        self.agent_params = self.specs['agent_params']

        self.db = None
        self.data_set_list = []

        # create evaluation core instance
        eval_module = importlib.import_module(self.specs['eval_core_package'])
        eval_cls = getattr(eval_module, self.specs['eval_core_class'])
        self.eval_core = eval_cls(design_specs_fname=self.specs['circuit_yaml_file'])

        self.circuit_content = read_yaml(self.specs['circuit_yaml_file'])
        self.init_pop_dir = self.circuit_content['database_dir']

        self.logger = Logger(log_path=self.specs['log_path'])
        self.logger.store_settings(y_fname, self.specs['circuit_yaml_file'])

        self.n_init_samples = self.agent_params['n_init_samples']
        self.n_new_samples = self.agent_params['n_new_samples']
        self.num_params_per_design = self.eval_core.num_params

        self.decision_box = DecisionBox(self.agent_params['ref_dsn_idx'],
                                        self.eval_core,
                                        self.logger)

        ea_module = importlib.import_module(self.specs['ea_module_name'])
        ea_cls = getattr(ea_module, self.specs['ea_class_name'])
        self.ea = ea_cls(eval_core=self.eval_core, **self.specs['ea_params'])

    def get_init_population(self):
                # load/create db
                fname = os.path.join(self.init_pop_dir, 'init_data.pickle')
                if os.path.isfile(fname):
                    with open(fname, 'rb') as f:
                        self.db = pickle.load(f)
                else:
                    self.db = self.eval_core.generate_data_set(self.n_init_samples, evaluate=True)
                    with open(fname, 'wb') as f:
                        pickle.dump(self.db, f)

                if len(self.db) >= self.n_init_samples:
                    random.shuffle(self.db)
                    self.db = self.db[:self.n_init_samples]
                else:
                    raise Warning('Number of init_samples is larger than the length of the '
                                  'initial data base, using the len(db) instead of n_init_samples')

                self.db = clean(self.db, self.eval_core)
                self.db = relable(self.db, self.eval_core)
                self.db = sorted(self.db, key=lambda x: x.cost)
                # HACK for paper
                # self.db = self.db[1:]
                self.logger.log_text("[INFO] Best cost in init_pop = {}".format(self.db[0].cost))

    def main(self):
        raise NotImplementedError
