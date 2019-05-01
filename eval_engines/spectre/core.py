import re
#import libpsf
import copy
import os
import jinja2
import os
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import yaml
import importlib
import IPython
from jinja2 import Environment, FileSystemLoader
debug = False

def get_config_info():
    # TODO
    config_info = dict()
    base_tmp_dir = os.environ.get('BASE_TMP_DIR', None)
    if not base_tmp_dir:
        raise EnvironmentError('BASE_TMP_DIR is not set in environment variables')
    else:
        config_info['BASE_TMP_DIR'] = base_tmp_dir
    return config_info

class SpectreWrapper(object):

    def __init__(self, netlist_loc, post_process_fn=None):
        """
        This Wrapper handles one netlist at a time, meaning that if there are multiple test benches
        for characterizations multiple instances of this class needs to be created

        :param netlist_loc: the template netlist used for circuit simulation
        """

        # suppose we have a config_info = {'section':'model_lib'}
        # config_info also contains BASE_TMP_DIR (location for storing simulation netlist/results)
        # implement get_config_info() later

        self.config_info = get_config_info()

        self.root_dir = self.config_info['BASE_TMP_DIR']
        self.num_process = self.config_info.get('NUM_PROCESS', 1)

        _, dsg_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.gen_dir, exist_ok=True)

        self.post_process = post_process_fn
        #raw_file = open(netlist_loc, 'r')
        #self.tmp_lines = raw_file.readlines()
        #raw_file.close()

        #jinja stuff
        dir_netlist, netlist_name = os.path.split(netlist_loc)
        file_loader = FileSystemLoader(searchpath=dir_netlist)
        jinja_env = Environment(loader=file_loader)
        self.template = jinja_env.get_template(netlist_name)

    def _get_design_name(self, state):
        """
        Creates a unique identifier fname based on the state
        :param state:
        :return:
        """
        fname = self.base_design_name
        for value in state.values():
            if isinstance(value,float):
              str_val = str(value)
              str_val = str_val.replace('.','')
            else:
              str_val = str(value)
            fname += "_" + str(str_val)
        return fname

    def _create_design(self, state, new_fname):
        # TODO: Replace these functions with jinja2
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')
       
        rendered_template = self.template.render(state)
        
        with open(fpath, 'w') as f:
            f.writelines(rendered_template)
            f.close()
        return design_folder, fpath

    def _simulate(self, fpath):
        # TODO run spectre
        command = ['spectre', '%s'%fpath, '>/dev/null 2>&1']
        exit_code = subprocess.call(command, cwd=os.path.dirname(fpath))
        info = 0
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
            # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self._get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self._create_design(state, dsn_name)
        info = self._simulate(fpath)
        results = self._parse_result(design_folder)
        if self.post_process:
            specs = self.post_process(results)
            return state, specs, info
        specs = results
        return state, specs, info

    def _parse_result(self, design_folder):
        # TODO: kourosh
        pass

    def run(self, states, design_names=None, verbose=False):
        # TODO: Use asyncio to instantiate multiple jobs for running parallel sims
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()
        return specs


class EvaluationEngine(object):
    def __init__(self, yaml_fname):

        self.design_specs_fname = yaml_fname
        with open(yaml_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        meas_module = importlib.import_module(self.ver_specs['measurement']['meas_module'])
        meas_cls = getattr(meas_module, self.ver_specs['measurement']['meas_class'])

        self.measMan = meas_cls(yaml_fname)
        self.params = self.measMan.params
        self.spec_range = self.measMan.spec_range
        self.params_vec = self.measMan.params_vec
        # minimum and maximum of each parameter
        # params_vec contains the acutal numbers but min and max should be the indices
        self.params_min = [0]*len(self.params_vec)

        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)

        self.id_encoder = IDEncoder(self.params_vec)


    @property
    def num_params(self):
        return len(self.params_vec)

    def generate_data_set(self, n=1):
        """
        :param n:
        :param evaluate:
        :return: a list of n Design objects with populated attributes (i.e. cost, specs, id)
        """
        valid_designs, tried_designs = [], []

        useless_iter_count = 0
        while len(valid_designs) <= n:
            design = {}
            for key, vec in self.params_vec.items():
                rand_idx = random.randrange(len(vec))
                design[key] = rand_idx
            design = Design(self.spec_range, self.id_encoder, list(design.values()))
            if design in tried_designs:
                if (useless_iter_count > n * 5):
                    raise ValueError("Random selection of a fraction of search space did not result in {}"
                                     "number of valid designs".format(n))
                useless_iter_count += 1
                continue
            design_result = self.evaluate([design])[0]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]
                valid_designs.append(design)
            tried_designs.append(design)
            print(len(valid_designs))

        return valid_designs[:n]

    def evaluate(self, design_list):
        """
        :param design_list:
        :return: a list of processed_results that the algorithm cares about, keywords should include
        cost and spec keywords with one scalar number as the value for each
        """
        results = []
        for design in design_list:
            try:
                result = self.measMan.evaluate(design)
                result['valid'] = True
            except Exception as e:
                result = {'valid': False}
                print(getattr(e, 'message', str(e)))
            results.append(result)
        return results

    def compute_penalty(self, spec_nums, spec_kwrd):
        return self.measMan.compute_penalty(spec_nums, spec_kwrd)


    def find_worst(self, spec_nums, spec_kwrd, ret_penalty=False):
        if not hasattr(spec_nums, '__iter__'):
            spec_nums = [spec_nums]

        penalties = self.compute_penalty(spec_nums, spec_kwrd)
        worst_penalty = max(penalties)
        worst_idx = penalties.index(worst_penalty)
        if ret_penalty:
            return spec_nums[worst_idx], worst_penalty
        else:
            return spec_nums[worst_idx]

#unit test
if __name__ == '__main__':

  #testing the cs amp functionality with Jinja2
  dsn_netlist = '/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/bag_deep_ckt/eval_engines/spectre/netlist_templates/cs_ac_16nm.scs'
  cs_env = SpectreWrapper(netlist_loc=dsn_netlist)
  
  state = {"nfin":2, "nf":4, "vb":0.5, "res":1000, "vdd":1.0}
  cs_env._create_design_and_simulate(state)

