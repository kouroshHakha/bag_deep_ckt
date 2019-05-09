from typing import List, Dict, Iterable, Union, Tuple, Any

import os
import subprocess
import yaml
import importlib
import random
import numpy as np
from multiprocessing import Queue
import threading
from jinja2 import Environment, FileSystemLoader

from eval_engines.util.core import IDEncoder, Design
from eval_engines.spectre.parser import SpectreParser

def get_config_info() -> Dict[str, Any]:
    """
    returns the config_info dictionary by getting the necessary variables from environment
    variables
    :return:
        a dictionary containing the variables used in SpectreWrapper class
    """
    config_info = dict()
    base_tmp_dir = os.environ.get('BASE_TMP_DIR', None)
    if not base_tmp_dir:
        raise EnvironmentError('BASE_TMP_DIR is not set in environment variables, please set it '
                               'up!')
    else:
        config_info['BASE_TMP_DIR'] = base_tmp_dir

    return config_info

class SpectreWrapper(object):

    def __init__(self, tb_dict: Dict) -> None:
        """
        This Wrapper handles one netlist at a time, meaning that if there are multiple test
        benches, multiple instances of this class needs to be created

        :param tb_dict:
            the template netlist used for circuit simulation
        """

        netlist_loc = tb_dict['netlist_template']
        if not os.path.isabs(netlist_loc):
            netlist_loc = os.path.abspath(netlist_loc)
        pp_module = importlib.import_module(tb_dict['tb_module'])
        pp_class = getattr(pp_module, tb_dict['tb_class'])
        self.post_process = getattr(pp_class, 'process')
        self.tb_params = tb_dict['tb_params']

        self.config_info = get_config_info()

        self.root_dir = self.config_info['BASE_TMP_DIR']
        self.num_process = self.config_info.get('NUM_PROCESS', 1)

        _, dsn_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0]
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.gen_dir, exist_ok=True)

        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

    def _get_design_name(self, state: Dict[str, Any]) -> str:
        """
        Creates a unique identifier fname based on the state. This function is used if dsn name
        is not provided to _create_design_and_simulate()
        :param state:
            dictionary mapping parameter key words to values
        :return:
            string fname
        """
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def _create_design(self, state: Dict[str, Any], new_fname: str) -> Tuple[str, str]:
        """
        creates the design netlist from the template.
        :param state:
            dictionary mapping parameter key words to values
        :param new_fname:
            new file name to be used for the netlist
        :return:
            (dir, file) where dir shows the directory to simulation folder, and file shows the
            netlist absolute location
        """
        output = self.template.render(**state)
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)
        fpath = os.path.join(design_folder, new_fname + '.scs')
        with open(fpath, 'w') as f:
            f.write(output)
            f.close()
        return design_folder, fpath

    @classmethod
    def _simulate(cls, fpath: str, debug: bool = False) -> int:
        """
        runs the spectre commands for simulation.
        :param fpath:
            netlist's absolute path
        :param debug:
            if true, descriptive statements get printed
        :return:
            an integer, if it's zero no error has occurred, if 1 some error in the spectre
            subprocess has occurred. In case of error one should go to the printed path
        """
        command = ['spectre', '%s'%fpath, '-format', 'psfbin' ,'> /dev/null 2>&1']
        log_file = os.path.join(os.path.dirname(fpath), 'log.txt')
        err_file = os.path.join(os.path.dirname(fpath), 'err_log.txt')
        exit_code = subprocess.call(command, cwd=os.path.dirname(fpath),
                                    stdout=open(log_file, 'w'), stderr=open(err_file, 'w'))
        info = 0
        if debug:
            print(command)
            print(fpath)
        if exit_code % 256:
            info = 1 # this means an error has occurred
            print("error occured during {command}\ncheck the log file: {fpath}".format(
                command=command, fpath=log_file))

        return info


    def _create_design_and_simulate(self, state: Dict[str, Any], dsn_name: str = None,
                                    verbose: bool = False) \
            -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        This function creates the new netlist and runs spectre commands on it and returns state,
        specs and error code tuple. The results are either post-processed by the post-processing
        function that the user provides through the yaml file, or it just returns the direct
        parse data.
        :param state:
            dictionary mapping parameter key words to values
        :param dsn_name:
            design name used for creating files and simulation folders, if it is None, a default
            verbose unique name is used.
        :param verbose:
            if True some extra statements will be printed
        :return:
            results from post-porccessing function (if provided) or the directly parsed data
        """
        if dsn_name:
            dsn_name = str(dsn_name)
        else:
            dsn_name = self._get_design_name(state)

        if verbose:
            print('state', state)
            print(dsn_name)

        design_folder, fpath = self._create_design(state, dsn_name)
        info = self._simulate(fpath)
        results = self._parse_result(design_folder)
        if self.post_process:
            specs = self.post_process(results, self.tb_params)
            return state, specs, info
        specs = results
        return state, specs, info

    @classmethod
    def _parse_result(cls,  design_folder: str) -> Dict[str, Any]:
        """
        Parses the spectre result in the design folder
        :param design_folder:
            absolute path to design folder
        :return:
            dictionary representing the result
        """
        _, folder_name = os.path.split(design_folder)
        raw_folder = os.path.join(design_folder, '{}.raw'.format(folder_name))
        res = SpectreParser.parse(raw_folder)
        return res

    def run(self, state: Dict[str, Any], design_name: str = None, verbose: bool = False) \
            -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        :param state:
            dictionary mapping parameter key words to values
        :param design_name:
            if None default design name will be used, otherwise the given design name will be used
        :param verbose:
            If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value),
            info: int)]
        """
        specs = self._create_design_and_simulate(state, design_name, verbose)
        return specs

class SubEngine(object):

    @classmethod
    def process(cls, results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class EvaluationEngine(object):

    def __init__(self, yaml_fname):

        self.design_specs_fname = yaml_fname
        with open(yaml_fname, 'r') as f:
            self.ver_specs = yaml.load(f)

        self.spec_range = self.ver_specs['spec_range']
        # params are interfaced using the index instead of the actual value
        params = self.ver_specs['params']

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in params.items():
            self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
            self.search_space_size = self.search_space_size * len(self.params_vec[key])

        # minimum and maximum of each parameter
        # params_vec contains the acutal numbers but min and max should be the indices
        self.params_min = [0]*len(self.params_vec)
        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)

        self.id_encoder = IDEncoder(self.params_vec)
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    @property
    def num_params(self):
        return len(self.params_vec)

    def generate_data_set(self, n: int = 1, debug: bool = False) -> List[Design]:
        """
        Generates n samples randomly, with their evaluation results. If debug is False errors are
        not raised and only valid designs (with no errors) will be returned.
        :param n:
            number of samples to be generated
        :param debug:
            if True evaluations will run in a single thread and any error encountered  along the
            way will raise an exception
        :return:
            List of design objects with their spec values.
        """
        valid_designs: List[Design] = list()
        tried_designs: List[Design] = list()

        useless_iter_count = 0
        while len(valid_designs) < n:
            design_list = list()
            for _ in range(n - len(valid_designs)):
                design = dict()
                for key, vec in self.params_vec.items():
                    rand_idx = random.randrange(len(vec))
                    design[key] = rand_idx
                design = Design(self.spec_range, self.id_encoder, list(design.values()))
                if design in tried_designs:
                    if useless_iter_count > n * 5:
                        raise ValueError("Random selection of a fraction of search space did not "
                                         "result in {} number of valid designs".format(n))
                    useless_iter_count += 1
                    continue
                design_list.append(design)
            design_results = self.evaluate(design_list, debug=debug)
            for dsn_res, design in zip(design_results, design_list):
                if dsn_res['valid']:
                    design.cost = dsn_res['cost']
                    for key in design.specs.keys():
                        design.specs[key] = dsn_res[key]
                    valid_designs.append(design)
                tried_designs.append(design)

            print("n designs valid so far: {}".format(len(valid_designs)))

        return valid_designs

    def evaluate(self, design_list: List[Design], debug: bool = False) -> List[Dict[str, float]]:
        """
        Evaluates designs in design_list
        :param design_list:
            List of designs
        :param debug:
            if True exceptions are raised if encountered, otherwise they are ignored
        :return:
            a list of dictionaries representing specifications
        """

        results = list()
        if debug:
            for design in design_list:
                results.append(self._main_task(design))
        else:
            t_list = list()
            queue = Queue()
            for i, design in enumerate(design_list):
                t = threading.Thread(target=self._main_task, args=(design, queue, i))
                t.start()
                t_list.append(t)

            for t in t_list:
                if t.is_alive():
                    t.join()
            queue.put('stop')  # put a stop flag in the queue

            # get the results in order
            results = [x for x in iter(queue.get, 'stop')]
            results = sorted(results, key=lambda x: x[0])
            results = [x[1] for x in results]

        return results

    def _main_task(self, design: Design, queue: Queue = None, index: int = None) \
            -> Dict[str, float]:
        """
        Main task for multi-tasking purporses. Runs the simulations and returns the results
        :param design:
            Design object
        :param queue:
            Multiprocessing Queue object for putting back the results in, if parallelization is
            invoked there is no guarantee that results are in order, that's why index is used
        :param index:
            keeps track of the thread order
        :return:
            a dictionary representing the result. Also queue object is updated
        """
        try:
            result = self._evaluate(design)
            result['valid'] = True
        except Exception as e:
            result = {'valid': False, 'message': str(e)}

        if queue:
            assert index is not None, "when queue is provided index should also be given"
            queue.put((index, result))

        return result

    def _evaluate(self, design: Design) -> Dict[str, float]:
        """
        Loops through the testbench templates and their spectreWrapper object nad tuns the
        corresponding updated netlist. It the gathers the results and calls the processing
        function (get_specs) that user needs to override.
        :param design:
            design object
        :return:
            dictionary representing the specifications. User has full control over what this
            function returns by overriding the specs_dict
        """
        state_dict = dict()
        for i, key in enumerate(self.params_vec.keys()):
            state_dict[key] = self.params_vec[key][design[i]]
        results = dict()
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.run(state_dict, design.id)

        specs_dict = self.get_specs(results, self.measurement_specs['meas_params'])
        specs_dict['cost'] = self.cost_fun(specs_dict)
        return specs_dict

    def find_worst(self, spec_nums: Union[Iterable[float], float], spec_kwrd: str,
                   ret_penalty: bool = False) -> Union[Tuple[float, float], float]:
        """
        Finds the worst number among spec_nums based on the penalty function provided by the user
        for a specific keyword.
        :param spec_nums:
            numbers representing different specification metrics
        :param spec_kwrd:
            keyword representing the specification name
        :param ret_penalty:
            if True, the penalty associated with that worst case is also returned (spec_num,
            penalty)
        :return:
            if ret_penalty is False only the worst specification number is returned, but if it's
            True the associated penalty is also returned (spec_num, penalty)
        """

        if not hasattr(spec_nums, '__iter__'):
            spec_nums = [spec_nums]

        penalties = self.compute_penalty(spec_nums, spec_kwrd)
        worst_penalty = max(penalties)
        worst_idx = penalties.index(worst_penalty)
        if ret_penalty:
            return spec_nums[worst_idx], worst_penalty
        else:
            return spec_nums[worst_idx]

    def cost_fun(self, specs_dict: Dict[str, float]) -> float:
        """
        returns the total cost for a given specification dictionary. This function will call
        compute_penalty which will be overwritten by the user based on the preference on how to
        infer cost value based on the specification dictionary.
        :param specs_dict:
            dictionary representing the performance
        :return:
            a scalar number representing th cost value of the design
        """
        cost = 0
        for spec in self.spec_range.keys():
            penalty = self.compute_penalty(specs_dict[spec], spec)[0]
            cost += penalty

        return cost

    def get_specs(self, results: Dict[str, Dict], params: Dict) -> Dict[str, float] :
        """
        This function gets all the individual post-processed test bench results, and implements
        higher level post-processing to fully specify the performance of the design that has been
        simulated
        :param results:
            dictionary mapping test bench key name to the post processed result of that test bench
        :param params:
            high level parameters comming from yaml file that might be needed for high level
            post-processing
        :return:
            a Dictionary mapping specification keywords in spec_range (look in the example
            yaml file) to the value of that spec
        """
        raise NotImplementedError

    def compute_penalty(self, spec_nums: Union[float, Iterable[float]], spec_kwrd: str) \
            -> Union[float, List[float]]:
        """
        Computes the penalty from the spec_kwrd for all spec_nums
        :param spec_nums:
            numbers representing specification values
        :param spec_kwrd:
            keyword representing the specification of interest
        :return:
        """
        raise NotImplementedError
