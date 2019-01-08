import pprint
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, Sequence
import yaml
import random
import time

from bag.io import read_yaml, open_file
from bag.core import BagProject
from bag.simulation.core import DesignManager
from bag.concurrent.core import batch_async_task

import os
import numpy as np
from copy import deepcopy
from bag_deep_ckt.util import *
import math
import abc
import multiprocessing
from shutil import copyfile
import IPython

class Phase1Error(Exception):
    pass

class BagEvalEngine(object, metaclass=abc.ABCMeta):


    def __init__(self, design_specs_fname):
        """
        :param design_specs_fname: The main yaml file that should have some specific structure,
        the template is given in ...

        """
        print('creating BAG project')
        self.bprj = BagProject()
        self.design_specs_fname = design_specs_fname
        self.ver_specs = read_yaml(self.design_specs_fname)

        root_dir = self.ver_specs['root_dir']
        self.swp_spec_dir = os.path.join(root_dir, 'gen_yamls')
        self.sim_specs_fname = os.path.join(self.swp_spec_dir, self.ver_specs['sim_spec_fname'])
        self.swp_spec_dir = os.path.join(self.swp_spec_dir, 'swp_spec_files')
        os.makedirs(self.swp_spec_dir, exist_ok=True)

        self.spec_range = self.ver_specs['spec_range']
        self.param_choices_layout = self.break_hierarchy(self.ver_specs['params']['layout_params'])
        self.param_choices_measurement = self.break_hierarchy(self.ver_specs['params']['measurements'])
        # self.params contains the dictionary corresponding to the params part of the main yaml file where empty
        # dicts are replaces with None
        self.params = self.break_hierarchy(self.ver_specs['params'])

        self.params_vec = {}
        self.search_space_size = 1
        for key, value in self.params.items():
            if value is not None:
                # self.params_vec contains keys of the main parameters and the corresponding search vector for each
                self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
                self.search_space_size = self.search_space_size * len(self.params_vec[key])

        self.id_encoder = IDEncoder(self.params_vec)

    @property
    def num_params(self):
        return len(self.params_vec)

    def break_hierarchy(self, main_dict):
        kwrds, values = self.break_hierarchy_into_lists(main_dict)
        # remove None value entries
        ret_dict = {}
        for k, v in zip(kwrds, values):
            if v == None:
                continue
            ret_dict[k] = v
        return ret_dict

    def break_hierarchy_into_lists(self, main_dict):
        """
        takes in a dictionary (of dictionaries) and will return a list of items (keys and values separated)
        with merged kwrds and corresponding values
        example:
            main_dict = {"1":{"1":{"1":111
                                   "2":112
                                   "3":113}
                              "2":{"1":121
                                   "2": {"1": 1221}}
                              "3":13}
                         "2": 2
                         "3": 3}

            ret_dict = {"1/1/1": 111, "1/1/2": 112, "1/1/3": 113, "1/2/1": 121, "1/2/2/1": 1221, "2": 2, "3": 3}
            ret_kwrds, ret_values = ret_dict.items()
        :param main_dict:
        :return: list of combined keys and corresponding values, if value in the end is empty dictionary it will be None
        """
        kwrds, values = [], []
        for k, v in main_dict.items():
            if isinstance(v, dict):
                # internet check if v is not empty is equivalent to not v
                if v:
                    ret_kwrds, ret_values = self.break_hierarchy_into_lists(v)
                    for ret_kwrd in ret_kwrds:
                        updated_key = self.merge_keys([k, ret_kwrd])
                        kwrds.append(updated_key)
                    values += ret_values
                else:
                    # especial base case
                    kwrds += [k]
                    values += [None]
            else:
                # base case
                kwrds += [k]
                values += [v]
        return kwrds, values

    def merge_keys(self, list_of_kwrds):
        """
        combines a list of kwrds to a single string separated by "/"
        :param list_of_kwrds:
        :return: merged_list
        """
        merged_list = "/".join(list_of_kwrds)
        return merged_list

    def unmerge_keys(self, merged_keys):
        """
        takes in a single string containing multiple kwrds separated by "/" and returns a list of those kwrds
        :param merged_keys:
        :return: list_of_unmerged_keys
        """
        list_of_unmerged_keys = str.split(merged_keys, "/")
        return list_of_unmerged_keys

    def decend(self, keys):
        um = self.unmerge_keys(keys)
        return self.merge_keys(um[1:])

    def update_with_unmerged_key(self, main_dict, keys, value):
        """
        a destructive method which updates a specific entry in a dictionary with the correct value, the hierarchy is
        given by the variable keys, if the hierarchy does not exist it will create the path and assign the value
        :param main_dict: the dictionary to be updated
        :param keys: a single string representing hierarchy of update
        :param value: the value to be replaced as the value
        :return:
        """
        list_of_kwrds = self.unmerge_keys(keys)
        current_ptr = main_dict
        for key in list_of_kwrds[:-1]:
            if key not in current_ptr:
                current_ptr[key] = {}
            current_ptr = current_ptr[key]
        current_ptr[list_of_kwrds[-1]] = value

    def generate_data_set(self, n=1, evaluate=False):
        start = time.time()
        # all the designs we have generated and ran simulation on They may or may not have failed
        tried_designs = []
        # a subset of all tried designs which have been valid, the length of this list should reach n in the end
        valid_designs = []
        # all designs that we are about to evaluate, the length of this list should be less than the number of cpus

        n_cpus = multiprocessing.cpu_count()
        n_cpus = min([n_cpus, n])
        # for _ in range(n):
        while (len(valid_designs) <= n):
            trying_designs = []
            useless_iter_count = 0
            while (len(trying_designs) <= n_cpus):
                design = {}
                for key, vec in self.params_vec.items():
                    rand_idx = random.randrange(len(vec))
                    # rand_value = self.params_vec[key][rand_idx]
                    rand_value = rand_idx
                    design[key] = rand_value
                # Imposing the constraints of layout generator
                design = self.impose_constraints(design)
                design = Design(self.spec_range, self.id_encoder, list(design.values()))
                if design in tried_designs:
                    if (useless_iter_count > n_cpus * 10):
                        raise ValueError("Random selection of a fraction of search space did not result in {}"
                                         "number of valid designs".format(n))
                    useless_iter_count += 1
                    continue
                tried_designs.append(design)
                trying_designs.append(design)

            if evaluate:
                design_results = self.evaluate(trying_designs)
                for design, design_result in zip(trying_designs, design_results):
                    if design_result['valid']:
                        design.cost = design_result['cost']
                        for key in design.specs.keys():
                            design.specs[key] = design_result[key]
                        valid_designs.append(design)

        generator_efficiency = len(valid_designs) / len(tried_designs)
        print("Genrator Efficiency: {}".format(generator_efficiency))
        print("avg time for simulating one instance: {}".format((time.time() - start)/len(tried_designs)))
        return valid_designs[:n]

    def evaluate(self, design_list):
        # type: (List[Design]) -> List
        swp_spec_file_list = []
        sweep_params_update = deepcopy(self.ver_specs['sweep_params'])
        # del template['sweep_params']['swp_spec_file']
        for dsn_num, design in enumerate(design_list):
            # 1. translate each list to a dict with layout_params and measurement_params indication
            # 2. write those dictionaries in the corresponding param.yaml and update self.ver_specs
            specs = deepcopy(self.ver_specs)
            layout_update = specs['layout_params']
            measurement_update = specs['measurements'][0]
            params_dict = dict(zip(self.params_vec.keys(), design))
            # imposing the constraint of layout generator
            self.impose_constraints(params_dict)

            # TODO: Still cannot handle multiple measurement manager units
            for key, value_idx in params_dict.items():
                next_key = self.decend(key)
                if next_key in self.param_choices_layout.keys():
                    self.update_with_unmerged_key(layout_update, next_key, self.params_vec[key][value_idx])
                elif next_key in self.param_choices_measurement.keys():
                    self.update_with_unmerged_key(measurement_update, next_key, self.params_vec[key][value_idx])

            specs['sweep_params']['swp_spec_file'] = ['params_'+str(design.id)]

            swp_spec_file_list.append('params_'+str(design.id))
            fname = os.path.join(self.swp_spec_dir, 'params_'+str(design.id)+'.yaml')
            with open_file(fname, 'w') as f:
                yaml.dump(specs, f)

        sweep_params_update['swp_spec_file'] = swp_spec_file_list
        self.ver_specs['sweep_params'].update(sweep_params_update)
        results = self.generate_and_sim()
        return self.process_results(results)

    def generate_and_sim(self):
        """
        phase 1 of evaluation is generation of layout, schematic, LVS and RCX
        If any of LVS or RCX fail results_ph1 will contain Exceptions for the corresponding instance
        We proceed to phase 2 only if phase 1 was successful.
        phase 2 is running the simulation with post extracted netlist view
        Then we aggregate the results of phase 1 and phase 2 in a single list, in the same order
        that designs were ordered, if phase 1 was failed the corresponding entry will contain
        a Phase1Error exception
        """
        results = []
        with open_file(self.sim_specs_fname, 'w') as f:
            yaml.dump(self.ver_specs, f)

        sim = DesignManager(self.bprj, self.sim_specs_fname)
        results_ph1 = sim.characterize_designs(generate=True, measure=False, load_from_file=False)
        # hacky: do parallel measurements, you should not sweep anything other than 'swp_spec_file' in sweep_params
        # the new yaml files themselves should not include any sweep_param
        start = time.time()
        impl_lib = self.ver_specs['impl_lib']
        coro_list = []
        file_list = self.ver_specs['sweep_params']['swp_spec_file']
        for ph1_iter_index, combo_list in enumerate(sim.get_combinations_iter()):
            dsn_name = sim.get_design_name(combo_list)
            specs_fname = os.path.join(self.swp_spec_dir, file_list[ph1_iter_index] + '.yaml')
            if isinstance(results_ph1[ph1_iter_index], Exception):
                continue
            coro_list.append(self.async_characterization(impl_lib, dsn_name, specs_fname))

        results_ph2 = batch_async_task(coro_list)
        print("sim time: {}".format(time.time() - start))
        # this part returns the correct order of results if some of the instances failed phase1 of evaluation
        ph2_iter_index = 0
        for ph1_iter_index, combo_list in enumerate(sim.get_combinations_iter()):
            if isinstance(results_ph1[ph1_iter_index], Exception):
                results.append(Phase1Error)
            else:
                results.append(results_ph2[ph2_iter_index])
                ph2_iter_index+=1
        pprint.pprint(results)

        return results

    async def async_characterization(self, impl_lib, dsn_name, specs_fname):
        sim = DesignManager(self.bprj, specs_fname)
        pprint.pprint(specs_fname)
        await sim.verify_design(impl_lib, dsn_name, load_from_file=False)
        # just copy the corresponding yaml file, so that everything is in one place
        base_fname = os.path.basename(specs_fname)
        copyfile(specs_fname, os.path.join(sim._root_dir, dsn_name+'/'+base_fname))
        # print('name: {}'.format(dsn_name))
        # dsn_name = list(sim.get_dsn_name_iter())[0]
        summary = sim.get_result(dsn_name)
        return summary

    @abc.abstractmethod
    def impose_constraints(self, design_dict):
        """
        This function takes the layout_params dictionary and applies the constraints
        imposed by the layout generator
        for example: design_dict['seg_dict/tail1'] = design_dict['seg_dict/in'] for when input pairs should have
        the same size as the tail transistor
        :param design_dict: dictionary containing layout params key words and their values.
        :return: the updated version of design_dict
        """
        raise NotImplementedError

    @abc.abstractmethod
    def process_results(self, results):
        """
        gets the results of reading summary.yaml as a list of dictionaries and maps it to a list of
        cost and individual metrics. concretely, as an example for opamp for two corners we have:
        results = [{'opamp_ac': {'funity': [blah, blah], 'pm': [blah, blah], 'gain': [blah, blah], 'corners', 'bw' }}]
        processed_results = [{'cost': blah, 'funity': blah, 'pm': blah}]
        we might need to write our own helper functions to interpret the results based on the measurementManager
        class
        :param results: a list of dictionaries containing the results of phase2
        :return: a list of processed_results that the algorithm cares about, keywords should include
        cost and spec keywords with one scalar number as the value for each
        """
        raise NotImplementedError
