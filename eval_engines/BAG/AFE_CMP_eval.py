import os
import time

from eval_engines.BAG.bagEvalEngine import BagEvalEngine, Phase1Error
from eval_engines.BAG.DeepCKTDesignManager import DeepCKTDesignManager
from bag.io import read_yaml
from bag_deep_ckt.util import *
import pdb
from bag.io import read_yaml, open_file
import yaml
import pprint
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, Sequence
import yaml
import random
import time

from bag.io import read_yaml, open_file
from bag.core import BagProject
from eval_engines.BAG.DeepCKTDesignManager import DeepCKTDesignManager
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
from bag.simulation.core import DesignManager
from bag import float_to_si_string
import pprint
import math


class SUBCKTERROR(Exception):
    pass

class CustomDesignManager(DeepCKTDesignManager):
    def get_design_name(self, combo_list):
        # type: (Sequence[Any, ...]) -> str
        """Generate cell names based on sweep parameter values."""

        name_base = self.specs['dsn_basename']
        suffix = ''
        for var, val in zip(self.swp_var_list, combo_list):
            if isinstance(val, str):
                val = os.path.splitext(os.path.basename(val))[0]
                suffix += '_%s_%s' % (var, val)
            elif isinstance(val, int):
                suffix += '_%s_%d' % (var, val)
            elif isinstance(val, float):
                suffix += '_%s_%s' % (var, float_to_si_string(val))
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name_base + suffix
    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values. Overwritten to incorporate
        the discrete evaluation problem, instead of a sweep"""

        # sanity check
        assert len(self.swp_var_list) == 1 and 'swp_spec_file' in self.swp_var_list , \
            "when using DeepCKTDesignManager for replacing file name, just (swp_spec_file) should be part of the " \
            "sweep_params dictionary"

        lay_params = deepcopy(self.specs['layout_params'])
        # yaml_fname = self.specs['root_dir']+'/gen_yamls/swp_spec_files/' + val_list[0] + '.yaml'
        yaml_fname = val_list[0]
        print(yaml_fname)
        updated_top_specs = read_yaml(yaml_fname)
        new_lay_params = updated_top_specs['layout_params']
        lay_params.update(new_lay_params)
        return lay_params

class AFE_CMP_EvaluationEngine(BagEvalEngine):

    def __init__(self, design_specs_fname):
        self.bprj = BagProject()
        self.design_specs_fname = design_specs_fname
        self.ver_specs = read_yaml(self.design_specs_fname)

        root_dir = os.path.abspath(self.ver_specs['root_dir'])
        self.gen_yamls_dir = os.path.join(root_dir, 'gen_yamls')
        self.top_level_dir = os.path.join(self.gen_yamls_dir, 'top_level')
        self.top_level_main_file = os.path.join(self.gen_yamls_dir, 'top_level.yaml')
        os.makedirs(self.top_level_dir, exist_ok=True)

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
        self._unique_suffix = 0
        self.temp_db = None

        self.subckts_template = self.ver_specs['sub_system_sim']
        self.subckts_yaml_dirs = dict()
        self.subckts_main_file = dict()
        for key in self.subckts_template.keys():
            directory = os.path.join(self.gen_yamls_dir, 'subckt_{}'.format(key))
            self.subckts_yaml_dirs[key] = directory
            self.subckts_main_file[key] = directory + '.yaml'
            os.makedirs(self.subckts_yaml_dirs[key], exist_ok=True)


    def evaluate(self, design_list):
        # type: (List[Design]) -> List

        subckts_yaml_files = dict(zip(self.subckts_template.keys(), [[]] * len(self.subckts_template)))
        top_level_yaml_files = []
        for dsn_num, design in enumerate(design_list):
            top_specs = deepcopy(self.ver_specs)
            layout_update = top_specs['layout_params']
            measurement_update = top_specs['measurements'][0]

            params_dict = dict(zip(self.params_vec.keys(), design))
            for key, value_idx in params_dict.items():
                params_dict[key] = self.params_vec[key][value_idx]
            self.impose_constraints(params_dict)

            for key, value in params_dict.items():
                next_key = self.decend(key)
                if next_key in self.param_choices_layout.keys():
                    self.update_with_unmerged_key(layout_update, next_key, value)
                elif next_key in self.param_choices_measurement.keys():
                    self.update_with_unmerged_key(measurement_update, next_key, value)

            # for each subckt we partition the updated layout into individual files
            subckts_template = deepcopy(self.subckts_template)
            for subckt_key, subckt in subckts_template.items():
                subckt['layout_params'].update(**layout_update[subckt_key])
                fname = os.path.join(self.subckts_yaml_dirs[subckt_key], 'params_{}.yaml'.format(str(design.id)))
                with open(fname, 'w') as f:
                    yaml.dump(subckt, f)
                subckts_yaml_files[subckt_key].append(fname)

            # for top level we create the individual yaml files
            fname = os.path.join(self.top_level_dir, 'params_top_' + str(design.id) + '.yaml')
            with open_file(fname, 'w') as f:
                yaml.dump(top_specs, f)
            top_level_yaml_files.append(fname)

        top_specs = deepcopy(self.ver_specs)
        for key, value in subckts_yaml_files.items():
            subckts_template = deepcopy(self.subckts_template)
            subckts_template[key]['sweep_params']['swp_spec_file'] = value
            subckts_template[key]['root_dir'] = os.path.join(top_specs['root_dir'], key)

            with open_file(self.subckts_main_file[key], 'w') as f:
                yaml.dump(subckts_template[key], f)


        top_specs['sweep_params']['swp_spec_file'] = top_level_yaml_files
        with open_file(self.top_level_main_file, 'w') as f:
            yaml.dump(top_specs, f)
        results = self.generate_and_sim()
        return self.process_results(results)


    def generate_and_sim_ckt(self, file_name, new_flag=True):
        results = []
        specs = read_yaml(file_name)

        sim =  CustomDesignManager(self.bprj, file_name)
        if self.temp_db is None:
            self.temp_db = sim.make_tdb()
        sim.set_tdb(self.temp_db)

        results_ph1 = sim.characterize_designs(generate=new_flag, measure=False, load_from_file=False)

        impl_lib = specs['impl_lib']
        coro_list = []
        file_list = specs['sweep_params']['swp_spec_file']
        # hacky: do parallel measurements, you should not sweep anything other than 'swp_spec_file' in sweep_params
        # the new yaml files themselves should not include any sweep_param
        for ph1_iter_index, combo_list in enumerate(sim.get_combinations_iter()):
            dsn_name = sim.get_design_name(combo_list)
            specs_fname = file_list[ph1_iter_index]
            if isinstance(results_ph1[ph1_iter_index], Exception):
                continue
            coro_list.append(self.async_characterization(impl_lib, dsn_name, specs_fname, load_from_file=not new_flag))

        results_ph2 = batch_async_task(coro_list)
        # this part returns the correct order of results if some of the instances failed phase1 of evaluation
        ph2_iter_index = 0
        for ph1_iter_index, combo_list in enumerate(sim.get_combinations_iter()):
            if isinstance(results_ph1[ph1_iter_index], Exception):
                results.append(results_ph1[ph1_iter_index])
            else:
                # if isinstance(results_ph2[ph2_iter_index], Exception):
                #     raise results_ph2[ph2_iter_index]
                results.append(results_ph2[ph2_iter_index])
                ph2_iter_index+=1

        return results

    def generate_and_sim(self):
        # TODO:
        #   We are assuming that we are doing just one corner
        results = []

        dtsa_results = self.generate_and_sim_ckt(self.subckts_main_file['dtsa_params'])

        specs = read_yaml(self.top_level_main_file)
        design_fname_list = specs['sweep_params']['swp_spec_file']
        indices_to_be_removed = []
        for dtsa_index, (fname, dtsa_result) in enumerate(zip(design_fname_list, dtsa_results)):
            if isinstance(dtsa_result, Exception):
                indices_to_be_removed.append(dtsa_index)
            elif isinstance(dtsa_result, Dict):
                # valid = True
                # for spec_kwrd in self.spec_range.keys():
                #     if spec_kwrd.startswith('dtsa'):
                #         spec_min, spec_max, _ = self.spec_range[spec_kwrd]
                #         dtsa_spec_kwrd = str.split(spec_kwrd, "/")[-1]
                #         if spec_max is None:
                #             valid = dtsa_result['comparator'][dtsa_spec_kwrd][0] >= spec_min
                #         else:
                #             valid = dtsa_result['comparator'][dtsa_spec_kwrd][0] <= spec_max
                #     if not valid:
                #         break
                #
                # if not valid:
                #     indices_to_be_removed.append(dtsa_index)
                pass
            else:
                raise ValueError("Unknown type for dtsa_result: {}".format(type(dtsa_result)))

        pprint.pprint("-"*30+"//yaml_files generated", stream=open("bag_eval_run.txt", "w"))
        pprint.pprint(design_fname_list, stream=open("bag_eval_run.txt", "a"))
        pprint.pprint("-"*30+"//dtsa_results:", stream=open("bag_eval_run.txt", "a"))
        pprint.pprint(dtsa_results, stream=open("bag_eval_run.txt", "a"))

        for dtsa_index in sorted(indices_to_be_removed, reverse=True):
            del design_fname_list[dtsa_index]

        pprint.pprint("-"*30+"//yaml_files that passed DTSA", stream=open("bag_eval_run.txt", "a"))
        pprint.pprint(design_fname_list, stream=open("bag_eval_run.txt", "a"))

        if design_fname_list:
            specs['sweep_params']['swp_spec_file'] = design_fname_list
            with open(self.top_level_main_file, 'w') as f:
                yaml.dump(specs, f)

            top_level_results = self.generate_and_sim_ckt(self.top_level_main_file)
        else:
            top_level_results = None

        pprint.pprint("-"*30+"//top_level_results", stream=open("bag_eval_run.txt", "a"))
        pprint.pprint(top_level_results, stream=open("bag_eval_run.txt", "a"))

        top_level_index = 0
        for dtsa_index in range(len(dtsa_results)):
            if isinstance(dtsa_results[dtsa_index], Exception):
                result = Exception('dtsa has a phase 1 error')
            else:
                result = {
                    'dtsa/v_charge':dtsa_results[dtsa_index]['comparator']['v_charge'],
                    'dtsa/v_reset':dtsa_results[dtsa_index]['comparator']['v_reset'],
                    'dtsa/v_out':dtsa_results[dtsa_index]['comparator']['v_out'],
                    'dtsa/ibias':dtsa_results[dtsa_index]['comparator']['ibias'],
                    'dtsa/sigma':dtsa_results[dtsa_index]['comparator']['sigma'],
                    'dtsa/offset':dtsa_results[dtsa_index]['comparator']['offset'],
                }
                if isinstance(top_level_results[top_level_index], Dict):
                    top_level_result = top_level_results[top_level_index]['photonic_link_AFE_rx']
                    new_result = {
                        'afe/cmrr': top_level_result['cmrr'],
                        'afe/ibias': top_level_result['ibias'],
                        'afe/r_afe': top_level_result['r_afe'],
                        'afe/f3db': top_level_result['f3db'],
                        'afe/rms_input_noise': top_level_result['rms_input_noise'],
                        'afe/rms_output_noise': top_level_result['rms_output_noise'],
                        # 'afe/tset': top_level_result['tset'],
                        'afe/eye_height': top_level_result['eye_height'],
                        'afe/eye_level_thickness_ratio': top_level_result['eye_level_thickness_ratio'],
                        'afe/eye_span_height':top_level_result['eye_span_height'],
                    }
                    result.update(**new_result)
                    # numpy array / List operations
                    n1 = np.array(result['dtsa/sigma'])
                    n2 = np.array(result['afe/rms_output_noise'])
                    result['top/tot_noise'] = np.sqrt(n1**2+n2**2).tolist()
                    min_sense = (9 * np.array(result['top/tot_noise']) + 0.001 + np.array(result['dtsa/offset'])) / np.array(result['afe/r_afe'])
                    result['top/min_sense'] = min_sense.tolist()
                    result['top/ibias'] = (np.array(result['afe/ibias']) + np.array(result['dtsa/ibias'])).tolist()
                else:
                    result = Exception('AFE has a phase 1 error')
                top_level_index += 1

            results.append(result)

        return results

    def get_worst_specs(self):
        result = dict()
        for kwrd in self.spec_range.keys():
            if kwrd.startswith('dtsa'):
                continue
            spec_min, spec_max, _ = self.spec_range[kwrd]
            if spec_max is None:
                result[kwrd] = 0
            else:
                result[kwrd] = 10*spec_max

        return result


    def impose_constraints(self, design_dict):
        # constraints:
        #   seg_dict and w_dict for transistors should be integer
        #   nser, npar, ndum for resistors should be integer
        #   l, w for resistors min and max are 0.5u and 50u respectively
        #   imposed by the parameter vector in yaml file

        cap_w, cap_h = (1, 1)
        aratio = 3
        for kwrd in design_dict.keys():
            step_4_indicator = self.decend(kwrd, step=4)
            step_3_indicator = self.decend(kwrd, step=3)
            step_2_indicator = self.decend(kwrd, step=2)
            if step_3_indicator.startswith('seg_dict') or step_3_indicator.startswith(
                    'w_dict') or step_3_indicator.startswith('n'):
                design_dict[kwrd] = int(design_dict[kwrd])
            if step_2_indicator.startswith('seg_dict') or step_2_indicator.startswith(
                    'w_dict') or step_2_indicator.startswith('n'):
                design_dict[kwrd] = int(design_dict[kwrd])
            if step_4_indicator.startswith('seg_dict') or step_4_indicator.startswith(
                    'w_dict') or step_4_indicator.startswith('n'):
                design_dict[kwrd] = int(design_dict[kwrd])
            # ctle_cap_w and ctle_cap_h constraint
            if kwrd == 'layout_params/tia_plus_ctle_params/ctle_params/cap_params/width':
                cap_w = design_dict[kwrd]
            elif kwrd == 'layout_params/tia_plus_ctle_params/ctle_params/cap_params/height':
                cap_h = design_dict[kwrd]

        if (cap_w / cap_h) > aratio:
            design_dict['layout_params/tia_plus_ctle_params/ctle_params/cap_params/width'] = aratio * design_dict['layout_params/tia_plus_ctle_params/ctle_params/cap_params/height']
        elif (cap_h / cap_w) > aratio:
            design_dict['layout_params/tia_plus_ctle_params/ctle_params/cap_params/height'] = aratio * design_dict['layout_params/tia_plus_ctle_params/ctle_params/cap_params/width']

        return design_dict

    def process_results(self, results):
        processed_results = []
        for result in results:
            processed_result = {'valid': True}
            cost = 0
            if isinstance(result, Dict):
                # result = result['photonic_link_AFE_rx']

                for spec in self.spec_range.keys():
                    processed_result[spec] = self.find_worst(result[spec], spec)
                    penalty = self.compute_penalty(processed_result[spec], spec)[0]
                    cost += penalty

                processed_result['cost'] = cost
            else:
                processed_result['valid'] = False

            processed_results.append(processed_result)

        return processed_results

if __name__ == '__main__':
    import pickle
    np.random.seed(10)
    random.seed(10)

    # fname = 'specs_design/AFE_DTSA.yaml'
    fname = 'specs_design/AFE_top.yaml'
    evalEngine = AFE_CMP_EvaluationEngine(design_specs_fname=fname)
    content = read_yaml(fname)
    dir = content['database_dir']

    start = time.time()
    sample_designs = evalEngine.generate_data_set(n=150, evaluate=True)
    print("time: {}".format(time.time() - start))
    os.makedirs(dir, exist_ok=True)

    data = sorted(sample_designs, key=lambda x:x.cost)

    with open(dir+"/init_data.pickle", 'wb') as f:
        pickle.dump(data, f)

    import pdb
    pdb.set_trace()

