import math
import time

from eval_engines.BAG.bagEvalEngine import BagEvalEngine
from bag.io import read_yaml
from bag_deep_ckt.util import *


class OpampEvaluationEngine(BagEvalEngine):
    def __init__(self, design_specs_fname):
        BagEvalEngine.__init__(self, design_specs_fname)
        self.ver_specs['measurements'][0]['find_cfb'] = False

    # def generate_data_set(self, n=1, evaluate=False):
    #
    #     designs = []
    #
    #     for _ in range(n):
    #         design = {}
    #         for key, value in self.params.items():
    #             rand_idx = random.randrange(len(self.params_vec[key]))
    #             # rand_value = self.params_vec[key][rand_idx]
    #             rand_value = rand_idx
    #             design[key] = rand_value
    #         # Imposing the constraints of layout generator
    #         design['tail1'] = design['in']
    #         design = Design(self.spec_range, list(design.values()))
    #         designs.append(design)
    #
    #     if evaluate:
    #         design_results = self.evaluate(designs)
    #         for i, design in enumerate(designs):
    #             design_result = design_results[i]
    #             if design_result['valid']:
    #                 design.cost = design_result['cost']
    #                 for key in design.specs.keys():
    #                     design.specs[key] = design_result[key]
    #     return designs


    # def generate_and_sim(self):
    #     """
    #     phase 1 of evaluation is generation of layout, schematic, LVS and RCX
    #     If any of LVS or RCX fail results_ph1 will contain Exceptions for the corresponding instance
    #     We proceed to phase 2 only if phase 1 was successful.
    #     phase 2 is running the simulation with post extracted netlist view
    #     Then we aggregate the results of phase 1 and phase 2 in a single list, in the same order
    #     that designs were ordered, if phase 1 was failed the corresponding entry will contain
    #     a Phase1Error exception
    #     """
    #     results = []
    #     with open_file(self.sim_specs_fname, 'w') as f:
    #         yaml.dump(self.ver_specs, f)
    #
    #     sim = DesignManager(self.bprj, self.sim_specs_fname)
    #     results_ph1 = sim.characterize_designs(generate=True, measure=False, load_from_file=False)
    #
    #     # hacky: do parallel measurements, you should not sweep anything other than 'swp_spec_file' in sweep_params
    #     # the new yaml files themselves should not include any sweep_param
    #     start = time.time()
    #     impl_lib = self.ver_specs['impl_lib']
    #     coro_list = []
    #     file_list = self.ver_specs['sweep_params']['swp_spec_file']
    #     for i, combo_list in enumerate(sim.get_combinations_iter()):
    #         dsn_name = sim.get_design_name(combo_list)
    #         specs_fname = os.path.join(self.swp_spec_dir, file_list[i] + '.yaml')
    #         if isinstance(results_ph1[i], Exception):
    #             continue
    #         coro_list.append(self.async_characterization(impl_lib, dsn_name, specs_fname))
    #
    #     results_ph2 = batch_async_task(coro_list)
    #     print("sim time: {}".format(time.time() - start))
    #     # this part returns the correct order of results if some of the instances failed phase1 of evaluation
    #     j = 0
    #     for i, combo_list in enumerate(sim.get_combinations_iter()):
    #         if isinstance(results_ph1[i], Exception):
    #             results.append(Phase1Error)
    #         else:
    #             results.append(results_ph2[j])
    #             j+=1
    #     pprint.pprint(results)
    #
    #     return results
    #
    # async def async_characterization(self, impl_lib, dsn_name, specs_fname):
    #     sim = DesignManager(self.bprj, specs_fname)
    #     pprint.pprint(specs_fname)
    #     await sim.verify_design(impl_lib, dsn_name, load_from_file=False)
    #     # print('name: {}'.format(dsn_name))
    #     # dsn_name = list(sim.get_dsn_name_iter())[0]
    #     summary = sim.get_result(dsn_name)['opamp_ac']
    #     return summary

    def impose_constraints(self, design_dict):
        design_dict['layout_params/seg_dict/tail1'] = design_dict['layout_params/seg_dict/in']
        return design_dict

    def process_results(self, results):
        # results = [{'funity': [blah, blah], 'pm': [blah, blah], 'gain': [blah, blah], 'corners', 'bw' }]
        # processed_results = [{'cost': blah, 'funity': blah, 'pm': blah}]
        processed_results = []
        for result in results:
            processed_result = {'valid': True}
            cost = 0
            if isinstance(result, Dict):
                result = result['opamp_ac']
                # just a couple of exception for definition of phase margin and gain in wierd cases
                # fow example when gain is less than unity, or when phase or gain are nan
                # in ac core pm is returned as nan if pm is larger than 180
                for i in range(len(result['funity'])):
                    if result['gain'][i] < 1.0:
                        result['pm'][i] = -180
                    if math.isnan(result['funity'][i]):
                        result['funity'][i] = 0
                    if math.isnan(result['pm'][i]):
                        result['pm'][i] = -180

                for spec in self.spec_range.keys():
                    processed_result[spec] = self.find_worst(result[spec], spec)
                    penalty = self.compute_penalty(processed_result[spec], spec)[0]
                    cost += penalty

                processed_result['cost'] = cost
            else:
                processed_result['valid'] = False

            processed_results.append(processed_result)

        return processed_results

    def find_worst(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]

        spec_min, spec_max = self.spec_range[spec_kwrd]
        if spec_min is not None:
            return min(spec_nums)
        if spec_max is not None:
            return max(spec_nums)

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    # penalty += abs(spec_num / spec_max - 1.0)
                    penalty += abs((spec_num - spec_max) / (spec_num + spec_max))
            if spec_min is not None:
                if spec_num < spec_min:
                    # penalty += abs(spec_num / spec_min - 1.0)
                    penalty += abs((spec_num - spec_min) / (spec_num + spec_min))
            penalties.append(penalty)
        return penalties

    # def evaluate(self, design_list):
    #     # type: (List[Design]) -> List
    #
    #     swp_spec_file_list = []
    #     template = deepcopy(self.ver_specs)
    #     sweep_params_update = deepcopy(self.ver_specs['sweep_params'])
    #     # del template['sweep_params']['swp_spec_file']
    #     for dsn_num, design in enumerate(design_list):
    #         # 1. translate each list to a dict with layout_params and measurement_params indication
    #         # 2. write those dictionaries in the corresponding param.yaml and update self.ver_specs
    #         specs = deepcopy(template)
    #         layout_update, measurement_update = {}, {}
    #         layout_update['seg_dict'] = {}
    #         for value_idx, key in zip(design, self.params.keys()):
    #             if key in self.param_choices_layout.keys():
    #                 layout_update['seg_dict'][key] = self.params_vec[key][value_idx]
    #             elif key in self.param_choices_measurement.keys():
    #                 measurement_update[key] = self.params_vec[key][value_idx]
    #
    #         # imposing the constraint of layout generator
    #         layout_update['seg_dict']['tail1'] = layout_update['seg_dict']['in']
    #
    #         specs['layout_params'].update(layout_update)
    #         specs['measurements'][0].update(measurement_update)
    #         specs['sweep_params']['swp_spec_file'] = ['params'+str(dsn_num)]
    #
    #         swp_spec_file_list.append('params'+str(dsn_num))
    #         fname = os.path.join(self.swp_spec_dir, 'params'+str(dsn_num)+'.yaml')
    #         with open_file(fname, 'w') as f:
    #             yaml.dump(specs, f)
    #
    #     sweep_params_update['swp_spec_file'] = swp_spec_file_list
    #     self.ver_specs['sweep_params'].update(sweep_params_update)
    #     results = self.generate_and_sim()
    #     return self.process_results(results)

def main():

    eval_core = OpampEvaluationEngine(design_specs_fname='specs_design/opamp_two_stage_1e8.yaml')
    content = read_yaml('specs_design/opamp_two_stage_1e8.yaml')
    db_dir = content['database_dir']

    np.random.seed(10)
    random.seed(10)

    start = time.time()
    sample_designs = eval_core.generate_data_set(n=10, evaluate=True)
    print("time for simulating one instance: {}".format((time.time() - start)))

    import pickle
    with open(db_dir+"/init_data.pickle", 'wb') as f:
        pickle.dump(sample_designs, f)

    with open(db_dir+"/init_data.pickle", 'rb') as f:
        data = pickle.load(f)
        a=1/0

    # designs = [[6, 6, 8, 8, 0, 3, 4, 8, 4, 31, 0]] # this design had a wierd phase behavior
    # designs = [[8, 6, 0, 8, 2, 0, 6, 7, 7, 21, 5]] # this design's funity for ff was nan
    # designs = [[8, 6, 0, 8, 2, 0, 6, 7, 7, 21, 5]]
    # results = eval_core.evaluate(designs)
    # fname = 'kourosh_test_opamp/swp_spec_files/param0.yaml'
    # data = read_yaml(fname)
    # pprint.pprint(data)

if __name__ == '__main__':
    main()
