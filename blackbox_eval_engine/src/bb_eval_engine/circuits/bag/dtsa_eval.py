import os

from bb_eval_engine.src.eval_engines.circuits.bag.bagEvalEngine import BagEvalEngine
from bag.io import read_yaml
from bag_deep_ckt.util import *


class DTSAEvaluationEngine(BagEvalEngine):
    def __init__(self, design_specs_fname):
        BagEvalEngine.__init__(self, design_specs_fname)

    def impose_constraints(self, design_dict):
        # no constraints for this layout generator so far (except for fingers being even)
        return design_dict

    def process_results(self, results):
        # TODO make it work across corners
        processed_results = []

        for result in results:
            processed_result = {'valid': True}
            cost = 0
            if isinstance(result, Dict):
                result = result['overdrive']
                for spec in self.spec_range.keys():
                    processed_result[spec] = self.find_worst(result[spec], spec)
                    penalty = self.compute_penalty(processed_result[spec], spec)[0]
                    cost += penalty
                processed_result['cost'] = cost
            else:
                processed_result['valid'] = False

            processed_results.append(processed_result)
        return processed_results

    # helper (optional) functions for process_results
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

if __name__ == '__main__':
    import pickle
    evalEngine = DTSAEvaluationEngine(design_specs_fname='specs_design/DTSA.yaml')
    content = read_yaml('specs_design/DTSA.yaml')
    dir = content['database_dir']

    np.random.seed(10)
    random.seed(10)

    start = time.time()
    sample_designs = evalEngine.generate_data_set(n=1, evaluate=True)
    print("time: {}".format(time.time() - start))
    os.makedirs(dir, exist_ok=True)
    with open(dir+"/init_data.pickle", 'wb') as f:
        pickle.dump(sample_designs, f)

    with open(dir+"/init_data.pickle", 'rb') as f:
        data = pickle.load(f)
        se = [x.cost for x in data]
        se = sorted(se, key=lambda x:x)
        a = 1/0