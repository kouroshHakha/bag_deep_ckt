import os

from eval_engines.src.eval_engines.circuits.bag.bagEvalEngine import BagEvalEngine
from bag.io import read_yaml
from bag_deep_ckt.util import *


class TIACTLEEvaluationEngine(BagEvalEngine):
    def __init__(self, design_specs_fname):
        BagEvalEngine.__init__(self, design_specs_fname)

    def impose_constraints(self, design_dict):
        # constraints:
        #   seg_dict and w_dict for transistors should be integer
        #   nser, npar, ndum for resistors should be integer
        #   l, w for resistors min and max are 0.5u and 50u respectively
        #   imposed by the parameter vector in yaml file

        for kwrd in design_dict.keys():
            step_3_indicator = self.decend(kwrd, step=3)
            if step_3_indicator.startswith('seg_dict') or step_3_indicator.startswith(
                    'w_dict') or step_3_indicator.startswith('n'):
                design_dict[kwrd] = int(design_dict[kwrd])

        return design_dict

    def process_results(self, results):
        processed_results = []
        for result in results:
            processed_result = {'valid': True}
            cost = 0
            if isinstance(result, Dict):
                result = result['noise']

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

    fname = 'specs_design/TIA_CTLE.yaml'
    evalEngine = TIACTLEEvaluationEngine(design_specs_fname=fname)
    content = read_yaml(fname)
    dir = content['database_dir']

    start = time.time()
    sample_designs = evalEngine.generate_data_set(n=150, evaluate=True)
    print("time: {}".format(time.time() - start))
    os.makedirs(dir, exist_ok=True)
    with open(dir+"/init_data.pickle", 'wb') as f:
        pickle.dump(sample_designs, f)

    with open(dir+"/init_data.pickle", 'rb') as f:
        data = pickle.load(f)
        se = [x.cost for x in data]
        se = sorted(se, key=lambda x:x)
        a = 1/0
