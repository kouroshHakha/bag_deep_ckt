from typing import Dict, Any
from eval_engines.src.eval_engines.circuits import EvaluationEngine, SubEngine
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import numpy as np
import pdb


class CSMeasMan(EvaluationEngine):

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)


    def get_specs(self, results_dict, params):
        _, specs_dict, _ = results_dict['ac_dc']
        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if not hasattr(spec_nums, '__iter__'):
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            spec_min, spec_max, w = self.spec_range[spec_kwrd]
            if spec_max is not None:
                if spec_num > spec_max:
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
            if spec_min is not None:
                if spec_num < spec_min:
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties



class ACTB(SubEngine):

    @classmethod
    def process(cls, results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        ac_result = results['ac']
        dc_results = results['dcOp']

        vout = ac_result['out']
        freq = ac_result['sweep_values']

        gain = cls.find_dc_gain(vout)
        bw = cls.find_bw(vout, freq)

        results = dict(
            gain=gain,
            bw=bw
        )
        return results

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_bw(cls, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return cls._get_best_crossing(freq, gain, gain_3dB)

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':

    set_seed(10)
    yname = 'bag_deep_ckt/eval_engines/spectre/specs_test/common_source.yaml'
    eval_core = CSMeasMan(yname)

    designs = eval_core.generate_data_set(n=100, debug=False)
    pdb.set_trace()