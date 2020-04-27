from bag_deep_ckt.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import pdb
import IPython
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

class fc_biasing_man(EvaluationEngine):

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)

    def get_specs(self, results_dict, params):
        specs_dict = dict()
        ac_dc = results_dict['ac_dc']
        for _, res, _ in ac_dc:
            specs_dict = res
        return specs_dict
       
class DCTB(object):

    @classmethod
    def process_dc(cls, results, params):
        dc_result = results['dcOp']
        vbiasp1,vbiasp2,vbiasn1,vbiasn2 = cls.find_op(dc_result)

        results = dict(
            vbiasp1 = vbiasp1,
            vbiasp2 = vbiasp2,
            vbiasn1 = vbiasn1,
            vbiasn2 = vbiasn2
        )
        return results

    @classmethod
    def find_op(self, dc_result):
        vbiasp1 = dc_result['vbiasp1']
        vbiasp2 = dc_result['vbiasp2']
        vbiasn1 = dc_result['vbiasn1']
        vbiasn2 = dc_result['vbiasn2']

        return vbiasp1, vbiasp2, vbiasn1, vbiasn2

if __name__ == '__main__':

    yname = 'bag_deep_ckt/eval_engines/spectre/specs_test/folded_cascode.yaml'
    eval_core = OpampMeasMan(yname)

    designs = eval_core.generate_data_set(n=1)
