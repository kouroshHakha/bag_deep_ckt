"""
This is an example of using blackbox_eval_engine setup for a cs_amp ac and dc simulation.
Look at the cs_amp.cir as well to understand the parser.
The parsing happens here depending on the format of netlist Template.
"""

from typing import Mapping, Any, Sequence

from pathlib import Path
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue
from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.util.design import Design


class CsAmpNgspiceWrapper(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:

        # use parse output here
        freq, vout, ibias = self.parse_output(state)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)

        spec = dict(
            bw=bw,
            gain=gain,
            ibias=ibias
        )

        return spec

    @classmethod
    def parse_output(cls, state):

        ac_fname = Path(state['ac'])
        dc_fname = Path(state['dc'])

        if not ac_fname.is_file():
            print(f"ac file doesn't exist: {ac_fname}")
        if not dc_fname.is_file():
            print(f"ac file doesn't exist: {dc_fname}")

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout = ac_raw_outputs[:, 1]
        ibias = - dc_raw_outputs[1]

        return freq, vout, ibias

    @classmethod
    def find_dc_gain(cls, vout):
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


class CSAmpFlow(NgspiceFlowManager):

    def interpret(self, design: Design, *args, **kwargs) -> Mapping[str, Any]:
        params_dict = design['value_dict']
        params_dict.update(self.update_netlist_model_paths(design, ['ac', 'dc'], name='ac_dc'))

        return params_dict

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):
        interpreted_designs = [self.interpret(design) for design in batch_of_designs]
        raw_results = self.ngspice_lut['ac_dc'].run(interpreted_designs, verbose=self.verbose)
        results = [res[1] for res in raw_results]

        return results
