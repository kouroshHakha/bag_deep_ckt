"""
This is an example of using ngspice.netlist.NgSpiceWrapper for a cs_amp ac and dc simulation.
Look at the cs_amp.cir as well to understand the parser.
The parsing happens here depending on the format of netlist Template.
"""

from typing import Mapping, Any, Sequence

from pathlib import Path
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, PathLike, StateValue
from bb_eval_engine.circuits.ngspice.engine import NgspiceEngineBase
from bb_eval_engine.circuits.ngspice.flow_manager import FlowManager
from bb_eval_engine.util.design import Design

class CsAmpClass(NgSpiceWrapper):

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

    def parse_output(self, state):

        ac_fname = Path(state['ac_path'])
        dc_fname = Path(state['dc_path'])

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

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


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



class CSAmpFlow(FlowManager):

    def __init__(self, netlist, **kwargs):

        num_process = kwargs.get('num_workers', 1)
        print(num_process)
        self.verbose = kwargs.get('verbose', False)
        self.ngspice_wrapper = CsAmpClass(num_process, netlist)


    def batch_evaluate(self, batch_of_designs: Sequence[Mapping[str, Any]], *args, **kwargs):

        raw_results = self.ngspice_wrapper.run(batch_of_designs, verbose=self.verbose)
        results = [res[1] for res in raw_results]

        return results


class CSAmpEvalEngine(NgspiceEngineBase):

    def interpret(self, design: Design):
        # keep the default
        params_dict = NgspiceEngineBase.interpret(self, design)
        netlist_path = Path(__file__).parent / 'netlist_temp' / 'ngspice_models' / '45nm_bulk.txt'
        wrapper = self.flow_manager.ngspice_wrapper
        id = design.id(self.id_encoder)
        params_dict['ac_path'] = str(wrapper.get_design_folder(id) / 'ac.csv')
        params_dict['dc_path'] = str(wrapper.get_design_folder(id) / 'dc.csv')
        params_dict['include'] = netlist_path.resolve()
        params_dict['id'] = id

        return params_dict
