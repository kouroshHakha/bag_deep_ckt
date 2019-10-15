"""
This is an example of using blackbox_eval_engine setup for a two_stage_opamp simulations.
It includes open-loop ac, transient, power supply rejection and common mode testbenches.
"""

from typing import Mapping, Any, Sequence

from pathlib import Path
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue
from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.util.design import Design

class TwoStageOpenLoop(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:

        # use parse output here
        freq, vout,  ibias = self.parse_output(state)
        gain = self.find_dc_gain(vout)
        ugbw = self.find_ugbw(freq, vout)
        phm = self.find_phm(freq, vout)


        spec = dict(
            ugbw=ugbw,
            gain=gain,
            phm=phm,
            Ibias=ibias
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
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_ugbw(cls, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = cls._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw
        else:
            return freq[0]

    @classmethod
    def find_phm(cls, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees

        # plt.subplot(211)
        # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # plt.subplot(212)
        # plt.plot(np.log10(freq[:200]), phase)

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = cls._get_best_crossing(freq, gain, val=1)
        if valid:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            return -180

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False


class TwoStageCommonModeGain(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:

        # use parse output here
        freq, vout = self.parse_output(state)
        gain = self.find_dc_gain(vout)


        spec = dict(
            cm_gain=gain,
        )

        return spec

    @classmethod
    def parse_output(cls, state):
        cm_fname = Path(state['cm'])

        if not cm_fname.is_file():
            print(f"cm file doesn't exist: {cm_fname}")

        cm_raw_outputs = np.genfromtxt(cm_fname, skip_header=1)
        freq = cm_raw_outputs[:, 0]
        vout_real = cm_raw_outputs[:, 1]
        vout_imag = cm_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return freq, vout

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]


class TwoStagePowerSupplyGain(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:

        # use parse output here
        freq, vout = self.parse_output(state)
        gain = self.find_dc_gain(vout)


        spec = dict(
            ps_gain=gain,
        )

        return spec

    @classmethod
    def parse_output(cls, state):
        ps_fname = Path(state['ps'])


        if not ps_fname.is_file():
            print(f"ps file doesn't exist: {ps_fname}")

        ps_raw_outputs = np.genfromtxt(ps_fname, skip_header=1)
        freq = ps_raw_outputs[:, 0]
        vout_real = ps_raw_outputs[:, 1]
        vout_imag = ps_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return freq, vout

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]

class TwoStageTransient(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:

        # use parse output here
        time, vout, vin = self.parse_output(state)
        # vout_norm = vout/vout[-1]
        # settling_time = self.get_tset(time, vout_norm, tot_err=0.01)


        spec = dict(
            time=time,
            vout=vout,
            vin=vin
        )

        return spec


    @classmethod
    def parse_output(cls, state):

        tran_fname = Path(state['tran'])

        if not tran_fname.is_file():
            print(f"ac file doesn't exist: {tran_fname}")

        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        time =  tran_raw_outputs[:, 0]
        vout =  tran_raw_outputs[:, 1]
        vin =   tran_raw_outputs[:, 3]

        return time, vout, vin


    @classmethod
    def get_tset(cls, t, vout, vin, fbck, tot_err=0.1, plt=False):

        # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here
        vin_norm = (vin-vin[0])/(vin[-1]-vin[0])
        ref_value = 1/fbck * vin
        y = (vout-vout[0])/(ref_value[-1]-ref_value[0])

        if plt:
            import matplotlib.pyplot as plt
            plt.plot(t, vin_norm/fbck)
            plt.plot(t, y)
            plt.figure()
            plt.plot(t, vout)
            plt.plot(t, vin)

        last_idx = np.where(y < 1.0 - tot_err)[0][-1]
        last_max_vec = np.where(y > 1.0 + tot_err)[0]
        if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
            last_idx = last_max_vec[-1]
            last_val = 1.0 + tot_err
        else:
            last_val = 1.0 - tot_err

        if last_idx == t.size - 1:
            return t[-1]
        f = interp.InterpolatedUnivariateSpline(t, y - last_val)
        t0 = t[last_idx]
        t1 = t[last_idx + 1]
        return sciopt.brentq(f, t0, t1)


class TwoStageFlow(NgspiceFlowManager):

    def __init__(self, *args, **kwargs):
        NgspiceFlowManager.__init__(self, *args, **kwargs)
        self.fb_factor = kwargs['feedback_factor']
        self.tot_err = kwargs['tot_err']

    def interpret(self, design: Design, *args, **kwargs) -> Mapping[str, Any]:

        mode = args[0]
        params_dict = design['value_dict']

        if mode == 'ol':
            path_vars = ['ac', 'dc']
        elif mode == 'cm':
            path_vars = ['cm']
        elif mode == 'ps':
            path_vars = ['ps']
        elif mode == 'tran':
            path_vars = ['tran']
        else:
            raise ValueError('invalid mode!')

        params_dict.update(self.update_netlist_model_paths(design, path_vars, name=mode))

        return params_dict

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):


        interpreted_designs = [self.interpret(design, 'ol') for design in batch_of_designs]
        raw_results = self.ngspice_lut['ol'].run(interpreted_designs, verbose=self.verbose)
        results_ol = [res[1] for res in raw_results]

        interpreted_designs = [self.interpret(design, 'cm') for design in batch_of_designs]
        raw_results = self.ngspice_lut['cm'].run(interpreted_designs, verbose=self.verbose)
        results_cm = [res[1] for res in raw_results]

        interpreted_designs = [self.interpret(design, 'ps') for design in batch_of_designs]
        raw_results = self.ngspice_lut['ps'].run(interpreted_designs, verbose=self.verbose)
        results_ps = [res[1] for res in raw_results]

        interpreted_designs = [self.interpret(design, 'tran') for design in batch_of_designs]
        raw_results = self.ngspice_lut['tran'].run(interpreted_designs, verbose=self.verbose)
        results_tran = [res[1] for res in raw_results]

        results = []

        for ol, cm, ps, tran in zip(results_ol, results_cm, results_ps, results_tran):
            results.append(self._get_specs(ol, cm, ps, tran))

        return results

    def _get_specs(self, result_ol, result_cm, result_ps, result_tran):
        fdbck = self.fb_factor
        tot_err = self.tot_err

        ugbw_cur = result_ol['ugbw']
        gain_cur = result_ol['gain']
        phm_cur = result_ol['phm']
        ibias_cur = result_ol['Ibias']

        # common mode gain and cmrr
        cm_gain_cur = result_cm['cm_gain']
        cmrr_cur = 20 * np.log10(gain_cur / cm_gain_cur)  # in db
        # power supply gain and psrr
        ps_gain_cur = result_ps['ps_gain']
        psrr_cur = 20 * np.log10(gain_cur / ps_gain_cur)  # in db

        # transient settling time and offset calculation
        t = result_tran['time']
        vout = result_tran['vout']
        vin = result_tran['vin']

        tset_cur = TwoStageTransient.get_tset(t, vout, vin, fdbck, tot_err=tot_err)
        offset_curr = abs(vout[0] - vin[0] / fdbck)

        specs_dict = dict(
            gain=gain_cur,
            ugbw=ugbw_cur,
            pm=phm_cur,
            ibias=ibias_cur,
            cmrr=cmrr_cur,
            psrr=psrr_cur,
            offset_sys=offset_curr,
            tset=tset_cur,
        )

        return specs_dict