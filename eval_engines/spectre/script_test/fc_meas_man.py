from bag_deep_ckt.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import pdb
import IPython
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import math
class FCMeasMan(EvaluationEngine):

    def __init__(self, yaml_fname):
        EvaluationEngine.__init__(self, yaml_fname)

    def get_specs(self, results_dict, params):
        specs_dict = dict()
        ac_dc = results_dict['ac_dc']
        for _, res, _ in ac_dc:
            specs_dict = res
        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        if type(spec_nums) is not list:
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

class TRTBNOISE(object):

    @classmethod
    def process_tr_noise(cls, results, params):
        max_freq = 1.0e+7

        noise_result = results['noise']['in']
        freq = results['noise']['sweep_values']
        intnoise = cls.find_intnoise(freq, noise_result)

        results = dict(
            noise=intnoise
        )
        return results

    @classmethod
    def find_intnoise(self,freq, noise):
      max_freq = 1.0e+7
      freq_arr = freq[freq < max_freq]
      noise = np.array(noise)
      noise_fr = noise[freq < max_freq]
      intnoise = math.sqrt(np.trapz(noise_fr**2, freq_arr))
      return intnoise 

class TRTBSWING(object):

    @classmethod
    def process_tr_swing(cls, results, params):
        tr_result = results['tran.tran']

        vout = tr_result['Voutac']
        vinac = tr_result['Vinac']
        t = tr_result['sweep_values']

        vswing = cls.find_swing(t, vout)

        results = dict(
            vswing=vswing
        )
        return results

    @classmethod
    def find_swing(self, time, vout):
      max_vol = max(vout)
      min_vol = min(vout)
      v_swing = max_vol - min_vol
      return v_swing 

class TRTBSS(object):

    @classmethod
    def process_tr_ss(cls, results, params):
        tr_result = results['tran.tran']

        vout = tr_result['Voutac']
        vinac = tr_result['Vinac']
        t = tr_result['sweep_values']

        settling_time = cls.find_tset(t, vout, vinac, fbck=1.0)

        results = dict(
            tset=settling_time
        )
        return results

    @classmethod
    def find_tset(self, time, vout, vin, fbck, tot_err=0.1, plot=False):
      vin_rise = []
      #for i,vol in enumerate(vin):
      #  if (vol <= 1.0):
      #    vin_rise.append(vol)
      #    if vin[i+1] <= vol:
            #vin_rise.append(vin[i+1])
      #      break

      #IPython.embed()
      #vin_rise = np.array(vin_rise)
      #t = time[:i+1]
      #vout = vout[:i+1]
     
      vin_rise_norm = (vin-vin[0])/(vin[-1]-vin[0])
      ref_value = 1/fbck * vin
      y = (vout-vout[0])/(ref_value[-1]-ref_value[0])
      
      if plot:
        import matplotlib.pyplot as plt
        plt.plot(time, vin_rise_norm/fbck)
        plt.plot(time,y)
        plt.figure()
        plt.plot(time,vout)
        plt.plot(time,vin_rise)
      last_idx = np.where(y < 1.0 - tot_err)[0][-1]
      last_max_vec = np.where(y > 1.0 + tot_err)[0]
      if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
        last_idx = last_max_vec[-1]
        last_val = 1.0 + tot_err
      else:
        last_val = 1.0 - tot_err

      if last_idx == time.size - 1:
        return time[-1]
      else:
        f = interp.InterpolatedUnivariateSpline(time, y-last_val)
        t0 = time[last_idx]
        t1 = time[last_idx + 1]
        return sciopt.brentq(f, t0, t1)

class TRTB(object):
    @classmethod
    def process_tr(cls, results, params):
        tr_result = results['tran.tran']

        vout = tr_result['Voutac']
        vinac = tr_result['Vinac']
        t = tr_result['sweep_values']

        slew_rate = cls.find_slew_rate(t,vout,vinac)

        results = dict(
            sr=slew_rate
        )
        return results

    @classmethod
    def find_slew_rate (self, time, vout, vinac):
        for i,each_vout in enumerate(vout):
            if (vinac[i]==1.0) and (vout[i+1] - each_vout < 0.001):
              break
        rise_arr = vout[0:i]
        per_rise_80 = 0.8*max(rise_arr)
        per_rise_20 = 0.2*max(rise_arr)
        sr_func = interp.interp1d(time[0:i], rise_arr, kind='quadratic')
        t_rise_20 = self._get_best_crossing(time[0:i],rise_arr, val=per_rise_20)
        t_rise_80 = self._get_best_crossing(time[0:i],rise_arr, val=per_rise_80)
        v_rise_20 = sr_func(t_rise_20[0])
        v_rise_80 = sr_func(t_rise_80[0])

        if (t_rise_80[0]-t_rise_20[0]) == 0:
          #print(rise_arr)
          #print(time[0:i])
          SR_rise = 0.0
        else:
          SR_rise = abs((v_rise_80-v_rise_20)/(10**6*(t_rise_80[0]-t_rise_20[0])))
      
        '''
        for i,each_vout in enumerate(vout):
          if each_vout > find_max:
            find_max = each_vout
            print('here')
            if vout[i+1] < find_max:
              'breaking here'
              break

        #shifting array up so that there is shift upward, to easily allow max/min calc
        rise_arr = vout[np.argmax(vinac<1.0)-1:]
        time = time[np.argmax(vinac<1.0)-1:]

        true_min = abs(rise_arr[0])
        rise_arr = rise_arr + true_min

        per_rise_80 = 0.8*max(rise_arr)
        per_rise_20 = 0.2*max(rise_arr)

        sr_func = interp.interp1d(time, rise_arr, kind='quadratic')
        t_rise_20 = self._get_best_crossing(time,rise_arr, val=per_rise_20)
        t_rise_80 = self._get_best_crossing(time,rise_arr, val=per_rise_80)
        v_rise_20 = sr_func(t_rise_20[0])-true_min
        v_rise_80 = sr_func(t_rise_80[0])-true_min

        SR_rise = abs((v_rise_80-v_rise_20)/(10**6*(t_rise_80[0]-t_rise_20[0])))
        '''
        #if SR_rise < SR_fall:
        return SR_rise
        #else:
        #  return SR_rise

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

class ACTB(object):

    @classmethod
    def process_ac(cls, results, params):
        ac_result = results['ac']

        vout = ac_result['Voutac']
        freq = ac_result['sweep_values']

        gain = cls.find_dc_gain(vout)
        ugbw,valid = cls.find_ugbw(freq, vout)
        phm = cls.find_phm(freq, vout)
        ibias = -1*results['dcOp']['V2:p']

        results = dict(
            gain=gain,
            funity = ugbw,
            pm = phm,
            ibias = ibias,
            valid = valid
        )
        return results

    @classmethod
    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw, valid
        else:
            return freq[0], valid

    @classmethod
    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees
        #
        #plt.subplot(211)
        #plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        #plt.subplot(212)
        #plt.plot(np.log10(freq[:200]), phase)
        #IPython.embed()
        #plt.show()

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
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

if __name__ == '__main__':

    yname = 'bag_deep_ckt/eval_engines/spectre/specs_test/folded_cascode.yaml'
    eval_core = OpampMeasMan(yname)

    designs = eval_core.generate_data_set(n=1)
