import gym
from gym import spaces

import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import itertools
from bag_deep_ckt.util import *
import pickle

from bag_deep_ckt.eval_engines.spectre.script_test.fc_meas_man import *

#Create script that initializes eval_engine and randomly selects parameters within it
CIR_YAML = "/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/bag_deep_ckt/eval_engines/spectre/specs_test/folded_cascode.yaml"

with open(CIR_YAML, 'r') as f:
  yaml_data = yaml.load(f)

# param array
params = yaml_data['params']
params_ac = []
params_id = list(params.keys())

for value in params.values():
  param_vec = np.arange(value[0], value[1], value[2])
  params_ac.append(param_vec)

#initialize sim environment
sim_env = FCMeasMan(CIR_YAML) 

rec_params = []
rec_specs = []

for j in range(100000):
  params_idx = [random.randint(0,len(i)-1) for i in params_ac]
  params = [params_ac[i][params_idx[i]] for i in range(len(params_id))]
  param_val = [OrderedDict(list(zip(params_id,params)))]

  #run param vals and simulate
  cur_specs = OrderedDict(sorted(sim_env.evaluate(param_val)[0][1].items(), key=lambda k:k[0]))
  cur_specs = np.array(list(cur_specs.values()))[:-1]

  rec_params.append(params)
  rec_specs.append(cur_specs)

  with open("params.txt","wb") as fp:
    pickle.dump(rec_params,fp)
  with open("specs.txt", "wb") as fp:
    pickle.dump(rec_specs, fp)

