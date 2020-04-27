import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from bag_deep_ckt.autockt.envs.spectre_vanilla_opamp_45nm import TwoStageAmp
from bag_deep_ckt.autockt.envs.spectre_fc import FoldedCascode 
import os
import argparse
import random
import IPython
import glob
from shutil import copyfile

#set arguments for input
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
parser.add_argument('--exp_name', '-exn', type=str)
args = parser.parse_args()

#set variable where the spectre log files will be saved
os.environ["BASE_TMP_DIR"] = "/tools/scratch/ksettaluri6/spectre_logs_" + str(random.randint(0,10000))

#save yaml file and specs generation for easier retrieval
cwd = os.getcwd()
gen_specs_folder = os.path.join(cwd, "bag_deep_ckt", "autockt", "gen_specs")
yaml_file = os.path.join(cwd, "bag_deep_ckt", "eval_engines", "spectre", "specs_test", "folded_cascode_w_biasing_high_gain.yaml")
list_files = glob.glob(gen_specs_folder + '/*')
gen_specs_file = max(list_files, key=os.path.getctime)
os.mkdir(os.path.join(os.path.expanduser('~'),"ray_results", args.exp_name))
copyfile(gen_specs_file, os.path.join(os.path.expanduser('~'),"ray_results",args.exp_name,str.split(gen_specs_file,'/')[-1]))
copyfile(yaml_file, os.path.join(os.path.expanduser('~'),"ray_results", args.exp_name, "folded_cascode.yaml"))

#initialize ray session with hyperparameters
ray.init()
config_validation = {
            "sample_batch_size": 200,
            "train_batch_size": 1200,
            "sgd_minibatch_size": 1200,
            "num_sgd_iter":3,
            "lr":1e-3,
            "vf_loss_coeff":0.5,
            "horizon":  60,#tune.grid_search([15,25]),
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 7,
            "env_config":{"generalize":False, "save_specs":False, "run_valid":True},
            }

config_train = {
            "sample_batch_size": 60,#200,
            "train_batch_size": 360,
            "sgd_minibatch_size": 360,
            "num_sgd_iter": 3,
            "lr":1e-3,
            "vf_loss_coeff": 0.5,
            "horizon":  20,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 7,
            "env_config":{"generalize":True, "save_specs":True, "run_valid":False},
            }

if not args.checkpoint_dir:
    trials = tune.run_experiments({
        args.exp_name: {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": FoldedCascode,
        "stop": {"episode_reward_mean": -0.02},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    tune.run_experiments({
        args.exp_name: {
        "run": "PPO",
        "config": config_train,
        "env": FoldedCascode,
        "stop": {"episode_reward_mean": -0.02},
        "restore": args.checkpoint_dir,
        "checkpoint_freq":1},
    })
