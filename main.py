import argparse
import yaml
import importlib
import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-yfname', '-f', type=str,
                        help='the main yaml file that sets the settings')
    parser.add_argument('--seed', '-s', type=int, default=10,
                        help='the main yaml file that sets the settings')

    args = parser.parse_args()

    util.set_random_seed(args.seed)

    with open(args.yfname, 'r') as f:
        setting = yaml.load(f)

    agent_module = importlib.import_module(setting['agent_module_name'])
    agent_cls = getattr(agent_module, setting['agent_class_name'])

    agent = agent_cls(args.yfname)
    agent.main()
