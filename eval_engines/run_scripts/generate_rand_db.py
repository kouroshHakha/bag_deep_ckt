import argparse
from argparse import Namespace
from pathlib import Path
from eval_engines.util.importlib import import_cls
from eval_engines.base import EvaluationEngineBase

import yaml
import pickle
import time


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('specs_fname', help='specs yaml file')
    parser.add_argument('db_path', help='location to store the db file')
    parser.add_argument('-n', '--number', dest='number', type=int, default=1,
                        help='number of individuals in the database')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='the seed used for generating the data base')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='True to make the underlying processes verbose')
    args = parser.parse_args()
    return args


def run_main(args: Namespace):

    with open(args.specs_fname, 'r') as f:
        specs = yaml.load(f, Loader=yaml.Loader)

    kwargs = {}
    if args.verbose:
        kwargs['verbose'] = True
    eval_engine_str = specs['eval_engine_cls']
    eval_engine_params = specs['eval_engine_params']
    eval_engine_cls = import_cls(eval_engine_str)
    eval_engine: EvaluationEngineBase = eval_engine_cls(specs=eval_engine_params, **kwargs)
    start = time.time()
    designs = eval_engine.generate_rand_designs(n=args.number, evaluate=True, seed=args.seed)

    db_path = Path(args.db_path)
    db_path.parent.mkdir(exist_ok=True, parents=True)
    with open(db_path, 'wb') as f:
        pickle.dump(designs, f, pickle.HIGHEST_PROTOCOL)

    print(f'data base stored in {str(db_path)}')
    print(f'random generation of {args.number} samples took {time.time() - start:.6} seconds')


if __name__ == '__main__':
    args = parse_args()
    run_main(args)