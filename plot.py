import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np

def get_dataset(log_dir):

    with open(log_dir, 'rb') as f:
        return pickle.load(f)

def plot_cost(data, legends):
    ax = plt.gca()
    max_itr = min([len(x) for x in data])
    avg_plt_objects = []
    min_plt_objects = []
    for exp, legend in zip(data, legends):
        db = []
        avg_to_plot = []
        min_to_plot = []
        n_evals = 0
        for i, offsprings in enumerate(exp):
            # print(offsprings)
            n_evals += len(offsprings)
            if any([x.cost == 0 for x in offsprings]):
                print("solution found on iter: {} neval: {}".format(i, n_evals))
            db += offsprings
            db_sorted = sorted(db, key=lambda x: x.cost)
            avg_cost = np.mean([x.cost for x in db_sorted[:25]])
            min_cost = db_sorted[0].cost
            avg_to_plot.append(avg_cost)
            min_to_plot.append(min_cost)


        avg_plt_objects.append(ax.plot(avg_to_plot, label='avg_'+legend))
        min_plt_objects.append(ax.plot(min_to_plot, '--', label='min_'+legend))

    ax.legend()
    ax.set_ylabel('cost')
    ax.set_xlabel('iteration')
    plt.show()

def print_best_design(data, legends):
    for exp, legend in zip(data, legends):
        db = []
        for offsprings in exp:
            db += offsprings

        db_sorted = sorted(db, key=lambda x: x.cost)
        solutions = [x for x in db_sorted if x.cost == 0]
        for sol in solutions:
            print("{} -> {}".format(sol, sol.specs))
        # best_sol = min(solutions, key=lambda x: x.specs['ibias_cur'])
        # print("{}: {} -> {}".format(legend, best_sol, best_sol.specs))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    args = parser.parse_args()

    data = []
    for log_dir in args.logdir:
        data.append(get_dataset(log_dir))

    plot_cost(data, args.legend)
    # print_best_design(data, args.legend)


