import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


def plot_acc(keys_to_plot):

    pardir = os.path.dirname(__file__)
    with open("bag_deep_ckt/experiments/DropOutAgent_vs_oracle/acc.pkl", 'rb') as f:
        df = pickle.load(f)
    n_rows = int(math.sqrt(len(keys_to_plot)))
    n_cols = int(math.ceil(len(keys_to_plot) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, dpi=150)
    for key, ax in zip(keys_to_plot, axs.ravel()):
        df.plot(y=key, ax=ax, legend=False)
        ax.set_title(key)
    plt.tight_layout()
    plt.savefig(os.path.join(pardir, "acc_categories.png"))

    confusion_keys = ["tt", "ff", "tf", "ft"]
    fig, axs = plt.subplots(2, 2, sharex=True, dpi=150)
    for key, ax in zip(confusion_keys, axs.ravel()):
        df.plot(y=key, ax=ax, legend=False)
        ax.set_title(key)
    plt.tight_layout()
    plt.savefig(os.path.join(pardir, "confusion_matrix.png"))

    accuracy_keys = ["a1", "a2", "a3", "a4"]
    fig, axs = plt.subplots(2, 2, sharex=True, dpi=150)
    for key, ax in zip(accuracy_keys, axs.ravel()):
        df.plot(y=key, ax=ax, legend=False)
    ax.set_title(key)
    plt.tight_layout()
    plt.savefig(os.path.join(pardir, "confusion_accs.png"))

if __name__ == '__main__':
    keys_to_plot = ["gain", "ugbw", "pm", "tset", "cmrr", "psrr", "offset_sys", "ibias", "total_acc"]
    sns.set_style("darkgrid")
    plot_acc(keys_to_plot)
    plt.close()