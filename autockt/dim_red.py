from sklearn.manifold import TSNE
import pickle
import numpy as np
import IPython
from typing import Optional, Mapping, Union
import random
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from sklearn.decomposition import PCA
import seaborn as sns

def _get_ax(ax):
  if ax is None:
    plt.close()
    ax = plt.axes()
  return ax

def _save_show_fig(fpath: Union[str, Path], show_fig: bool, **kwargs):
  #fpath: Path = Path(fpath)
  #fpath.parent.mkdir(parents=True, exist_ok=True)
  #if fpath:
  #  plt.savefig(fpath, **kwargs)
  if show_fig:
    plt.show()

def scatter2d(data: np.ndarray, labels: Optional[np.ndarray] = None, label_mapping: Optional[Mapping[int, str]] = None, ax=None, fpath='', show_fig=True, fig_title=None, **kwargs):
  random.seed(20)
  ax = _get_ax(ax)
  markers = (',', '+', 'o', '*')
  colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
  marker_color = list(itertools.product(markers, colors))
  random.shuffle(marker_color)
  marker_color = iter(marker_color)
  #if labels is not None:
  #  for label in np.unique(labels):
  #    pos = (labels == label)
  #    if label_mapping is None:
  #      _label = label                                                                                      else:
  #      _label = label_mapping[label]
  #    marker, color = next(marker_color)
  #    ax.scatter(data[pos, 0], data[pos, 1], marker=marker, color=color, label=_label, **kwargs)
  #else:
  ax.scatter(data[:, 0], data[:, 1], marker='o', **kwargs)
  ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
  
  if fig_title is None:
    ax.set_title(fig_title)
  _save_show_fig(fpath, show_fig, dpi=400, bbox_inches="tight")

def tsne_scatter2d(data: np.ndarray, labels: Optional[np.ndarray] = None, label_mapping: Optional[Mapping[int,str]] = None, seed: Optional[int] = None, ax=None, fpath='', show_fig=False, title=None, **kwargs):
  data_hat = TSNE(n_components=2,perplexity=300, random_state=seed).fit_transform(data)
  scatter2d(data_hat, labels, label_mapping, ax, fpath, show_fig, title, **kwargs)

def pca(data,labels):
  pca = PCA(n_components = 3)
  pca_result = pca.fit_transform(data)

  res = {}
  res['pca-one'] = pca_result[:,0]
  res['pca-two'] = pca_result[:,1]
  res['pca_three'] = pca_result[:,2]

  print('Explained variation per principal component:{}'.format(pca.explained_variance_ratio_))

  #visualize
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(res['pca-one'],res['pca-two'],c=labels)
  ax.set_xlabel('PCA dimension 1')
  ax.set_ylabel('PCA dimension 2')
  ax.set_title('Objectives initially given to AutoCkt')
  plt.show()

  IPython.embed()

def load_data(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data

def lookup(spec):
  goal_spec = [10.0e+8, 500, 75.0, 1.0, 1.0e-9]
  norm_spec = (spec-goal_spec)/(goal_spec+spec)
  return norm_spec

def error(x, y):
   
  IPython.embed()
  return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def main():
  data = np.array(load_data('/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/obs_reached_test'))
  data_nreached = np.array(load_data('/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/obs_nreached_test'))
 
  data = np.array(data)
  data_nreached = np.array(data_nreached)

  for i,each_data in enumerate(data):
    data[i] = lookup(each_data)
  
  for i,each_data in enumerate(data_nreached):
    data_nreached[i] = lookup(each_data)

  IPython.embed()
  #data_params = np.array(load_data('/tools/projects/ksettaluri6/BAG2_TSMC16FFC_tutorial/params.txt'))

  #act_data_rem = []
  #for i,each_dp in enumerate(data):
  #  if not(-1.80000e+02 in each_dp):
      #data = np.delete(data, axis=0, obj=i)
  #   act_data_rem.append(each_dp)
  #act_data_rem.append(data[0])

  labels = 1*np.ones(len(data))
  labels = np.append(labels, 2*np.ones(len(data_nreached)))
  data_tot = np.append(data, data_nreached, axis = 0)
  IPython.embed()
  #labels = []
  #for i,each_dp in enumerate(data):
  #  if not(-1.80000e+02 in each_dp):
  #    labels.append(1)
  #  else:
  #    labels.append(0)
  pca(data_tot,labels)
  #pca(data, 1*np.ones(len(data)))
  #tsne_scatter2d(data)

if __name__ == '__main__':
  main()
