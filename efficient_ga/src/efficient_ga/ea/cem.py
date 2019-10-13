from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import numpy as np
from .ea import EA
from util import Design


class CEM(EA):
    """
    The vanilla implementation of Cross Entropy method with gaussian underlying distribution
    """
    def __init__(self, eval_core, n, dist_type='gaussian'):
        self.eval_core = eval_core
        dim = len(self.eval_core.params)
        self.n_new_samples = n
        self.type = dist_type
        if self.type == 'gaussian':
            self.mu = np.random.uniform(dim)
            self.var = np.random.uniform(low=0.001, size=dim)
        elif self.type == 'kde':
            self.kde = None

    def fit(self, data):
        if self.type == 'gaussian':
            self.mu = np.mean(data, axis=0)
            self.var = np.var(data, axis=0)
            # print("[debug] cem variance = {}".format(self.var))
        elif self.type == 'kde':
            self.kde = gaussian_kde(np.transpose(data))

    def sample(self, n):
        if self.type == 'gaussian':
            samples = multivariate_normal.rvs(self.mu, np.diag(self.var), n)
        elif self.type == 'kde':
            samples = self.kde.resample(n)
            samples = samples.transpose()
        else:
            raise ValueError('type {} is not a valid distribution type'.format(self.type))
        return samples

    def get_next_generation_candidates(self, parent1, parent2):
        offsprings = self.sample(self.n_new_samples)
        if offsprings.ndim == 1:
            offsprings = offsprings[None, :]

        dsns = list()
        for dsn in offsprings:
            dsn = np.clip(dsn, self.eval_core.params_min, self.eval_core.params_max)
            dsn = np.floor(dsn).astype(int)
            dsn = Design(self.eval_core.spec_range,
                         self.eval_core.id_encoder,
                         dsn.tolist())
            dsns.append(dsn)

        return dsns

    def prepare_for_generation(self, db, n):
        # fit on top n/2
        db = sorted(db, key=lambda x: x.cost)
        self.fit(db[:n//2])
