import numpy as np


class Server(object):

    def __init__(self, epsilon_inf, epsilon_1, m):
        if epsilon_1 >= epsilon_inf:
            raise ValueError("epsilon_1 should be strictly less than epsilon_inf")

        self.epsilon_inf = float(epsilon_inf)
        self.epsilon_1 = float(epsilon_1)
        self.m = int(m)

        self.f = 2.0 / (1.0 + np.exp(self.epsilon_inf / 2.0))
        self.p = (
            (np.exp(self.epsilon_inf / 2.0) - np.exp(self.epsilon_1 / 2.0)) /
            ((np.exp(self.epsilon_inf / 2.0) - 1.0) * (np.exp(self.epsilon_1 / 2.0) + 1.0))
        )

    def estimate_distribution(self, reports):
        reports = np.array(reports, dtype=float)
        if reports.ndim == 3:
            agg = reports.reshape(-1, self.m)
        elif reports.ndim == 2:
            # shape == (N, m)
            agg = reports
        else:
            raise ValueError("Unexpected shape for reports")

        avg_unary = agg.mean(axis=0)

        est = (avg_unary - (1.0 - self.f) / 2.0) / (self.f - (1.0 - self.f) / 2.0)

        est = np.clip(est, 0.0, None)
        if est.sum() == 0.0:
            est = np.ones_like(est) / len(est)
        else:
            est = est / est.sum()

        return est

    def estimate_mean(self, reports):
        dist = self.estimate_distribution(reports)
        indices = np.arange(len(dist))
        mu_hat = float(np.sum(indices * dist))
        return mu_hat

