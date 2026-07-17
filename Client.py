import numpy as np
from itertools import product
from minepy import MINE


class CorrelationCal(object):
    def __init__(self, fixthreshold=0.8):
        self.fixthreshold = fixthreshold

    def _joint_prob(self, data):
        data = np.array(data)
        values = np.unique(data)
        pairs = list(product(values, repeat=2))
        n = data.size
        matrix = []

        for (x, y) in pairs:
            count = np.sum((data == x) | (data == y))
            prob = float(count) / n if n > 0 else 0.0
            if prob >= self.fixthreshold:
                matrix.append([(x, y), prob])
        return matrix

    def _adjacent_transitions(self, data):
        data = np.array(data).flatten()
        if data.size < 2:
            return 0.0

        states = np.unique(data)
        idx_map = {s: i for i, s in enumerate(states)}
        trans = np.zeros((len(states), len(states)), dtype=float)

        for i in range(len(data) - 1):
            a, b = data[i], data[i + 1]
            trans[idx_map[a], idx_map[b]] += 1.0

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        trans = trans / row_sums
        return float(trans.max())

    def _inter_day_mic(self, data):
        data = np.array(data)
        if data.ndim == 1:
            return 0.0
        if data.shape[0] < 2:
            return 0.0

        mine = MINE()
        seq1 = data[0, :]
        seq2 = data[1, :]
        mine.compute_score(seq1, seq2)
        return float(mine.mic())

    def max_corr(self, data):
        joint_list = self._joint_prob(data)
        joint_max = max([p for (_, p) in joint_list], default=0.0)

        adj_max = self._adjacent_transitions(data)
        inter_mic = self._inter_day_mic(data)

        return max(joint_max, adj_max, inter_mic)


class Client(object):

    def __init__(self, epsilon_inf, epsilon_1, m, thresh, budget,
                 fixthreshold=0.8, corr=True):
        if epsilon_1 >= epsilon_inf:
            raise ValueError("epsilon_1 should be strictly less than epsilon_inf")

        self.epsilon_inf = float(epsilon_inf)
        self.epsilon_1 = float(epsilon_1)
        self.m = int(m)
        self.thresh = int(thresh)
        self.initial_budget = float(budget)
        self.budget = float(budget)
        self.corr = bool(corr)
        self.fixthreshold = float(fixthreshold)
        # f: probability of keeping true bit in first round
        self.f = 2.0 / (1.0 + np.exp(self.epsilon_inf / 2.0))
        # p: probability used in second round adjustment
        self.p = (
            (np.exp(self.epsilon_inf / 2.0) - np.exp(self.epsilon_1 / 2.0)) /
            ((np.exp(self.epsilon_inf / 2.0) - 1.0) * (np.exp(self.epsilon_1 / 2.0) + 1.0))
        )

        self.memo = {}

        self.corr_cal = CorrelationCal(fixthreshold=self.fixthreshold)

    def ue(self, input_data):
        input_data = np.array(input_data, dtype=int).flatten()
        ue_data = np.zeros((len(input_data), self.m), dtype=int)
        for i, v in enumerate(input_data):
            if 0 <= v < self.m:
                ue_data[i, v] = 1
        return ue_data

    def _apply_correlation_noise(self, unary_vec, maxcorr):
        if not self.corr:
            noisy = unary_vec.copy().astype(float)
            for i in range(len(noisy)):
                if noisy[i] == 1:
                    if np.random.rand() > self.f:
                        noisy[i] = 0
                else:
                    if np.random.rand() < (1.0 - self.f) / self.m:
                        noisy[i] = 1
            return noisy

        d = self.m
        c = maxcorr
        C = np.exp(self.epsilon_inf) + d - 1.0 + c
        p = np.exp(self.epsilon_inf) / C
        q = (c + 1.0) / C

        noisy = unary_vec.copy().astype(float)
        for i in range(len(noisy)):
            if noisy[i] == 1:
                if np.random.rand() > (p - q):
                    noisy[i] = 0
            else:
                if np.random.rand() < q:
                    noisy[i] = 1
        return noisy

    def _first_round(self, ue_data, maxcorr):
        noisy_data = []
        for row in ue_data:
            noisy_row = self._apply_correlation_noise(row, maxcorr)
            noisy_data.append(noisy_row)
        return np.array(noisy_data)

    def _second_round(self, first_round_output):
        data = first_round_output.copy().astype(float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] == 1:
                    if np.random.rand() > self.p:
                        data[i, j] = 0
                else:
                    if np.random.rand() < (1.0 - self.p) / self.m:
                        data[i, j] = 1
        return data

    def report(self, values):
        values = np.array(values, dtype=int).flatten()
        ue_data = self.ue(values)

        maxcorr = self.corr_cal.max_corr(values)

        key = tuple(values.tolist())
        if key in self.memo:
            first_round_out = self.memo[key]
        else:

            if self.budget <= 0.0:
                return np.zeros_like(ue_data)

            first_round_out = self._first_round(ue_data, maxcorr)

            if len(values) >= self.thresh:
                self.memo[key] = first_round_out

            self.budget -= self.epsilon_inf * (len(values) / float(self.thresh))

        second_round_out = self._second_round(first_round_out)

        return second_round_out
