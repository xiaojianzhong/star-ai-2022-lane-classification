import time

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class Metric:
    def __init__(self):
        self.l = None
        self.n = None
        self.pred = None
        self.t = None
        self.start = None
        self.end = None
        self.reset()

    def reset(self):
        self.l = 0.
        self.n = 0
        self.pred = np.empty((0), dtype=np.int)
        self.t = np.empty((0), dtype=np.int)
        self.start = time.time()
        self.end = self.start

    def add(self, l, pred, t):
        self.l += l
        self.n += 1
        self.pred = np.concatenate([self.pred, pred], axis=0)
        self.t = np.concatenate([self.t, t], axis=0)
        self.end = time.time()

    def loss(self):
        return self.l / self.n

    def precision(self):
        return precision_score(self.t, self.pred, zero_division=1)

    def recall(self):
        return recall_score(self.t, self.pred, zero_division=1)

    def f1(self):
        return f1_score(self.t, self.pred, zero_division=1)

    def seconds(self):
        return int(self.end - self.start)
