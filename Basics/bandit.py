import numpy as np
from numpy.random import beta, normal, uniform
from torch import norm


class Bandit:
    def __init__(self, p: float, samples: int = 0, mean: float = 0.0):
        self._p = p
        self.samples = samples
        self.mean = mean

    def pull(self) -> int:
        val = uniform() < self._p
        self.samples += 1
        self.mean += 1.0 / self.samples * (val - self.mean)
        return val


class ThompsonSamplingBandit(Bandit):
    def __init__(self, p: float):
        super().__init__(p)
        self._alpha = 1
        self._beta = 1

    def pull(self) -> int:
        result = super().pull()
        self._alpha += result
        self._beta += 1 - result
        return result

    def sample(self) -> float:
        return beta(self._alpha, self._beta)

class GaussianBandit:
    def __init__(self, mean_true: float, precision_true: float):
        self._mean_true = mean_true
        self._precision = precision_true
        self._std_true = 1.0/np.sqrt(precision_true)
        self.samples = 0
        self.mean = 0.0

    def pull(self) -> int:
        val = normal(self._mean_true, self._std_true)
        self.samples += 1
        self.mean += 1.0 / self.samples * (val - self.mean)
        return val

class GaussianThompsonSamplingBandit(GaussianBandit):
    def __init__(self, mean_true: float, precision_true: float):
        super().__init__(mean_true, precision_true)
        self._mean_pred = 0.0
        self._precision_pred = precision_true

    def pull(self) -> int:
        result = super().pull()
        self._precision_pred += self._precision
        self._mean_pred += self._precision/self._precision_pred*(result - self._mean_pred)
        return result

    def sample(self) -> float:
        return normal(self._mean_pred, 1.0/np.sqrt(self._precision_pred))
