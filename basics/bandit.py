from numpy.random import beta, uniform


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
        if result == 1:
            self._alpha += 1
        else:
            self._beta += 1
        return result

    def sample(self) -> float:
        return beta(self._alpha, self._beta)
