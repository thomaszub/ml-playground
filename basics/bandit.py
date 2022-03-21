import random


class Bandit:
    def __init__(self, p: float, samples: int = 0, mean: float = 0.0):
        self._p = p
        self.samples = samples
        self.mean = mean

    def pull(self) -> int:
        val = random.random() < self._p
        self.samples += 1
        self.mean += 1.0 / self.samples * (val - self.mean)
        return val
