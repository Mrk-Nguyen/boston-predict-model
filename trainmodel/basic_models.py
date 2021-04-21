import numpy as np
from collections.abc import Iterable

# Build base class model
class MeanRegressor(object):
    def __init__(self):
        super().__init__()
        self._mean = 0
    
    def fit(self: object, X: Iterable, Y: Iterable) -> object:
        self._mean = np.mean(Y)
        return self
    
    def predict(self: object, X: Iterable) -> Iterable:
        return np.full(len(X), self._mean)

# Build random regressor model
class RandomRegressor(object):
    def __init__(self):
        super().__init__()
        self._max = 100
        self._min = 0
    
    def fit(self: object, X: Iterable, Y: Iterable) -> object:
        self._max, self._min = max(Y), min(Y)
        return self
    
    def predict(self: object, X: Iterable) -> Iterable:
        np.random.seed(291)
        return np.random.rand(len(X)) * (self._max - self._min) + self._min