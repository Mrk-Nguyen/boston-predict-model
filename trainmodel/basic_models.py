import numpy as np
from collections.abc import Iterable

# Build base class model
class MeanRegressor(object):
    def __init__(self):
        super().__init__()
        self._mean = 0
    
    def fit(self: object, X: Iterable, Y: Iterable) -> object:
        return self
    
    def predict(self: object, X: Iterable) -> Iterable:
        return []

# Build random regressor model
class RandomRegressor(object):
    def __init__(self):
        super().__init__()
        self._max = 100
        self._min = 0
    
    def fit(self: object, X: Iterable, Y: Iterable) -> object:
        return self
    
    def predict(self: object, X: Iterable) -> Iterable:
        return []