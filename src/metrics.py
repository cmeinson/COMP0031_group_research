from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

# how do we wanna do metrics?

# we can do just a simple metrcs class with all the mathy functions and then a separate evaluator class?

class Metrics:
    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array) -> None:
        # might need more attributes idk
        self._X = X # not even sure if needed
        self._y = y
        self._preds = preds
        pass

    
    def accuracy(self) -> float:
        return np.mean(self._y == self._preds)

    # etc other metrics

