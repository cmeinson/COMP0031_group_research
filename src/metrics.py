import numpy as np
import pandas as pd

# how do we wanna do metrics?

# we can do just a simple metrcs class with all the mathy functions and then a separate evaluator class?

class Metrics:
    # All availabel metrics:
    ACC = "accuracy"

    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array) -> None:
        # might need more attributes idk
        self._X = X # not even sure if needed
        self._y = y
        self._preds = preds
        pass

    def get(self, metric_name):
        if metric_name == self.ACC:
            return self.accuracy()
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)
    
    def accuracy(self) -> float:
        return np.mean(self._y == self._preds)

    # etc other metrics

