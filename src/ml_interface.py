from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from sklearn.svm import SVC


class Model:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """
        other - idk do we need any params here?
        """
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """
        X_train
        y_train
        sensitive attributes names list
        method - ml algo name to use for the main model training
        method_bias - method name if needed for the bias mitigation
        other - dictionary of any other parms that we might wish to pass?
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """
        X_test
        other - dictionary of any other parms that we might wish to pass?

        returns y_preds
        """
        raise NotImplementedError


class BaseModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method = "EN", method_bias = None, other: Dict[str, Any] = {}):
        if (method == "EN"):
            self._model = ElasticNet()
        else:
            #idk some default
            self._model = SVC()
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        return self._model.predict(X)