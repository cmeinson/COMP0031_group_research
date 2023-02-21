from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB




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
    SV = "SupportVectorClassifier"
    KN = "KNearestNeighbours"
    NN = "NeuralNetwork"
    RF = "RandomForest"
    NB = "NaiveBayes"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        if method == self.SV:
            self._model = SVC()
        elif method == self.KN:
            # example of how to pass hyperparams through "other"
            k = 3 if ("KNN_k" not in other) else other["KNN_k"] 
            self._model = KNeighborsClassifier(k)
        elif method == self.NN:
            self._model = MLPClassifier()
        elif method == self.RF:
            self._model = RandomForestClassifier(n_estimators=10)
        elif method == self.NB:
            self._model = GaussianNB()
        else:
            raise RuntimeError("Invalid ml method name: ", method)
            
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        return self._model.predict(X)