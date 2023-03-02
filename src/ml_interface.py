from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

class Model:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyperparams we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        raise NotImplementedError

    def train(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        :param sensitive_atributes: names of sensitive attributes to be protected
        :type sensitive_atributes: List[str]
        :param method:  ml algo name to use for the main model training
        :type method: _type_
        :param method_bias: method name if needed for the bias mitigation, defaults to None
        :type method_bias: _type_, optional
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        raise NotImplementedError


class BaseModel(Model):
    SV = "SupportVectorClassifier"
    KN = "KNearestNeighbours"
    NN = "NeuralNetwork"
    RF = "RandomForest"
    NB = "NaiveBayes"
    LR = "LogisticRegression"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        self._model = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
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
        elif method == self.LR:
            self._model = LogisticRegression()
        else:
            raise RuntimeError("Invalid ml method name: ", method)
        
        self.encode_data(X)
        self._model.fit(self.processor.fit_transform(X), y)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        return self._model.predict(self.processor.transform(X))
    
    def encode_data(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_processor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_processor = StandardScaler()
        self.processor = ColumnTransformer([
            ('OneHotEncoder', categorical_processor, categorical_columns),
            ('StandardScaler', numerical_processor, numerical_columns)])
