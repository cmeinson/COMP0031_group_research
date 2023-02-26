from .ml_interface import Model
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression

class FairBalanceModel(Model):
    LOGR = "LogisticRegression" 

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._model = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_attributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        :param sensitive_attributes: names of sensitive attributes to be protected
        :type sensitive_attributes: List[str]
        :param method:  ml algo name to use for the main model training
        :type method: _type_
        :param method_bias: method name if needed for the bias mitigation, defaults to None
        :type method_bias: _type_, optional
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional
        """

        if method == self.LOGR:
            self._model = LogisticRegression(max_iter=100000)
        else:
            raise RuntimeError("Invalid ml method name: ", method)
            
        sample_weight = self.FairBalance(X, y, sensitive_attributes)
        self._model.fit(X, y, sample_weight)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        return self._model.predict(X)

    def FairBalance(self, X, y, A):
        # X: independent variables (2-d pd.DataFrame)
        # y: the dependent variable (1-d np.array)
        # A: the name of the sensitive attributes (list of string)
        groups_class = {}
        group_weight = {}
        for i in range(len(y)):
            key_class = tuple([X[a][i] for a in A] + [y[i]])
            key = key_class[:-1]
            if key not in group_weight:
                group_weight[key] = 0
            group_weight[key] += 1
            if key_class not in groups_class:
                groups_class[key_class] = []
            groups_class[key_class].append(i)
        sample_weight = np.array([1.0]*len(y))
        for key in groups_class:
            weight = group_weight[key[:-1]]/len(groups_class[key])
            for i in groups_class[key]:
                sample_weight[i] = weight
        # Rescale the total weights to len(y)
        sample_weight = sample_weight * len(y) / sum(sample_weight)
        return sample_weight

    # def data_preprocessing(self, X):
    #     numerical_columns_selector = selector(dtype_exclude=object)
    #     numerical_columns = numerical_columns_selector(X)
    #     numerical_preprocessor = StandardScaler()

    #     categorical_columns_selector = selector(dtype_include=object)
    #     categorical_columns = categorical_columns_selector(X)
    #     categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore')

    #     preprocessor = ColumnTransformer([
    #         ('OneHotEncoder', categorical_preprocessor, categorical_columns),
    #         ('StandardScaler', numerical_preprocessor, numerical_columns)])

    #     preprocessor.fit_transform(X)

    


