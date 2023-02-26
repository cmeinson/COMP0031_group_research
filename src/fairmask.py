from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


class FairMaskModel(Model):
    RF = "RandomForestClassifier"
    DT = "DecisionTreeRegressor"
    LOG = "LogisticRegression-NOT IMPLEMENTED"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._mask_models = None
        self._model = None
        self._sensitive = None
        self._method_bias = None

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
        self._method_bias = method_bias
        self._mask_models = {}
        self._sensitive = sensitive_attributes

        X_non_sens = X.copy()
        print(X_non_sens)
        X_non_sens.drop(self._sensitive, axis=1)

        # Build the mask_model for predicting each protected attribute
        for attr in sensitive_attributes: # ngl this code very sketchy but whatever will just copy what they did for now 
            mask_model = self._get_model(method_bias)

            if method_bias == self.DT or method_bias == self.LOG:
                mask_model.fit(X_non_sens, X[attr])
            else:
                clf = self._get_model(method)
                clf.fit(X_non_sens, X[attr])
                y_proba = clf.predict_proba(X_non_sens)
                y_proba = [each[1] for each in y_proba]
                mask_model.fit(X_non_sens, y_proba)
            self._mask_models[attr] =mask_model 
        # mask the attributes
        X_masked = self._mask(X)

        # Build the model for the actual prediction
        self._model = self._get_model(method)
        self._model.fit(X_masked, y)
             

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        X_masked = self._mask(X)
        return self._model.predict(X_masked)

    def _get_model(self, method):
        if method == self.RF:
            return RandomForestClassifier()
        elif method == self.DT:
            return DecisionTreeRegressor()
        else:
            raise RuntimeError("Invalid ml method name: ", method)
        
    def _mask(self, X: pd.DataFrame):
        X_out = X.copy()
        X_non_sens = X.copy()
        X_non_sens.drop(self._sensitive, axis=1)

        for attr in self._sensitive: 
            mask_model = self._mask_models[attr]
            mask = mask_model.predict(X_non_sens)
            if self._method_bias == self.DT or self._method_bias == self.LOG:
                mask = self.reg2clf(mask, threshold=0.5) # TODO: I am 100% sure there is a better way than this! to do that
            X_out.loc[:, attr] = mask
        return X_out
    
    def reg2clf(self,protected_pred,threshold=.5):
        out = []
        for each in protected_pred:
            if each >=threshold:
                out.append(1)
            else: out.append(0)
        return out


    