from .ml_interface import Model
import pandas as pd
import random,csv
import numpy as np
import math,copy,os
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import tree
import sys

class FairMaskModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super(FairMaskModel, self).__init__(other)
        self.clf1 = RandomForestClassifier()
        self.clf2 = DecisionTreeRegressor()

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

        for i in range(other['rep']):
            X = copy.deepcopy(X.loc[:, X.columns != 'Probability'])
            y = copy.deepcopy(y['Probability'])

            reduced = list(X.columns)
            reduced.remove(sensitive_attributes)
            X_reduced, y_reduced = X.loc[:, reduced], X[sensitive_attributes]

            # Build model to predict the protect attribute
            self.clf2 = copy.deepcopy(other['base2'])
            self.clf1 = copy.deepcopy(other['base1'])
            self.clf1.fit(X_reduced, y_reduced)
            
            y_proba = self.clf.predict_proba(X_reduced)
            y_proba = [each[1] for each in y_proba]
            if isinstance(self.clf2, DecisionTreeClassifier) or isinstance(self.clf1, LogisticRegression):
                self.clf2.fit(X_reduced, y_reduced)
            else:
                self.clf2.fit(X_reduced, y_proba)



    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        #need access to the protected attributes -> maybe passed through other again? 
        X_test_reduced = X.loc[:, X.columns != other['sensitive_attributes']]
        protected_pred = self.clf2.predict(X_test_reduced)
        if isinstance(self.clf1, DecisionTreeRegressor) or isinstance(self.clf1, LinearRegression):
            protected_pred = self.reg2clf(protected_pred, thresh=.5)
        # Build model to predict the taget attribute Y
        clf = copy.deepcopy(self.clf1)

        X.loc[:, other['sensitive_attributes']] = protected_pred
        preds = clf.predict(X)
        return preds
    
    def reg2clf(self,protected_pred,threshold=.5):
        out = []
        for each in protected_pred:
            if each >=threshold:
                out.append(1)
            else: out.append(0)
        return out


    