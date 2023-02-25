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

    def reg2clf(self,protected_pred,threshold=.5):
        out = []
        for each in protected_pred:
            if each >=threshold:
                out.append(1)
            else: out.append(0)
        return out


    def predict(self, df, base_clf, base2, keyword, ratio=.2, rep=10, thresh=.5):
        """ Parameters description

        - df: input dataset
        - base_clf: classification model - trained with masked protected attributes
        - base2: extrapolation model - trained on non-protected attributes and predict test data
        - keyword: sensitive attributes
        - ratio: test data ratio 
        - rep: repetitions
        - thresh: threshold that determines when an attribute is sensitive or not, we can actually standarize this or set it as 
        a default for both algorithms
        """ 

        dataset_orig = df.dropna()
        #Train data using MinMaxScaler witih boundaries 0-1

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

        for i in range(rep):
            dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=ratio, random_state=i)
            
            #split data to train and test -> X data, Y probabilities

            X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
            y_train = copy.deepcopy(dataset_orig_train['Probability'])
            X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
            y_test = copy.deepcopy(dataset_orig_test['Probability'])

            #remove sensitive column (e.g. "sex")
            reduced = list(X_train.columns)

            #reduced dataset without the sensitive columns
            X_reduced, y_reduced = X_train.loc[:, reduced], X_train[keyword]

            # Build model to predict the protect attribute
            clf1 = copy.deepcopy(base2)

            clf = copy.deepcopy(base_clf)
            clf.fit(X_reduced, y_reduced)
            y_proba = clf.predict_proba(X_reduced)
            y_proba = [each[1] for each in y_proba]
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_reduced, y_reduced)
            else:
                clf1.fit(X_reduced, y_proba)
            #             clf1.fit(X_reduced,y_reduced)

            X_test_reduced = X_test.loc[:, X_test.columns != keyword]
            protected_pred = clf1.predict(X_test_reduced)
            if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
                protected_pred = self.reg2clf(protected_pred, threshold=thresh)
            # Build model to predict the taget attribute Y
            
            clf2 = copy.deepcopy(base_clf)

            X_test.loc[:, keyword] = protected_pred
            y_pred = clf2.predict(X_test)

        return y_pred