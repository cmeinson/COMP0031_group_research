from .ml_interface import Model
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression

class FairBalanceModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super(FairBalanceModel, self).__init__(other)
        self.clf = LogisticRegression(max_iter=100000)
        self.groups = {}

    #Need to change how this method is called because FairBalance needs access to the whole dataset
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

        X_train, X_test, y_train, y_test = self.train_test_split(X, y, sensitive_attributes, test_size=0.5)
        self.data_preprocessing(X_train)
        sample_weight = self.FairBalance(X_train, y_train, sensitive_attributes)
        self.clf.fit(X_train, y_train, sample_weight)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        preds = self.clf.predict(X)
        return preds

    def FairBalance(self, X, y, A):
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
        sample_weight = np.array([1.0] * len(y))
        for key in groups_class:
            weight = group_weight[key[:-1]] / len(groups_class[key])
            for i in groups_class[key]:
                sample_weight[i] = weight
        sample_weight = sample_weight * len(y) / sum(sample_weight)
        return sample_weight


    #This should be moved to be part of the classes implementing Data interface
    def data_preprocessing(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        numerical_columns = numerical_columns_selector(X)
        numerical_preprocessor = StandardScaler()

        categorical_columns_selector = selector(dtype_include=object)
        categorical_columns = categorical_columns_selector(X)
        categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore')

        preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

        preprocessor.fit_transform(X)

    #Modify the classes methods of get_train_data(), get_test_data() to account for this?
    def train_test_split(self, X, y, sensitive_attributes, test_size = 0.5):
        #Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(y)):
            key = tuple([X[a][i] for a in sensitive_attributes] + [y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key]) * test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)

        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y[train]
        y_test = y[test]
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        return X_train, X_test, y_train, y_test


