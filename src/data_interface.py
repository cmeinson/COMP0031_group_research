from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    race_pos_label = None
    race_all_splits = []
    sensitive = []

    def __init__(self, preprocessing:str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        raise NotImplementedError
    
    def _all_races_in_test_and_train(self):
        if (self._X_test_cat is None): return False
        return set(self._X_test_cat['race'].unique()) == set(self._X_train_cat['race'].unique())
    
    def new_data_split(self) -> None:
        """Changes the data split"""
        self._X_test_cat = None
        self._X_train_cat = None
        while (not self._all_races_in_test_and_train()): # to avoid crashing tests
            self._X_train_cat, self._X_test_cat, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=self._test_ratio)
        #print("in test")
        #print(self._X_test_cat['race'].value_counts())
        self.set_race_pos_label(self.race_pos_label)


    def set_race_pos_label(self, new):
        self.race_pos_label = new
        self._X_test = self.copy_with_bin_race(self._X_test_cat, self.race_pos_label)
        self._X_train = self.copy_with_bin_race(self._X_train_cat, self.race_pos_label)


    def get_all_test_data(self) -> List[Tuple[pd.DataFrame, np.array]]:
        out = []
        for l in self.race_all_splits:
            out.append((self.copy_with_bin_race(self._X_test_cat, l), self._y_test.copy()))
        return out

    def copy_with_bin_race(self, X, pos_label):
        X_new = X.copy()
        X_new['race'] = np.where(X_new['race'] != pos_label, 0, 1)
        return X_new
    
    def merge_races(self, remove: List[str], into ):
        for rem in remove:
            self.data['race'] = np.where(self.data['race'] == rem, into, self.data['race'])

    def split_cat_cols(self, attr):
        cats = self.data.dropna()[attr].unique()
        self._X_test = self._split_col(attr, cats, self._X_test_cat)
        self._X_train = self._split_col(attr, cats, self._X_train_cat)
        if attr in self.sensitive:
            self.sensitive = list(set(self.sensitive) - set([attr])) + list(cats)
        
    def _split_col(self, attr, cats, X):
        for cat in cats:
            X[cat] = np.where(X[attr] == cat, 1, 0)
        
        return X.drop([attr], axis=1)
    
    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        """Returns the training data where
        X: is the df with all attriutes, with accordig column names
        y: the outcome for each row (e.g. the default credit, is income above 50k, did reoffend?)
        Duplicares everything for safety reasons:)

        :return: training data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_train.copy(), self._y_train.copy())

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        """
        :return: test data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_test.copy(), self._y_test.copy())

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        raise NotImplementedError
    
    def update_race_pos_label(self, new):
        self.race_pos_label = new

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return self.sensitive
    


class DummyData(Data):
    sensitive = ["sensitive", "sensitive2"]

    def __init__(self, preprocessing = None, test_ratio=0.2) -> None:
        data = [[0.31, 0, 0], [0.41, 0, 1], [0.51, 0, 0], [0.71, 0, 1], [0.81, 0, 0], [0.91, 0, 1], 
                [0.2, 1, 0], [0.3, 1, 1], [0.4, 1, 0], [0.5, 1, 1], [0.6, 1, 0], [0.7, 1, 1]]
        self._X = pd.DataFrame(data, columns=["attr", "sensitive", "sensitive2"])
        self._y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,])
        self._test_ratio = test_ratio
        self.new_data_split()        



