from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
FairBalance Adult Data columns: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, 
                    race, sex, capital-gain, capital-loss, hours-per-week, native-country

FairMask Adult Data columns: age, education-num, race, sex, capital-gain, capital-loss, hours-per-week
"""

class AdultData(Data):
    race_pos_label = "White"
    race_all_splits = ["White", 'Asian-Pac-Islander', "Black", 'Amer-Indian-Eskimo', 'Other']

    def __init__(self, preprocessing:str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the data folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train
        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        self._test_ratio = test_ratio
        self.data = pd.read_csv('data/adult.csv')

    

        self.race_all_splits = ["White", 'Asian-Pac-Islander', "Black", 'Amer-Indian-Eskimo', 'Other']
        if preprocessing == "FairBalance merge races" or preprocessing == "FairMask merge races":
            self.race_all_splits = ["White", 'Asian-Pac-Islander', "Black", 'Other']
            self.merge_races()

        # Do default pre-processing from Preprocessing.ipynb
        self.pre_processing()

        # Split into input and output

        self._X = pd.DataFrame(self.data)
        self._y = self.data['Probability'].to_numpy()
        
        if preprocessing == "FairBalance" or preprocessing == "FairBalance merge races":
            self._X = self.fairbalance_columns(self._X)
        else:
            self._X = self.fairmask_columns(self._X)


        # Create train-test split
        self.new_data_split()   

    def merge_races(self, remove: List[str] = ['Amer-Indian-Eskimo'], into = "Other"):
        for rem in remove:
            self.data['race'] = np.where(self.data['race'] == rem, into, self.data['race'])

    def new_data_split(self) -> None:
        """Changes the data split"""
        self._X_train_cat, self._X_test_cat, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=self._test_ratio)
        self.update_race_pos_label(self.race_pos_label)

    def update_race_pos_label(self, new):
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

    def pre_processing(self):
        self.data = self.data.dropna()

        # Binarize sex, race and income (probability)
        self.data['sex'] = np.where(self.data['sex'] == 'Male', 1, 0)
        #self.dataset_orig['race'] = np.where(self.dataset_orig['race'] != 'White', 0, 1)
        self.data['Probability'] = np.where(self.data['Probability'] == '<=50K', 0, 1)

        # Discretize age
        self.data['age'] = np.where(self.data['age'] >= 70, 70, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 60 ) & (self.data['age'] < 70), 60, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 50 ) & (self.data['age'] < 60), 50, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 40 ) & (self.data['age'] < 50), 40, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 30 ) & (self.data['age'] < 40), 30, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 20 ) & (self.data['age'] < 30), 20, self.data['age'])
        self.data['age'] = np.where((self.data['age'] >= 10 ) & (self.data['age'] < 10), 10, self.data['age'])
        self.data['age'] = np.where(self.data['age'] < 10, 0, self.data['age'])

    def fairmask_columns(self, X):
        return X.drop(['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'Probability'], axis=1)

    def fairbalance_columns(self, X):
        return X.drop(['fnlwgt', 'education', 'Probability'], axis=1)
    
    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        
        #return ['sex', 'race', 'age', 'Probability'] (is age, income a sensitive attribute?)
        return ['race', 'sex']