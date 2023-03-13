from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd

"""
FairBalance Adult Data columns: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, 
                    race, sex, capital-gain, capital-loss, hours-per-week, native-country

FairMask Adult Data columns: age, education-num, race, sex, capital-gain, capital-loss, hours-per-week
"""

class AdultData(Data):
    race_pos_label = "White"
    race_all_splits = ["White", 'Asian-Pac-Islander', "Black", 'Amer-Indian-Eskimo', 'Other']
    sensitive = ['sex', 'race']

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
            self.merge_races(['Amer-Indian-Eskimo'],  "Other")

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
