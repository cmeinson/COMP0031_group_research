from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class AdultData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, preprocessing:str = None, tests_ratio = 0.2) -> None:
        """
        - reads the according dataset from the data folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """

        self.dataset_orig = pd.read_csv('data/adult.csv')

        # Do default pre-processing from Preprocessing.ipynb
        self.pre_processing()

            # Split into input and output
            self._X = pd.DataFrame(self.dataset_orig, columns=["age", "education-num", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"])
            self._y = self.dataset_orig['Probability'].to_numpy()

        elif preprocessing == "FairBalancePreprocessing":
            self.FairBalancePreprocessing()
        
        if preprocessing == "FairBalance":
            self._preprocess_fairbalance(self._X)
        elif preprocessing == "FairMask":
            pass # Nothing special as of yet
        
        # Create train-test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=tests_ratio)

    def pre_processing(self):
        self.dataset_orig = self.dataset_orig.dropna()
        self.dataset_orig = self.dataset_orig.drop(['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'], axis=1)

        # Binarize sex, race and income (probability)
        self.dataset_orig['sex'] = np.where(self.dataset_orig['sex'] == 'Male', 1, 0)
        self.dataset_orig['race'] = np.where(self.dataset_orig['race'] != 'White', 0, 1)
        self.dataset_orig['Probability'] = np.where(self.dataset_orig['Probability'] == '<=50K', 0, 1)

        # Discretize age
        self.dataset_orig['age'] = np.where(self.dataset_orig['age'] >= 70, 70, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 60 ) & (self.dataset_orig['age'] < 70), 60, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 50 ) & (self.dataset_orig['age'] < 60), 50, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 40 ) & (self.dataset_orig['age'] < 50), 40, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 30 ) & (self.dataset_orig['age'] < 40), 30, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 20 ) & (self.dataset_orig['age'] < 30), 20, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 10 ) & (self.dataset_orig['age'] < 10), 10, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where(self.dataset_orig['age'] < 10, 0, self.dataset_orig['age'])

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        
        #return ['sex', 'race', 'age', 'Probability'] (is age, income a sensitive attribute?)
        return ['race', 'sex']

    #Probability we need some general methods if fairmask uses some common stuff
    def FairBalancePreprocessing(self):
        self.dataset_orig = self.dataset_orig.dropna()
        self.dataset = self.dataset_orig[["age", "workclass", "education-num" , "marital-status", "occupation", "relationship", "race",
                                          "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "Probability"]]

        self.dataset['sex'] = np.where(self.dataset['sex'] == 'Male', 1, 0)
        self.dataset['race'] = np.where(self.dataset['race'] != 'White', 0, 1)
        self.dataset['Probability'] = np.where(self.dataset['Probability'] == '<=50K', 0, 1)

        independent = self.dataset.keys().tolist()
        dependent = independent.pop(-1)

        self._X = self.dataset[independent]
        self._y = np.array(self.dataset[dependent])

        
