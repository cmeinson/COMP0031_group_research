from sklearn.model_selection import train_test_split
from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd


class CompasData(Data):
    race_pos_label = "Caucasian"
    race_all_splits = ["Caucasian", "African-American", "Other", "Hispanic"]
   
    def __init__(self, preprocessing=None, test_ratio=0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        self._test_ratio = test_ratio
        self.data = pd.read_csv('data/compas-scores-two-years.csv')

        # Do default preprocessing
        self.pre_processing()

        self._X = pd.DataFrame(self.data, columns=["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"])
        self._y = self.data['Probability'].to_numpy()

        if preprocessing == "FairBalance":
            self._preprocess_fairbalance(self._X)
        elif preprocessing == "FairMask":
            pass # Nothing special as of yet

        # Create train-test split
        self.new_data_split()

    def new_data_split(self) -> None:
        """Changes the data split"""
        self._X_train, self._X_test_cat, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=self._test_ratio)
        self._X_test = self.copy_with_bin_race(self._X_test_cat, self.race_pos_label)
        self._X_train = self.copy_with_bin_race(self._X_train, self.race_pos_label)

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
        # preprocessing done according to preprocessing.ipynb
        self.data = self.data.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age', 'juv_fel_count', 'decile_score',
             'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text', 'screening_date',
             'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
             'start', 'end', 'event'], axis=1)

        self.data = self.data.dropna()
        #self.data['race'] = np.where(self.data['race'] != self.race_pos_label, 0, 1)
        self.data['sex'] = np.where(self.data['sex'] == 'Female', 0, 1)
        self.data['age_cat'] = np.where(self.data['age_cat'] == 'Greater than 45', 45, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == '25 - 45', 25, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == 'Less than 25', 0, self.data['age_cat'])
        self.data['c_charge_degree'] = np.where(self.data['c_charge_degree'] == 'F', 1, 0)

        self.data.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
        self.data['Probability'] = np.where(self.data['Probability'] == 0, 1, 0)

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex', 'race'] # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError

compas = CompasData()