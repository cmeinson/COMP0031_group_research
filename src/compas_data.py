from sklearn.model_selection import train_test_split
from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd

"""
African-American    3696
Caucasian           2454
Hispanic             637
Other                377
Asian                 32 (merged with Other)
Native American       18 (merged with Other)
""" 
class CompasData(Data):
    race_pos_label = "Caucasian"
    race_all_splits = ["Caucasian", "Other", "Hispanic", "African-American"]
    sensitive = ['sex', 'race']
   
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
        
        if preprocessing == "FairBalance merge races" or preprocessing == "FairMask merge races":
            self.race_all_splits = ["Caucasian", "Other", "African-American"]
            self.merge_races(["Hispanic", "Asian", "Native American"], "Other")
        else:
            self.race_all_splits = ["Caucasian", "Other", "Hispanic", "African-American"]
            self.merge_races(["Asian", "Native American"], "Other")
            

        # Do default preprocessing
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

    def fairbalance_columns(self, X):
        return X[['sex', 'age', 'age_cat', 'race',
                  'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'priors_count', 'c_charge_degree', 'c_charge_desc']]
    
    def fairmask_columns(self, X):
        return X[["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"]]

    def pre_processing(self):
        # preprocessing done according to preprocessing.ipynb
        self.data = self.data.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'decile_score',
             'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'is_recid', 'r_case_number',
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

