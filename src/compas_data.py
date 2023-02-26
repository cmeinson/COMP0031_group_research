from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

class CompasData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    # does reading and cleaning go here or do we add extra functions for that?
    def __init__(self, preprocessing=None, tests_ratio=0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        self.data = pd.read_csv('data/compas-scores-two-years.csv')

        # Do default preprocessing
        self.pre_processing()

        self._X = pd.DataFrame(self.data)
        self._y = self.data['Probability'].to_numpy()

        if preprocessing == "FairBalance":
            self.FairBalance(self._X)
        elif preprocessing == "FairMask":
            pass # Nothing special as of yet

        # Create train-test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=tests_ratio)

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

        self.data['race'] = np.where(self.data['race'] != 'Caucasian', 0, 1)
        self.data['sex'] = np.where(self.data['sex'] == 'Female', 0, 1)
        self.data[''] = np.where(self.data['age_cat'] == 'Greater than 45', 45, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == '25 - 45', 25, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == 'Less than 25', 0, self.data['age_cat'])

        self.data.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
        self.data['Probability'] = np.where(self.data['Probability'] == 0, 1, 0)

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex', 'age_cat', 'race']
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError

    def FairBalance(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_processor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_processor = StandardScaler()
        self.processor = ColumnTransformer([
            ('OneHotEncoder', categorical_processor, categorical_columns),
            ('StandardScaler', numerical_processor, numerical_columns)])
