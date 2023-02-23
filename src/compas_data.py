from .data_interface import Data
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class CompasData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, tests_ratio=0.2) -> None:
        # does reading and cleaning go here or do we add extra functions for that?
        dataset = open('../data/compas-scores-raw.csv')
        lines = dataset.readlines()
        self.headers = lines[0].split(',')
        self.data = {}

        # set column headers as keys in dictionary
        for col in self.headers:
            self.data[col] = []

        # set data for keys in dictionary
        for row in range(1, len(lines)):
            vals = lines[row].split(',')
            # update key value pair
            for i in range(0, len(self.headers)):
                curr = self.data.get(list(self.data)[i])
                curr.append(vals[i])
                self.data[list(self.data)[i]] = curr

        self.X = pd.DataFrame(self.data)

        #print(self.headers)
        #print(self.data)
        #print(self.X)
        # raise NotImplementedError

    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        # returns (X, y)
        return (self.X, self.headers)
        # raise NotImplementedError

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        test_col = self.get_sensitive_column_names()
        test_data = {}
        for i in range(0, len(test_col)):
            test_data[test_col[i]] = self.data.get(list(self.data)[i])

        test_dataframe = pd.DataFrame(test_data)
        return (test_dataframe, test_col)
        # returns (X, y)
        #raise NotImplementedError

    def get_sensitive_column_names(self) -> List[str]:
        # returns a list of names
        sensitive_col = []
        common_sensitive_attributes = ['race', 'gender', 'sex', 'ethnic', 'birth']

        # check whether column name indicates sensitive attribute
        for attr in common_sensitive_attributes:
            for column in self.headers:
                if attr in column.casefold():
                    # print('checking ' + attr + ' against ' + column)
                    sensitive_col.append(column)
        return sensitive_col
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError