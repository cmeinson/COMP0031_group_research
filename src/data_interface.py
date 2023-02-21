from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, tests_ratio = 0.2) -> None:
        # does reading and cleaning go here or do we add extra functions for that?
        raise NotImplementedError

    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        # returns (X, y)
        raise NotImplementedError

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        # returns (X, y)
        raise NotImplementedError

    def get_sensitive_column_names(self) -> List[str]:
        # returns a list of names
        raise NotImplementedError

    #def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError



class DummyData(Data):
    def __init__(self, test_ratio = 0.2) -> None:
        data = [[0.1, 0],[0.4, 0],[0.5, 0],[0.3, 1],[0.4, 1],[0.7, 1]]
        self._X = pd.DataFrame(data, columns=["attr", "sensitive"])
        self._y = np.array([0,1,1,0,1,1])
        self._X_train,  self._X_test, self._y_train, self._y_test =  train_test_split(self._X, self._y, test_size = test_ratio)

    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        return (self._X_train, self._y_train)

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        return (self._X_test, self._y_test)

    def get_sensitive_column_names(self) -> List[str]:
        return ["sensitive"]
