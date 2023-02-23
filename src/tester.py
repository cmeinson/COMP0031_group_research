from os import path
import pandas as pd
from .data_interface import Data, DummyData
from .compas_data import CompasData
from .ml_interface import Model, BaseModel
from .metrics import Metrics
from typing import List, Dict, Any
from .fairbalance import FairBalanceModel

class Tester:
    # Avalable datasets for testing:
    ADULT_D = "Adult Dataset"
    COMPAS_D = "Compas Dataset"
    DUMMY_D = "Dummy Dataset"

    # Available bias mitigation methods
    FAIRBALANCE = "FairBalance Bias Mitigation"
    FAIRMASK = "FairMask Bias Mitigation"
    BASE_ML = "No Bias Mitigation"

    def __init__(self, output_filename) -> None:
        self._initd_data = {}
        self._file = output_filename + ".csv"

    def run_test(self, metric_names: List[str], dataset: str,
                 bias_mit: str, ml_method: str, bias_ml_method: str = None,
                 data_preprocessing: str = None, sensitive_attr: List[str] = None, other={}):

        data = self._get_dataset(dataset,data_preprocessing)
        model = self._get_model(bias_mit)

        X, y = data.get_train_data()
        if not sensitive_attr:
            sensitive_attr = data.get_sensitive_column_names()
        model.train(X, y, sensitive_attr, ml_method, bias_ml_method, other)

        X, y = data.get_test_data()
        preds = model.predict(X, other)
        evals = self._evaluate(Metrics(X, y, preds), metric_names)
        self.save_test_results(evals, dataset, bias_mit, ml_method, bias_ml_method, sensitive_attr)
        return X, y, preds


    def _evaluate(self, metrics: Metrics, metric_names: List[str]):
        evals = {}
        for name in metric_names:
            evals[name] = metrics.get(name)
        return evals 

    def _get_dataset(self, name:str, preprocessing:str) -> Data:
        dataset_description = name if not preprocessing else name+preprocessing
        if dataset_description in self._initd_data:
            return self._initd_data[dataset_description]

        data = None
        if name == self.ADULT_D:
            pass
        elif name == self.COMPAS_D:
            data = CompasData(preprocessing)
        elif name == self.DUMMY_D:
            data = DummyData(preprocessing)  # default
        else:
            raise RuntimeError("Incorrect dataset name ", name)

        self._initd_data[dataset_description] = data
        return data

    def _get_model(self, name) -> Model:
        if name == self.FAIRMASK:
            pass
        elif name == self.FAIRBALANCE:
            return FairBalanceModel()
        elif name == self.BASE_ML:
            return BaseModel()  # default
        else:
            raise RuntimeError("Incorrect method name ", name)

    def save_test_results(self, evals: Dict[str, Any], dataset: str,
                          bias_mit: str, ml_method: str, bias_ml_method: str,
                          sensitive_attr: List[str], other={}):

        entry = {
            "timestamp": [pd.Timestamp.now()],
            "Dataset": [dataset],
            "Bias Mitigation": [bias_mit],
            "ML method": [ml_method],
            "ML bias mit": [bias_ml_method],
            "Sensitive attrs": [sensitive_attr]
        }
        entry.update({key: [evals[key]] for key in evals})
        res = pd.DataFrame(entry)
        

        if self._file and path.exists(self._file):
            res = pd.concat([res, pd.read_csv(self._file)], ignore_index=True)
        res.to_csv(self._file, index=False)