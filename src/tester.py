from os import path
import pandas as pd
from .data_interface import Data, DummyData
from .ml_interface import Model, BaseModel
from .metrics import Metrics
from typing import List, Dict, Any


class Tester:
    def __init__(self, output_filename) -> None:
        self._initd_data = {}
        self._file = output_filename + ".csv"

    def run_test(self, metric_names: List[str], dataset: str, bias_mit: str, 
                 ml_method: str, bias_ml_method: str = None,
                 sensitive_attr: List[str] = None, other={}):

        data = self._get_dataset(dataset)
        model = self._get_model(bias_mit)

        X, y = data.get_train_data()
        if not sensitive_attr:
            sensitive_attr = data.get_sensitive_column_names()
        model.fit(X, y, sensitive_attr, ml_method, bias_ml_method, other)

        X, y = data.get_test_data()
        preds = model.predict(X, other)
        evals = self._evaluate(Metrics(X, y, preds), metric_names)
        self.save_test_results(evals, dataset, bias_mit, ml_method, bias_ml_method, sensitive_attr)
        return X, y, preds

    def _evaluate(self, metrics: Metrics, metric_names):
        evals = {}
        if "acc" in metric_names:
            evals["acc"] = metrics.accuracy()
            print(evals)
        if "rec" in metric_names:
            pass
        if "etc" in metric_names:
            pass
        return evals    

    def _get_dataset(self, name) -> Data:
        if name in self._initd_data:
            return self._initd_data[name]

        data = None
        if name == "Adult":
            pass
        elif name == "Compas":
            pass
        elif name == "Default":
            data = DummyData()  # default
        else:
            raise RuntimeError("Incorrect dataset name ", name)

        self._initd_data[name] = data
        return data

    def _get_model(self, name) -> Model:
        if name == "FairMask":
            pass
        elif name == "FairBalance":
            pass
        elif name == "Default":
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
            "Sensitive attrs": sensitive_attr
        }
        entry.update({key: [evals[key]] for key in evals})
        res = pd.DataFrame(entry)
        

        if self._file and path.exists(self._file):
            res = pd.concat([res, pd.read_csv(self._file)], ignore_index=True)
        res.to_csv(self._file, index=False)