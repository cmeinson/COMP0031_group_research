import os
from src import *
import numpy as np
import matplotlib.pyplot as plt
import time

n_repetitions = 50
same_data_split = True
results_filename = "fairbalance_reweighing"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
mls = [(Tester.BASE_ML, Model.LG_R, None, "FairBalance", other_fb), 
       (Tester.FAIRBALANCE, Model.LG_R, None, "FairBalance", other_fb), 

       (Tester.REWEIGHING, Model.LG_R, None, "FairBalance", other), 
]
metric_names = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre, oth in mls:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth)

