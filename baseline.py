import os
from src import *
import numpy as np
import matplotlib.pyplot as plt
import time

n_repetitions = 50
same_data_split = True
results_filename = "fairbalance_fairmask"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
#FairMask preprocessing
mls1 = [(Tester.FAIRMASK, Model.RF_C, Model.DT_R, None, other_fb), 
       (Tester.FAIRBALANCE, Model.LG_R, None, None, other_fb), 
       (Tester.REWEIGHING, Model.LG_R, None, None, other_fb), 
       (Tester.BASE_ML, Model.LG_R, None, None, other_fb), 
]

#FairBalance preprocessing
mls2 = [(Tester.FAIRMASK, Model.RF_C, Model.RF_C, "FairBalance", other_fb), 
       (Tester.FAIRBALANCE, Model.LG_R, None, "FairBalance", other_fb), 
       (Tester.REWEIGHING, Model.LG_R, None, "FairBalance", other_fb), 
       (Tester.BASE_ML, Model.LG_R, None, "FairBalance", other_fb), 
]

metric_names = [Metrics.ACC, Metrics.F1, Metrics.PRE, Metrics.REC ,Metrics.M_EOD, Metrics.M_AOD, Metrics.DF, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre, oth in mls2:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth)

