import os
from src import *

# Just an example for now

n_repetitions = 1
same_data_split = True
results_filename = "heheh"
other = {"save_intermediate": False}

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
mls = [(Tester.BASE_ML, Model.LG_R, None, "FairBalance"), 
       (Tester.FAIRBALANCE, Model.LG_R, None, "FairBalance"), 
       (Tester.BASE_ML, Model.RF_C, None, None), 
       (Tester.FAIRMASK, Model.RF_C, Model.DT_R, None)
]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre in mls:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = other)