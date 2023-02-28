import os
from src import *

# Just an example for now

repetitions = 5
results_filename = "testtt"
other = {"save_intermediate": True}

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
mls = [(Tester.BASE_ML, BaseModel.LR, None, "FairBalance"), 
       (Tester.FAIRBALANCE, FairBalanceModel.LOGR, None, "FairBalance"), 
       (Tester.BASE_ML, BaseModel.RF, None, None), 
       (Tester.FAIRMASK, FairMaskModel.RF, FairMaskModel.DT, None)
]

metric_names = Metrics.get_all_names()

# All the following constants are just strings so caould also easily be read from a file or whatever. just for convenience:

for i in range(1): # diff data splits
    tester = Tester(os.path.join("results",results_filename))
    for dataset in datasets:
        for bias_mit, method, method2, pre in mls:
            X, y, preds = tester.run_test(metric_names, dataset, bias_mit, method, method2, repetitions, data_preprocessing=pre, other = other)