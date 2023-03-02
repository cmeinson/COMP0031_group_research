import os
from src import *
import warnings
warnings.filterwarnings("ignore")

results_filename = "FairBalance"

metric_names = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]

datasets = [Tester.ADULT_D,Tester.COMPAS_D]
mls = [(Tester.FAIRBALANCE, FairBalanceModel.LOGR, None, "FairBalance"),
       (Tester.BASE_ML, BaseModel.LR, None, "FairBalance")
      ]

repetitions = 10

for i in range(1): # diff data splits
    tester = Tester(os.path.join("results",results_filename))
    for dataset in datasets:
        for bias_mit, method, method2, preprocessing in mls:
            X, y, preds = tester.run_test(metric_names, dataset, bias_mit, method, method2, repetitions, data_preprocessing=preprocessing)






