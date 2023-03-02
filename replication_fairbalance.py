import os
from src import *
import warnings
warnings.filterwarnings("ignore")

results_filename = os.path.join("results","FairBalance.csv")

metric_names = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]

datasets = [Tester.ADULT_D,Tester.COMPAS_D]
mls = [(Tester.BASE_ML, Model.LG_R, None, "FairBalance"), 
       (Tester.FAIRBALANCE, Model.LG_R, None, "FairBalance"),
      ]

n_repetitions = 10
same_data_split = True

if __name__ == "__main__":
    tester = Tester(results_filename)
    for dataset in datasets:
        for bias_mit, method, method2, pre in mls:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre)






