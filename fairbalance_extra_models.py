import os
from src import *

"""
Testing the performance in mitigating bias with different ML models that
support fit(X,y,[sample_weight]) and sparse matrix
"""
n_repetitions = 5
same_data_split = True
results_filename = "fairbalance"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
models = [Model.LG_R, Model.DT_C, Model.RF_C, Model.SV_C, Model.AB_C]
metric_names = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]
results_file = os.path.join("results",results_filename +".csv")

if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for model in models:
            tester.run_test(metric_names, dataset, Tester.FAIRBALANCE, model, None, n_repetitions, same_data_split, data_preprocessing="FairBalance", other = other_fb)