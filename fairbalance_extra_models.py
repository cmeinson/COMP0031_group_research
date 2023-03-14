import os
from src import *
import numpy as np
import matplotlib.pyplot as plt
import time

"""
Testing the performance in mitigating bias with different ML models that
support fit(X,y,[sample_weight]) and sparse arrays
"""
n_repetitions = [1, 5, 10, 20]
same_data_split = True
results_filename = "fairbalance_diff_models"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
models = [Model.LG_R, Model.DT_C, Model.RF_C, Model.SV_C, Model.AB_C]
metric_names = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]
results_file = os.path.join("results",results_filename +".csv")
models_runtimes_adult = []
models_runtimes_compas = []

if __name__ == "__main__":
    tester = Tester(results_file)
    for n in n_repetitions:
        print("No. of repetitions: ", n)
        for dataset in datasets:
            if dataset == Tester.ADULT_D:
                for model in models:
                    start = time.time()
                    tester.run_test(metric_names, dataset, Tester.FAIRBALANCE, model, None, n, same_data_split, data_preprocessing="FairBalance", other = other_fb)
                    end = time.time()
                    models_runtimes_adult.append((end-start)/n)
            elif dataset == Tester.COMPAS_D:
                for model in models:
                    start = time.time()
                    tester.run_test(metric_names, dataset, Tester.FAIRBALANCE, model, None, n, same_data_split, data_preprocessing="FairBalance", other = other_fb)
                    end = time.time()
                    models_runtimes_compas.append((end-start)/n)
        print("Adult runtimes: ", models_runtimes_adult)
        print("Compas runtimes: ", models_runtimes_compas)

