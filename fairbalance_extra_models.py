import os
from src import *
import numpy as np
import matplotlib.pyplot as plt
import time

"""
Testing the performance in mitigating bias with different ML models that
support fit(X,y,[sample_weight]) and sparse arrays
"""
n_repetitions = 5
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
    for dataset in datasets:
        if dataset == Tester.ADULT_D:
            for model in models:
                start = time.time()
                tester.run_test(metric_names, dataset, Tester.FAIRBALANCE, model, None, n_repetitions, same_data_split, data_preprocessing="FairBalance", other = other_fb)
                end = time.time()
                models_runtimes_adult.append((end-start)/n_repetitions)
        elif dataset == Tester.COMPAS_D:
            for model in models:
                start = time.time()
                tester.run_test(metric_names, dataset, Tester.FAIRBALANCE, model, None, n_repetitions, same_data_split, data_preprocessing="FairBalance", other = other_fb)
                end = time.time()
                models_runtimes_compas.append((end-start)/n_repetitions)

print("Adult: ", models_runtimes_adult)
fig, ax = plt.subplots()
ax.bar(models, models_runtimes_adult)
ax.set_xlabel('Model')
ax.set_ylabel('Avg of running time')
ax.set_title('Avg of running times of different ML models on Adult dataset')
plt.legend(['Time (sec)'], loc='upper right')
plt.show()

print("Compas: ", models_runtimes_compas)
fig, ax = plt.subplots()
ax.bar(models, models_runtimes_compas)
ax.set_xlabel('Model')
ax.set_ylabel('Avg of running time')
ax.set_title('Avg of running times of different ML models on Compas dataset')
plt.legend(['Time (sec)'], loc='upper right')
plt.show()