import os
from src import *
import time
import numpy as np
import matplotlib.pyplot as plt

"""
Testing the performance in mitigating bias with a single protected attribute at a time 
vs multiple protected attributes
"""
n_repetitions = 5
same_data_split = True
results_filename = "fairbalance_single_attr"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
model = Model.LG_R
sensitive_attr = ['race', 'sex']
metric_names_single = [Metrics.ACC, Metrics.F1, Metrics.EOD, Metrics.AOD]
metric_names_multiple = [Metrics.ACC, Metrics.F1, Metrics.M_EOD, Metrics.M_AOD]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        #Passing 1 sensitive attribute
        for attr in sensitive_attr:
            tester.run_test(metric_names_single, dataset, Tester.FAIRBALANCE, model, None, n_repetitions, same_data_split, data_preprocessing="FairBalance", sensitive_attr = [attr], other = other_fb)
        #Passing 2 sensitive attributes
        tester.run_test(metric_names_multiple, dataset, Tester.FAIRBALANCE, model, None, n_repetitions, same_data_split, data_preprocessing="FairBalance", other = other_fb)
    



