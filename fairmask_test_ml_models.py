import os
from src import *
import time
import numpy as np
import matplotlib.pyplot as plt

n_repetitions = 2
same_data_split = True
results_filename = "heheh"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")

ml_methods_for_testing = [Model.DT_C, Model.RF_C , Model.KN_C, Model.SV_C, Model.NN_C, Model.NB_C, Model.GB_C]
datasets =  [Tester.ADULT_D, Tester.COMPAS_D]

if __name__ == "__main__":
    tester = Tester(results_file)

    for method1 in ml_methods_for_testing:
       models_runtimes_adult = []
       models_runtimes_compas = []
       for method2 in ml_methods_for_testing:
              for dataset in datasets:
                     start = time.time()
                     tester.run_test(metric_names, dataset, Tester.FAIRMASK, method1, method2,  n_repetitions, same_data_split, data_preprocessing=None, other = other)
                     end = time.time()     
                     if dataset == tester.ADULT_D:
                            models_runtimes_adult.append((end-start)/n_repetitions)
                     else:
                            models_runtimes_compas.append((end-start)/n_repetitions)
              
       # print("Adult: ", models_runtimes_adult)
       fig, ax = plt.subplots()
       ax.bar(ml_methods_for_testing, models_runtimes_adult)
       ax.set_xlabel('Model')
       ax.set_ylabel('Avg of running time')
       ax.set_title(f'Avg of running times of different ML models on Adult dataset for method {method1}')
       plt.legend(['Time (sec)'], loc='upper right')
       plt.show()

       # print("Compas: ", models_runtimes_compas)
       fig, ax = plt.subplots()
       ax.bar(ml_methods_for_testing, models_runtimes_compas)
       ax.set_xlabel('Model')
       ax.set_ylabel('Avg of running time')
       ax.set_title(f'Avg of running times of different ML models on Compas dataset for method {method1}')
       plt.legend(['Time (sec)'], loc='upper right')
       plt.show()

              
