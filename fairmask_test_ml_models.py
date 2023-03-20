from email.charset import BASE64
import os
from src import *
import numpy as np
import matplotlib.pyplot as plt

from src import reweighing

n_repetitions = 3
same_data_split = True
results_filename = "FairmMask Results"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")

ml_methods_for_testing = [Model.DT_C, Model.KN_C, Model.NN_C, Model.NB_C, Model.GB_C, Model.RF_C,Model.SV_C]
datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
bias_mit = [Tester.BASE_ML, Tester.FAIRMASK]

increased_bias = 0


def more_bias(bias1,bias2):
       keys = list(bias1.keys())

       for i in range(len(bias1)):
              if i in [0,1,4,5,6,7,12,13,20,21,22,23]:
                     return i,(bias1[keys[i]] > bias2[keys[i]])
              elif i in [2,3,10,11,18,19]:
                     return i,(bias1[keys[i]] < bias2[keys[i]])

if __name__ == "__main__":
    tester = Tester(results_file)

    for method1 in ml_methods_for_testing:
       models_runtimes_adult = []
       models_runtimes_compas = []
       for method2 in ml_methods_for_testing:
              for dataset in datasets:
                     
                     time, evals = tester.run_test(metric_names, dataset, Tester.FAIRMASK, method1, method2,  n_repetitions, 
                                   same_data_split, data_preprocessing=None, other = other)
                     if dataset == tester.ADULT_D:
                            models_runtimes_adult.append(time/n_repetitions)
                     else:
                            models_runtimes_compas.append(time/n_repetitions)

                     #        if bias == "No Bias Mitigation":
                     #               bias_noth = evals
                     #        elif bias == "FairMask Bias Mitigation":
                     #               bias_fm = evals
                     # metric, biased = more_bias(bias_fm, bias_noth)
                     # if biased:
                     #        increased_bias+=1

                     #        print(f"Increased bias in {increased_bias} instances, on {metric}")
                     # else:
                     #        print("noth")
              
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
