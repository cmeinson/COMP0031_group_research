import os
from src import *
import numpy as np
import matplotlib.pyplot as plt

from src import reweighing

n_repetitions = 3
same_data_split = True
results_filename = "FairmMask Results 2"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")

ml_methods_for_testing = [Model.DT_C, Model.KN_C, Model.NN_C, Model.NB_C, Model.GB_C, Model.RF_C,Model.SV_C]
datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
bias_mit = [Tester.BASE_ML, Tester.FAIRMASK, Tester.REWEIGHING, Tester. FAIRBALANCE]

increased_bias = 0

def more_bias(fairmask,noth):
       keys = list(noth.keys())

       for i in range(len(fairmask)):
              if i not in [16,17,18,19]:
                     return i,(fairmask[keys[i]] > noth[keys[i]])

if __name__ == "__main__":
       tester = Tester(results_file)
       file = open("times.txt","w")
       total = 0
       metrics_increased_bias = {}
       for method1 in [Model.LG_R]:
              models_runtimes_adult = []
              models_runtimes_compas = []
              for method2 in ml_methods_for_testing:
                     for dataset in datasets:
                            for ml_or_not in bias_mit:
                                   time, evals = tester.run_test(metric_names, dataset, ml_or_not, method1, method2,  n_repetitions, 
                                                 same_data_split, data_preprocessing=None, other = other)
                                   if dataset == tester.ADULT_D:
                                          models_runtimes_adult.append(time/n_repetitions)
                                   else:
                                          models_runtimes_compas.append(time/n_repetitions)

                                   file.write(f"For method1 = {method1} and method2 = {method2} in dataset {dataset} time needed was {time}\n")

                                   if ml_or_not == "No Bias Mitigation":
                                          bias_noth = evals
                                   elif ml_or_not == "FairMask Bias Mitigation":
                                          bias_fm = evals
                            metric, biased = more_bias(bias_fm, bias_noth)
                            if biased:
                                   increased_bias+=1

                                   metrics_increased_bias[metric] = metrics_increased_bias.get(metric, 0) + 1
                            total+=1
                     
              # # print("Adult: ", models_runtimes_adult)
              # fig, ax = plt.subplots()
              # ax.bar(ml_methods_for_testing, models_runtimes_adult)
              # ax.set_xlabel('Model')
              # ax.set_ylabel('Avg of running time')
              # ax.set_title(f'Avg of running times of different ML models on Adult dataset for method {method1}')
              # plt.legend(['Time (sec)'], loc='upper right')
              # plt.show()

              # # print("Compas: ", models_runtimes_compas)
              # fig, ax = plt.subplots()
              # ax.bar(ml_methods_for_testing, models_runtimes_compas)
              # ax.set_xlabel('Model')
              # ax.set_ylabel('Avg of running time')
              # ax.set_title(f'Avg of running times of different ML models on Compas dataset for method {method1}')
              # plt.legend(['Time (sec)'], loc='upper right')
              # plt.show()

       print(f"Increased bias in {increased_bias} instances out of {total}")
       print(metrics_increased_bias)
       file.close()
