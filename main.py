import os
from src import *

# Just an example for now

n_repetitions = 2
same_data_split = True
results_filename = "heheh"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [Tester.ADULT_D, Tester.COMPAS_D]
mls = [(Tester.FAIRMASK, Model.GB_C, Model.KN_C, None, other), 
       (Tester.BASE_ML, Model.RF_C, None, None, other), 
       (Tester.FAIRMASK, Model.SV_C, Model.NB_C, None, other)
]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre, oth in mls:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth)