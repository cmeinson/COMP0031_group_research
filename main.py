import os
from src import *

# Just an example for now


n_repetitions = 50
same_data_split = True
results_filename = "heheh"
other = {Tester.OPT_SAVE_INTERMID: False}

other_fb = other.copy()
other_fb[BaseModel.OPT_FBALANCE] = True


datasets =  [Tester.COMPAS_D]
mls = [(Tester.BASE_ML, Model.LG_R, None, "FairBalance", other_fb), 
       (Tester.FAIRBALANCE, Model.LG_R, None, "FairBalance", other_fb), 

       (Tester.BASE_ML, Model.RF_C, None, None, other), 
       (Tester.FAIRMASK, Model.RF_C, Model.DT_R, None, other)
]
metric_names = Metrics.get_all_names()
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre, oth in mls:
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth)