import os
from src import *

# Just an example for now

results_filename = "alltest"

metric_names = Metrics.get_attribute_dependant() + Metrics.get_attribute_independant() + Metrics.get_subgroup_dependant()

# All the following constants are just strings so caould also easily be read from a file or whatever. just for convenience:
tester = Tester(os.path.join("results",results_filename))

for dataset in [Tester.ADULT_D, Tester.COMPAS_D]:
    for bias_mit, method, method2 in [(Tester.BASE_ML, BaseModel.LR, None), 
                                      (Tester.FAIRBALANCE, FairBalanceModel.LOGR, None), 
                                      (Tester.BASE_ML, BaseModel.RF, None), 
                                      (Tester.FAIRMASK, FairMaskModel.RF, FairMaskModel.DT)]:
        X, y, preds = tester.run_test(metric_names, dataset, bias_mit, method, method2)
        print(y, preds)