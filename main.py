import os
from src import *

# Just an example for now

results_filename = "test"

metric_names = Metrics.get_attribute_dependant() + Metrics.get_attribute_independant() + Metrics.get_subgroup_dependant()

# All the following constants are just strings so caould also easily be read from a file or whatever. just for convenience:
dataset = Tester.COMPAS_D
bias_mit = Tester.FAIRBALANCE
ml_method = FairBalanceModel.LOGR

tester = Tester(os.path.join("results",results_filename))
X, y, preds = tester.run_test(metric_names, dataset, bias_mit, ml_method, sensitive_attr=["sex","race"])
print(y, preds)