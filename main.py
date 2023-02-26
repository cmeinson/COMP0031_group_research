import os
from src import *

# Just an example for now

results_filename = "test"

metric_names = [Metrics.ACC]

# All the following constants are just strings so caould also easily be read from a file or whatever. just for convenience:
dataset = Tester.COMPAS_D
bias_mit = Tester.FAIRMASK
ml_method = FairMaskModel.RF
ml_method_bias = FairMaskModel.DT

tester = Tester(os.path.join("results",results_filename))
X, y, preds = tester.run_test(metric_names, dataset, bias_mit, ml_method, ml_method_bias)
print(X)
print(y, preds)