import os
from src import *

# Just an example for now

results_filename = "test"

metric_names = [Metrics.ACC]

# All the following constants are just strings so caould also easily be read from a file or whatever. just for convenience:
dataset = Tester.DUMMY_D
bias_mit = Tester.BASE_ML
ml_method = BaseModel.SV

tester = Tester(os.path.join("results",results_filename))
X, y, preds = tester.run_test(metric_names, dataset, bias_mit, ml_method)
print(X)
print(y, preds)