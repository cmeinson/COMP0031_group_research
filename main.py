from src import *

# Just an example for now

tester = Tester("test")

metric_names = ["acc"]
dataset = "Default"
bias_mit = "Default"
ml_method = "SVC"
X, y, preds = tester.run_test(metric_names, dataset, bias_mit, ml_method)
print(X)
print(y, preds)