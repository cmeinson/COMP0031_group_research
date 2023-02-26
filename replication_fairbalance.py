import os
from src import *
import warnings
warnings.filterwarnings("ignore")

results_filename = "FairBalanceAdult"

metric_names = [Metrics.ACC]

dataset = Tester.ADULT_D
bias_mit = Tester.FAIRBALANCE
ml_method = BaseModel.LR

tester = Tester(os.path.join("results",results_filename))
X, y, preds = tester.run_test(metric_names, dataset, bias_mit, ml_method, 
                              data_preprocessing="FairBalancePreprocessing")


