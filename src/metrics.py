import numpy as np
import pandas as pd
from typing import List
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference, average_odds_difference, disparate_impact_ratio, df_bias_amplification

# how do we wanna do metrics?

# we can do just a simple metrcs class with all the mathy functions and then a separate evaluator class?

class Metrics:
    # All available metrics:
    #performance metrics
    ACC = "accuracy"
    PRE = "precision"
    REC = "recall"
    F1 = "f1score"

    #fairness metrics
    AOD = "[AOD] Average Odds Difference"
    EOD = "[EOD] Equal Opportunity Difference"
    SPD = "[SPD] Statistical Parity Difference"
    DI = "[DI] Disparate Impact "
    SF = "[SF] Statistical Parity Subgroup Fairness"
    DF = "[DF] Differential Fairness"

    warnings.simplefilter("ignore")

    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array) -> None:
        # might need more attributes idk
        self._X = X # not even sure if needed
        self._y = y
        self._preds = preds
        self.groups = {}

    def get_subgroup_dependant():
        # metrics that need a list of attributes as input to create subgroups
        return [Metrics.DF]

    def get_attribute_dependant():
        # metrics that need a single attribute as input
        return [Metrics.AOD, Metrics.EOD, Metrics.SPD]

    def get_attribute_independant():
        # metrics independant of attributes
        return [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.F1, Metrics.DI]

    def get(self, metric_name, attr: str or List[str] = None):
        if metric_name == self.ACC:
            return self.accuracy()
        elif metric_name == self.PRE:
            return self.precision()
        elif metric_name == self.REC:
            return self.recall()
        elif metric_name == self.F1:
            return self.f1score()
        elif metric_name == self.AOD:
            return self.aod(attr)
        elif metric_name == self.EOD:
            return self.eod(attr)
        elif metric_name == self.SPD:
            return self.spd(attr)
        elif metric_name == self.DI:
            return self.di()
         #elif metric_name == self.SF:
            #return self.sf(attr)
        elif metric_name == self.DF:
            return self.df(attr)
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)
    
    def accuracy(self) -> float:
        return accuracy_score(self._y, self._preds)
        #return np.mean(self._y == self._preds)

    # etc other metrics

    def precision(self) -> float:
        return precision_score(self._y, self._preds)

    def recall(self) -> float:
        return recall_score(self._y, self._preds)
    
    def f1score(self) -> float:
        return f1_score(self._y, self._preds)

    def aod(self, attr) -> float:
        return np.mean(average_odds_difference(pd.Series(self._y), self._preds))

    def eod(self, attr) -> float:
        return np.mean(equal_opportunity_difference(pd.Series(self._y), self._preds))

    def spd(self, attr) -> float:
        return np.mean(statistical_parity_difference(pd.Series(self._y)))

    def di(self) -> float:
        return np.mean(disparate_impact_ratio(pd.Series(self._y)))
    
    def df(self, attr) -> float:
        return np.mean(disparate_impact_ratio(pd.Series(self._y), self._preds))
    
    #def sf(self, attr) -> float:
    




    
    """""
    #df taken from https://arxiv.org/pdf/1807.08362.pdf
    
    def df(self, attr) -> float:
        noOfClasses = 2
        concParam = 1.0
        dirichletAlpha = concParam/noOfClasses
        intersectGroups = np.unique(attr,axis=0)
        countsClassOne = np.zeros((len(intersectGroups)))
        countsTotal = np.zeros((len(intersectGroups)))
        for i in range(len(self._preds)):
            index=np.where((intersectGroups==attr[0]).all(axis=0))[0][0]
            countsTotal[index] += 1
            if self._preds[i] == 1:
                countsClassOne[index] += 1
        probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concParam)
        epsilonSmoothed = self.differentialFairnessBinaryOutcome(probabilitiesForDFSmoothed)
        return epsilonSmoothed
    
    def differentialFairnessBinaryOutcome(self, probabilitiesOfPositive):
        epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
        for i in  range(len(probabilitiesOfPositive)):
            epsilon = 0.0
            for j in range(len(probabilitiesOfPositive)):
                if i == j:
                    continue
                else:
                    epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])-np.log(probabilitiesOfPositive[j])))
                    epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))-np.log((1-probabilitiesOfPositive[j]))))
            epsilonPerGroup[i] = epsilon
        epsilon = max(epsilonPerGroup)
        return epsilon
    """