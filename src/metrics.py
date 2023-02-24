import numpy as np
import pandas as pd
from typing import List

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
    AOD = "aod"
    EOD = "eod"
    SPD = "spd"
    DI = "di"
    FR = "fr"
    SF = "[SF] Statistical Parity Subgroup Fairness"
    DF = "[DF] Differential Fairness"

    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array) -> None:
        # might need more attributes idk
        self._X = X # not even sure if needed
        self._y = y
        self._preds = preds
        self.groups = {}

    def get_subgroup_dependant():
        # metrics that need a list of attributes as input to create subgroups
        return [Metrics.SF, Metrics.DF]

    def get_attribute_dependant():
        # metrics that need a single attribute as input
        return [Metrics.AOD, Metrics.EOD, Metrics.SPD]

    def get_attribute_independant():
        # mterics independant of attributes
        return [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.F1]

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
            return self.aod()
        elif metric_name == self.EOD:
            return self.eod()
        elif metric_name == self.SPD:
            return self.spd()
        #elif metric_name == self.DI:
            #return self.di()
        #elif metric_name == self.FR:
            #return self.fr()
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)
    
    def accuracy(self) -> float:
        return np.mean(self._y == self._preds)

    # etc other metrics

    def precision(self) -> float:
        target1 = np.mean(self._y)[1] <= np.mean(self._y)[0]
        conf = self.conf()
        if target1:
            prec = conf['tp'] / (conf['tp'] + conf['fp'])
        else:
            prec = conf['tn'] / (conf['tn'] + conf['fn'])
        return prec

    def recall(self) -> float:
        target1 = np.mean(self.y)[1] <= np.mean(self.y)[0]
        conf = self.conf()
        if target1:
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
        else:
            tpr = conf['tn'] / (conf['tn'] + conf['fp'])
        return tpr

    def f1score(self) -> float:
        prec = self.precision()
        tpr = self.recall()
        f1 = 2*tpr*prec/(tpr+prec)
        return f1

    def aod(self, a=None) -> float:
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.conf(ind0)
            conf1 = self.conf(ind1)
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])
            fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
            return 0.5 * (tpr1 + fpr1 - tpr0 - fpr0)
        else:
            aos = self.aos()
            return max(aos.values()) - min(aos.values())

    def eod(self, a=None) -> float:
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.conf(ind0)
            conf1 = self.conf(ind1)
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            return tpr1 - tpr0
        else:
            tprs = self.prs("t")
            return max(tprs.values())-min(tprs.values())

    def spd(self, a=None) -> float:
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.conf(ind0)
            conf1 = self.conf(ind1)
            pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
            pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
            return pr1 - pr0
        else:
            prs = self.prs("n")
            return max(prs.values())-min(prs.values())


    #def di(self) -> float:

    #def fr(self) -> float:

    def conf(self, sub=None):
        if sub is None:
            sub = range(len(self._y))
        y = self._y[sub]
        y_pred = self.y_pred[sub]
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(y)):
            if y[i]==0 and y_pred[i]==0:
                conf['tn']+=1
            elif y[i]==1 and y_pred[i]==1:
                conf['tp'] += 1
            elif y[i]==0 and y_pred[i]==1:
                conf['fp'] += 1
            elif y[i]==1 and y_pred[i]==0:
                conf['fn'] += 1
        return conf
    
    def aos(self):
        tprs = self.prs("t")
        fprs = self.prs("f")
        for key in tprs:
            aos = {key: (tprs[key]+fprs[key])/2}
        return aos
    
    def prs(self, type):
        prs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.conf(sub)
            if type == "t":
                pr = conf['tp'] / (conf['tp'] + conf['fn'])
            elif type == "f":
                pr = conf['fp'] / (conf['fp'] + conf['tn'])
            else:
                pr = (conf['tp'] + conf['fp']) / len(sub)
            prs[group] = pr
        return prs