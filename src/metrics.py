import numpy as np
import pandas as pd
from typing import List, Callable
import warnings
from sklearn import metrics
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
#from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference, average_odds_difference, disparate_impact_ratio, df_bias_amplification

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
    DI = "[DI] Disparate Impact"
    FR = "[FR] Flip Rate"

    SF = "[SF] Statistical Parity Subgroup Fairness"
    ONE_SF = "[SF] Statistical Parity Subgroup Fairness for One Attribute"

    DF = "[DF] Differential Fairness"
    ONE_DF = "[DF] Differential Fairness for One Attribute"

    M_EOD = "[MEOD] M Equal Opportunity Difference"
    M_AOD = "[MEOD] M Average Odds Difference"

    warnings.simplefilter("ignore")

    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array, predict: Callable[[pd.DataFrame], np.array]) -> None:
        # might need more attributes idk
        self._X = X # not even sure if needed
        self._X.reset_index(drop=True, inplace=True) # TODO: would it be better for everyone if this was done in the data class?
        self._y = y
        self._preds = preds
        self._predict = predict
        self.groups = defaultdict(list)
        self._round = lambda x: round(x,5)

    def get_all_names():
        return Metrics.get_attribute_dependant() + Metrics.get_attribute_independant() + Metrics.get_subgroup_dependant()

    def get_subgroup_dependant():
        # metrics that need a list of attributes as input to create subgroups
        return [Metrics.SF, Metrics.DF, Metrics.M_EOD, Metrics.M_AOD]

    def get_attribute_dependant():
        # metrics that need a single attribute as input
        return [Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.FR, Metrics.ONE_SF, Metrics.ONE_DF]

    def get_attribute_independant():
        # metrics independant of attributes
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
            return self.aod(attr)
        elif metric_name == self.EOD:
            return self.eod(attr)
        elif metric_name == self.SPD:
            return self.spd(attr)
        elif metric_name == self.DI:
            return self.di(attr)
        elif metric_name == self.SF:
            return self.sf(attr)
        elif metric_name == self.ONE_SF:
            return self.oneSF(attr)
        elif metric_name == self.DF:
            return self.df(attr)
        elif metric_name == self.ONE_DF:
            return self.oneDF(attr)
        elif metric_name == self.M_EOD:
            return self.meod(attr)
        elif metric_name == self.M_AOD:
            return self.maod(attr)
        elif metric_name == self.FR:
            return self.fr(attr)
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)

    def accuracy(self) -> float:
        return accuracy_score(self._y, self._preds)
        #return np.mean(self._y == self._preds)

    # etc other metrics

    def precision(self) -> float:
        return self._round(precision_score(self._y, self._preds))

    def recall(self) -> float:
        return self._round(recall_score(self._y, self._preds))

    def f1score(self) -> float:
        return self._round(f1_score(self._y, self._preds))

    def aod(self, attribute):
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
        tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
        fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])
        fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
        return abs(self._round(0.5 * (tpr1 + fpr1 - tpr0 - fpr0)))

    def eod(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
        tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
        return abs(self._round(tpr1 - tpr0))

    def spd(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
        pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
        return abs(self._round(pr1 - pr0))

    def di(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
        pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
        di = pr1/pr0
        return self._round(abs(1-di))

    def fr(self, attribute):
        X_flip = self.flip_X(attribute)
        preds_flip = self._predict(X_flip)
        total = self._X.shape[0]
        same = np.count_nonzero(self._preds==preds_flip)
        return self._round((total-same)/total)
    
    # df taken from https://github.com/rashid-islam/Differential_Fairness/blob/1e90232fdb688a538bb534319ff3c9a7e0dee576/differential_fairness.py
    
    def df(self, attributes) -> float:
        intersectGroups = np.unique(attributes,axis=0)
        countsClassOne = np.zeros((len(intersectGroups)))
        countsTotal = np.zeros((len(intersectGroups)))
        for i in range(len(self._preds)):
            index=np.where((sorted(intersectGroups)==sorted(attributes)))[0]
            countsTotal[index] += 1
            if self._preds[i] == 1:
                countsClassOne[index] += 1
        probabilitiesForDFSmoothed = (countsClassOne + 0.5) /(countsTotal + 1.0)
        epsilonSmoothed = self.dfBinary(probabilitiesForDFSmoothed)
        return self._round(epsilonSmoothed)
    
    def oneDF(self, attributes) -> float:
        intersectGroups = np.unique(attributes)
        countsClassOne = np.zeros((len(intersectGroups)))
        countsTotal = np.zeros((len(intersectGroups)))
        for i in range(len(self._preds)):
            index=np.where((intersectGroups == [attributes]))[0]
            countsTotal[index] += 1
            if self._preds[i] == 1:
                countsClassOne[index] += 1
        probabilitiesForDFSmoothed = (countsClassOne + 0.5) /(countsTotal + 1.0)
        epsilonSmoothed = self.dfOneBinary(probabilitiesForDFSmoothed)
        return self._round(epsilonSmoothed)
    
    def dfBinary(self, probabilitiesOfPositive):
        epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
        for i in range(len(probabilitiesOfPositive)):
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
    
    def dfOneBinary(self, probabilitiesOfPositive):
        epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
        for i in range(len(probabilitiesOfPositive)):
            epsilon = 0.0
            epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])))
            epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))))
            epsilonPerGroup[i] = epsilon
        epsilon = max(epsilonPerGroup)
        return epsilon

    def sf(self, attributes) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        sgf = self.sgf()
        return self._round(sum(sgf.values())/len(sgf.values()))
    
    def oneSF(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        sgf = self.sgf()
        return self._round(sum(sgf.values())/len(sgf.values()))

    def meod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        tprs = self.tprs()
        return abs(self._round(max(tprs.values())-min(tprs.values())))

    def maod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        aos = self.aos()
        return abs(self._round(max(aos.values()) - min(aos.values())))

    def confusionMatrix(self, sub=None):
        if sub is None:
            sub = range(len(self._y))
        y = self._y[sub]
        y_pred = self._preds[sub]
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

    def tprs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def fprs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            fprs[group] = fpr
        return fprs

    def aos(self):
        tprs = self.tprs()
        fprs = self.fprs()
        aos = {key: (tprs[key]+fprs[key])/2 for key in tprs}
        return aos

    def prs(self):
        prs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            pr = (conf['tp']+conf['fp']) / len(sub)
            prs[group] = pr
        return prs
    
    def sgf(self):
        sgf = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            if (conf['tp']+conf['fp']+conf['tn']+conf['fn']) == 0:
                sg = 0 
            else:
                sg = conf['tp']/(conf['tp']+conf['fp']+conf['tn']+conf['fn'])
            sg = conf['fp'] / (conf['fp'] + conf['tn'])
            sgf[group] = sg
        return sgf

    def flip_X(self,attribute):
        X_flip = self._X.copy()
        X_flip[attribute] = np.where(X_flip[attribute]==1, 0, 1)
        return X_flip
