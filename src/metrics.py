import numpy as np
import pandas as pd
from typing import List, Callable
import warnings
from sklearn import metrics
import copy
import math
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# how do we wanna do metrics?

# we can do just a simple metrcs class with all the mathy functions and then a separate evaluator class?
def guard(nr): # to avoid div by 0
    if nr == 0:
        return 1e-10
    return nr


class Metrics:
    # All available metrics:
    #performance metrics
    ACC = "accuracy"
    PRE = "precision"
    REC = "recall"
    F1 = "f1score"

    #fairness metrics
    A_AOD = "[AOD] Abs Average Odds Difference"
    A_EOD = "[EOD] Abs Equal Opportunity Difference"
    A_SPD = "[SPD] Abs Statistical Parity Difference"
    AOD = "[AOD] Average Odds Difference"
    EOD = "[EOD] Equal Opportunity Difference"
    SPD = "[SPD] Statistical Parity Difference"
    DI = "[DI] Disparate Impact"
    FR = "[FR] Flip Rate" #8

    SF = "[SF] Statistical Parity Subgroup Fairness"
    SF_INV = "[SF] Statistical Parity Subgroup Fairness if 0 was the positive label"
    ONE_SF = "[SF] Statistical Parity Subgroup Fairness for One Attribute"

    DF = "[DF] Differential Fairness"
    DF_INV = "[DF] Differential Fairness if 0 was the positive label"
    ONE_DF = "[DF] Differential Fairness for One Attribute" #10

    POS = "[+%] Proportion of Positive labels for the group"
    T_POS = "[+%] Proportion of Positive true labels for the group"

    A_M_EOD = "[MEOD] Abs M Equal Opportunity Difference"
    A_M_AOD = "[MEOD] Abs M Average Odds Difference"

    M_EOD = "[MEOD] M Equal Opportunity Difference"
    M_AOD = "[MEOD] M Average Odds Difference" #12

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
        return [Metrics.SF, Metrics.SF_INV, Metrics.DF, Metrics.DF_INV, Metrics.M_EOD, Metrics.M_AOD]


    def get_attribute_dependant():
        # metrics that need a single attribute as input
        return [Metrics.POS, Metrics.T_POS, Metrics.A_AOD, Metrics.A_EOD, Metrics.A_SPD, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.FR, Metrics.ONE_SF, Metrics.ONE_DF]
    
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
        elif metric_name == self.A_AOD:
            return abs(self.aod(attr))
        elif metric_name == self.A_EOD:
            return abs(self.eod(attr))
        elif metric_name == self.A_SPD:
            return abs(self.spd(attr))
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
        elif metric_name == self.SF_INV:
            return self.sf(attr, outcome='n')
        elif metric_name == self.ONE_SF:
            return self.sf([attr])
        elif metric_name == self.DF:
            return self.df(attr)
        elif metric_name == self.DF_INV:
            return self.df(attr, outcome='n')
        elif metric_name == self.ONE_DF:
            return self.df([attr])
        elif metric_name == self.M_EOD:
            return self.meod(attr)
        elif metric_name == self.M_AOD:
            return self.maod(attr)
        elif metric_name == self.A_M_EOD:
            return abs(self.meod(attr))
        elif metric_name == self.A_M_AOD:
            return abs(self.maod(attr))
        elif metric_name == self.FR:
            return self.fr(attr)
        elif metric_name == self.POS:
            return self.pos(attr)
        elif metric_name == self.T_POS:
            return self.true_pos(attr)
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)

    def accuracy(self) -> float:
        return accuracy_score(self._y, self._preds)
        #return np.mean(self._y == self._preds)

    
    def precision(self) -> float:
        return self._round(precision_score(self._y, self._preds))

    def recall(self) -> float:
        return self._round(recall_score(self._y, self._preds))

    def f1score(self) -> float:
        return self._round(f1_score(self._y, self._preds))

    def true_pos(self, attribute) -> float:
        total, count = 0,0
        for i in range(len(self._y)):
            if (self._X[attribute][i] == 1):
                count += 1
                total += self._y[i]
        return total / guard(count)
    
    def pos(self, attribute) -> float:
        total, count = 0,0
        for i in range(len(self._y)):
            if (self._X[attribute][i] == 1):
                count += 1
                total += self._preds[i]
        return total / guard(count)

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
        tpr0 = conf0['tp'] / guard(conf0['tp'] + conf0['fn'])
        tpr1 = conf1['tp'] / guard(conf1['tp'] + conf1['fn'])
        fpr0 = conf0['fp'] / guard(conf0['fp'] + conf0['tn'])
        fpr1 = conf1['fp'] / guard(conf1['fp'] + conf1['tn'])
        return (self._round(0.5 * (tpr1 + fpr1 - tpr0 - fpr0)))


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
        tpr0 = conf0['tp'] / guard(conf0['tp'] + conf0['fn'])
        tpr1 = conf1['tp'] / guard(conf1['tp'] + conf1['fn'])
        return (self._round(tpr1 - tpr0))

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
        pr0 = (conf0['tp']+conf0['fp']) / guard(len(ind0))
        pr1 = (conf1['tp']+conf1['fp']) / guard(len(ind1))
        return (self._round(pr1 - pr0))

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
        pr0 = (conf0['tp']+conf0['fp']) / guard(len(ind0))
        pr1 = (conf1['tp']+conf1['fp']) / guard(len(ind1))
        di = pr1/guard(pr0)
        return self._round((1-di))

    def fr(self, attribute):
        X_flip = self.flip_X(attribute)
        preds_flip = self._predict(X_flip)
        total = self._X.shape[0]
        same = np.count_nonzero(self._preds==preds_flip)
        return self._round((total-same)/guard(total))

    def get_subgroup_attr_vals(self, attrs_unique_vals):
        subgroups = [[]]
        for attr_vals in attrs_unique_vals:
            new_subgroups = []
            for subg in subgroups:
                for attr_val in attr_vals:
                    new_subgroups.append(subg+ [attr_val])
            subgroups = new_subgroups
        return subgroups
    
    def get_subgroup_conf_and_size(self, attributes, attribute_vals):
        ind = set(range(len(self._y)))
        for i in range(len(attributes)):
            ind = ind & set(np.where(self._X[attributes[i]] == attribute_vals[i])[0])
        return self.confusionMatrix(sorted(ind)), len(ind)  # TODO: do indices have to be sorted?            

    def sf(self, attributes: List[str], outcome = 'p') -> float:
        attr_values = [np.unique(self._X[a]) for a in attributes]
        subgroups = self.get_subgroup_attr_vals(attr_values)

        sample_size = len(self._y)
        conf = self.confusionMatrix()
        prob_pos = conf[outcome] / sample_size
        ans = 0
        for subgroup in subgroups:
            group_conf, group_size = self.get_subgroup_conf_and_size(attributes, subgroup)
            if group_size!= 0:
                group_prob_pos = group_conf[outcome] / group_size
                group_prob = group_size / sample_size

                group_ans = abs(prob_pos - group_prob_pos)*group_prob
                ans = max(group_ans, ans)
        return ans
    
    def df(self, attributes: List[str], outcome = 'p') -> float:
        attr_values = [np.unique(self._X[a]) for a in attributes]
        subgroups = self.get_subgroup_attr_vals(attr_values)

        ans_min, ans_max = 1, 1
        for subgroup1 in subgroups:
            conf1, size1 = self.get_subgroup_conf_and_size(attributes, subgroup1)
            for subgroup2 in subgroups:
                conf2, size2 = self.get_subgroup_conf_and_size(attributes, subgroup2)
                if size1!=0 and size2!=0:
                    prob_pos1 = conf1[outcome] / size1
                    prob_pos2 = conf2[outcome] / size2
                    if (prob_pos1 == prob_pos2):
                        ans = 1
                    else:
                        if prob_pos1==0 or prob_pos2==0:
                            return 3 # TODO: idk what to do in this case
                        ans = prob_pos1 / prob_pos2
                    ans_max = max(ans, ans_max)
                    ans_min = min(ans, ans_min)            
        ans = max(math.log(ans_max), -math.log(ans_min))
        return ans


    def meod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        tprs = self.tprs()
        return (self._round(max(tprs.values())-min(tprs.values())))

    def maod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        aos = self.aos()
        return (self._round(max(aos.values()) - min(aos.values())))

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
        conf["p"] = conf['tp'] + conf['fp']
        conf["n"] = conf['tn'] + conf['fn']
        return conf

    def tprs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            tpr = conf['tp'] / guard(conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def fprs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            fpr = conf['fp'] / guard(conf['fp'] + conf['tn'])
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
            pr = (conf['tp']+conf['fp']) / guard(len(sub))
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
                sg = conf['tp']/ guard(conf['tp']+conf['fp']+conf['tn']+conf['fn'])
            sg = conf['fp'] / guard(conf['fp'] + conf['tn'])
            sgf[group] = sg
        return sgf

    def flip_X(self,attribute):
        X_flip = self._X.copy()
        X_flip[attribute] = np.where(X_flip[attribute]==1, 0, 1)
        return X_flip
    

