import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class MyGradientBoostingClassifier:
    
    def __init__(self, n_estimators=100, lr=0.04, max_depth=20,max_features='auto',criterion='mae', min_samples_leaf=1,verbose=False):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.criterion=criterion
        self.verbose = verbose
        
        self.estimator_list = None
        self.is_first = True
        self.F = None
        self.score_list = list()
        
    def fit(self, train_X, train_y,valid_X,valid_y):
        self.estimator_list = list()
        self.F = np.zeros_like(train_y, dtype=float)
        multiple = np.array([5 if i == 1 else 1 for i in train_y])
        
        for i in range(1, self.n_estimators + 1):
            # get negative gradients
            # neg_grads = train_y - self.logit(self.F)
            neg_grads = multiple * (train_y - self.logit(self.F))
            base = DecisionTreeRegressor(max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,max_features=self.max_features)
            base.fit(train_X, neg_grads)
            train_preds = base.predict(train_X)
            self.estimator_list.append(base)
            
            if self.is_first:
                self.F = train_preds
                self.is_first = False
            else:
                self.F += self.lr * train_preds
                
            train_preds = self.logit(self.F) 
#             train_score = r2_score(train_y, train_preds)
            train_roc_auc_score = sklearn.metrics.roc_auc_score(np.array(train_y),train_preds)
            valid_preds = self.predict(valid_X)
#             valid_score = r2_score(valid_y, valid_preds)
            valid_roc_auc_score = sklearn.metrics.roc_auc_score(np.array(valid_y),valid_preds)
            iter_score = dict(iter=i,train_roc_auc_score = train_roc_auc_score,valid_roc_auc_score=valid_roc_auc_score)
            self.score_list.append(iter_score)
            if self.verbose:
                print(iter_score)
                
    def predict(self, X):
        F = np.zeros_like(len(X), dtype=float)
        is_first = True
        for base in self.estimator_list:
            preds = base.predict(X)
            if is_first:
                F = preds
                is_first = False
            else:
                F += self.lr * preds
        return self.logit(F)
    
    @staticmethod
    def logit(F):
        return 1.0 / (1.0 + np.exp(-F))