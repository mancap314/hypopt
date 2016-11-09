#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import normalize, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score

data = {}

space4dt = {
    'max_depth': [1,20],
    'max_features': [1,7],
    'criterion': ('gini', 'entropy'),
}

space4gbc = {
    'loss': ('deviance', 'exponential'),
    'learning_rate': [0.05, 0.3],
    'n_estimators': [50, 200],
    'max_depth': [1, 20],
    'min_samples_split': [1, 6],
}

space4knn = {
    'n_neighbors': [1, 100],
}

space4lor = {
    'C': [1, 20],
    'max_iter': [100, 300],
    'solver': ('newton-cg', 'lbfgs','liblinear', 'sag'),
}

space4lr = {
    'fit_intercept': (True, False),
    'normalize': (True, False),
}

space4rf = {
    'max_depth': [1,20],
    'max_features': [1,5],
    'n_estimators': [1,20],
    'criterion': ('gini', 'entropy'),
}

space4svc = {
    'C': [1, 5],
    'kernel': ('linear', 'sigmoid', 'rbf'),
    'gamma':  [0.5, 10],
}

space4xgb = {
    'n_estimators' : [100, 1000],
    'learning_rate' : [0.025, 0.5],
    'max_depth' : [1, 13],
    'min_child_weight' : [1, 7],
    'subsample' : [0.5, 1],
    'gamma' : [0.5, 1],
    'colsample_bytree' : [0.5, 1],
}

method = {
    'decision_tree': {'space': space4dt, 'method': DecisionTreeClassifier()},
    'gbc': {'space': space4gbc, 'method': GradientBoostingClassifier()},
    'knn': {'space': space4knn, 'method': KNeighborsClassifier()},
    'linear_regression': {'space': space4lr, 'method': LinearRegression()},
    'logistic_regression': {'space': space4lor, 'method': LogisticRegression()},
    'random_forest': {'space': space4rf, 'method': RandomForestClassifier()},
    'svm': {'space': space4svc, 'method': SVC()},
    'xgb': {'space': space4xgb, 'method': XGBClassifier()}
    }

cmethod = {
    'decision_tree': {'space': space4dt, 'method': DecisionTreeClassifier},
    'gbc': {'space': space4gbc, 'method': GradientBoostingClassifier},
    'knn': {'space': space4knn, 'method': KNeighborsClassifier},
    'linear_regression': {'space': space4lr, 'method': LinearRegression},
    'logistic_regression': {'space': space4lor, 'method': LogisticRegression},
    'random_forest': {'space': space4rf, 'method': RandomForestClassifier},
    'svm': {'space': space4svc, 'method': SVC},
    'xgb': {'space': space4xgb, 'method': XGBClassifier}
    }

def find_best(dataset, alg):
    global data
    data = dataset
    meth = method[alg]['method']
    space = method[alg]['space']
    clf = GridSearchCV(meth, space, cv=5, verbose=3, scoring='precision') # scoring changed
    clf.fit(dataset['train']['X'], dataset['train']['y'])
    params = clf.best_params_
    print 'params:', params
    y_true, y_pred = dataset['test']['y'], clf.predict(dataset['test']['X']).astype(int)
    print 'classification report for {0}:'.format(alg)
    ps = precision_score(y_true, y_pred)
    print 'precision_score:', ps
    cr = classification_report(y_true, y_pred)
    print cr
    print 'best_score:', clf.best_score_ #ok pour ponderation
    params.update({'score': ps})
    return params

def make_prediction(methods, dataset, datatest):
    t_score, space = 0, {}
    t_prediction_control = [0]*dataset['test']['X'].shape[0]
    t_prediction_submit = [0]*datatest['X'].shape[0]
    for mtd in methods:
        params = find_best(dataset, mtd)
        score = params['score']
        del params['score']
        metho = cmethod[mtd]['method']
        meth = metho(**params)
        meth.fit(dataset['train']['X'], dataset['train']['y'])
        prediction_control = meth.predict(dataset['test']['X'])
        space.update({mtd: (1, 10)})
    t_prediction_control = np.array([round(tpc/t_score) for tpc in t_prediction_control]).astype(int)
    print 'score (control):', precision_score(dataset['test']['y'], t_prediction_control)
    print classification_report(dataset['test']['y'], t_prediction_control)
    t_prediction_submit = np.array([round(tps/t_score) for tps in t_prediction_submit]).astype(int)
    return t_prediction_submit

#find optimal weights
#build dictionary of search_space step by step
#optimize through missclassification in test dataset...

#fit models on all trainin sets before submitting?

