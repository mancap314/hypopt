#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import normalize, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

X, y, alg = [], [], ''

space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,7)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),
    'random_state': 1,
}

space4gbc = {
    'loss': hp.choice('loss', ['deviance', 'exponential']),
    'learning_rate': hp.uniform('learning_rate', 0.05, 0.3),
    'n_estimators': hp.choice('n_estimators', range(50, 200)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_sample_split': hp.choice('min_samples_split', range(1, 6)),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),
}

space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 100)),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),
}

space4lor = {
    'C': hp.uniform('C', 0, 20),
    'max_iter': hp.choice('max_iter', range(100, 300)),
    'solver': hp.choice('solver', ['newton-cg', 'lbfgs','liblinear', 'sag']),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0,1]),
}

space4lr = {
    'capture_intercept': hp.choice('capture_intercept', [True, False]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),

}

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),
    'random_state': 1,
}

space4svc = {
    'C': hp.uniform('C', 1, 5),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf']),
    'gamma': hp.uniform('gamma', 0.5, 10),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0,1]),
}

space4xgb = {
    'n_estimators' : hp.choice('n_estimators', range(100, 1000)),
    'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'max_depth' : hp.choice('max_depth', range(1, 13)),
    'min_child_weight' : hp.choice('min_child_weight', range(1, 7)),
    'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'objective': 'binary:logistic',
}

choices = {
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
    },
    'logistic_regression': {
        'solver': ['newton-cg', 'lbfgs','liblinear', 'sag'],
        'multi_class': ['ovr', 'multinomial'],
    },
    'linear_regression': {
        'capture_intercept': [True, False]
    },
    'logistic_regression': {
        'solver': ['newton-cg', 'lbfgs','liblinear', 'sag'],
    },
    'random_forest': {
        'criterion': ['gini', 'entropy'],
    },
    'svc': {
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
    }
}

method = {
    'decision_tree': {'space': space4dt, 'method': DecisionTreeClassifier},
    'gbc': {'space': space4gbc, 'method': GradientBoostingClassifier},
    'knn': {'space': space4knn, 'method': KNeighborsClassifier},
    'linear_regression': {'space': space4lr, 'method': LinearRegression},
    'logistic_regression': {'space': space4lor, 'method': LogisticRegression},
    'random_forest': {'space': space4rf, 'method': RandomForestClassifier},
    'svm': {'space': space4svc, 'method': SVC},
    'xgb': {'space': space4xgb, 'method': XGBClassifier}
    }

def ghyperopt(meth, trials, ylim=[0, 1]):
    """
    draws the performance of the trials by parameter and parameter value
    :param meth: method tested
    :param trials: object trial
    :param ylim: optional, accuracy domain to draw
    :return: plot with the performance of the tested models by parameter
    """
    parameters = method[meth]['space'].keys()
    print 'parameters', parameters
    ncols = int(min(len(parameters), 3))
    nrows = int(round(len(parameters)/ncols))
    print 'ncols =', ncols, ', nrows =', nrows
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10*nrows))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        ys = np.array(ys)
        j, k = i // ncols, i % ncols
        axes[j][k].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(i/len(parameters)))
        axes[j][k].set_title(val)
        axes[j][k].set_ylim(ylim)
    plt.show()

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']

    clf = method[alg]['method'](**params)
    kf = KFold(X_.shape[0], n_folds=5, random_state=1)
    return cross_val_score(clf, X_, y, cv = kf).mean()

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

def find_best(vars, target, talgo, max_evals=300, algo=tpe.suggest):
    global X, y, alg
    X, y, alg = vars, target, talgo
    trials = Trials()
    space = method[alg]['space']
    best = fmin(f, space, algo=algo, max_evals=max_evals, trials=trials)
    return {'best': best, 'trials': trials}

def best_accuracy(trials):
    return max([-t['result']['loss'] for t in trials.trials])

def fitted_algo(x, y, params):
    if 'accuracy' in params: #accuracy obtained in training set not relevant here
        del params['accuracy']
    if 'normalize' in params:
        if params['normalize'] == 1:
            x = normalize(x)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            x = scale(x)
        del params['scale']

    alg = params['algo']
    del params['algo']

    malg = method[alg]['method'](**params)
    malg.fit(x, y)
    return malg

def replace_choices(algo, model):
    if algo not in choices.keys():
        return model
    choice_params = choices[algo]
    for cp in choice_params:
        ind = model[cp]
        value = choices[algo][cp][ind]
        model[cp] = value
    return model

def make_params(algo, accuracy, params):
    params = replace_choices(algo, params)
    new_entries = {'algo': algo, 'accuracy': accuracy}
    params.update(new_entries)
    return params

def run_method(X, y, method):
    tstart = time.time()
    sol = find_best(X, y, method)
    tstop = time.time()
    print "time needed for {0}: {1}sec:".format(method, str(tstop-tstart))
    print 'best configuration found for {0}:'.format(method)
    print sol['best']
    print 'accuracy for this configuration: {0}%'.format(round(best_accuracy(sol['trials'])*100, 2))
    params = make_params(method, best_accuracy(sol['trials']), sol['best'])
    print "params before save:"
    print params
    np.save('methods/{0}.npy'.format(method), params)

def load_fitted_method(method, X, y):
    params = np.load('methods/{0}.npy'.format(method)).item()
    print 'params:'
    print params
    accuracy = params['accuracy']
    return {'fitted_algo': fitted_algo(X, y, params), 'accuracy': accuracy}

def get_models(methods, X, y):
    models = []
    for method in methods:
        model = load_fitted_method(method, X, y)
        models.append({'method': method, 'model': model})
    return models

#print replace_choices('dt', {'max_features': 2, 'normalize': 0, 'scale': 1, 'criterion': 1, 'max_depth': 4})
#OK
#print make_params('dt', 0.845987988, {'max_features': 2, 'normalize': 0, 'scale': 1, 'criterion': 1, 'max_depth': 4})
#OK

#définir méthode qui renvoie des prédiction pour le meilleur modèle
