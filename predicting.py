#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import hypopt
import preproc
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

titanic = preproc.preproc("train.csv")

#variables = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
variables = ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]
X = titanic[variables].astype(float)

non_numeric_col = ['Sex', 'Title']
le = LabelEncoder()
for feature in non_numeric_col:
    X[feature] = le.fit_transform(X[feature])

y = titanic["Survived"].astype('category')

methods = ['random_forest', 'xgb', 'decision_tree']

models = hypopt.get_models(methods, X, y)

predictions = []
for model in models:
    print 'method:', model['method']
    accuracy = model['model']['accuracy']
    print 'accuracy =', accuracy
    alg = model['model']['fitted_algo']
    prediction = alg.predict_proba(X)[:,1]
    prediction = [round(pred) for pred in prediction]
    predictions.append({'model': model, 'accuracy': accuracy, 'prediction': prediction})

y = np.array(y).astype('float')
for prediction in predictions:
    pred = prediction['prediction']
    accuracy = sum((pred == y).astype('int')) / len(y)
    print 'accuracy on train dataset = {0}%'.format(round(100 * accuracy, 2))

sum_acc = sum([pred['accuracy'] for pred in predictions])
preds = [0] * X.shape[0]
for prediction in predictions:
    pred = prediction['prediction']
    preds = [preds[i] + pred[i] * prediction['accuracy'] / sum_acc for i in range(len(preds))]

preds = np.array([round(pred) for pred in preds]).astype('float')
accuracy = sum((preds == y).astype('int')) / len(y)
print 'mixed accuracy on train dataset = {0}%'.format(round(100 * accuracy, 2))

######
# test data set
titanic = preproc.preproc("test.csv")
X = titanic[variables].astype(float)

non_numeric_col = ['Sex', 'Title']
le = LabelEncoder()
for feature in non_numeric_col:
    X[feature] = le.fit_transform(X[feature])

predictions = []
for model in models:
    print 'method:', model['method']
    accuracy = model['model']['accuracy']
    print 'accuracy =', accuracy
    alg = model['model']['fitted_algo']
    prediction = alg.predict_proba(X)[:,1]
    prediction = [round(pred) for pred in prediction]
    predictions.append({'model': model, 'accuracy': accuracy, 'prediction': prediction})

preds = [0] * X.shape[0]
for prediction in predictions:
    pred = prediction['prediction']
    preds = [preds[i] + pred[i] * prediction['accuracy'] / sum_acc for i in range(len(preds))]

preds = np.array([round(pred) for pred in preds]).astype('int')

submission = pd.DataFrame({
    "PassengerId": titanic["PassengerId"],
    "Survived": preds
})

submission.to_csv("kaggle3.csv", index=False)
