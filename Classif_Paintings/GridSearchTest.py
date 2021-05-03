#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:43:09 2018

@author: gonthier
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np

# Crete dataset
X,y = datasets.make_classification(n_samples=5000, n_features=20, n_classes=2)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

scores = ['precision']
classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
cs = np.logspace(-5, -2, 20)
param_grid = dict(C=cs)

for score in scores:
    grid = GridSearchCV(classifier, refit=True,scoring =
                        make_scorer(average_precision_score,needs_threshold=True),param_grid=param_grid,
                        n_jobs=-1)
    index_posEx = np.where(y_train>= 1)[0]
    index_negEx = np.where(y_train < 1)[0]
    index_posEx_keep = index_posEx[0:3]
    index_keep = np.concatenate((index_negEx,index_posEx_keep))
    
    y_train_reduce = y_train[index_keep]
    X_train_reduce = X_train[index_keep]
    print("Number of positive examples :",np.sum(y_train_reduce))
    print("Number of positive examples :",np.abs(np.sum(1-y_train_reduce)))
    for i in range(500):
        grid.fit(X_train_reduce, y_train_reduce)
