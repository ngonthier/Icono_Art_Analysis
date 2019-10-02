#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:10:11 2019

@author: gonthier
"""


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score
import numpy as np


def PredictOnTestSet(X_test,dicoclf,clf='LinearSVC'):
    dico_pred = {}
    for c in  dicoclf.keys():
        clf =  dicoclf[c]
        y_predict_confidence_score = clf.decision_function(X_test)
        y_predict_test = clf.predict(X_test)
        dico_pred[c] = [y_predict_confidence_score,y_predict_test]
    return(dico_pred)
    
def TrainClassifierOnAllClass(X,y,clf='LinearSVC',gridSearch=True):
    number_samples,num_classes = y.shape
    
    dico_clf = {}
    
    for c in range(num_classes):
        y_c = y[:,c]
        classifier_c = TrainLinearSVC(X,y_c,clf=clf,gridSearch=gridSearch)
        dico_clf[c] = classifier_c
        
    return(dico_clf)

def TrainLinearSVC(X,y,clf='LinearSVC',class_weight=None,gridSearch=True,n_jobs=-1,
                 C_finalSVM=1,cskind=None):
    """
    @param clf : LinearSVC, defaultSGD or SGDsquared_hinge  
    Trained on one class uniquely
    """
    if cskind =='' or cskind is None:
        # default case
        cs = np.logspace(-5, -2, 20)
        cs = np.hstack((cs,[0.01,0.2,1.,2.,10.,100.]))
    elif cskind=='small':
        cs = np.logspace(-5, 3, 9)
    param_grid = dict(C=cs)
    # TODO  class_weight='balanced' TODO add this parameter ! 
    if gridSearch:
        if clf == 'LinearSVC':
            clf = LinearSVC(penalty='l2',class_weight=class_weight, 
                            loss='squared_hinge',max_iter=1000,dual=True)
            param_grid = dict(C=cs)
        elif clf == 'defaultSGD':
            clf = SGDClassifier(max_iter=1000, tol=0.0001)
            param_grid = dict(alpha=cs)
        elif clf == 'SGDsquared_hinge':
            clf = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
            param_grid = dict(alpha=cs)
    
        classifier = GridSearchCV(clf, refit=True,
                                  scoring =make_scorer(average_precision_score,
                                                       needs_threshold=True),
                                  param_grid=param_grid,n_jobs=n_jobs)
    else:
        # ,class_weight='balanced'
        if clf == 'LinearSVC':
            classifier = LinearSVC(penalty='l2',class_weight=class_weight,
                                   loss='squared_hinge',max_iter=1000,dual=True,C=C_finalSVM)
        elif clf == 'defaultSGD':
            classifier = SGDClassifier(max_iter=1000)
        elif clf == 'SGDsquared_hinge':
            classifier = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
    
    classifier.fit(X,y)
    
    return(classifier)