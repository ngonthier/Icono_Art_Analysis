#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:19:24 2018

@author: gonthier
"""

from sklearn.metrics import precision_score,recall_score,f1_score
y_true = [-1,1,-1,1,1,1,-1]
y_pred =  [1,1,1,-1,1,1,-1]
precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
F1_computed_with_recall_precision = 2*(precision*recall)/(precision+recall)
print(precision,recall,f1,F1_computed_with_recall_precision)