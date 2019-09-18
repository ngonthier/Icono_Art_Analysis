#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:14:19 2019

In this script we are looking at the transfer learning of a VGG with some 
statistics imposed on the features maps of the layers

@author: gonthier
"""


def learn_and_eval(target_dataset,source_dataset,final_clf,features,constrNet):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
    TODO : linear SVM - MLP - perceptron - MLP one per class - MLP with finetuning of the net
    @param : features : which features we will use
    TODO : fc2, fc1, max spatial, max et min spatial
    @param : constrNet the constrained net used
    TODO : VGGAdaIn, VGGAdaIn seulement sur les features qui répondent trop fort, VGGGram
    """
    
    return(0)
