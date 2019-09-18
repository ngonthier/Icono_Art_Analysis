#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:14:19 2019

In this script we are looking at the transfer learning of a VGG with some 
statistics imposed on the features maps of the layers

@author: gonthier
"""

from trouver_classes_parmi_K import TrainClassif
import numpy as np
import os.path
from Study_Var_FeaturesMaps import get_dict_stats

def compute_ref_stats(dico,type_ref='mean'):
    """
    This function compute a reference statistics on the statistics of the whole dataset
    """
    for key in dico.keys():
        continue
    return(0)

def learn_and_eval(target_dataset,source_dataset,final_clf,features,constrNet,kind_method):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
    TODO : linear SVM - MLP - perceptron - MLP one per class - MLP with finetuning of the net
    @param : features : which features we will use
    TODO : fc2, fc1, max spatial, max et min spatial
    @param : constrNet the constrained net used
    TODO : VGGAdaIn, VGGAdaIn seulement sur les features qui répondent trop fort, VGGGram
    @param : kind_method the type of methods we will use : TL or FT
    """
    style_layers = ['block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
           ]
    # Compute statistics on the source_dataset
    if not(source_dataset is None):
        if constrNet=='VGGAdaIn':
            whatToload = 'varmean'
        number_im_considered = np.inf
        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                   whatToload,saveformat='h5')
        
        
        # Compute the reference statistics
        ref_stats = compute_ref_stats(dict_stats)
        
        
    if kind_method=='TL': # Transfert Learning
        # Compute bottleneck features on the target dataset
        return(0)
    
    
    
    
    
    return(0)
