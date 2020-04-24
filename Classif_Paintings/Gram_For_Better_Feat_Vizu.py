# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:04 2020

The goal of this script is to look at the gram matrices of a fine-tuned network
To see the eigen values of its to see if we can have a better features visualisation

@author: gonthier
"""

import numpy as np

from Study_Var_FeaturesMaps import get_dict_stats

from CompNet_FT_lucidIm import get_fine_tuned_model

def compute_mean_var_of_GramMatrix():
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    
    model_name = 'RASTA_small01_modif'
    source_dataset = 'RASTA'
    number_im_considered = None
    style_layers = ['mixed4d_pre_relu']
    whatToload = 'all'
    
    fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)
    
    dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                   whatToload,saveformat='h5',set=set_,getBeforeReLU=False,\
                   Net=constrNet,style_layers_imposed=[],\
                   list_mean_and_std_source=[],list_mean_and_std_target=[],\
                   cropCenter=cropCenter,BV=True,sizeIm=224,model_alreadyLoaded=fine_tuned_model,\
                   name_model=model_name,\
                   randomCropValid=False,classe=None)
    
    stats_layer = dict_stats[style_layers[0]]
    del dict_stats
    number_img = len(stats_layer)
    [cov,mean] = stats_layer[0]
    features_size,_ = cov.shape
#    matrix_of_cov_matrix = np.empty((number_img,features_size,features_size),dtype=np.float32)
#    matrix_of_mean_matrix = np.empty((number_img,features_size),dtype=np.float32)
#    print('matrix_of_cov_matrix',matrix_of_cov_matrix.shape)
#    list_cov = []
#    list_mean = []
    
    matrix_of_cov_matrix = np.memmap('cov_matrix .dat', dtype=np.float32,
              mode='w+', shape=(number_img,features_size,features_size))
    matrix_of_mean_matrix = np.memmap('mean_matrix .dat', dtype=np.float32,
              mode='w+', shape=(number_img,features_size))
    for i in range(number_img):
        [cov,mean] = stats_layer[i]
        matrix_of_cov_matrix[i,:,:] = cov
        matrix_of_mean_matrix[i,:] = mean
    
#    del stats_layer
    
    
    
    
    

