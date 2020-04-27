# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:04 2020

The goal of this script is to look at the gram matrices of a fine-tuned network
To see the eigen values of its to see if we can have a better features visualisation

@author: gonthier
"""

import numpy as np
import os
import pathlib

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
    # Dans le cas de RASTA ce fichier fait plus de 60Go
    
    stats_layer = dict_stats[style_layers[0]]
    del dict_stats
    number_img = len(stats_layer)
    [cov,mean] = stats_layer[0]
    features_size,_ = cov.shape
    
    mean_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
    mean_squared_value_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
    for i in range(number_img):
        [cov,mean] = stats_layer[i]
        mean_cov_matrix += cov/number_img
        mean_squared_value_cov_matrix += (cov**2)/number_img
    squared_mean_cov_matrix = mean_cov_matrix**2
    var_cov_matrix = mean_squared_value_cov_matrix - squared_mean_cov_matrix
    std_cov_matrix = np.sqrt(var_cov_matrix)
    
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of Mean of the cov matrices'.format(np.mean(mean_cov_matrix),np.median(mean_cov_matrix),np.min(mean_cov_matrix),np.max(mean_cov_matrix)))
    #Mean 9.48e+01, median 8.80e+01, min -1.40e+03 and max 4.97e+03 of Mean of the cov matrices
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of abs Mean of the cov matrices'.format(np.mean(np.abs(mean_cov_matrix)),np.median(np.abs(mean_cov_matrix)),np.min(np.abs(mean_cov_matrix)),np.max(np.abs(mean_cov_matrix))))
    #Mean 1.47e+02, median 1.16e+02, min 6.99e-04 and max 4.97e+03 of abs Mean of the cov matrices
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of std of the cov matrices'.format(np.mean(std_cov_matrix),np.median(std_cov_matrix),np.min(std_cov_matrix),np.max(std_cov_matrix)))
    #Mean 3.62e+02, median 3.35e+02, min 8.02e+01 and max 3.00e+03 of std of the cov matrices
    
    where_std_sup_mean = np.where(std_cov_matrix > mean_cov_matrix)
    print('Number of time std > mean in the 528*528 = 278784 matrice :',len(where_std_sup_mean[0]))
    # 271913
    where_std_sup_abs_mean = np.where(std_cov_matrix > np.abs(mean_cov_matrix))
    print('Number of time std > mean in the 528*528 = 278784 matrice :',len(where_std_sup_abs_mean[0]))
    # 271679
    
    output_path = os.path.join('CompModifModel','InceptionV1','model_name')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    path_data = os.path.join(output_path,style_layers[0]+'_mean_cov_matrix.npy')
    np.save(path_data, mean_cov_matrix)
    path_data = os.path.join(output_path,style_layers[0]+'_std_cov_matrix.npy')
    np.save(path_data, std_cov_matrix)
    
    
    
    
#    matrix_of_cov_matrix = np.empty((number_img,features_size,features_size),dtype=np.float32)
#    matrix_of_mean_matrix = np.empty((number_img,features_size),dtype=np.float32)
#    print('matrix_of_cov_matrix',matrix_of_cov_matrix.shape)
#    list_cov = []
#    list_mean = []
    
#    matrix_of_cov_matrix = np.memmap('cov_matrix.dat', dtype=np.float32,
#              mode='w+', shape=(number_img,features_size,features_size))
#    matrix_of_mean_matrix = np.memmap('mean_matrix.dat', dtype=np.float32,
#              mode='w+', shape=(number_img,features_size))
#    for i in range(number_img):
#        [cov,mean] = stats_layer[i]
#        matrix_of_cov_matrix[i,:,:] = cov
#        matrix_of_mean_matrix[i,:] = mean
    
#    del stats_layer
    
    
    
    
    

