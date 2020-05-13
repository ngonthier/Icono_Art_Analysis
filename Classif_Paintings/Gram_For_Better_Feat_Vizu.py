# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:04 2020

The goal of this script is to look at the gram matrices of a fine-tuned network
To see the eigen values of its to see if we can have a better features visualisation

Pour le moment, on calcule les covariances après sous-traction de la moyenne spatiale
d'une image donnée et non pas après soustraction de la moyenne globale sur tout le 
dataset

@author: gonthier
"""

import numpy as np
from numpy import linalg as LA
import os
import platform
import pathlib
import time
import glob
import matplotlib.pyplot as plt

import tempfile
import json
import h5py

from tensorflow.python.keras.models import load_model

from sklearn.decomposition import PCA
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate,Activation,Dense,Flatten,Input,Dropout,InputLayer

from Study_Var_FeaturesMaps import get_dict_stats

from CompNet_FT_lucidIm import get_fine_tuned_model,convert_finetuned_modelToFrozenGraph

import lucid_utils

from IMDB import get_database

from infere_layers_info import get_dico_layers_type_all_layers_fromNet

from googlenet import LRN,PoolHelper

from utils_keras import fix_layer0

import Activation_for_model
from StatsConstr_ClassifwithTL import predictionFT_net
from CompNet_FT_lucidIm import get_fine_tuned_model

def plot_PCAlike_featureVizu_basedOnGramMatrix(model_name = 'RASTA_small01_modif',classe = None,\
                                   layer='mixed4d_pre_relu'):
    """
    In this function we will first compute the spatial mean per feature on layer
    Then compute the Gram matrix after the substraction of this mean on the whole 
    dataset
    """
    # TODO
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    
    if 'RASTA' in model_name:
        source_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        source_dataset = 'IconArt_v1'
    number_im_considered = None
    style_layers = [layer]
    whatToload = 'all'
    if not(classe is None):
        classe_str = classe
    else:
        classe_str = ''
    
    #classe = 'Color_Field_Painting'
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    path_output_lucid_im = os.path.join(output_path,'PCAlucid')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if classe is None:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_std_cov_matrix.npy')
    else:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_std_cov_matrix.npy')

    K.set_learning_phase(0)
    suffix = ''
    suffix_str = suffix
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    if not (os.path.isfile(path_data_cov_matrix) and os.path.isfile(path_data_std_cov_matrix)):
    
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',set=set_,getBeforeReLU=False,\
                       Net=constrNet,style_layers_imposed=[],\
                       list_mean_and_std_source=[],list_mean_and_std_target=[],\
                       cropCenter=cropCenter,BV=True,sizeIm=224,model_alreadyLoaded=fine_tuned_model,\
                       name_model=model_name,\
                       randomCropValid=False,classe=classe)
        # Dans le cas de RASTA ce fichier fait plus de 60Go
        
        
        stats_layer = dict_stats[layer]
        print('len(stats_layer)',len(stats_layer))
        del dict_stats
        number_img = len(stats_layer)
        [cov,mean] = stats_layer[0]
        features_size,_ = cov.shape
        
        mean_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        mean_squared_value_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        for i in range(number_img):
            [cov,mean] = stats_layer[i]
            mean_cov_matrix += cov/number_img
            cov_squared = cov**2
            mean_squared_value_cov_matrix += (cov_squared)/number_img
            
            isnan_cov_squared = np.isnan(cov_squared)
            where_is_nan = np.where(isnan_cov_squared)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,cov[ci,cj],cov_squared[ci,cj])
            isnan_mean_cov = np.isnan(mean_cov_matrix)
            where_is_nan = np.where(isnan_mean_cov)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,isnan_mean_cov[ci,cj])
            
        squared_mean_cov_matrix = mean_cov_matrix**2
        var_cov_matrix = mean_squared_value_cov_matrix - squared_mean_cov_matrix
        var_cov_matrix = np.clip(var_cov_matrix,0.0,np.inf)
        std_cov_matrix = np.sqrt(var_cov_matrix)
        
        if classe is None:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
        else:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
    else:
        mean_cov_matrix = np.load(path_data_cov_matrix)
        std_cov_matrix = np.load(path_data_std_cov_matrix)
        
    features_size,_ = mean_cov_matrix.shape
    print(style_layers[0],str(classe))
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

#    mixed4d_pre_relu None
#    Mean 9.48e+01, median 8.80e+01, min -1.40e+03 and max 4.97e+03 of Mean of the cov matrices
#    Mean 1.47e+02, median 1.16e+02, min 6.99e-04 and max 4.97e+03 of abs Mean of the cov matrices
#    Mean 3.62e+02, median 3.35e+02, min 8.02e+01 and max 3.00e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 271913
#    Number of time std > mean in the 528*528 = 278784 matrice : 271679
#    
#    mixed4d_pre_relu Abstract_Art
#    Mean 1.23e+02, median 1.11e+02, min -1.29e+03 and max 3.57e+03 of Mean of the cov matrices
#    Mean 1.74e+02, median 1.38e+02, min 3.37e-03 and max 3.57e+03 of abs Mean of the cov matrices
#    Mean 3.47e+02, median 3.21e+02, min 6.84e+01 and max 2.71e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 266607
#    Number of time std > mean in the 528*528 = 278784 matrice : 266325
#    
#    mixed4d_pre_relu Color_Field_Painting
#    Mean 9.37e+01, median 8.63e+01, min -9.74e+02 and max 2.47e+03 of Mean of the cov matrices
#    Mean 1.21e+02, median 9.77e+01, min 2.04e-05 and max 2.47e+03 of abs Mean of the cov matrices
#    Mean 2.44e+02, median 2.27e+02, min 4.11e+01 and max 2.10e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 263827
#    Number of time std > mean in the 528*528 = 278784 matrice : 263111
    
    # Compute the eigen values 
    #del std_cov_matrix,where_std_sup_abs_mean,where_std_sup_mean
    eigen_values, eigen_vectors = LA.eig(mean_cov_matrix)
    
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
    plt.figure()
    plt.scatter(np.arange(0,len(eigen_values)),eigen_values,s=4)
    plt.ylabel('Eigen Value')
    plt.savefig(os.path.join(path_output_lucid_im,'EigenValues_'+layer+'_'+classe_str+'.png'),\
                dpi=300)
    plt.close()
    
    print('Eigen values 10 first value',eigen_values[0:10])
    #print('First eigen vector :',eigen_vectors[:,0])
    print('Max imag part first vector :',np.max(np.imag(eigen_vectors[:,0])))
    eigen_vectors = np.real(eigen_vectors)

    num_components_draw = 10
    
    name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
    if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
        name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,path=path_lucid_model,suffix=suffix)
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
    for comp_number in range(num_components_draw):
        weights = eigen_vectors[:,comp_number]
        #weights = weights[0:1]
        #print('weights',weights)
        #time.sleep(.300)
        prexif_name = '_PCA'+str(comp_number)
        if not(classe is None):
           prexif_name += '_'+classe 
        index_features_withinLayer_all = np.arange(0,features_size)
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights,\
                         index_features_withinLayer=index_features_withinLayer_all,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_pos = prexif_name + '_PosContrib'
        where_pos = np.where(weights>0.)[0]
        weights_pos = list(weights[where_pos])
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights_pos,\
                         index_features_withinLayer=where_pos,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_pos,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_neg = prexif_name + '_NegContrib'
        where_neg = np.where(weights>0.)[0]
        weights_neg = list(-weights[where_neg])
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights_neg,\
                         index_features_withinLayer=where_neg,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_neg,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        where_max = np.argmax(weights)
        prexif_name_max = prexif_name+  '_Max'+str(where_max)
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=[1.],\
                         index_features_withinLayer=[where_max],\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        where_min = np.argmin(weights)
        prexif_name_max = prexif_name+  '_Min'+str(where_min)
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=[1.],\
                         index_features_withinLayer=[where_min],\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
def PCAbased_FeaVizu_deepmodel_meanGlobal(model_name = 'RASTA_small01_modif',
                                          classe = None,\
                                          layer='mixed4d_pre_relu'):
    """
    Fct en cours, non fini
    """
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    
    if 'RASTA' in model_name:
        source_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        source_dataset = 'IconArt_v1'
    number_im_considered = None
    style_layers = [layer]
    whatToload = 'all'
    if not(classe is None):
        classe_str = classe
    else:
        classe_str = ''
    
    #classe = 'Color_Field_Painting'
    
    KindOfMeanReduciton = ''
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    
    path_output_lucid_im = os.path.join(output_path,'PCAGramGlobalMeanlucid')
    str_stats = 'cov_global_mean'

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if classe is None:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_mean_'+str_stats+'_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_std_'+str_stats+'_matrix.npy')
    else:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_std_cov_matrix.npy')

    K.set_learning_phase(0)
    suffix = ''
    suffix_str = suffix
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    if not (os.path.isfile(path_data_cov_matrix) and os.path.isfile(path_data_std_cov_matrix)):
    
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',set=set_,getBeforeReLU=False,\
                       Net=constrNet,style_layers_imposed=[],\
                       list_mean_and_std_source=[],list_mean_and_std_target=[],\
                       cropCenter=cropCenter,BV=True,sizeIm=224,model_alreadyLoaded=fine_tuned_model,\
                       name_model=model_name,\
                       randomCropValid=False,classe=classe,\
                       KindOfMeanReduciton=KindOfMeanReduciton)
        # Dans le cas de RASTA ce fichier fait plus de 60Go
        
        
        stats_layer = dict_stats[layer]
        print('len(stats_layer)',len(stats_layer))
        del dict_stats
        number_img = len(stats_layer)
        [cov,mean] = stats_layer[0]
        features_size,_ = cov.shape
        
        mean_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        mean_squared_value_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        for i in range(number_img):
            [cov,mean] = stats_layer[i]
            mean_cov_matrix += cov/number_img
            cov_squared = cov**2
            mean_squared_value_cov_matrix += (cov_squared)/number_img
            
            isnan_cov_squared = np.isnan(cov_squared)
            where_is_nan = np.where(isnan_cov_squared)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,cov[ci,cj],cov_squared[ci,cj])
            isnan_mean_cov = np.isnan(mean_cov_matrix)
            where_is_nan = np.where(isnan_mean_cov)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,isnan_mean_cov[ci,cj])
            
        squared_mean_cov_matrix = mean_cov_matrix**2
        var_cov_matrix = mean_squared_value_cov_matrix - squared_mean_cov_matrix
        var_cov_matrix = np.clip(var_cov_matrix,0.0,np.inf)
        std_cov_matrix = np.sqrt(var_cov_matrix)
        
        if classe is None:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
        else:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
    else:
        mean_cov_matrix = np.load(path_data_cov_matrix)
        std_cov_matrix = np.load(path_data_std_cov_matrix)
 
def compute_global_mean_cov_matrices_onBigSet(    
                                    constrNet = 'InceptionV1',
                                    cropCenter = True,
                                    set_ = 'train',
                                    model_name = 'RASTA_small01_modif',
                                    classe = None,\
                                    layer='mixed4d_pre_relu',
                                    KindOfMeanReduciton = 'global',
                                    verbose=False):
    """
    For a given layer and a given dataset given class
    """

    if 'RASTA' in model_name:
        source_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        source_dataset = 'IconArt_v1'
    number_im_considered = None
    style_layers = [layer]
    whatToload = 'all'
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    if KindOfMeanReduciton=='instance':
        str_stats = 'cov'
    if KindOfMeanReduciton=='global':
        str_stats = 'covGlobalMean'
    elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
        str_stats = 'gram'
    else:
        raise(NotImplementedError)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if classe is None:
        path_data_mean_matrix = os.path.join(output_path,style_layers[0]+'_mean_matrix.npy')
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_mean_'+str_stats+'_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_std_'+str_stats+'_matrix.npy')
    else:
        path_data_mean_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_mean_matrix.npy')
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_std_cov_matrix.npy')

    K.set_learning_phase(0)

    if not (os.path.isfile(path_data_cov_matrix) and os.path.isfile(path_data_std_cov_matrix) and os.path.isfile(path_data_mean_matrix)):
    
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(source_dataset)
        df_train = df_label[df_label['set']=='train']

        if KindOfMeanReduciton=='global':
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,list_layers=style_layers,
                                                                                                    stats_on_layer='mean')
            list_spatial_means = predictionFT_net(mean_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                                                  Net=constrNet,cropCenter=cropCenter)
            print('spatial_means len and shape',len(list_spatial_means),list_spatial_means[0].shape)
            list_mean_spatial_means = []
            dict_global_means = {}
            array_spatial_means = np.array(np.vstack(list_spatial_means))
            mean_spatial_means = np.mean(array_spatial_means,axis=0)
            print('mean_spatial_means',mean_spatial_means.shape)
            dict_global_means[layer] = mean_spatial_means
#            array_spatial_means
#            
#            
#            for layername,spatial_means in zip(style_layers,list_spatial_means):
#                mean_spatial_means = np.mean(np.array(list_spatial_means),axis=0)
#                print('mean_spatial_means',mean_spatial_means.shape)
#                list_mean_spatial_means += [mean_spatial_means]
#                dict_global_means[layername] = mean_spatial_means

        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',set=set_,getBeforeReLU=False,\
                       Net=constrNet,style_layers_imposed=[],\
                       list_mean_and_std_source=[],list_mean_and_std_target=[],\
                       cropCenter=cropCenter,BV=True,sizeIm=224,model_alreadyLoaded=fine_tuned_model,\
                       name_model=model_name,\
                       randomCropValid=False,classe=classe,\
                       KindOfMeanReduciton=KindOfMeanReduciton,\
                       dict_global_means=dict_global_means)
        # Dans le cas de RASTA ce fichier fait plus de 60Go
        
        
        stats_layer = dict_stats[layer]
        print('len(stats_layer)',len(stats_layer))
        del dict_stats
        number_img = len(stats_layer)
        [cov,mean] = stats_layer[0]
        features_size,_ = cov.shape
        
        mean_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        mean_squared_value_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        for i in range(number_img):
            [cov,mean] = stats_layer[i]
            mean_cov_matrix += cov/number_img
            cov_squared = cov**2
            mean_squared_value_cov_matrix += (cov_squared)/number_img
            
            isnan_cov_squared = np.isnan(cov_squared)
            where_is_nan = np.where(isnan_cov_squared)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,cov[ci,cj],cov_squared[ci,cj])
            isnan_mean_cov = np.isnan(mean_cov_matrix)
            where_is_nan = np.where(isnan_mean_cov)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,isnan_mean_cov[ci,cj])
            
        squared_mean_cov_matrix = mean_cov_matrix**2
        var_cov_matrix = mean_squared_value_cov_matrix - squared_mean_cov_matrix
        var_cov_matrix = np.clip(var_cov_matrix,0.0,np.inf)
        std_cov_matrix = np.sqrt(var_cov_matrix)
        
        np.save(path_data_mean_matrix, mean_spatial_means)
        np.save(path_data_cov_matrix, mean_cov_matrix)
        np.save(path_data_std_cov_matrix, std_cov_matrix)
        
    else:
        mean_spatial_means = np.load(path_data_mean_matrix)
        mean_cov_matrix = np.load(path_data_cov_matrix)
        std_cov_matrix = np.load(path_data_std_cov_matrix)
        
    features_size,_ = mean_cov_matrix.shape
    if verbose:
        print(style_layers[0],str(classe))
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
       
    return(mean_cov_matrix,std_cov_matrix,mean_spatial_means)
    
    
def print_rect_matrix(matrix,name,path,title=''):
    features_size,_ = matrix.shape
    dpi = 600
    fig,ax = plt.subplots()
    pos = ax.imshow(matrix,interpolation='none')
    fig.colorbar(pos, ax=ax)
    plt.title(title)
    plt.savefig(os.path.join(path,name+'_colormap.png'),dpi=dpi)
    plt.close()
    fig,ax = plt.subplots()
    pos = ax.imshow(np.log(1.+np.abs(matrix)),interpolation='none')
    fig.colorbar(pos, ax=ax)
    plt.title(title+' log(1+ abs(value))')
    plt.savefig(os.path.join(path,name+'_log_colormap.png'),dpi=dpi)
    plt.close()
    
    ## Cela semble beaucoup trop long a faire !
#    fig, ax = plt.subplots()
#    for i in range(features_size):
#        for j in range(features_size):
#            c = '{0:.2e}'.format(matrix[i][j])
#            ax.text(i+0.5, j+0.5, str(c), va='center', ha='center')
#    
#    plt.savefig(os.path.join(path,name+'_values.png'),dpi=dpi)
    
def print_values(list_labels,list_arrays,path,name):
    assert(len(list_labels)==len(list_arrays))
    dpi = 600
    plt.figure()
    for label,array in zip(list_labels,list_arrays):
        x = np.arange(0,len(array))
        plt.plot(x, array, label=label,linestyle = 'None',
                 marker='o',alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(path,name+'_plots.png'),dpi=dpi)
    
    
def print_stats_matrices(model_name = 'RASTA_small01_modif',
                         list_classes=[None],layer='mixed4d_pre_relu'):
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)

    if KindOfMeanReduciton=='instance':
        path_output_lucid_im = os.path.join(output_path,'PCAlucid')
    if KindOfMeanReduciton=='global':
        path_output_lucid_im = os.path.join(output_path,'PCACovGMeanlucid')
    elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
        path_output_lucid_im = os.path.join(output_path,'PCAGramlucid')
    else:
        raise(NotImplementedError)
    
    list_means = []
    list_cov = []
    
    for classe in list_classes:
        print('classe',classe)
        
        mean_cov_matrix,std_cov_matrix,mean_spatial_means = compute_global_mean_cov_matrices_onBigSet(constrNet=constrNet,
                                                  cropCenter=cropCenter,
                                                  set_=set_,
                                                  model_name = model_name,
                                                   classe = classe,\
                                                   layer=layer,
                                                   KindOfMeanReduciton=KindOfMeanReduciton)
        
        features_size,_ = mean_cov_matrix.shape
        
        if np.isnan(mean_cov_matrix).any():
            print('There is nan value in mean_cov_matrix')
        if np.isnan(std_cov_matrix).any():
            print('There is nan value in std_cov_matrix')
            
        name = 'GlobalCov_'+str(classe)
        title = 'GlobalCov '+str(classe)
        print_rect_matrix(matrix=mean_cov_matrix,name=name,path=path_output_lucid_im,
                          title=title)
        name = 'StdGlobalCov_'+str(classe)
        title = 'StdGlobalCov '+str(classe)
        print_rect_matrix(matrix=std_cov_matrix,name=name,path=path_output_lucid_im,
                          title=title)
        list_means += [mean_spatial_means]
        list_cov += [np.ravel(mean_cov_matrix)]
        
    name = 'Means'
    print_values(list_labels=list_classes,list_arrays=list_means,path=path_output_lucid_im,name=name)
    
    name = 'Covs'
    print_values(list_labels=list_classes,list_arrays=list_cov,path=path_output_lucid_im,name=name)
    

    
def PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',
                               classe = None,\
                               layer='mixed4d_pre_relu',
                               plot_FeatVizu=False):
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    
    mean_cov_matrix,std_cov_matrix,mean_spatial_means = compute_global_mean_cov_matrices_onBigSet(constrNet=constrNet,
                                              cropCenter=cropCenter,
                                              set_=set_,
                                              model_name = model_name,
                                               classe = classe,\
                                               layer=layer,
                                               KindOfMeanReduciton=KindOfMeanReduciton)
    
    features_size,_ = mean_cov_matrix.shape
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    
    if KindOfMeanReduciton=='instance':
        path_output_lucid_im = os.path.join(output_path,'PCAlucid')
    if KindOfMeanReduciton=='global':
        path_output_lucid_im = os.path.join(output_path,'PCACovGMeanlucid')
    elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
        path_output_lucid_im = os.path.join(output_path,'PCAGramlucid')
    else:
        raise(NotImplementedError)
        

    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    if not(classe is None):
        classe_str = classe
    else:
        classe_str = ''
    suffix = ''
    suffix_str = suffix
    # Compute the eigen values 
    #del std_cov_matrix,where_std_sup_abs_mean,where_std_sup_mean
    
    print('plot_FeatVizu',plot_FeatVizu)
    if plot_FeatVizu:
    
        eigen_values, eigen_vectors = LA.eig(mean_cov_matrix)
        
        pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
        plt.figure()
        plt.scatter(np.arange(0,len(eigen_values)),eigen_values,s=4)
        plt.ylabel('Eigen Value')
        plt.savefig(os.path.join(path_output_lucid_im,'EigenValues_'+layer+'_'+classe_str+'.png'),\
                    dpi=300)
        plt.close()
        
        print('Eigen values 10 first value',eigen_values[0:10])
        #print('First eigen vector :',eigen_vectors[:,0])
        print('Max imag part first vector :',np.max(np.imag(eigen_vectors[:,0])))
        eigen_vectors = np.real(eigen_vectors)
    
        num_components_draw = 10
        
        name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
        if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
            name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                       constrNet=constrNet,path=path_lucid_model,suffix=suffix)
        
        if constrNet=='VGG':
            input_name_lucid ='block1_conv1_input'
        elif constrNet=='InceptionV1':
            input_name_lucid ='input_1'
        for comp_number in range(num_components_draw):
            weights = eigen_vectors[:,comp_number]
            #weights = weights[0:1]
            #print('weights',weights)
            #time.sleep(.300)
            prexif_name = '_PCA'+str(comp_number)
            if not(classe is None):
               prexif_name += '_'+classe 
            index_features_withinLayer_all = np.arange(0,features_size)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                             layer_to_print=layer,weights=weights,\
                             index_features_withinLayer=index_features_withinLayer_all,\
                             path_output=path_output_lucid_im,prexif_name=prexif_name,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            prexif_name_pos = prexif_name + '_PosContrib'
            where_pos = np.where(weights>0.)[0]
            weights_pos = list(weights[where_pos])
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                             layer_to_print=layer,weights=weights_pos,\
                             index_features_withinLayer=where_pos,\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_pos,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            prexif_name_neg = prexif_name + '_NegContrib'
            where_neg = np.where(weights>0.)[0]
            weights_neg = list(-weights[where_neg])
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                             layer_to_print=layer,weights=weights_neg,\
                             index_features_withinLayer=where_neg,\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_neg,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            where_max = np.argmax(weights)
            prexif_name_max = prexif_name+  '_Max'+str(where_max)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                             layer_to_print=layer,weights=[1.],\
                             index_features_withinLayer=[where_max],\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            where_min = np.argmin(weights)
            prexif_name_max = prexif_name+  '_Min'+str(where_min)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                             layer_to_print=layer,weights=[1.],\
                             index_features_withinLayer=[where_min],\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            # Faire que les positives après
            
    #        lucid_utils.print_images(model_path=name_pb,list_layer_index_to_print,path_output='',prexif_name='',\
    #                 input_name='block1_conv1_input',Net='VGG')
    #    
    #    
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
        
def produce_latex_text(model_name = 'RASTA_small01_modif',
                       layer_tab=['mixed4d_pre_relu'],classe_tab = [None],
                       folder_im_latex='im'):
    
    path_base  = os.path.join('C:\\','Users','gonthier')
    ownCloudname = 'ownCloud'
    
    if not(os.path.exists(path_base)):
        path_base  = os.path.join(os.sep,'media','gonthier','HDD')
        ownCloudname ='owncloud'
    
    path_to_im_local = os.path.join(folder_im_latex,model_name,'PCAlucid')
    
    folder_im_this_model = os.path.join(path_base,ownCloudname,'Mes_Presentations_Latex','2020-04_Feature_Visualisation',path_to_im_local)
    
    list_all_image = glob.glob(folder_im_this_model+'\*.png')
    #print(list_all_image)
    
    file_path = os.path.join(folder_im_this_model,'printImages.tex')
    file = open(file_path,"w") 
    num_components_draw = 10
    for layer in layer_tab:
        for classe in classe_tab:
            if classe is None:
                classe_str =''
                latex_str = ''
            else:
                classe_str = '_'+classe
                latex_str = r" - %s" % classe.replace('_','\_') 
                latex_str += r" classe only"
            
            for i in range(num_components_draw):
                base_name_im = layer + 'concat__PCA'+str(i)+classe_str
                base_name_im_max = layer + 'concat__PCA'+str(i)+'_Max'
                base_name_im_min = layer + 'concat__PCA'+str(i)+'_Min'
                name_main_image = base_name_im + '_Deco'+'_toRGB.png'
                name_maxcontrib_image = base_name_im + '_PosContrib_Deco'+'_toRGB.png'
                name_mincontrib_image = base_name_im + '_NegContrib_Deco'+'_toRGB.png'
                for name_local in list_all_image:
                    _,name_local = os.path.split(name_local)
                    #print(name_local)
                    if base_name_im_max in name_local:
                        #print('Max')
                        name_local_tab = name_local.split('_')
                        for elt in name_local_tab:
                            if 'Max' in elt:
                                max_index = elt.replace('Max','')
                        name_max_image = base_name_im_max + max_index + '_Deco_toRGB.png'
                    elif base_name_im_min in name_local:
                        #print('Min')
                        name_min_image = name_local
                        name_local_tab = name_local.split('_')
                        for elt in name_local_tab:
                            if 'Min' in elt:
                                min_index = elt.replace('Min','')
                        name_min_image = base_name_im_min + min_index + '_Deco_toRGB.png'
                                
                # Text for Positive contrib slide
                newline = " \n"
                str_beg = r"\frame{  " +newline
                str_beg += r" \frametitle{%s" % model_name.replace('_','\_')
                str_beg += r" - %s" % layer.replace('_','\_')
                str_beg += r" - component %s" % str(i) 
                str_beg +=  latex_str 
                str_beg +=  "} \n " 
                str_beg += r"\begin{figure}[!tbp] " +newline
                str_beg += r"\begin{minipage}[b]{0.29\textwidth}   "+newline
                path_to_im = os.path.join(path_to_im_local,name_main_image).replace("\\", "/")
                str_beg += r"\includegraphics[width=\textwidth]{%s} \\ " % path_to_im
                str_beg += newline
                str_beg += r"{\scriptsize All contribution}"  +newline
                str_beg += r"\end{minipage} \hfill"  +newline
                str_pos = str_beg+  r"\begin{minipage}[b]{0.29\textwidth} " +newline
                path_to_im = os.path.join(path_to_im_local,name_maxcontrib_image).replace("\\", "/")
                str_pos += r"\includegraphics[width=\textwidth]{%s} \\  " % path_to_im
                str_pos += newline
                str_pos += r"{\scriptsize Pos contribution}  " +newline
                str_pos += r"\end{minipage} \hfill " +newline
                str_pos += r"\begin{minipage}[b]{0.29\textwidth} " +newline
                path_to_im = os.path.join(path_to_im_local,name_max_image).replace("\\", "/")
                str_pos += r"\includegraphics[width=\textwidth]{%s} \\  " % path_to_im
                str_pos += newline
                str_pos += r"{\scriptsize Max contribution %s  } "% max_index
                str_pos += newline
                str_pos += r"\end{minipage} \hfill " +newline
                str_pos += r"\end{figure} " +newline
                str_pos += "} \n "
                
                str_neg = str_beg+  r"\begin{minipage}[b]{0.29\textwidth} " +newline
                path_to_im = os.path.join(path_to_im_local,name_mincontrib_image).replace("\\", "/")
                str_neg += r"\includegraphics[width=\textwidth]{%s} \\ " % path_to_im
                str_neg += newline
                str_neg += r"{\scriptsize Neg contribution} "+newline
                str_neg += r"\end{minipage} \hfill " +newline
                str_neg += r"\begin{minipage}[b]{0.29\textwidth}  "+newline
                path_to_im = os.path.join(path_to_im_local,name_min_image).replace("\\", "/")
                str_neg += r"\includegraphics[width=\textwidth]{%s} \\  "% path_to_im
                str_neg += newline
                str_neg += r"{\scriptsize Min contribution  %s  } "% min_index
                str_neg += newline
                str_neg += r"\end{minipage} \hfill   " +newline
                str_neg += r"\end{figure} " +newline
                str_neg += "} \n "
                
                file.write(str_pos)
                file.write(str_neg)

    file.close()
        
def Generate_Im_class_conditionated(model_name='IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                                    constrNet = 'InceptionV1',
                                    classe='Mary',layer='mixed4d_pre_relu'):
    """
    L idee de cette fonction est la suivante en deux étapes :
        Step 1 : creer le block au milieu du reseau qui maximise de maniere robuste
        la réponse a une des classes 
        Step 2 : faire la PCA de ce block de features
        Step 3 : generer l'image qui maximise la premiere composante de cette 
        decomposition
        
    """
    
    if 'IconArt_v1' in model_name:
        dataset = 'IconArt_v1'
    elif 'RASTA'  in model_name:
        dataset = 'RASTA'
    else:
        raise(ValueError('The dataset is unknown'))
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    
    assert(classe in classes)
    
    index_class = classes.index(classe) # Index de la classe en question
    #print(classe,index_class)
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
        #trainable_layers_name = get_trainable_layers_name()
    elif constrNet=='InceptionV1_slim':
        input_name_lucid ='input_1'
        #trainable_layers_name = trainable_layers()
    elif constrNet=='ResNet50':
        input_name_lucid ='input_1'
        raise(NotImplementedError('Not implemented yet with ResNet for print_images'))
    else:
        raise(NotImplementedError(constrNet + ' is not implemented sorry.'))
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    path_output_lucid_im = os.path.join(output_path,'PCAlucid')


    #matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    suffix = ''
    
    K.set_learning_phase(0)
    #with K.get_session().as_default(): 
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    #print('#### ',model_name)
    output_path_with_model = os.path.join(output_path,model_name+suffix)
    pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
    
    net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
    
    #print(net_finetuned.summary())
    
    suffix_str = suffix
    name_pb_full_model = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
    if not(os.path.isfile(os.path.join(path_lucid_model,name_pb_full_model))):
        name_pb_full_model = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,
                                   path=path_lucid_model,suffix=suffix)
        
    layer_concerned = net_finetuned.get_layer(layer)
    layer_output = layer_concerned.output
    dim_shape = layer_output.shape.dims
    new_input_shape = []
    for dim in dim_shape:
        new_input_shape += [dim.value]
    spatial_dim = new_input_shape[1]
    block_shape = [new_input_shape[1],new_input_shape[2],new_input_shape[3]]
    output_layer_name = net_finetuned.layers[-1].name
    dico =  get_dico_layers_type_all_layers_fromNet(net_finetuned)
    print(output_layer_name,net_finetuned.layers[-1])
    name_pb = convert_Part_finetuned_modelToFrozenGraph(model=net_finetuned,
                                                        model_name=model_name,
                                                        new_input_layer_name=layer,
                                                        constrNet=constrNet,
                                                        path=path_lucid_model,
                                                        suffix=suffix)

    #print('output_layer_name',output_layer_name)
    
    index_features_withinLayer_all = [[output_layer_name,index_class]]
    new_input_name_lucid = 'new_input_1'
    prexif_name=''
    constrNet_local = 'GenericFeatureMaps_'+constrNet

    #print(dico)
    # TODO a finir ici
    
    ROBUSTNESS = False # Robustness = True do not work 
    DECORRELATE = False
    output_im_list = lucid_utils.get_feature_block_that_maximizeGivenOutput(model_path=os.path.join(path_lucid_model,name_pb)
                                ,list_layer_index_to_print=index_features_withinLayer_all,
                                input_name=new_input_name_lucid,
                                sizeIm=spatial_dim,\
                                DECORRELATE = DECORRELATE,ROBUSTNESS  = ROBUSTNESS,
                                dico=dico,\
                                image_shape=block_shape)

    feature_block = np.array(output_im_list[0][0])
    print('feature_block',feature_block.shape,np.max(feature_block),np.min(feature_block))
    feature_block_reshaped = feature_block.reshape((-1,feature_block.shape[-1]))
    features_size = feature_block.shape[-1]
    return(0)
    pca = PCA(n_components=None,copy=True,whiten=False)
    pca.fit(feature_block_reshaped)
    eigen_vectors = pca.components_
    print('pca_comp',eigen_vectors.shape)
    first_vector = eigen_vectors[0,:]
              
    num_components_draw = 1

    
    path_output_lucid_im = os.path.join(output_path,'PCAlucid_classCond')

    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
    for comp_number in range(num_components_draw):
        weights = eigen_vectors[:,comp_number]
        #weights = weights[0:1]
        #print('weights',weights)
        #time.sleep(.300)
        prexif_name = '_PCA'+str(comp_number)
        if not(classe is None):
           prexif_name += '_'+classe 
        index_features_withinLayer_all = np.arange(0,features_size)
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                         layer_to_print=layer,weights=weights,\
                         index_features_withinLayer=index_features_withinLayer_all,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_pos = prexif_name + '_PosContrib'
        where_pos = np.where(weights>0.)[0]
        weights_pos = list(weights[where_pos])
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                         layer_to_print=layer,weights=weights_pos,\
                         index_features_withinLayer=where_pos,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_pos,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_neg = prexif_name + '_NegContrib'
        where_neg = np.where(weights>0.)[0]
        weights_neg = list(-weights[where_neg])
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                         layer_to_print=layer,weights=weights_neg,\
                         index_features_withinLayer=where_neg,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_neg,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        where_max = np.argmax(weights)
        prexif_name_max = prexif_name+  '_Max'+str(where_max)
        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                         layer_to_print=layer,weights=[1.],\
                         index_features_withinLayer=[where_max],\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)



def layers_unique(liste):
    new_liste= []
    new_liste_name= []
    for elt in liste:
        if elt.name in new_liste_name:
            continue
        else:
            new_liste_name += [elt.name]
            new_liste += [elt]
        
    return(new_liste)
    
def convert_Part_finetuned_modelToFrozenGraph(model,model_name,new_input_layer_name,
                                              constrNet='InceptionV1',path='',
                                              suffix=''):
    
#    tf.keras.backend.clear_session()
#    tf.reset_default_graph()
#    K.set_learning_phase(0)
    
#    if constrNet=='VGG':
#        input_name_lucid ='block1_conv1_input'
#        raise(NotImplementedError)
#    elif constrNet=='InceptionV1':
#        input_name_lucid ='input_1'
#        #trainable_layers_name = get_trainable_layers_name()
#    elif constrNet=='InceptionV1_slim':
#        input_name_lucid ='input_1'
#        #trainable_layers_name = trainable_layers()
#        raise(NotImplementedError)
#    elif constrNet=='ResNet50':
#        input_name_lucid ='input_1'
#        raise(NotImplementedError)
#    else:
#        raise(NotImplementedError)

    layer = model.get_layer(new_input_layer_name)
    layer_output = layer.output
    dim_shape = layer_output.shape.dims
    new_input_shape = []
    for dim in dim_shape:
        new_input_shape += [dim.value]
    new_input_shape.pop(0) #remove the batch size dim
    new_input_shape_with_batch = [None] + new_input_shape
    new_input = Input(shape=new_input_shape,name='new_input_1') 
    print(new_input)
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    
    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if (layer.name not in network_dict['input_layers_of'][layer_name]):
                    network_dict['input_layers_of'][layer_name].append(layer.name)
    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    firstTime_here = True
    new_input_layer_passed = False
    for layer in model.layers[1:]:
        #print(layer.name)
        # Determine input tensors
        if new_input_layer_passed and not(firstTime_here):
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
            
        if new_input_layer_passed:
                
            if isinstance(layer, Concatenate) and firstTime_here: # We will skip it !
                continue
            
            if firstTime_here:
                firstTime_here = False
            else:
                #layer_input = layer_input[0,...]
                if len(layer_input) >1:
                    layer_input = layers_unique(layer_input)
                if len(layer_input) == 1:
                    layer_input = layer_input[0]
                    if len(layer_input.shape)==5:
                        layer_input = layer_input[0,...]
            
            x = layer(layer_input)
            #print(x,layer_input)
            network_dict['new_output_tensor_of'].update({layer.name: x})
            
        if layer.name == new_input_layer_name:
            
            layer_input = new_input
            new_input_layer_passed = True
            
    net_finetuned_truncated = Model(inputs=new_input,outputs=x)
    
        
    include_optimizer = False
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        net_finetuned_truncated.save(model_path,include_optimizer=include_optimizer)
        #fix_layer0(model_path, new_input_shape_with_batch, 'float32')
        curr_session = tf.get_default_session()
        # close current session
        if curr_session is not None:
            curr_session.close()
        # reset graph
        K.clear_session()
        # create new session
        s = tf.Session()
        K.set_session(s)
        
        if constrNet=='InceptionV1':
            custom_objects = {'PoolHelper': PoolHelper,'LRN':LRN}
        
        net_finetuned_truncated= load_model(model_path, custom_objects=custom_objects) # load_model return the model
    finally:
        os.remove(model_path)
    #print(net_finetuned_truncated.summary())
#    del model
    
    if path=='':
        os.makedirs('./model', exist_ok=True)
        path ='model'
    else:
        os.makedirs(path, exist_ok=True)
    frozen_graph = lucid_utils.freeze_session(K.get_session(),
                              output_names=[out.op.name for out in net_finetuned_truncated.outputs])
    if not(suffix=='' or suffix is None):
        suffix_str = '_'+suffix
    else:
        suffix_str = ''
    name_pb = 'tf_graph_'+constrNet+model_name+new_input_layer_name+suffix_str+'.pb'
    
    nodes_tab = [n.name for n in frozen_graph.node]
    #print(nodes_tab)
    tf.io.write_graph(frozen_graph,logdir= path,name= name_pb, as_text=False)
    #input('wait')
    # Ici cela ne marche pas car on a encore l ancien graph de lautre reseau !
    return(name_pb)            
    
    
def compute_mean_and_covariance_forTrainSet(dataset,model_name,constrNet,
                                            layers_concerned=[],
                                            stats_on_layer='mean',
                                            suffix='',cropCenter = True,
                                            FTmodel=True):
    """
    This function will compute the mean activation of each features maps for all
    the layers and also the layers_concerned and then the
    cumulated covariance matrix of those feature maps also with substraction 
    of the spatial mean obtained on the whole dataset
    For a given network
    @param : FTmodel : in the case of finetuned from scratch if False use the initialisation
    networks
    """
    K.set_learning_phase(0) #IE no training
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']

    if model_name=='pretrained':
        base_model = Activation_for_model.get_Network(constrNet)
    else:
        # Pour ton windows il va falloir copier les model .h5 finetuné dans ce dossier la 
        # C:\media\gonthier\HDD2\output_exp\Covdata\RASTA\model
        if 'RandInit' in model_name:
            FT_model,init_model = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if FTmodel:
                base_model = FT_model
            else:
               base_model = init_model 
        else:
            output = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if len(output)==2:
                base_model, init_model = output
            else:
                base_model = output
    mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(base_model,list_layers=layers_concerned,
                                                                                                          stats_on_layer=stats_on_layer)
    #print(model.summary())
    list_spatial_means = predictionFT_net(mean_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter)
    print('spatial_means len and shape',len(list_spatial_means),list_spatial_means[0].shape)
    list_mean_spatial_means = []
    for spatial_means in list_spatial_means:
        mean_spatial_means = np.mean(spatial_means,axis=0)
        list_mean_spatial_means += [mean_spatial_means]
    
    #stats_on_layer = 'cov_instance_mean'
    stats_on_layer = 'cov_global_mean'
    #stats_on_layer = 'gram'
    cov_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(base_model,list_layers=layers_concerned,
                                                                                        stats_on_layer=stats_on_layer,
                                                                                            list_means=list_mean_spatial_means)
    list_covs = predictionFT_net(cov_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter)
    print('list_covs len and shape',len(list_covs),list_covs[0].shape)
    list_mean_covs = []
    for covs in list_covs:
        mean_covs = np.mean(covs,axis=0)
        list_mean_covs += [mean_covs]
        
    return(list_mean_spatial_means,list_mean_covs)
    
    
    

    

    
    
if __name__ == '__main__':
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',classe = None,\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',classe ='Color_Field_Painting',\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',classe ='Abstract_Art',\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',classe ='Northern_Renaissance',\
#                                   layer='mixed4d_pre_relu')
#    produce_latex_text(model_name = 'RASTA_small01_modif',
#                       layer_tab=['mixed4d_pre_relu'],classe_tab = [None,'Color_Field_Painting','Abstract_Art','Northern_Renaissance'],
#                       folder_im_latex='im')
    
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe = None,\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe ='Mary',\
#                                   layer='mixed4d_pre_relu')

#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe = None,\
#                                   layer='mixed4b_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe ='Mary',\
#                                   layer='mixed4b_pre_relu')
#    produce_latex_text(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',
#                       layer_tab=['mixed4d_pre_relu','mixed4b_pre_relu'],classe_tab = [None,'Mary'],
#                       folder_im_latex='im')
    
    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = None,\
                                   layer='mixed4d',plot_FeatVizu=False)
    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
                                   layer='mixed4d',classe='Mary')
    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
                                   layer='mixed4d',classe='ruins')
    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
                                   layer='mixed4d',classe='nudity')
    
    print_stats_matrices(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                         list_classes=[None,'Mary','ruins','nudity'],layer='mixed4d')
    
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = None,\
#                                   layer='mixed4c_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = 'Mary',\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = 'Mary',\
#                                   layer='mixed4c_pre_relu')
    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                               classe = None,\
                                   layer='mixed4d')
    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                               classe ='Northern_Renaissance',\
                                   layer='mixed4d')
    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                               classe ='Abstract_Art',\
                                   layer='mixed4d')
    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                               classe ='Ukiyo-e',\
                                layer='mixed4d')
    
    
    
    

