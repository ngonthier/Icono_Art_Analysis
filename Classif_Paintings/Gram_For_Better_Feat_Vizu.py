# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:04 2020

The goal of this script is to look at the gram matrices of a fine-tuned network
To see the eigen values of its to see if we can have a better features visualisation

Pour le moment, on calcule les covariances après sous-traction de la moyenne spatiale
d'une image donnée et non pas après soustraction de la moyenne globale sur tout le 
dataset

Remarques tu peux parfois avoir l'erreur suivante : 
UnknownError: 2 root error(s) found.(0) Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
    
Il te faudra alors peut etre vider le cache qui se trouve a l'endroit suivant : 
    AppData Roaming NVIDIA ComputeCache

https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in



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
from sklearn.cluster import KMeans,MeanShift
import scipy 

from tensorflow.python.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralCoclustering,SpectralBiclustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.metrics import average_precision_score

from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate,Activation,Dense,Flatten,Input,Dropout,InputLayer

from Study_Var_FeaturesMaps import get_dict_stats
from CompNet_FT_lucidIm import get_fine_tuned_model,convert_finetuned_modelToFrozenGraph,\
    get_path_pbmodel_pretrainedModel
import lucid_utils
from IMDB import get_database
from infere_layers_info import get_dico_layers_type_all_layers_fromNet
from googlenet import LRN,PoolHelper
from utils_keras import fix_layer0

import Activation_for_model
from StatsConstr_ClassifwithTL import predictionFT_net

from inceptionV1_keras_utils import get_dico_layers_type

from Stats_Fcts import get_Model_cov_mean_features,get_Model_gram_mean_features,get_Model_cov_mean_features_global_mean
from Stats_Fcts import load_resize_and_process_img
from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator

import pickle
import matplotlib

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
 
def compute_global_clustering_cov_matrices_onBigSet(    
                                    constrNet = 'InceptionV1',
                                    cropCenter = True,
                                    set_ = 'train',
                                    model_name = 'RASTA_small01_modif',
                                    classe = None,\
                                    layer='mixed4d_pre_relu',
                                    KindOfMeanReduciton = 'global',
                                    verbose=False,
                                    source_dataset=None,
                                    clustering=''):
    """
    For a given layer and a given dataset given class
    """
    if model_name=='pretrained':
        assert(not(source_dataset is None))
    else:
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
        
        if model_name=='pretrained':
            fine_tuned_model = Activation_for_model.get_Network(constrNet)
        else:
            fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(source_dataset)
        df_train = df_label[df_label['set']=='train']
        if not(classe is None):
            df_train_c = df_train[df_train[classe]==1.0]
        else:
            df_train_c = df_train

        if KindOfMeanReduciton=='global':
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,list_layers=style_layers,
                                                                                                    stats_on_layer='mean')
            list_spatial_means = predictionFT_net(mean_model,df_train_c,x_col=item_name,y_col=classes,path_im=path_to_img,
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
        
        size_data = int(((features_size+1)*features_size)//2)
        X_data = np.empty(shape=(number_img,size_data))
        
        for i in range(number_img):
            [cov,mean] = stats_layer[i]
            triu_matrices = cov[np.triu_indices(features_size)]
            X_data[i,:] = triu_matrices
        
        n_clusters = 3
        ### Ici on veut un clustering 
        
        scoclustering = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
        #scoclustering = SpectralBiclustering(n_clusters=n_clusters, random_state=0)
        
        scoclustering.fit(X_data)
        row_labels_= scoclustering.row_labels_
        column_labels_ = scoclustering.column_labels_
        
        for l_i in range(n_clusters):
            image_clustered_together = X_data[np.where(row_labels_==l_i)]
            
            
            #for r_i in range(n_clusters):
                
        
        

def compute_global_mean_cov_matrices_onBigSet(    
                                    constrNet = 'InceptionV1',
                                    cropCenter = True,
                                    set_ = 'train',
                                    model_name = 'RASTA_small01_modif',
                                    classe = None,\
                                    layer='mixed4d_pre_relu',
                                    KindOfMeanReduciton = 'global',
                                    verbose=False,
                                    source_dataset=None):
    """
    For a given layer and a given dataset given class
    """
    if model_name=='pretrained':
        assert(not(source_dataset is None))
    else:
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
        
        if model_name=='pretrained':
            fine_tuned_model = Activation_for_model.get_Network(constrNet)
        else:
            fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(source_dataset)
        df_train = df_label[df_label['set']=='train']
        if not(classe is None):
            df_train_c = df_train[df_train[classe]==1.0]
        else:
            df_train_c = df_train

        if KindOfMeanReduciton=='global':
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,list_layers=style_layers,
                                                                                                    stats_on_layer='mean')
            list_spatial_means = predictionFT_net(mean_model,df_train_c,x_col=item_name,y_col=classes,path_im=path_to_img,
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
    
    
    
    
def print_hist_diag_matrix(matrix,name,path,title=''):
    diag = np.diagonal(matrix)
    plt.figure()
    plt.hist(diag, bins = 100)
    plt.yscale('log')
    dpi = 600
    plt.title(title)
    plt.savefig(os.path.join(path,name+'_diagHisto.png'),dpi=dpi)
    plt.close()
    
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
#    fig,ax = plt.subplots()
#    pos = ax.imshow(np.log(1.+np.abs(matrix)),interpolation='none')
#    fig.colorbar(pos, ax=ax)
#    plt.title(title+' log(1+ abs(value))')
#    plt.savefig(os.path.join(path,name+'_log_colormap.png'),dpi=dpi)
#    plt.close()
    
    ## Cela semble beaucoup trop long a faire !
#    fig, ax = plt.subplots()
#    for i in range(features_size):
#        for j in range(features_size):
#            c = '{0:.2e}'.format(matrix[i][j])
#            ax.text(i+0.5, j+0.5, str(c), va='center', ha='center')
#    
#    plt.savefig(os.path.join(path,name+'_values.png'),dpi=dpi)
    
def print_values(list_labels,list_arrays,path,name,title=''):
    assert(len(list_labels)==len(list_arrays))
    dpi = 600
    plt.figure()
    for label,array in zip(list_labels,list_arrays):
        x = np.arange(0,len(array))
        plt.plot(x, array, label=label,linestyle = 'None',
                 marker='o',alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(path,name+'_plots.png'),dpi=dpi)
    
    
def print_stats_matrices(model_name = 'RASTA_small01_modif',
                         list_classes=[None],layer='mixed4d_pre_relu',
                         source_dataset=None):
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    
    if model_name=='pretrained':
        assert(not(source_dataset is None))
        str_folder = source_dataset
    else:
        str_folder = ''
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)

    if KindOfMeanReduciton=='instance':
        path_output_lucid_im = os.path.join(output_path,'PCAlucid'+str_folder)
    if KindOfMeanReduciton=='global':
        path_output_lucid_im = os.path.join(output_path,'PCACovGMeanlucid'+str_folder)
    elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
        path_output_lucid_im = os.path.join(output_path,'PCAGramlucid'+str_folder)
    else:
        raise(NotImplementedError)
    
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
    
    list_means = []
    list_cov = []
    
    for classe in list_classes:
        print('classe',classe)
        
        mean_cov_matrix,std_cov_matrix,mean_spatial_means = compute_global_mean_cov_matrices_onBigSet(constrNet=constrNet,
                                                  cropCenter=cropCenter,\
                                                  set_=set_,\
                                                  model_name = model_name,
                                                   classe = classe,\
                                                   layer=layer,\
                                                   KindOfMeanReduciton=KindOfMeanReduciton,\
                                                   source_dataset=source_dataset)
        
        features_size,_ = mean_cov_matrix.shape
        
        if np.isnan(mean_cov_matrix).any():
            print('There is nan value in mean_cov_matrix')
        if np.isnan(std_cov_matrix).any():
            print('There is nan value in std_cov_matrix')
            
        name = 'GlobalCov_'+layer+'_'+str(classe)
        title = 'GlobalCov '+layer+' '+str(classe)
        print_rect_matrix(matrix=mean_cov_matrix,name=name,path=path_output_lucid_im,
                          title=title)
        name = 'StdGlobalCov_'+layer+'_'+str(classe)
        title = 'StdGlobalCov '+layer+' '+str(classe)
        print_rect_matrix(matrix=std_cov_matrix,name=name,path=path_output_lucid_im,
                          title=title)
        
        name = 'GlobalCov_'+layer+'_'+str(classe)
        title = 'GlobalCov '+layer+' '+str(classe) + ' Histogram of diagonal values'
        print_hist_diag_matrix(mean_cov_matrix,name,path=path_output_lucid_im,
                               title=title)
        
        list_means += [mean_spatial_means]
        list_cov += [np.ravel(mean_cov_matrix)]
        
        
    name = 'Means_'+layer
    title = 'Means '+layer
    print_values(list_labels=list_classes,list_arrays=list_means,path=path_output_lucid_im,name=name,
                title=title)
    name = 'Covs_'+layer
    title = 'Covs '+layer
    print_values(list_labels=list_classes,list_arrays=list_cov,path=path_output_lucid_im,name=name,
                 title=title)
    
def Clust_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
                               classe = None,\
                               layer='mixed4d',\
                               plot_FeatVizu=True,\
                               source_dataset=None,
                               num_components_draw = 10,
                               strictMinimum=False,cossim=False,
                               clustering='PCA'):
    """
    Clustering on one image !
    """
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    KindOfMeanReduciton=None
    
    if 'IconArt_v1' in model_name:
        dataset = 'IconArt_v1'
    elif 'RASTA'  in model_name:
        dataset = 'RASTA'
    else:
        raise(ValueError('The dataset is unknown'))
    
    if model_name=='pretrained':
        fine_tuned_model = Activation_for_model.get_Network(constrNet)
    else:
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

    model_alreadyLoaded = fine_tuned_model

    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    
    style_layers = [layer]
    if KindOfMeanReduciton=='global':
        mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,list_layers=style_layers,
                                                                                                stats_on_layer='mean')
        list_spatial_means = predictionFT_net(mean_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                                              Net=constrNet,cropCenter=cropCenter)
    if KindOfMeanReduciton=='instance':
        net_get_cov = get_Model_cov_mean_features(style_layers,model_alreadyLoaded)
    elif KindOfMeanReduciton is None or  KindOfMeanReduciton=='':
        net_get_cov = get_Model_gram_mean_features(style_layers,model_alreadyLoaded)
    elif KindOfMeanReduciton=='global':
        net_get_cov = get_Model_cov_mean_features_global_mean(style_layers,model_alreadyLoaded,
                                                   dict_global_means)
    else:
        raise(ValueError(KindOfMeanReduciton))
        
    randomCropValid = False
    Net = constrNet 
    sizeIm = 224
    # Select one image only
    if not(classe is None):
        df_train = df_train[df_label[classe]==1]
    list_imgs = df_train[item_name].values 
    
    number_im_considered = 10
    
    for i in range(number_im_considered):
        img_name = list_imgs[i] + '.jpg'
        image_path = os.path.join(path_to_img,img_name)
        print(image_path)
        
    #    for i,image_path in enumerate(list_imgs):
    #        if number_im_considered is None or i < number_im_considered:
    #            if i%itera==0: print(i,image_path)
        head, tail = os.path.split(image_path)
        short_name = '.'.join(tail.split('.')[0:-1])
    #    if not(set is None or set==''):
    #        if not(short_name in images_in_set):
    #            # The image is not in the set considered
    #            continue
        # Get the covairances matrixes and the means
        try:
            #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
            # Erreur : 
            #FileNotFoundError: [Errno 2] No such file or directory: 'data/RASTA_LAMSADE/wikipaintings_full/wikipaintings_train/Northern_Renaissance/hans-holbein-the-younger_henry-viii-handing-over-a-charter-to-thomas-vicary-commemorating-the-joining-of-the-barbers-and-1541.jpg'
            try:
                if cropCenter:
                    image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                            crop_size=sizeIm,interpolation='lanczos:center')
                    # For VGG or ResNet with classification head size == 224
                elif randomCropValid:
                    image_array= load_and_crop_img(path=image_path,Net=Net,target_size=256,
                                            crop_size=sizeIm,interpolation='lanczos:center')
                else:
                    image_array = load_resize_and_process_img(image_path,Net=Net,max_dim=sizeIm)
            except FileNotFoundError as error:
                # A workaround because name too long for windows system file !
                if 'hans-holbein-the-younger_henry-viii-handing-over-a-charter-to-thomas-vicary-commemorating-the-joining-of-the-barbers-and-1541' in image_path:
                    image_path = image_path.replace('hans-holbein-the-younger_henry-viii-handing-over-a-charter-to-thomas-vicary-commemorating-the-joining-of-the-barbers-and-1541','hans-holbein-henry-viii-handing')
                    if cropCenter:
                        image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                                crop_size=sizeIm,interpolation='lanczos:center')
                        # For VGG or ResNet with classification head size == 224
                    elif randomCropValid:
                        image_array= load_and_crop_img(path=image_path,Net=Net,target_size=256,
                                                crop_size=sizeIm,interpolation='lanczos:center')
                    else:
                        image_array = load_resize_and_process_img(image_path,Net=Net,max_dim=sizeIm)
                else:
                    raise(error)
            net_cov_mean = net_get_cov.predict(image_array, batch_size=1)
        except IndexError as e:
            print(e)
            print(i,image_path)
            raise(e)
    
    
        cov,mean = net_cov_mean
        cov = cov[0,:,:]
        
        plt.figure()
        plt.imshow(cov)
        plt.title('Cov Matrices of '+short_name)
    plt.show()
    input('enter to close')
    plt.close('all')
    
#    n_clusters = 9
#    ### Ici on veut un clustering 
#    
#    scoclustering = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
#    # To cluster row and columns at the same time : https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-coclustering-py
#    scoclustering.fit(cov)
#    row_labels_= scoclustering.row_labels_
#    column_labels_ = scoclustering.column_labels_
#    
#    indices = []
#    for i,mx in enumerate(cov):
#        if np.sum(mx, axis=1) == 0:
#            indices.append(i)
#    
#    mask = np.ones(mtx.shape[0], dtype=bool)
#    mask[indices] = False
#    mtx = mtx[mask] 
    
#    for l_i in range(n_clusters):
#        cluster_cov_i = cov[]
#        image_clustered_together = cov[np.where(row_labels_==l_i)]


def plot_confusion_matrix(conf_arr,classes,title=''):

    new_conf_arr = []
    for row in conf_arr:
        new_conf_arr.append(row / sum(row))

    plt.matshow(new_conf_arr)
    plt.yticks(range(25), classes)
    plt.xticks(range(25), classes, rotation=90)
    plt.colorbar()
    plt.title(title, y=-0.01) # To put it at the bottom
    plt.show()

def get_confusion_matrices_for_list_models(list_model_name=['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif'],
                                           list_constrNet=['InceptionV1']):
    model_name = list_model_name[0]
    if 'IconArt_v1' in model_name:
        dataset = 'IconArt_v1'
    elif 'RASTA' in model_name:
        dataset = 'RASTA'
    else:
        raise(ValueError('The dataset is unknown'))
        
    cropCenter = True

    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    
    if len(list_constrNet)==1:
        list_constrNet = [list_constrNet[0]]*len(list_model_name)

    
    df_val = df_label[df_label['set']==str_val]
    df_test = df_label[df_label['set']=='test']
    
    list_pred = []
    list_precision = []
    list_top1_acc = []
    list_ap = []
    
    mean_pred = None
    bagging_method = 'mean'
    bagging_method = 'val_weighted_top1_acc'
    bagging_method = 'val_weighted_ap'
    
    for model_name,constrNet in zip(list_model_name,list_constrNet):
    
        print(model_name)
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)
        labels = df_test[classes].values
        val_labels = df_val[classes].values
        
        labels_multiclass = []
        for i in range(len(labels)):
            gt_label_i = labels[i,:]
            where = np.where(gt_label_i==1)[0][0]
            labels_multiclass += [classes[where]]
            
        #labels_multiclass = np.where(labels==1)
        val_pred = predictionFT_net(fine_tuned_model,df_val,x_col=item_name,y_col=classes,path_im=path_to_img,
                                              Net=constrNet,cropCenter=cropCenter)
        if bagging_method=='val_weighted_top1_acc':
            raise(NotImplementedError)
            top_1_acc_val = tf.keras.metrics.top_k_categorical_accuracy(val_labels,val_pred,k=1).eval()
            list_top1_acc += [top_1_acc_val]  
        if bagging_method=='val_weighted_ap':
            ap_list = []
            for j in range(num_classes):
                ap_j = average_precision_score(val_labels[:,j],val_pred[:,j],average=None)
                ap_list += [ap_j]
            list_ap += [np.array(ap_list)]  

        test_pred = predictionFT_net(fine_tuned_model,df_test.copy(),x_col=item_name,y_col=classes,path_im=path_to_img,
                                     Net=constrNet,cropCenter=cropCenter)
        curr_session = tf.get_default_session()
        # close current session
        if curr_session is not None:
            curr_session.close()
        # reset graph
        K.clear_session()
        # create new session
        s = tf.InteractiveSession()
        K.set_session(s)
        
        if bagging_method=='mean':
            if mean_pred is None:
                mean_pred = test_pred
            else:
                mean_pred += test_pred

        list_pred += [test_pred]
        
    pred = np.zeros_like(list_pred[0])
        
    if bagging_method=='mean':
        pred = mean_pred/len(model_name)
    elif bagging_method=='val_weighted_top1_acc':
        for j in range(num_classes):
            list_j = []
            weights = []
            for pred_classif_i,top_1_acc_val in zip(list_pred,list_top1_acc):
                print('top_1_acc_val',top_1_acc_val.shape)
                test_pred_j = pred_classif_i[:,j]
                list_j += [test_pred_j]
                weights += [top_1_acc_val[j]]
            test_pred_j_all = np.vstack(test_pred_j)
            print(test_pred_j_all.shape)
            print(len(weights))
            test_pred_j_voting = np.average(test_pred_j_all, axis=1,
                   weights=weights)
            pred[:,j]  = test_pred_j_voting
    elif bagging_method=='val_weighted_ap':
        for j in range(num_classes):
            list_j = []
            weights = []
            for pred_classif_i,ap in zip(list_pred,list_ap):
                test_pred_j = pred_classif_i[:,j]
                list_j += [test_pred_j]
                weights += [ap[j]]
            test_pred_j_all = np.vstack(list_j)
            test_pred_j_voting = np.average(test_pred_j_all, axis=0,
                   weights=weights)
            pred[:,j]  = test_pred_j_voting
    print(pred.shape)
    # Ici on pourrai
    preds_top_class = []
    for i in range(len(pred)): 
        index = np.argsort(pred[i,:])[::-1][0]
        preds_top_class += [classes[index]]
   
#    conf_arr = confusion_matrix(labels, preds)
   
    top_k_accs = []
    top_k = [1,3,5]
    for k in top_k:
        top_k_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(labels,pred,k=k).eval())
        top_k_accs += [top_k_acc]
        print('Top {0} accuracy : {1:.2f} %'.format(k,top_k_acc*100))
        
    conf_arr = confusion_matrix(labels_multiclass, preds_top_class)
    plot_confusion_matrix(conf_arr,classes,title='')
    
    ## Results
    # Avec mean bagging ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif']
#    Top 1 accuracy : 56.81 %
#    Top 3 accuracy : 82.75 %
#    Top 5 accuracy : 91.11 %
    
    # Avec mean ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif','RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200']
#    Top 1 accuracy : 55.94 %
#    Top 3 accuracy : 82.60 %
#    Top 5 accuracy : 91.03 %
    # val_weighted_top1_acc
#    get_confusion_matrices_for_list_models(list_model_name=['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif'],
#                                           list_constrNet=['InceptionV1'])
# Avec ap weigthed bagging ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif']
#Top 1 accuracy : 56.88 %
#Top 3 accuracy : 82.81 %
#Top 5 accuracy : 91.29 %
# Avec ap weighted bagging ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200','RASTA_small01_modif','RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200']
#Top 1 accuracy : 56.44 %
#Top 3 accuracy : 82.73 %
#Top 5 accuracy : 91.33 %
    
    
    
    
def get_confusion_matrices_for_given_model(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                                           constrNet='InceptionV1'):
    if 'IconArt_v1' in model_name:
        dataset = 'IconArt_v1'
    elif 'RASTA'  in model_name:
        dataset = 'RASTA'
    else:
        raise(ValueError('The dataset is unknown'))
        
    cropCenter = True

    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    
    fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)
    df_test = df_label[df_label['set']=='test']
    labels = df_test[classes].values
    
    labels_multiclass = []
    for i in range(len(labels)):
        gt_label_i = labels[i,:]
        where = np.where(gt_label_i==1)[0][0]
        labels_multiclass += [classes[where]]
        
    #labels_multiclass = np.where(labels==1)
    pred = predictionFT_net(fine_tuned_model,df_test,x_col=item_name,y_col=classes,path_im=path_to_img,
                                          Net=constrNet,cropCenter=cropCenter)
    
    preds_top_class = []
    for i in range(len(pred)): 
        index = np.argsort(pred[i,:])[::-1][0]
        preds_top_class += [classes[index]]
   
#    conf_arr = confusion_matrix(labels, preds)
    top_k_accs = []
    top_k = [1,3,5]
    for k in top_k:
        top_k_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(labels,pred,k=k).eval())
        top_k_accs += [top_k_acc]
        print('Top {0} accuracy : {1:.2f} %'.format(k,top_k_acc*100))
        
    conf_arr = confusion_matrix(labels_multiclass, preds_top_class)
    #print('conf_arr.shape',conf_arr.shape)
    plot_confusion_matrix(conf_arr,classes,title='') #model_name+' '+constrNet
## RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200
#    Top 1 accuracy : 49.05 %
#    Top 3 accuracy : 77.19 %
#    Top 5 accuracy : 87.71 %
## RASTA_small01_modif
#Top 1 accuracy : 55.18 %
#Top 3 accuracy : 82.25 %
#Top 5 accuracy : 91.06 %
## RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200
#Top 1 accuracy : 47.21 %
#Top 3 accuracy : 76.03 %
#Top 5 accuracy : 86.87 %
# RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200
#Top 1 accuracy : 47.19 %
#Top 3 accuracy : 76.23 %
#Top 5 accuracy : 86.38 %    

def topK_features_per_class_list_of_model():
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
#    model_name_list = ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                       'RASTA_small01_modif',
#                       'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                       'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                       'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200'
#                       ]
    model_name_list = [
                       'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200'
                       ] # a faire plus tard
    
    for model_name in model_name_list:
        for stats_on_layer in ['mean','max']:
            vizu_topK_feature_per_class(model_name =model_name,\
                                       layer='mixed4d',\
                                       source_dataset=None,
                                       num_components_draw = 10,
                                       stats_on_layer=stats_on_layer)
            
def topK_features_per_class_list_of_modelpretrained():
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
#    model_name_list = ['RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                       'RASTA_small01_modif',
#                       'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                       'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                       'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200'
#                       ]
    model_name_list = [
                       'pretrained',
                       'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                       'RASTA_small01_modif',
                       'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
                       'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
                       'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG'
                       ] # a faire plus tard
    
    # Tu n'as pas fini pour RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200 
    # il faudra relancer cela 
    
    for layer in ['mixed4d','mixed5a']:
        for model_name in model_name_list:
            for stats_on_layer in ['mean','max','meanFirePos_minusMean','meanFirePos']:
                for selection_feature in [None,'ClassMinusGlobalMean']:
                    vizu_topK_feature_per_class(model_name =model_name,\
                                               layer=layer,\
                                               source_dataset='RASTA',
                                               num_components_draw = 10,
                                               stats_on_layer=stats_on_layer,
                                               selection_feature=selection_feature)
                    
def test_meanFirePos():
    
    for selection_feature in [None]:
        vizu_topK_feature_per_class(model_name ='RASTA_small01_modif',\
                                   layer='mixed4d',\
                                   source_dataset='RASTA',
                                   num_components_draw = 2,
                                   stats_on_layer='meanFirePos_minusMean',# meanFirePos
                                   selection_feature=selection_feature)

def vizu_topK_feature_per_class(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
                               layer='mixed4d',\
                               source_dataset=None,
                               num_components_draw = 10,
                               cossim=False,
                               constrNet = 'InceptionV1',
                               dot_vector=True,
                               stats_on_layer='mean',
                               selection_feature=None):
    """
    Le but de cette fonction est d'afficher les k features avec la plus forte 
    réponses 
    
    @param : If selection_feature is None we do nothing
        if selection_feature=='TopOnlyForClass' Top For this class and not for the other model'
        if selection_feature=='ClassMinusGlobalMean'
    """
    
    
    if 'IconArt_v1' in model_name:
        dataset = 'IconArt_v1'
    elif 'RASTA'  in model_name:
        dataset = 'RASTA'
    else:
        if not(source_dataset is None):
            dataset = source_dataset
        else:
            raise(ValueError('The dataset is unknown'))
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    
    clustering = None
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    
    df_train = df_label[df_label['set']=='train']

    
    if model_name=='pretrained':
        add_end_folder_name = '_'+dataset
    else:
        add_end_folder_name = ''
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
    
    path_output_lucid_im = os.path.join(output_path,'AllFeatures')
        
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True)
    

    name_dico = 'DicoOrderLayer_'+layer
    if not(stats_on_layer=='mean'):
        name_dico +='_'+stats_on_layer
    if selection_feature=='TopOnlyForClass':
       name_dico +='_TopOnlyForClass'
    if selection_feature=='ClassMinusGlobalMean':
       name_dico +='_ClassMinusGlobalMean'
    name_dico += '.pkl'
    path_dico = os.path.join(path_output_lucid_im,name_dico)
    
    ROBUSTNESS = True
    DECORRELATE = True
    
    if model_name=='pretrained':
        path_lucid_model = os.path.join('')
    else:
        path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    suffix = ''
    
    if not(os.path.isfile(path_dico)):

        if model_name=='pretrained':
           fine_tuned_model = Activation_for_model.get_Network(constrNet)
        else:
           fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)
           path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
        
        style_layers = [layer]
        
        if stats_on_layer=='meanFirePos_minusMean':
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,
                                                                                                     list_layers=style_layers,
                                                                                                     stats_on_layer='meanAfterRelu')
        else:
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,
                                                                                                     list_layers=style_layers,
                                                                                                     stats_on_layer=stats_on_layer)
        
        suffix_str = suffix                                                                     
        if not(model_name=='pretrained'):
            name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
            if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
                name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                           constrNet=constrNet,path=path_lucid_model,suffix=suffix)
            if constrNet=='VGG':
                input_name_lucid ='block1_conv1_input'
            elif constrNet=='InceptionV1':
                input_name_lucid ='input_1'
            elif constrNet=='InceptionV1_slim':
                input_name_lucid ='input_1'
        
        else:
            name_pb,input_name_lucid = get_path_pbmodel_pretrainedModel(constrNet='InceptionV1')

        dico_most_response_feat = {}


            
        if stats_on_layer=='meanFirePos_minusMean':
            list_spatial_means_all = predictionFT_net(mean_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                                                  Net=constrNet,cropCenter=cropCenter)
            array_spatial_means_all = np.array(np.vstack(list_spatial_means_all))
            total_number_im = len(list_spatial_means_all)
            mean_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(fine_tuned_model,
                                                                                                     list_layers=style_layers,
                                                                                                     stats_on_layer=stats_on_layer,
                                                                                                     list_means=list_spatial_means_all)
        if selection_feature=='ClassMinusGlobalMean':
            list_spatial_means_all = predictionFT_net(mean_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                                                  Net=constrNet,cropCenter=cropCenter)
            array_spatial_means_all = np.array(np.vstack(list_spatial_means_all))
            total_number_im = len(list_spatial_means_all)
            

        # Loop on the classe
        for classe in classes:
            print('=== For ',classe,'===')
            
            if selection_feature=='TopOnlyForClass':
                df_train_not_c = df_train[df_train[classe]!=1.]
                list_spatial_means_notc = predictionFT_net(mean_model,df_train_not_c,x_col=item_name,y_col=classes,path_im=path_to_img,
                                                  Net=constrNet,cropCenter=cropCenter)
                array_spatial_means_notc = np.array(np.vstack(list_spatial_means_notc))
            
            df_train_c = df_train[df_train[classe]==1.]
            list_spatial_means = predictionFT_net(mean_model,df_train_c,x_col=item_name,y_col=classes,path_im=path_to_img,
                                                  Net=constrNet,cropCenter=cropCenter)
            total_number_img_in_c = len(list_spatial_means)
    
            array_spatial_means = np.array(np.vstack(list_spatial_means))
            if stats_on_layer=='mean':
                mean_spatial_means = np.mean(array_spatial_means,axis=0)
                if selection_feature=='TopOnlyForClass':
                    mean_spatial_means_notc = np.mean(array_spatial_means_notc,axis=0)
                elif selection_feature=='ClassMinusGlobalMean':
                    mean_spatial_means_all = np.mean(array_spatial_means_all,axis=0)

            elif stats_on_layer=='max': # In this case we take the max of the mean
                mean_spatial_means = np.max(array_spatial_means,axis=0)
                if selection_feature=='TopOnlyForClass':
                    mean_spatial_means_notc = np.max(array_spatial_means_notc,axis=0)
                elif selection_feature=='ClassMinusGlobalMean':
                    mean_spatial_means_all = np.max(array_spatial_means_all,axis=0)
                    
            elif stats_on_layer=='meanFirePos' or stats_on_layer=='meanFirePos_minusMean': # In this case we take the mean of th meanFirePos
                mean_spatial_means = np.mean(array_spatial_means,axis=0)
                if selection_feature=='TopOnlyForClass':
                    mean_spatial_means_notc = np.mean(array_spatial_means_notc,axis=0)
                elif selection_feature=='ClassMinusGlobalMean':
                    mean_spatial_means_all = np.mean(array_spatial_means_all,axis=0)

            else:
                raise(ValueError(stats_on_layer))
            #print('mean_spatial_means',mean_spatial_means.shape)
            features_size  = len(mean_spatial_means)
            num_features = features_size
            
            if clustering is None or clustering=='random' or clustering=='equal':
                classe_str = ''
            else:
                if not(classe is None):
                    classe_str = classe
                else:
                    classe_str = ''
                    
            if selection_feature=='TopOnlyForClass':
               mean_spatial_means = mean_spatial_means - mean_spatial_means_notc
            elif selection_feature=='ClassMinusGlobalMean':
               lambda_c = (total_number_im) / (total_number_im - total_number_img_in_c)
               mean_spatial_means = lambda_c * mean_spatial_means -lambda_c* mean_spatial_means_all

            argsort_mean_spatial = np.argsort(mean_spatial_means)[::-1]
        
            dico_most_response_feat[classe] = argsort_mean_spatial
        
            for comp_number in range(num_components_draw):
                weights = np.zeros(shape=(features_size,))
                index_feature = argsort_mean_spatial[comp_number]
                weights[index_feature] = 1.
                prexif_name = '_Feat'+str(index_feature)
                
                obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer,Net=constrNet)

                if DECORRELATE:
                    ext='_Deco'
                else:
                    ext=''
                
                if ROBUSTNESS:
                  ext+= ''
                else:
                  ext+= '_noRob'
                name_base = layer  + kind_layer+'_'+prexif_name+ext+'_toRGB.png'
                name_output = os.path.join(path_output_lucid_im,name_base)
                print(name_output)
                if not(os.path.isfile(name_output)):
                    index_features_withinLayer_all = np.arange(0,features_size)
                    lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights,\
                         index_features_withinLayer=index_features_withinLayer_all,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=224,
                         cossim=cossim,dot_vector=dot_vector,
                         num_features=num_features,ROBUSTNESS=ROBUSTNESS,
                         DECORRELATE=DECORRELATE)
                    
    
        with open(path_dico, 'wb') as handle:
            pickle.dump(dico_most_response_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(path_dico, 'rb') as handle:
            dico_most_response_feat = pickle.load(handle)
            
            
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    
    if selection_feature is None:
        path_output_composition = os.path.join(output_path,'AllFeatures','TopK')
    elif selection_feature=='TopOnlyForClass':
        path_output_composition = os.path.join(output_path,'AllFeatures','TopOnlyForClass')
    elif selection_feature=='ClassMinusGlobalMean':
        path_output_composition = os.path.join(output_path,'AllFeatures','ClassMinusGlobalMean')
    pathlib.Path(path_output_composition).mkdir(parents=True, exist_ok=True)
            
    # Maintenant on va faire des prints avec les images et les différentes classes
    for classe in classes:
        argsort_mean_spatial = dico_most_response_feat[classe]
       
        plt.rcParams["figure.figsize"] = [num_components_draw,2]
        fig, axes = plt.subplots(1, num_components_draw) # squeeze=False for the case of one figure only
        fig.suptitle(classe)

        for comp_number in range(num_components_draw):
            ax = axes[comp_number]
            index_feature = argsort_mean_spatial[comp_number]
            prexif_name = '_Feat'+str(index_feature)
            
            obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer,Net=constrNet)

            if DECORRELATE:
                ext='_Deco'
            else:
                ext=''
            
            if ROBUSTNESS:
              ext+= ''
            else:
              ext+= '_noRob'
            name_base = layer  + kind_layer+'_'+prexif_name+ext+'_toRGB.png'
            
            if not(os.path.isfile(name_base)):
                suffix_str = suffix                                                                     
                if not(model_name=='pretrained'):
                    name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
                    if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
                        name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                                   constrNet=constrNet,path=path_lucid_model,suffix=suffix)
                    if constrNet=='VGG':
                        input_name_lucid ='block1_conv1_input'
                    elif constrNet=='InceptionV1':
                        input_name_lucid ='input_1'
                    elif constrNet=='InceptionV1_slim':
                        input_name_lucid ='input_1'
                
                else:
                    name_pb,input_name_lucid = get_path_pbmodel_pretrainedModel(constrNet='InceptionV1')
                lucid_utils.print_images(model_path=os.path.join(path_lucid_model,name_pb),
                                         list_layer_index_to_print=[layer,index_feature],
                                         path_output=path_output_lucid_im,prexif_name=prexif_name,\
                                         input_name=input_name_lucid,Net=constrNet,sizeIm=224,
                                         ROBUSTNESS=ROBUSTNESS,
                                         DECORRELATE=DECORRELATE)
            name_output = os.path.join(path_output_lucid_im,name_base)
            print(name_output)
            img = plt.imread(name_output)
            print(img)
            ax.imshow(img, interpolation='none')
            ax.set(title=str(index_feature))
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            
        if selection_feature=='TopOnlyForClass':
            name_fig = classe +'ClassMinusGlobalMean'+str(num_components_draw)+'_Features.png'
        elif selection_feature=='TopOnlyForClass':
            name_fig = classe +'_ClassMinusGlobalMean'+str(num_components_draw)+'_Features.png'
        else:
            name_fig = classe +'_Top'+str(num_components_draw)+'_Features.png'
            
        if stats_on_layer=='max':
            name_fig = str(1)+ 'max_' + name_fig 
        elif not(stats_on_layer=='mean'):
            name_fig = stats_on_layer+'_' + name_fig
        name_fig = layer + '_' + name_fig
        path_fig = os.path.join(path_output_composition,name_fig)
        plt.savefig(path_fig,dpi=600,bbox_inches='tight')
        plt.close()
            
            
            

    
def PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',\
                               classe = None,\
                               layer='mixed4d_pre_relu',\
                               plot_FeatVizu=True,\
                               source_dataset=None,
                               num_components_draw = 10,
                               strictMinimum=False,cossim=False,
                               clustering='PCA',
                               constrNet = 'InceptionV1',
                               dot_vector=True):
    """
    @param : clustering PCA or PCAsubset or none or random and we will do all the feature one by one
        PCAsubset : means that we will extract only some component and do some withening of the covariance matrices
    
    @param : dot_vector : we will use the dot product between the vector direction and the layers 
    """
    matplotlib.use('Agg')
    cropCenter = True
    set_ = 'train'
    KindOfMeanReduciton='global'
    
    mean_cov_matrix,std_cov_matrix,mean_spatial_means = compute_global_mean_cov_matrices_onBigSet(constrNet=constrNet,
                                              cropCenter=cropCenter,\
                                              set_=set_,\
                                              model_name = model_name,\
                                           classe = classe,\
                                           layer=layer,\
                                           KindOfMeanReduciton=KindOfMeanReduciton,\
                                           source_dataset=source_dataset)
    
    features_size,_ = mean_cov_matrix.shape
    num_features = features_size
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    
    if clustering is None: 
        path_output_lucid_im = os.path.join(output_path,'AllFeatures')
    elif clustering=='random':
        path_output_lucid_im = os.path.join(output_path,'RandDir')
    elif clustering=='equal' or clustering=='equalHotElement':
        path_output_lucid_im = os.path.join(output_path,'Equal')
    else:
        if KindOfMeanReduciton=='instance':
            path_output_lucid_im = os.path.join(output_path,'PCAlucid')
        if KindOfMeanReduciton=='global':
            path_output_lucid_im = os.path.join(output_path,'PCACovGMeanlucid')
        elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
            path_output_lucid_im = os.path.join(output_path,'PCAGramlucid')
        else:
            raise(NotImplementedError)
        
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True)
    
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    if clustering is None or clustering=='random' or clustering=='equal':
        classe_str = ''
    else:
        if not(classe is None):
            classe_str = classe
        else:
            classe_str = ''
    suffix = ''
    suffix_str = suffix
    # Compute the eigen values 
    #del std_cov_matrix,where_std_sup_abs_mean,where_std_sup_mean
    
    print('plot_FeatVizu',plot_FeatVizu,'for ',classe)
    if plot_FeatVizu:
    
        if not(model_name=='pretrained'):
            name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
            if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
                name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                           constrNet=constrNet,path=path_lucid_model,suffix=suffix)
            if constrNet=='VGG':
                input_name_lucid ='block1_conv1_input'
            elif constrNet=='InceptionV1':
                input_name_lucid ='input_1'
            elif constrNet=='InceptionV1_slim':
                input_name_lucid ='input_1'
        
        else:
            name_pb,input_name_lucid = get_path_pbmodel_pretrainedModel(constrNet='InceptionV1')
           
        
        if clustering is None:
            for comp_number in range(features_size):
                weights = np.zeros(shape=(features_size,))
                weights[comp_number] = 1.
                prexif_name = '_Feat'+str(comp_number)
                #print(prexif_name)
                do_lucidVizu_forPCA_all_case(path_lucid_model,
                                            name_pb,
                                             prexif_name,features_size,
                                             layer,weights,path_output_lucid_im,
                                             input_name_lucid,constrNet,
                                             strictMinimum=True,cossim=cossim)
        elif clustering=='random':
            for comp_number in range(num_components_draw):
                weights = np.random.random((features_size,))
                prexif_name = '_Random'+str(comp_number)
                #print(prexif_name)
                do_lucidVizu_forPCA_all_case(path_lucid_model,
                                            name_pb,
                                             prexif_name,features_size,
                                             layer,weights,path_output_lucid_im,
                                             input_name_lucid,constrNet,
                                             strictMinimum=True,cossim=cossim)
        elif clustering=='equal':
            weights = np.ones((features_size,))
            prexif_name = '_Equal'
            do_lucidVizu_forPCA_all_case(path_lucid_model,
                                        name_pb,
                                         prexif_name,features_size,
                                         layer,weights,path_output_lucid_im,
                                         input_name_lucid,constrNet,
                                         strictMinimum=True,cossim=cossim,
                                         dot_vector=dot_vector,num_features=num_features)
        elif clustering=='equalHotElement':
            weights = np.zeros((features_size,))
            diag = np.diag(mean_cov_matrix)
            weights[np.where(diag>0.)] = 1.
            prexif_name = '_EqualForFireElets' + classe_str
            do_lucidVizu_forPCA_all_case(path_lucid_model,
                                        name_pb,
                                         prexif_name,features_size,
                                         layer,weights,path_output_lucid_im,
                                         input_name_lucid,constrNet,
                                         strictMinimum=True,cossim=cossim,
                                         dot_vector=dot_vector,num_features=num_features)
            
        elif clustering=='PCA':
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
            #print('Max imag part first vector :',np.max(np.imag(eigen_vectors[:,0])))
            eigen_vectors = np.real(eigen_vectors) 
    #        if KindOfMeanReduciton=='global':
    #            eigen_vectors = eigen_vectors + np.reshape(mean_spatial_means,(1,features_size))
    #        prexif_name_spm = '_SPM'
    #        if not(classe is None):
    #           prexif_name_spm += '_'+classe 
    #        index_features_withinLayer_all = np.arange(0,features_size)
    #        lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
    #                         layer_to_print=layer,weights=mean_spatial_means,\
    #                         index_features_withinLayer=index_features_withinLayer_all,\
    #                         path_output=path_output_lucid_im,prexif_name=prexif_name_spm,\
    #                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
            
            for comp_number in range(num_components_draw):
                weights = eigen_vectors[:,comp_number]
                prexif_name = '_PCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                print(prexif_name)
                do_lucidVizu_forPCA_all_case(path_lucid_model,
                                            name_pb,
                                             prexif_name,features_size,
                                             layer,weights,path_output_lucid_im,
                                             input_name_lucid,constrNet,
                                             strictMinimum=strictMinimum,cossim=cossim,
                                             dot_vector=dot_vector,num_features=num_features)
        elif clustering=='corr': 
            # On va convertir la matrice de covariance en matrice de correlation
            D = np.diag(np.sqrt(np.diag(mean_cov_matrix)))
            DInv = np.linalg.inv(D)
            corr_matrix = np.matmul(DInv,np.matmul(mean_cov_matrix,DInv)) # correlation matrix
            eigen_values, eigen_vectors = LA.eig(corr_matrix)

            pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
            plt.figure()
            plt.scatter(np.arange(0,len(eigen_values)),eigen_values,s=4)
            plt.ylabel('Eigen Value')
            plt.savefig(os.path.join(path_output_lucid_im,'EigenValues_CorrMatrix'+layer+'_'+classe_str+'.png'),\
                        dpi=300)
            plt.close()
            
            eigen_vectors = np.real(eigen_vectors) 
         
            for comp_number in range(num_components_draw):
                weights = eigen_vectors[:,comp_number]
                prexif_name = '_PCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                print(prexif_name)
                do_lucidVizu_forPCA_all_case(path_lucid_model,
                                            name_pb,
                                             prexif_name,features_size,
                                             layer,weights,path_output_lucid_im,
                                             input_name_lucid,constrNet,
                                             strictMinimum=strictMinimum,cossim=cossim,
                                             dot_vector=dot_vector,num_features=num_features)
                
        elif clustering=='PCAsubset':
            # In this case we only keep the top value of the gram matrice 
            # Then whithen it and then do the diagonalisation
            
            
            list_mean_cov_matrix_trui = list(np.abs(mean_cov_matrix[np.triu_indices(features_size)]))
            #print(list_mean_cov_matrix_trui)
            decile = np.percentile(list_mean_cov_matrix_trui, 99) # last decile
            #print(decile)
            where_mean_cov_matrix_big = np.where(np.abs(mean_cov_matrix)>=decile)
            mask = np.zeros_like(mean_cov_matrix)
            mask[where_mean_cov_matrix_big] = 1
            masked_cov = mean_cov_matrix*mask
            
            tmp_cov = masked_cov.copy()
            feature_no_supprimer = []
            
            for ii in range(features_size):
                i = features_size-1-ii
                
                if np.sum(tmp_cov[:,i])==0 or tmp_cov[i,i]==0:
                    masked_cov = np.delete(masked_cov,i,0)
                    masked_cov = np.delete(masked_cov,i,1)
                    # On supprime ligne et colum
                else:
                    # On normalise ligne et colonne par la variance
#                    var_ii = masked_cov[i,i]
#                    masked_cov[:,i] /= var_ii
#                    masked_cov[i,:] /= var_ii
#                    masked_cov[i,i] = 1.
                    feature_no_supprimer += [i]
            feature_no_supprimer = np.sort(feature_no_supprimer)
                    
            # On va convertir la matrice de covariance en matrice de correlation
            D = np.diag(np.sqrt(np.diag(masked_cov)))
            DInv = np.linalg.inv(D)
            masked_cov = np.matmul(DInv,np.matmul(masked_cov,DInv)) # correlation matrix 
            #print(masked_cov)
              
#            n_clusters = 9
#            scoclustering = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
#            # To cluster row and columns at the same time : https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-coclustering-py
#            scoclustering.fit(cov)
#            row_labels_= scoclustering.row_labels_
#            column_labels_ = scoclustering.column_labels_
#            
#            for i in range(n_clusters):
#                sub_cov = 
                    
            #print('masked_cov.shape',masked_cov.shape)
            limited_features_size = masked_cov.shape[0]
            eigen_values, eigen_vectors = LA.eig(masked_cov)
#            print('eigen_values',np.sort(eigen_values))
            plt.figure()
            plt.plot(eigen_values)
            plt.show()
#            input('wait')
            eigen_vectors = np.real(eigen_vectors)
            #eigen_vectors = np.matmul(D,np.matmul(eigen_vectors,D))
            
            # Here each column is an eigen vectors
            
#            ica = FastICA(random_state=0)
#            S_ica_ = ica.fit(np.transpose(eigen_vectors,[1,0]))
#            components_vectors = S_ica_.components_
#            # Here each line is an components
#            
#            kurtosis = scipy.stats.kurtosis(components_vectors, axis=0)
#            decreasing_order = np.argsort(kurtosis)[::-1]
#            eigen_vectors_order = components_vectors[decreasing_order,:]
#            eigen_vectors =np.transpose(eigen_vectors_order,[1,0])
            
            # Reprojection dans une matrice de taille features_size * features_size
            eigen_vectors_all = np.zeros((features_size,features_size))
            # Il faudrait peut être prendre minus 1 ici ou quelque chose comme cela
            for j in range(limited_features_size):
                local_eigen_vector = eigen_vectors[:,j]
                eigen_vectors_all[feature_no_supprimer,j] = local_eigen_vector
                # On replace le vecteur propre de petite taille dans le grand
            
            for comp_number in range(num_components_draw):
                weights = eigen_vectors_all[:,comp_number]
                prexif_name = '_subsetPCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 

                do_lucidVizu_forPCA_all_case(path_lucid_model,
                                            name_pb,
                                             prexif_name,features_size,
                                             layer,weights,path_output_lucid_im,
                                             input_name_lucid,constrNet,
                                             strictMinimum=strictMinimum,
                                             cossim=cossim,
                                             dot_vector=dot_vector,num_features=num_features)
                
        else:
            raise(ValueError(clustering))
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
    
def produce_latex_textV2(model_name = 'RASTA_small01_modif',
                       layer_tab=['mixed4d_pre_relu'],classe_tab = [None],
                       folder_im_latex='im',
                       num_components_draw = 10,
                       KindOfMeanReduciton='global'):
    
    constrNet = 'InceptionV1'
    
    if constrNet=='InceptionV1':
        dico = get_dico_layers_type()
    else:
        raise(NotImplementedError)
    
    
    
    path_base  = os.path.join('C:\\','Users','gonthier')
    ownCloudname = 'ownCloud'
    
    if not(os.path.exists(path_base)):
        path_base  = os.path.join(os.sep,'media','gonthier','HDD')
        ownCloudname ='owncloud'
        
    if KindOfMeanReduciton=='instance':
        path_to_im_local = os.path.join(folder_im_latex,model_name,'PCAlucid')
    if KindOfMeanReduciton=='global':
        path_to_im_local = os.path.join(folder_im_latex,model_name,'PCACovGMeanlucid')
    elif KindOfMeanReduciton=='' or KindOfMeanReduciton is None:
        path_to_im_local = os.path.join(folder_im_latex,model_name,'PCAGramlucid')
    else:
        raise(NotImplementedError)
    
    folder_im_this_model = os.path.join(path_base,ownCloudname,'Mes_Presentations_Latex','2020-04_Feature_Visualisation',path_to_im_local)
    
    list_all_image = glob.glob(folder_im_this_model+'\*.png')
    #print(list_all_image)
    
    file_path = os.path.join(folder_im_this_model,'printImages.tex')
    file = open(file_path,"w") 
    
    for layer in layer_tab:
        typelayer = dico[layer]
        for classe in classe_tab:
            if classe is None:
                classe_str =''
                latex_str = ''
            else:
                classe_str = '_'+classe
                latex_str = r" - %s" % classe.replace('_','\_') 
                latex_str += r" classe only"
            
            for i in range(num_components_draw):
                base_name_im = layer + typelayer+'__PCA'+str(i)+classe_str
                base_name_im_max = layer + typelayer+'__PCA'+str(i)+'_Max'
                base_name_im_min = layer + typelayer+'__PCA'+str(i)+'_Min'
                name_main_image = base_name_im + '_Deco'+'_toRGB.png'
                name_negdir = base_name_im + '_NegDir_Deco'+'_toRGB.png'
                name_maxcontrib_image = base_name_im + '_PosContrib_Deco'+'_toRGB.png'
                name_mincontrib_image = base_name_im + '_NegContrib_Deco'+'_toRGB.png'
                print(list_all_image)
                name_max_image = None
                name_min_image = None
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
                           
                if name_min_image is None:
                    print(base_name_im_min,'not found !')
                    return(0)
                if name_max_image is None:
                    print(base_name_im_max,'not found !')
                    return(0)
                # Text for Positive contrib slide
                newline = " \n"
                str_beg = r"\frame{  " +newline
                str_beg += r" \frametitle{%s" % model_name.replace('_','\_')
                str_beg += r" - %s" % layer.replace('_','\_')
                str_beg += r" - component %s" % str(i) 
                str_beg +=  latex_str 
                str_beg +=  "} \n " 
                str_pos = str_beg + r"\begin{figure}[!tbp] " +newline
                str_pos += r"\begin{minipage}[b]{0.29\textwidth}   "+newline
                path_to_im = os.path.join(path_to_im_local,name_main_image).replace("\\", "/")
                str_pos += r"\includegraphics[width=\textwidth]{%s} \\ " % path_to_im
                str_pos += newline
                str_pos += r"{\scriptsize All contribution}"  +newline
                str_pos += r"\end{minipage} \hfill"  +newline
                str_pos +=  r"\begin{minipage}[b]{0.29\textwidth} " +newline
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
                
                str_neg = str_beg
                str_neg += r"\begin{figure}[!tbp] " +newline
                str_neg += r"\begin{minipage}[b]{0.29\textwidth}   "+newline
                path_to_im = os.path.join(path_to_im_local,name_negdir).replace("\\", "/")
                str_neg += r"\includegraphics[width=\textwidth]{%s} \\ " % path_to_im
                str_neg += newline
                str_neg += r"{\scriptsize All contribution Neg Direction}"  +newline
                str_neg += r"\end{minipage} \hfill"  +newline
                str_neg +=  r"\begin{minipage}[b]{0.29\textwidth} " +newline
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
        
def _old_Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                                    constrNet = 'InceptionV1',
                                    classe='Northern_Renaissance',layer='mixed4d_pre_relu'):
    """
    L idee de cette fonction est la suivante en deux étapes :
        Step 1 : creer le block au milieu du reseau qui maximise de maniere robuste
        la réponse a une des classes 
        Step 2 : faire la PCA de ce block de features
        Step 3 : generer l'image qui maximise la premiere composante de cette 
        decomposition
        
    Cela ne fonctionne pas du tout !!!
        
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


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,weights,path_output_lucid_im,
                                 input_name_lucid,constrNet,
                                 strictMinimum=False,cossim=False,
                                 sizeIm=256,dot_vector=False,
                                 num_features=None,onlyPos=False):
    print(prexif_name)
    index_features_withinLayer_all = np.arange(0,features_size)
    lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                     layer_to_print=layer,weights=weights,\
                     index_features_withinLayer=index_features_withinLayer_all,\
                     path_output=path_output_lucid_im,prexif_name=prexif_name,\
                     input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                     cossim=cossim,dot_vector=dot_vector,num_features=num_features)
   
    if onlyPos: # Print only the positive image
        return(0)
    
    minus_weights = -weights
    prexif_name_negDir =prexif_name+ '_NegDir' # Negative direction
    print(prexif_name_negDir)
    index_features_withinLayer_all = np.arange(0,features_size)
    lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                     layer_to_print=layer,weights=minus_weights,\
                     index_features_withinLayer=index_features_withinLayer_all,\
                     path_output=path_output_lucid_im,prexif_name=prexif_name_negDir,\
                     input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                     cossim=cossim,dot_vector=dot_vector,num_features=num_features)
    
    if not(strictMinimum):
        print('strictMinimum',strictMinimum)
        prexif_name_pos = prexif_name + '_PosContrib'
        where_pos = np.where(weights>0.)[0]
        if len(where_pos)>0:
            weights_pos = list(weights[where_pos])
            print(prexif_name_pos)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                         layer_to_print=layer,weights=weights_pos,\
                         index_features_withinLayer=where_pos,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_pos,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                         cossim=cossim,dot_vector=dot_vector,num_features=num_features)
            
            where_max = np.argmax(weights)
            prexif_name_max = prexif_name+  '_Max'+str(where_max)
            print(prexif_name_max)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                             layer_to_print=layer,weights=[1.],\
                             index_features_withinLayer=[where_max],\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                             cossim=cossim,dot_vector=dot_vector,num_features=num_features)
    #            
        prexif_name_neg = prexif_name + '_NegContrib'
        where_neg = np.where(weights<0.)[0]
        weights_neg = list(-weights[where_neg])
        if len(weights_neg)>0:
            print(prexif_name_neg)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                             layer_to_print=layer,weights=weights_neg,\
                             index_features_withinLayer=where_neg,\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_neg,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                             cossim=cossim,dot_vector=dot_vector,num_features=num_features)
    
            where_min = np.argmin(weights)
            prexif_name_max = prexif_name+  '_Min'+str(where_min)
            print(prexif_name_max)
            lucid_utils.print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb_full_model),
                             layer_to_print=layer,weights=[1.],\
                             index_features_withinLayer=[where_min],\
                             path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                             input_name=input_name_lucid,Net=constrNet,sizeIm=sizeIm,
                             cossim=cossim,dot_vector=dot_vector,num_features=num_features)

def do_whitening(full_activations):
    correl = np.matmul(full_activations.T, full_activations) / len(full_activations)
    correl = correl.astype("float32")
    S = np.linalg.inv(correl)
    S = S.astype("float32")
    return S  
      
def Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                                    constrNet = 'InceptionV1',
                                    classe='Northern_Renaissance',layer='mixed4d',
                                    num_components_draw = 3,clustering = 'WHC',
                                    number_of_blocks = 1,strictMinimum=True,
                                    whiten=False,cossim=False,
                                    n_clusters = 5):
    """
    L idee de cette fonction est la suivante en deux étapes :
        Step 1 : creer le block au milieu du reseau qui maximise de maniere robuste
        la réponse a une des classes 
        Step 2 : faire la PCA de ce block de features
        Step 3 : generer l'image qui maximise la premiere composante de cette 
        decomposition
        
    Il faudrait peut etre faire un clustering avec connectivity = True
    
    Pour trouver les régions de l'images les differents objects ou morceaux d'objects

    https://scikit-learn.org/stable/modules/clustering.html
    
    Ou bien retirer la base, faire quelque chose quoi
    
    @param :     clustering = None # Diagonalisation on the Gram matrices
    clustering = 'MeanShift' # MeanShift clustering and then plot the center and the principal component of the cluster
    clustering = 'KMeans' # KMeans clustering and then plot the center and the principal component of the cluster
    clustering = 'ICA' # ICA decomposition indeed of Eigenvalues ones (PCA like)
    'IPCA' : PCA and then ICA
    WHC : AgglomerativeClustering with connectivity
    NMF : Non maximum factorization : a tester
    
    @param : whiten if True whiten the activation before using it
    @param : cossim if True use a cosine similarity based loss
        
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
    path_output_lucid_im = os.path.join(output_path,'PCAlucid_classCond')
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 

    #matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    suffix = ''
    
    #K.set_learning_phase(0)
    #with K.get_session().as_default(): 
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    #print('#### ',model_name)
    output_path_with_model = os.path.join(output_path,model_name+suffix)
    pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
    K.set_learning_phase(0)
    net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
    
    # We will compute the max and min value of the given layer on the whole train set
    print('We will compute the bounds')
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    if not(classe is None):
       df_train = df_train[df_train[classe]==1.0]
    #print(df_train.head(5))
    maxmin_model = Activation_for_model.get_Model_that_output_StatsOnActivation_forGivenLayers(model=tf.keras.models.clone_model(net_finetuned),
                                                                                         list_layers=[layer],stats_on_layer='max&min')
    maxmin_model.trainable = False
    cropCenter = True
    #print(maxmin_model.summary())
    maxmin_act = predictionFT_net(maxmin_model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter)
    max_act, min_act =  maxmin_act

    max_per_channel = np.max(max_act,axis=0)
    min_per_channel = np.min(min_act,axis=0)

#    list_outputs_name,max_act = Activation_for_model.compute_OneValue_Per_Feature(dataset,model_name,constrNet,stats_on_layer='max',
#                                 suffix='',cropCenter = True,FTmodel=True)
#    list_outputs_name,min_act = Activation_for_model.compute_OneValue_Per_Feature(dataset,model_name,constrNet,stats_on_layer='min',
#                                 suffix='',cropCenter = True,FTmodel=True)
#    for layer_name_inlist,activations_l in zip(list_outputs_name,max_act):
#        if layer_name_inlist==layer:
#            
#    for layer_name_inlist,activations_l in zip(list_outputs_name,min_act):
#        if layer_name_inlist==layer:
#            min_per_channel = np.min(activations_l,axis=-1)
#            
#    print('min and max med of max_per_channel',np.min(max_per_channel),np.max(max_per_channel),np.median(max_per_channel))
#    print('max_per_channel',max_per_channel.shape)
#    print('min and max med of min_per_channel',np.min(min_per_channel),np.max(min_per_channel),np.median(min_per_channel))
#    print(' min_per_channel',min_per_channel.shape)
#    
    
    
    layer_concerned = net_finetuned.get_layer(layer)
    
    # Ici il faudrait calculer le max et le min possible pour cette feature maps
    
    # Tester pour mixed4d mais pas pour mixed4d_pre_relu
    part_model = get_truncated_keras_model(model=net_finetuned,
                                           model_name=model_name,
                                           new_input_layer_name=layer,
                                           constrNet=constrNet,
                                           batch_size=1,
                                           reshape=False)  
    part_model.trainable = False

    loss_c = part_model.output[0][index_class]
    grad_symbolic = K.gradients(loss_c, part_model.inputs[0])[0]
#    grad_symbolic_minus_loss = K.gradients(-loss_c, part_model.inputs[0])[0]
    #grad_symbolic = K.gradients(loss_c, part_model.get_layer('new_input_1').input)[0]
#    symbolic_loss_c_plus_gradient = K.function(part_model.inputs[0], [loss_c,grad_symbolic])
#    symbolic_minus_loss_c_and_gradient = K.function(part_model.inputs[0], [-loss_c,grad_symbolic_minus_loss])
#    #symbolic_loss_c_plus_gradient = K.function([part_model.get_layer('new_input_1').input], [loss_c,grad_symbolic])
#    symbolic_loss_c = K.function(part_model.inputs[0], loss_c)
#    symbolic_minus_loss = K.function(part_model.inputs[0], -loss_c)
#    iterate_only_grad = K.function([part_model.input], [grad_symbolic])
#    iterate = K.function([part_model.input], [grad_symbolic,loss_c])
    
    layer_output = layer_concerned.output
    dim_shape = layer_output.shape.dims
    new_input_shape = []
    for dim in dim_shape:
        new_input_shape += [dim.value]
    new_input_shape.pop(0) #remove the batch size dim
    #number_of_blocks = 50
    
    variable_shape = [number_of_blocks] + new_input_shape
    part_model.build(variable_shape)
    max_x_value= np.max(max_per_channel)
    min_x_value= np.min(min_per_channel)
    variable = max_x_value*np.random.random(variable_shape) +  min_x_value
    # np.random.random(variable_shape) entre 0  et 1 normalement 

    grads = normalize(grad_symbolic)
    # this function returns the loss and grads given the input picture
    iterate = K.function(part_model.inputs[0], [loss_c, grads])
    print('Start computing the mid-block !')
    epochs = 10000
    step = 0.5
    for _ in range(epochs):
        loss_value, grads_value = iterate(variable)
        variable += grads_value * step
        variable = np.clip(variable,a_min=min_x_value,a_max=max_x_value)
    # Avec cela on va a une loss de l ordre de 0.75
    
    # TODO : il faudrait faire un truc du genre Integrated Gradient avec une baseline a zero
    # Voir un truc du genre smooth gradient
    
    print('Final Loss : ',loss_value)
    print('Bounds of the block + std',np.max(variable),np.min(variable),np.std(variable))
    print('variable.shape',variable.shape)
 
#    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
#    # Compute the gradients for a list of variables.
#    with tf.GradientTape() as tape:
#      loss_c = part_model.output[0][index_class]
#    vars = part_model.inputs[0]
#    grads = tape.gradient(loss_c, vars)
#    
#    # Process the gradients, for example cap them, etc.
#    # capped_grads = [MyCapper(g) for g in grads]
#    processed_grads = [process_gradient(g) for g in grads]
#    
#    # Ask the optimizer to apply the processed gradients.
#    opt.apply_gradients(zip(processed_grads, var_list))



    suffix_str =  suffix_str = suffix
    name_pb_full_model = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
    if not(os.path.isfile(os.path.join(path_lucid_model,name_pb_full_model))):
        name_pb_full_model = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,
                                   path=path_lucid_model,suffix=suffix)
        
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
    else:
        raise(NotImplementedError)
        
    features_size = new_input_shape[-1]
    h_time_w_size = new_input_shape[0]*new_input_shape[1]
    h = new_input_shape[0]
    w = new_input_shape[1]
    
    print('clustering',clustering)
    str_param = ''
    if whiten:
        var_tmp = variable.reshape((-1,variable.shape[-1]))
        scaler = StandardScaler()
        var_whiten = scaler.fit_transform(var_tmp)
#        var_whiten = do_whitening(var_tmp)
        print(var_whiten.shape)
        variable = var_whiten.reshape(variable.shape)
        str_param += 'Whiten_'
    if cossim:
        str_param +='cossim_'
    
    
    if clustering is None:

        if not(number_of_blocks==1):
            feature_block_reshaped = variable.reshape((number_of_blocks,-1,variable.shape[-1]))
            gram_matrix = np.matmul(np.transpose(feature_block_reshaped,[0,2,1]),feature_block_reshaped)/h_time_w_size
            gram_matrix = np.mean(gram_matrix,axis=0)
        else:
            feature_block_reshaped = variable.reshape((-1,variable.shape[-1]))
            gram_matrix = np.matmul(np.transpose(feature_block_reshaped,[1,0]),feature_block_reshaped)/h_time_w_size
            
        # We will compute the Gram matrices of this features block
        name = 'GramMatrixCond_'+str(classe)
        path = path_output_lucid_im
        title = 'GramMatrixCond '+str(classe)
        print_rect_matrix(gram_matrix,name,path,title=title)   
    #    pca = PCA(n_components=None,copy=True,whiten=False)
    #    print('feature_block_reshaped.shape',feature_block_reshaped.shape)
    #    pca.fit(feature_block_reshaped)
    #    print('Eigen values 10 first value',pca.singular_values_[0:10])
    #    print('Explained ratio 10 first',pca.explained_variance_ratio_[0:10])
    #    eigen_vectorsPCA = pca.components_
        #print('pca_comp',eigen_vectors.shape)
        
        eigen_values, eigen_vectors = LA.eig(gram_matrix)
        print('Eigen values 10 first value',eigen_values[0:10])
        eigen_vectors = np.real(eigen_vectors) 
        
        for comp_number in range(num_components_draw):
            weights = eigen_vectors[:,comp_number]
            #weights = weights[0:1]
            #print('weights',weights)
            #time.sleep(.300)
    
            if not(number_of_blocks==1):
                prexif_name =str_param+ 'MeanGramOn'+str(number_of_blocks)+'_PCA'+str(comp_number)
            else:
                prexif_name =str_param+ '_PCA'+str(comp_number)
            if not(classe is None):
               prexif_name += '_'+classe 
            do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,weights,path_output_lucid_im,
                                 input_name_lucid,constrNet,strictMinimum,cossim)
            
    elif clustering=='KMeans':
        
        feature_block_reshaped = variable.reshape((-1,features_size))

        str_kmeans = 'C'+str(n_clusters)+'Means'
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(feature_block_reshaped)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of clusters : %d" % n_clusters_)
        
        for i,center in enumerate(cluster_centers):
            #print(center.shape)
            prexif_name = str_param
            prexif_name += str_kmeans +'center'+str(i)
            if not(number_of_blocks==1):
                prexif_name += 'NumSample'+str(number_of_blocks)
            if not(classe is None):
               prexif_name += '_'+classe 
            do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,center,path_output_lucid_im,
                                 input_name_lucid,constrNet,strictMinimum,cossim)
            
        for c in labels_unique:
            position_label_c = np.where(labels==c)[0]
            feature_block_cluster_c = feature_block_reshaped[position_label_c,:]
            gram_matrix = np.matmul(np.transpose(feature_block_cluster_c,[1,0]),feature_block_cluster_c)/h_time_w_size
            eigen_values, eigen_vectors = LA.eig(gram_matrix)
            print('Eigen values 1 first value',eigen_values[0:1])
            eigen_vectors = np.real(eigen_vectors) 
            for comp_number in range(num_components_draw):
                weights = eigen_vectors[:,comp_number]
                prexif_name =str_param+ str_kmeans+'cluster'+str(c)
                prexif_name += '_PCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                     prexif_name,features_size,
                                     layer,weights,path_output_lucid_im,
                                     input_name_lucid,constrNet,strictMinimum,cossim)
    elif clustering=='NMF':
        variable = np.clip(variable,a_min=0.,a_max=np.inf)
        feature_block_reshaped = variable.reshape((-1,features_size))

        str_nmf = 'NMF'+str(n_clusters)
        
        nmf = NMF(n_components=n_clusters, init='random', random_state=0)
        W = nmf.fit(feature_block_reshaped)
        H = nmf.components_
        print('Non Negative Matrix Factorization done.')
        for i,center in enumerate(H):
            #print(center.shape)
            prexif_name = str_param
            prexif_name += str_nmf +'_center'+str(i)
            if not(number_of_blocks==1):
                prexif_name += '_NumSample'+str(number_of_blocks)
            if not(classe is None):
               prexif_name += '_'+classe 
            do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,center,path_output_lucid_im,
                                 input_name_lucid,constrNet,strictMinimum,cossim,
                                 onlyPos=strictMinimum)
            
    elif clustering=='MeanShift':
        # Meme avec le mean shift on jette l'information spatiale !
        feature_block_reshaped = variable.reshape((-1,features_size))
        ms = MeanShift()
        ms.fit(feature_block_reshaped)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of estimated clusters : %d" % n_clusters_)
        
        for i,center in enumerate(cluster_centers):
            print(center.shape)
            prexif_name = str_param
            prexif_name += 'MScenter'+str(i)
            if not(number_of_blocks==1):
                prexif_name += 'NumSample'+str(number_of_blocks)
            if not(classe is None):
               prexif_name += '_'+classe 
            do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,center,path_output_lucid_im,
                                 input_name_lucid,constrNet,strictMinimum,cossim)
            
        for c in labels_unique:
            position_label_c = np.where(labels==c)[0]
            feature_block_cluster_c = feature_block_reshaped[position_label_c,:]
            gram_matrix = np.matmul(np.transpose(feature_block_cluster_c,[1,0]),feature_block_cluster_c)/h_time_w_size
            eigen_values, eigen_vectors = LA.eig(gram_matrix)
            print('Eigen values 1 first value',eigen_values[0:1])
            eigen_vectors = np.real(eigen_vectors) 
            for comp_number in range(num_components_draw):
                weights = eigen_vectors[:,comp_number]
                prexif_name =str_param+ 'MScluster'+str(c)
                prexif_name += '_PCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                     prexif_name,features_size,
                                     layer,weights,path_output_lucid_im,
                                     input_name_lucid,constrNet,strictMinimum,cossim)
                
    elif clustering=='ICA':
        
        feature_block_reshaped = variable.reshape((-1,features_size))
        ica = FastICA(random_state=0)
        S_ica_ = ica.fit(feature_block_reshaped)
        components_vectors = S_ica_.components_
        for comp_number in range(num_components_draw):
            weights = components_vectors[comp_number,:]
            #weights = weights[0:1]
            #print('weights',weights)
            #time.sleep(.300)
    
            if not(number_of_blocks==1):
                prexif_name = str_param+'MeanGramOn'+str(number_of_blocks)+'_ICA'+str(comp_number)
            else:
                prexif_name = str_param+'_ICA'+str(comp_number)
            if not(classe is None):
               prexif_name += '_'+classe 
            do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                 prexif_name,features_size,
                                 layer,weights,path_output_lucid_im,
                                 input_name_lucid,constrNet,strictMinimum,cossim)
    elif clustering=='IPCA':
        # https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-24
        feature_block_reshaped = variable.reshape((-1,features_size))
        pca = PCA(n_components=None,copy=True,whiten=False)
        pca.fit(feature_block_reshaped)
        eigen_vectorsPCA = pca.components_
        ica = FastICA(random_state=0)
        S_ica_ = ica.fit(eigen_vectorsPCA)
        components_vectors = S_ica_.components_
        
        kurtosis = scipy.stats.kurtosis(components_vectors, axis=0)
        # Axis 0 because each line is a independant component here
        decreasing_order = np.argsort(kurtosis)[::-1]
        
        for comp_number,index_com_vectors in enumerate(decreasing_order):
            if comp_number < num_components_draw:
                weights = components_vectors[index_com_vectors,:]
                #weights = weights[0:1]
                #print('weights',weights)
                #time.sleep(.300)
        
                if not(number_of_blocks==1):
                    prexif_name =str_param+ 'MeanGramOn'+str(number_of_blocks)+'_IPCA'+str(comp_number)
                else:
                    prexif_name = str_param+'_IPCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                     prexif_name,features_size,
                                     layer,weights,path_output_lucid_im,
                                     input_name_lucid,constrNet,strictMinimum,cossim)
    elif clustering=='WHC':
        
        # Define the structure A of the data. 
        # Pixels connected to their neighbors.
        assert(number_of_blocks==1)
        
        connectivity = grid_to_graph(n_x=h, n_y=w, n_z=1)
        feature_block_reshaped = variable.reshape((-1,features_size))
        ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                               connectivity=connectivity)
        ward.fit(feature_block_reshaped)

        labels = ward.labels_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of clusters : %d" % n_clusters_)

        for c in labels_unique:
            position_label_c = np.where(labels==c)[0]
            feature_block_cluster_c = feature_block_reshaped[position_label_c,:]
            gram_matrix = np.matmul(np.transpose(feature_block_cluster_c,[1,0]),feature_block_cluster_c)/h_time_w_size
            eigen_values, eigen_vectors = LA.eig(gram_matrix)
            print('Eigen values 1 first value',eigen_values[0:1])
            eigen_vectors = np.real(eigen_vectors) 
            for comp_number in range(num_components_draw):
                weights = eigen_vectors[:,comp_number]
                prexif_name = str_param+ 'WHCcluster'+str(c)
                prexif_name += '_PCA'+str(comp_number)
                if not(classe is None):
                   prexif_name += '_'+classe 
                do_lucidVizu_forPCA_all_case(path_lucid_model,name_pb_full_model,
                                     prexif_name,features_size,
                                     layer,weights,path_output_lucid_im,
                                     input_name_lucid,constrNet,
                                     strictMinimum,cossim)
        

## Tentative de faire avec LBFGS mais cela n'a pas fonctionner
#    number_iter = 10
#    lr = 1.0
#    for i in range(number_iter):
#      gradient,loss_value = iterate(variable)
#      print(i,loss_value)
#      print('bound grad',np.max(gradient),np.min(gradient),gradient.shape)
#      variable += lr*gradient
#      print('boudn var',np.max(variable),np.min(variable),variable.shape)
#      
##      
#    max_x_value= 1
##    variable = max_x_value*np.random.random(variable_shape) # il faudrait peut etre 
##
##    number_iter = 30000
##    loss_value = 0.
##    itera = 0
##    while (loss_value <0.9 and itera <number_iter):
##        gradient,loss_value = iterate(variable)
##        variable += lr*gradient
#        
#    # Tentative avec LBFGSde scipy : ne marche pas
#    import scipy
#    x0 = max_x_value*np.random.random(variable_shape)
#    x0 = max_x_value*np.random.random([1,14,14,528])
#    x0 = max_x_value*np.random.random([1*14*14*528])
#
#
#    x_f,f,d =scipy.optimize.fmin_l_bfgs_b(symbolic_minus_loss, x0, fprime=grad_symbolic_minus_loss) # If fprime==None, then func returns the function value and the gradient (f, g = func(x, *args)), )
#
#
#    x_f,f,d =scipy.optimize.fmin_l_bfgs_b(symbolic_minus_loss_c_and_gradient, x0, fprime=None) # If fprime==None, then func returns the function value and the gradient (f, g = func(x, *args)), )
    #res = scipy.optimize.minimize(symbolic_minus_loss, x0, method='L-BFGS-B')
#    input_block = part_model.inputs[0]
#    x_tf_var = tf.Variable(max_x_value*np.random.random(variable_shape))
#    loss_output = symbolic_minus_loss(x_tf_var)
#    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_output,method='L-BFGS-B')
#    trainable_variables = tf.trainable_variables()
#    sess = tf.Session()
#    sess.run(tf.global_variables_initializer())    
#    
#    optimizer.minimize(sess)
    
    # Avec une optimisation a la TF : ne marche pas non plus
#    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
#    
#    @tf.function()
#    def train_step(input_var):
#      with tf.GradientTape() as tape:
#        loss = symbolic_loss_c(input_var)
#    
#      grad = tape.gradient(loss, input_var)
#      opt.apply_gradients([(grad, input_var)])
#      #image.assign(clip_0_1(image))
#
#    x_tf_var = tf.Variable(max_x_value*np.random.random(variable_shape))
#
#    epochs = 10
#    steps_per_epoch = 100
#    
#    step = 0
#    for n in range(epochs):
#      for m in range(steps_per_epoch):
#        step += 1
#        train_step(x_tf_var)
#        print(".", end='')
    


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
    
def get_truncated_keras_model(model,model_name,new_input_layer_name,
                              constrNet='InceptionV1',batch_size=None,
                              reshape=True):
    layer = model.get_layer(new_input_layer_name)
    layer_output = layer.output
    dim_shape = layer_output.shape.dims
    new_input_shape = []
    for dim in dim_shape:
        new_input_shape += [dim.value]
    new_input_shape.pop(0) #remove the batch size dim
    new_input_shape_with_batch = [batch_size] + new_input_shape
    if reshape:
        assert(not(batch_size is None))
        first_new_input = Input(shape=new_input_shape[0]*new_input_shape[1]*new_input_shape[2],name='new_input_1')
        new_input = tf.keras.layers.Reshape(new_input_shape)(first_new_input)
    else:
        new_input = Input(shape=new_input_shape,name='new_input_1') 
        first_new_input = new_input
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    
    # Set the input layers of each layer
    list_layer_name = []
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            list_layer_name += [layer_name]
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if (layer.name not in network_dict['input_layers_of'][layer_name]):
                    network_dict['input_layers_of'][layer_name].append(layer.name)
    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    assert(new_input_layer_name in list_layer_name)

    # Iterate over all layers after the input
    new_input_layer_passed = False
    for layer in model.layers[1:]:
        #print('==',layer.name,new_input_layer_passed,network_dict['input_layers_of'])
        if new_input_layer_passed:
            
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]

            if len(layer_input) >1:
                layer_input = layers_unique(layer_input)
            if len(layer_input) == 1:
                layer_input = layer_input[0]
                if len(layer_input.shape)==5:
                    layer_input = layer_input[0,...]
            
            x = layer(layer_input)
            network_dict['new_output_tensor_of'].update({layer.name: x})
            
        if layer.name == new_input_layer_name:
            
            layer_input = new_input
            new_input_layer_passed = True
            network_dict['new_output_tensor_of'].update({new_input_layer_name: layer_input})
            
            
    net_finetuned_truncated = Model(inputs=first_new_input,outputs=x)
    
        
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
    return(net_finetuned_truncated)
    
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

    net_finetuned_truncated= get_truncated_keras_model(model=model,
                                                       model_name=model_name,
                                                       new_input_layer_name=new_input_layer_name,
                                                       constrNet=constrNet)
    
    
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
    

#    PCAbased_FeaVizu_deepmodel(model_name = 'pretrained',\
#                                   layer='mixed4d',classe='nudity',
#                                   source_dataset='IconArt_v1',plot_FeatVizu=True)
#    
#    print_stats_matrices(model_name = 'pretrained',
#                         list_classes=[None,'Mary','ruins','nudity'],layer='mixed4d',
#                         source_dataset='IconArt_v1')
#    
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = None,\
#                                   layer='mixed4d',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
#                                   layer='mixed4d',classe='Mary',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
#                                   layer='mixed4d',classe='ruins',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',\
#                                   layer='mixed4d',classe='nudity',plot_FeatVizu=True,num_components_draw=3)
#    
#    produce_latex_textV2(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                       layer_tab=['mixed4d'],classe_tab = [None,'Mary','ruins','nudity'],
#                       folder_im_latex='im',
#                       num_components_draw = 3,
#                       KindOfMeanReduciton='global')
#    
#    print_stats_matrices(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                         list_classes=[None,'Mary','ruins','nudity'],layer='mixed4d')
#    print_stats_matrices(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200_LastEpoch',
#                         list_classes=[None,'Mary','ruins','nudity'],layer='mixed4d')
#    
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = None,\
#                                   layer='mixed4c_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = 'Mary',\
#                                   layer='mixed4d_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',classe = 'Mary',\
#                                   layer='mixed4c_pre_relu')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = None,\
#                                   layer='mixed4d',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe ='Northern_Renaissance',\
#                                   layer='mixed4d',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe ='Abstract_Art',\
#                                   layer='mixed4d',plot_FeatVizu=True,num_components_draw=3)
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe ='Ukiyo-e',\
#                                layer='mixed4d',plot_FeatVizu=True,num_components_draw=3)
#    
#    print_stats_matrices(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                         list_classes=[None,'Northern_Renaissance','Abstract_Art','Ukiyo-e'],
#                         layer='mixed4d',
#                         source_dataset='RASTA')
#    print_stats_matrices(model_name = 'pretrained',
#                         list_classes=[None,'Northern_Renaissance','Abstract_Art','Ukiyo-e'],
#                         layer='mixed4d',
#                         source_dataset='RASTA')
#    
#    produce_latex_textV2(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                       layer_tab=['mixed4d'],classe_tab = [None,'Northern_Renaissance','Abstract_Art','Ukiyo-e'],
#                       folder_im_latex='im',
#                       num_components_draw = 3,
#                       KindOfMeanReduciton='global')
   
    ## Calcul pour le Week-end 
    
    # Toutes les features visualisation pour la couche donnée pour IconArt et RASTA
#    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = None,\
#                               layer='mixed4d',
#                               clustering=None)
# A faire tourner plus tard 
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = 'Northern_Renaissance',\
#                               layer='mixed4d',
#                               clustering=None)
#    
#    # 10 aléatoires
##    PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
##                               classe = None,\
##                               layer='mixed4d',
##                               clustering='random')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = 'Northern_Renaissance',\
#                               layer='mixed4d',
#                               clustering='random')
#    
#   #  Faire les 10 premieres components pour chacun des deux datasets et pour quelques classes
#    
##    for classe in [None,'Mary','ruins','nudity','angel']:
##        PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
##                               classe = classe,\
##                               layer='mixed4d',
##                               clustering='PCA')
#    for classe in [None,'Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art','Post-Impressionism','Realism']:
#        PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='mixed4d',
#                               clustering='PCA')
    #for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art','Post-Impressionism','Realism','Impressionism','Magic_Realism']:
#    for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e']:
#        PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='Mixed_5c_Concatenated',
#                               clustering='PCA',
#                               constrNet='InceptionV1_slim',
#                               num_components_draw = 2,
#                               strictMinimum=False)
#        PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='Mixed_5c_Concatenated',
#                               clustering='PCAsubset',
#                               constrNet='InceptionV1_slim',
#                               num_components_draw = 2,
#                               strictMinimum=False)
#        
        
        
        
        
    # PCA subset non teste 
#    for classe in [None,'Mary','ruins','nudity','angel']:
#        PCAbased_FeaVizu_deepmodel(model_name = 'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='mixed4d',
#                               clustering='PCAsubset')
#    for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art','Post-Impressionism','Realism']:
#        PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='mixed4d',
#                               clustering='PCAsubset')
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                           classe = None,\
#                           layer='mixed4d',
#                           clustering='equal')
#    for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e']:
#        PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                               classe = classe,\
#                               layer='mixed4d',
#                               clustering='equalHotElement')

    
#    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                                    constrNet = 'InceptionV1',
#                                    classe='Northern_Renaissance',layer='mixed4d',
#                                    num_components_draw = 3,clustering = 'WHC',
#                                    number_of_blocks = 1,strictMinimum=True,
#                                    whiten=False,cossim=False)
#    for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e']:
#        for cossim in [False]:
#            for clustering in ['Kmeans','PCA','IPCA','MeanShift']:
#                for whiten in [False]:
#                 
#                    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                                        constrNet = 'InceptionV1',
#                                        classe=classe,layer='mixed4d',
#                                        num_components_draw =5,clustering = clustering,
#                                        number_of_blocks = 20,strictMinimum=True,
#                                        whiten=whiten,cossim=cossim)
#    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                        constrNet = 'InceptionV1',
#                        classe='Northern_Renaissance',layer='mixed4d',
#                        num_components_draw =5,clustering = 'NMF',
#                        number_of_blocks = 2,strictMinimum=True,
#                        whiten=False,cossim=False)
#    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                        constrNet = 'InceptionV1',
#                        classe='Abstract_Art',layer='mixed4d',
#                        num_components_draw =5,clustering = 'NMF',
#                        number_of_blocks = 2,strictMinimum=True,
#                        whiten=False,cossim=False)
    
    # Nouveau test 19/06/20
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200',
#                               classe = 'Northern_Renaissance',\
#                               layer='mixed4d',
#                               clustering='PCA',
#                               num_components_draw=2,
#                               strictMinimum=False)
    # TODO cela ne marche pas et je ne sais pas pourquoi !!!
#    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200',
#                        constrNet = 'InceptionV1',
#                        classe='Northern_Renaissance',layer='mixed4d',
#                        num_components_draw =5,clustering = 'NMF',
#                        number_of_blocks = 2,strictMinimum=True,
#                        whiten=False,cossim=False)
#    Generate_Im_class_conditionated(model_name='RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200',
#                        constrNet = 'InceptionV1',
#                        classe='Abstract_Art',layer='mixed4d',
#                        num_components_draw =5,clustering = 'NMF',
#                        number_of_blocks = 2,strictMinimum=True,
#                        whiten=False,cossim=False)
    # PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
    #                            classe = 'Northern_Renaissance',\
    #                            layer='Mixed_4e_Concatenated',
    #                            clustering='PCAsubset',
    #                            num_components_draw=2,
    #                            strictMinimum=False,
    #                            constrNet='InceptionV1_slim')
    
    # for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art',
    #                'Post-Impressionism','Realism']:
    #     PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',
    #                           classe = classe,\
    #                           layer='mixed4d',
    #                           clustering='PCA',
    #                           num_components_draw=3)
    
#    produce_latex_textV2(model_name = 'RASTA_small01_modif',
#                       layer_tab=['mixed4d_pre_relu'],classe_tab = ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art',
#                                 'Post-Impressionism','Realism'],
#                               folder_im_latex='im',
#                               num_components_draw = 3,
#                               KindOfMeanReduciton='global')
    
    topK_features_per_class_list_of_modelpretrained()
    
#    PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',
#                                       classe = 'Northern_Renaissance',\
#                                       layer='mixed4d',
#                                       clustering='corr')
    
    for layer in ['mixed4d','mixed5a']:
        for kind_method in ['corr','PCA','PCAsubset']:
        
            for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art','Post-Impressionism','Realism']:
                PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                                       classe = classe,\
                                       layer=layer,
                                       clustering=kind_method)
            for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e','Pop_Art','Post-Impressionism','Realism']:
                PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_small01_modif',
                                       classe = classe,\
                                       layer=layer,
                                       clustering=kind_method)
    
    
    for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e']:
        for cossim in [False]:
            for clustering in ['NMF','Kmeans','PCA','IPCA','WHC']:
                for whiten in [False,True]:
                    for layer in ['mixed4d','mixed5a']:
                        if clustering=='WHC':
                            number_of_blocks = 1
                        else:
                            number_of_blocks = 20
                        Generate_Im_class_conditionated(model_name='RASTA_small01_modif',
                                            constrNet = 'InceptionV1',
                                            classe=classe,layer=layer,
                                            num_components_draw =3,clustering = clustering,
                                            number_of_blocks = number_of_blocks,strictMinimum=True,
                                            whiten=whiten,cossim=cossim)
    
    # for classe in ['Abstract_Art','Ukiyo-e']:
    #     PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
    #                               classe = classe,\
    #                               layer='mixed4d',
    #                               clustering='PCA',
    #                               num_components_draw=3,strictMinimum=False)
    #     PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
    #                               classe = classe,\
    #                               layer='mixed4d',
    #                               clustering='PCA',
    #                               num_components_draw=3,strictMinimum=False)
            
    # for classe in ['Northern_Renaissance','Abstract_Art','Ukiyo-e']:
    #     PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
    #                           classe = classe,\
    #                           layer='mixed4d',
    #                           clustering='equalHotElement')
    #     PCAbased_FeaVizu_deepmodel(model_name = 'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
    #                           classe = classe,\
    #                           layer='mixed4d',
    #                           clustering='equalHotElement')
    
    
    
    

