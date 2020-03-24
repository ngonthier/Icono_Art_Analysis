#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:24:07 2020

In this script we will load the Model before and after fine-tuning and do 
some deep-dream of the features maps of the weights that change the most


Les codes recensés ici peuvent etre utiles : https://github.com/tensorflow/lucid

@author: gonthier
"""

import tensorflow as tf
import os
import matplotlib
from keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras import backend as K
import numpy as np

from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index,numeral_layers_index_bitsVersion,\
    Precompute_Cumulated_Hist_4Moments,load_Cumulated_Hist_4Moments,get_list_im
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
    load_resize_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model,\
    MLP_model,Perceptron_model,vgg_adaDBN,ResNet_AdaIn,ResNet_BNRefinements_Feat_extractor,\
    ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction,ResNet_cut,vgg_suffleInStats,\
    get_ResNet_ROWD_meanX_meanX2_features,get_BaseNorm_meanX_meanX2_features,\
    get_VGGmodel_meanX_meanX2_features,add_head_and_trainable,extract_Norm_stats_of_ResNet,\
    vgg_FRN,get_those_layers_output
from StatsConstr_ClassifwithTL import learn_and_eval
import cv2

import pickle
import pathlib
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from keras_resnet_utils import getBNlayersResNet50,getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion,\
    fit_generator_ForRefineParameters

import lucid_utils

def get_fine_tuned_model(model_name,constrNet='VGG'):
    
    opt_option_small=[0.1,0.001]
    opt_option_big=[0.1,0.001]
    
    list_models_name = ['IconArt_v1_small_modif','IconArt_v1_big_modif','RASTA_small_modif','RASTA_big_modif']
    if not(model_name in list_models_name):
        raise(NotImplementedError)
        
    if 'small' in  model_name:
        opt_option = opt_option_small
    elif 'big' in model_name:
        opt_option = opt_option_big
        
    if 'RASTA' in model_name:
        target_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        target_dataset = 'IconArt_v1'
        
    source_dataset = 'imagenet'   
    weights = 'imagenet'
    
    features = 'block5_pool'
    normalisation = False
    getBeforeReLU = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset= 'ImageNet'
    kind_method=  'FT'
    transformOnFinalLayer='GlobalAveragePooling2D'           
    final_clf = 'MLP2'
    
    computeGlobalVariance = False
    optimizer='SGD'
    
    return_best_model=True
    epochs=20
    cropCenter=True
    SGDmomentum=0.9
    decay=1e-4

    returnStatistics = True    
    net_finetuned = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers=[],weights=weights,\
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                           ReDo=False,
                           returnStatistics=returnStatistics,cropCenter=cropCenter,\
                           optimizer=optimizer,opt_option=opt_option,epochs=epochs,\
                           SGDmomentum=SGDmomentum,decay=decay,return_best_model=return_best_model)
    return(net_finetuned)

def convert_finetuned_modelToFrozenGraph(model_name,constrNet='VGG',path=''):
    K.set_learning_phase(0)
    with K.get_session().as_default(): 
        #images = tf.placeholder("float32", [None, 224, 224, 3], name="input")
    
        # <Code to construct & load your model inference graph goes here>
        model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
        
        #  ! il va falloir ajouter des noeuds / node pre_relu !
        
        
        opt_option_small=[0.1,0.001]
        opt_option_big=[0.1,0.001]
        
        list_models_name = ['IconArt_v1_small_modif','IconArt_v1_big_modif','RASTA_small_modif','RASTA_big_modif']
        if not(model_name in list_models_name):
            raise(NotImplementedError)
            
        if 'small' in  model_name:
            opt_option = opt_option_small
        elif 'big' in model_name:
            opt_option = opt_option_big
            
        if 'RASTA' in model_name:
            target_dataset = 'RASTA'
        elif 'IconArt_v1' in model_name:
            target_dataset = 'IconArt_v1'
            
        source_dataset = 'imagenet'   
        weights = 'imagenet'
        
        features = 'block5_pool'
        normalisation = False
        getBeforeReLU = False
        final_clf= 'LinearSVC' # Don t matter
        source_dataset= 'ImageNet'
        kind_method=  'FT'
        transformOnFinalLayer='GlobalAveragePooling2D'           
        final_clf = 'MLP2'
        
        computeGlobalVariance = False
        optimizer='SGD'
        
        return_best_model=True
        epochs=20
        cropCenter=True
        SGDmomentum=0.9
        decay=1e-4
    
        returnStatistics = True    
        net_finetuned = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                               constrNet,kind_method,style_layers=[],weights=weights,\
                               normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                               ReDo=False,
                               returnStatistics=returnStatistics,cropCenter=cropCenter,\
                               optimizer=optimizer,opt_option=opt_option,epochs=epochs,\
                               SGDmomentum=SGDmomentum,decay=decay,return_best_model=return_best_model)
        if path=='':
            os.makedirs('./model', exist_ok=True)
            path ='model'
        else:
            os.makedirs(path, exist_ok=True)
        frozen_graph = lucid_utils.freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
        name_pb = 'tf_graph_'+constrNet+model_name+'.pb'
        tf.io.write_graph(frozen_graph,logdir= path,name= name_pb, as_text=False)
    
    
    return(name_pb)

def get_gap_between_weights(list_name_layers,list_weights,net_finetuned):
    finetuned_layers = net_finetuned.layers
        
    dict_layers_argsort = {}
    dict_layers_relative_diff = {}
    j = 0
    for finetuned_layer in finetuned_layers:
        # check for convolutional layer
        layer_name = finetuned_layer.name
        if not('conv' in layer_name):
            continue
        # get filter weights
        if not(layer_name in list_name_layers):
            continue
        o_filters, o_biases = list_weights[j]
        j+=1
        f_filters, f_biases = finetuned_layer.get_weights()
        print(layer_name, f_filters.shape)
        num_filters = o_filters.shape[-1]
        # Norm 2 between the weights of the filters
            
        diff_filters = o_filters - f_filters
        norm2_filter = np.mean(o_filters**2,axis=(0,1,2))
        norm1_filter = np.mean(np.abs(o_filters),axis=(0,1,2))
        diff_squared = diff_filters**2
        diff_abs = np.abs(diff_filters)
        mean_squared = np.mean(diff_squared,axis=(0,1,2))
        mean_abs = np.mean(diff_abs,axis=(0,1,2))
        relative_diff_squared = mean_squared / norm2_filter
        relative_diff_abs = mean_abs / norm1_filter
        print('== For layer :',layer_name,' ==')
        print('= Absolute squared of difference =')
        print_stats_on_diff(mean_squared)
        print('= Absolute abs of difference =')
        print_stats_on_diff(mean_abs)
        print('= Relative squared of difference =')
        print_stats_on_diff(relative_diff_squared)
        print('= Relative abs of difference =')
        print_stats_on_diff(relative_diff_abs)
        
        dict_layers_relative_diff[layer_name] = relative_diff_abs
        argsort = np.argsort(mean_squared)[::-1]
        dict_layers_argsort[layer_name] = argsort
        
    return(dict_layers_relative_diff,dict_layers_argsort)

def print_stats_on_diff(np_list,k=1):
    print('Max :',np.max(np_list),'Median :',np.median(np_list),'Mean :',np.mean(np_list))
    argsort = np.argsort(np_list)[::-1]
    for i in range(k):
        print('Top ',i,': index =',argsort[i],' value :',np_list[argsort[i]])

def Comparaison_of_FineTunedModel():
    """
    This function will load the two models (deep nets) before and after fine-tuning 
    and then compute the difference between the weights and finally run a 
    deep dream on the feature maps of the weights that have the most change
    """
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution

       
    
    Model_dict = {}
    list_markers = ['o','s','X','*']
    alpha = 0.7
    
    dict_of_dict_hist = {}
    dict_of_dict = {}
    constrNet = 'VGG'
    
    weights = 'imagenet'
    
    imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
    net_layers = imagenet_model.layers
       
    list_weights = []
    list_name_layers = []
    for original_layer in net_layers:
        # check for convolutional layer
        layer_name = original_layer.name
        if not('conv' in layer_name):
            continue
        # get filter weights
        o_weights = original_layer.get_weights() # o_filters, o_biases
        list_weights +=[o_weights]
        list_name_layers += [layer_name]
    
    opt_option_small=[0.1,0.001]
    opt_option_big=[0.1,0.001]
    
    list_models_name = ['IconArt_v1_small_modif','IconArt_v1_big_modif','RASTA_small_modif','RASTA_big_modif','random']
    list_models_name = ['IconArt_v1_small_modif']
    opt_option_tab = [opt_option_small,opt_option_big,opt_option_small,opt_option_big,None]
    
    K.set_learning_phase(0)
    list_layer_index_to_print_base_model = []
    for model_name in list_models_name:
        if not(model_name=='random'):
            net_finetuned = get_fine_tuned_model(model_name)
            name_pb = convert_finetuned_modelToFrozenGraph(model_name,constrNet='VGG',path='')
            dict_layers_relative_diff,dict_layers_argsort = get_gap_between_weights(list_name_layers,list_weights,net_finetuned)
            
            list_layer_index_to_print = []
            for key in dict_layers_argsort.keys():
                top1 = dict_layers_argsort[key][0]
                list_layer_index_to_print += [[key,top1]]
                list_layer_index_to_print_base_model += [[key,top1]]
            
            
            
        
    
        
    
 
if __name__ == '__main__': 
    DeepDream_withFinedModel()    
        
        
        