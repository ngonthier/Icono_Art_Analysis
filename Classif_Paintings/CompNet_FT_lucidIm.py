#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:24:07 2020

In this script we will load the Model before and after fine-tuning and do 
some deep-dream of the features maps of the weights that change the most
(relative way)

Les codes recensés ici peuvent etre utiles : https://github.com/tensorflow/lucid

@author: gonthier
"""

# InvalidArgumentError: Invalid device ordinal value (1). Valid range is [0, 0]. 	while setting up XLA_GPU_JIT device number 1
# This error can be due to a missing export cuda device, normally when launching your own environnement need to execute this command line with the right number
# can be set in the activate.sh file of the conda env

import tensorflow as tf
import os
import matplotlib
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.python.keras.layers import Conv2D

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
from googlenet import inception_v1_oldTF as Inception_V1

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
import platform
import pickle

list_finetuned_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
                        'IconArt_v1_small001_modif_LastEpoch','IconArt_v1_big001_modif_LastEpoch',
                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
                        'RASTA_small01_modif','RASTA_big01_modif',
                        'RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_LastEpoch','RASTA_big001_modif_LastEpoch',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif_dataAug',
                        'RASTA_small01_modif_ep120',
                        'RASTA_small01_modif_dataAug_ep120',
                        'RASTA_small01_modif_deepSupervision_ep120',
                        'RASTA_big001_modif_dataAug',
                        'RASTA_big001_modif_ep120',
                        'RASTA_big001_modif_dataAug_ep120',
                        'RASTA_big001_modif_deepSupervision_ep120',
                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch','RASTA_big001_modif_LastEpoch'
                        ]

def get_random_net(constrNet='VGG'):
    seed = 0
    tf.set_random_seed(seed) # For  tf v1
    if constrNet=='VGG':
        randomNet = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None)
    elif constrNet=='InceptionV1':
        randomNet = Inception_V1(include_top=False, weights=None)
    return(randomNet)

def get_fine_tuned_model(model_name,constrNet='VGG',suffix=''):
    
    opt_option_small=[0.1,0.001]
    opt_option_small01=[0.1,0.01]
    opt_option_big=[0.001]
    opt_option_big01=[0.01]
    
    if not(model_name in list_finetuned_models_name):
        raise(NotImplementedError)
        
    if 'small001' in  model_name:
        opt_option = opt_option_small
    elif 'big001' in model_name:
        opt_option = opt_option_big
    elif 'small01' in  model_name:
        opt_option = opt_option_small01
    elif 'big01' in model_name:
        opt_option = opt_option_big01
    else:
        print('Model unknown :',model_name)
        raise(NotImplementedError)
        
    if 'LastEpoch' in model_name:
        return_best_model=False
    else:
        return_best_model=True
        
    if 'dataAug' in model_name:
        dataAug=True
    else:
        dataAug=False
        
    if 'ep120' in model_name:
        epochs=120
    else:
        epochs=20
        
    if 'deepSupervision' in model_name:
        deepSupervision=True
    else:
        deepSupervision=False
        
    if 'RASTA' in model_name:
        target_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        target_dataset = 'IconArt_v1'
    else:
        raise(NotImplementedError)

    weights = 'imagenet'
    
    if constrNet=='VGG':
        features = 'block5_pool'
        final_clf = 'MLP2'
        transformOnFinalLayer='GlobalAveragePooling2D' 
    elif constrNet=='InceptionV1':
        features = 'avgpool'
        final_clf = 'MLP1'
        transformOnFinalLayer=None
    else:
        raise(ValueError(constrNet + ' is unknown in this function'))
        
    normalisation = False
    source_dataset= 'ImageNet'
    kind_method=  'FT'
    optimizer='SGD'
    
    
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
                           SGDmomentum=SGDmomentum,decay=decay,return_best_model=return_best_model,\
                           pretrainingModif=True,suffix=suffix,deepSupervision=deepSupervision,\
                           dataAug=dataAug)
    return(net_finetuned)

def convert_finetuned_modelToFrozenGraph(model_name,constrNet='VGG',path='',suffix=''):
    
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    K.set_learning_phase(0)

    list_models_name_all = list_finetuned_models_name + ['random']
    if not(model_name in list_models_name_all):
        raise(NotImplementedError)
        
    if model_name=='random':
        net_finetuned = get_random_net(constrNet)
    else:
        net_finetuned = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
        
    if path=='':
        os.makedirs('./model', exist_ok=True)
        path ='model'
    else:
        os.makedirs(path, exist_ok=True)
    frozen_graph = lucid_utils.freeze_session(K.get_session(),
                              output_names=[out.op.name for out in net_finetuned.outputs])
    if not(suffix=='' or suffix is None):
        suffix_str = '_'+suffix
    else:
        suffix_str = ''
    name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
    
    #nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(nodes_tab)
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
#        if not('conv' in layer_name):
#            continue
        # get filter weights
        if not(layer_name in list_name_layers):
            continue
        if isinstance(finetuned_layer, Conv2D) :
            o_filters, o_biases = list_weights[j]
            j+=1
            f_filters, f_biases = finetuned_layer.get_weights()
#            print(layer_name, f_filters.shape)
            #num_filters = o_filters.shape[-1]
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
            argsort = np.argsort(relative_diff_abs)[::-1]
            dict_layers_argsort[layer_name] = argsort
        
    return(dict_layers_relative_diff,dict_layers_argsort)

def print_stats_on_diff(np_list,k=3):
    print('Max :',np.max(np_list),'Median :',np.median(np_list),'Mean :',np.mean(np_list))
    argsort = np.argsort(np_list)[::-1]
    for i in range(k):
        print('Top ',i,': index =',argsort[i],' value :',np_list[argsort[i]])

def get_imageNet_weights(Net):
    weights = 'imagenet'
    
    if Net=='VGG':
        imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
    elif Net == 'InceptionV1':
        imagenet_model = Inception_V1(include_top=False, weights=weights)
    else:
        raise(NotImplementedError)
        
    net_layers = imagenet_model.layers
       
    list_weights = []
    list_name_layers = []
    for original_layer in net_layers:
        #print(original_layer.name,isinstance(original_layer, Conv2D))
        # check for convolutional layer
        layer_name = original_layer.name
        if isinstance(original_layer, Conv2D) :
            # get filter weights
            o_weights = original_layer.get_weights() # o_filters, o_biases
            list_weights +=[o_weights]
            list_name_layers += [layer_name]
    return(list_weights,list_name_layers)

def print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path='',\
                                    constrNet='InceptionV1'):
    if constrNet=='VGG':
        # For the original pretrained imagenet VGG
        lucid_utils.print_images(model_path=os.path.join('model','tf_vgg19.pb'),list_layer_index_to_print=list_layer_index_to_print_base_model\
                      ,path_output=output_path,prexif_name='ImagnetVGG',input_name='input_1',Net=constrNet)
    elif constrNet=='InceptionV1':
        # For the original pretrained imagenet InceptionV1 from Lucid to keras to Lucid
        lucid_utils.print_images(model_path=os.path.join('model','tf_inception_v1.pb'),list_layer_index_to_print=list_layer_index_to_print_base_model\
                      ,path_output=output_path,prexif_name='ImagnetVGG',input_name='input_1',Net=constrNet)
 

def Comparaison_of_FineTunedModel(constrNet = 'VGG'):
    """
    This function will load the two models (deep nets) before and after fine-tuning 
    and then compute the difference between the weights and finally run a 
    deep dream on the feature maps of the weights that have the most change
    """
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
    else:
        raise(NotImplementedError(constrNet + ' is not implemented sorry.'))
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    list_weights,list_name_layers = get_imageNet_weights(Net=constrNet)
    
    list_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
                        'RASTA_small01_modif','RASTA_big01_modif',
                        'RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision']
    
    list_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
                        'RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision']
    # Semble diverger dans le cas de InceptionV1  :'RASTA_big01_modif',
    list_models_name = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch']
    list_models_name = ['RASTA_big001_modif_LastEpoch',
                        'RASTA_small01_modif_dataAug',
                        'RASTA_small01_modif_ep120',
                        'RASTA_small01_modif_dataAug_ep120',
                        'RASTA_small01_modif_deepSupervision_ep120',
                        'RASTA_big001_modif_dataAug',
                        'RASTA_big001_modif_ep120',
                        'RASTA_big001_modif_dataAug_ep120',
                        'RASTA_big001_modif_deepSupervision_ep120']
    #list_models_name = ['random']
    #opt_option_tab = [opt_option_small,opt_option_big,opt_option_small,opt_option_big,None]
    
    suffix_tab = ['','1'] # In order to have more than once the model fine-tuned with some given hyperparameters
    
    K.set_learning_phase(0)
    #with K.get_session().as_default(): 
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    dict_list_layer_index_to_print_base_model = {}
    
    num_top = 3
    for model_name in list_models_name:
        print('#### ',model_name)
        
        if not(model_name=='random'):
            for suffix in suffix_tab:
                output_path_with_model = os.path.join(output_path,model_name+suffix)
                pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
                
                net_finetuned = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
                dict_layers_relative_diff,dict_layers_argsort = get_gap_between_weights(list_name_layers,\
                                                                                list_weights,net_finetuned)
                save_file = os.path.join(output_path_with_model,'dict_layers_relative_diff.pkl')
                with open(save_file, 'wb') as handle:
                    pickle.dump(dict_layers_relative_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                              constrNet=constrNet,path=path_lucid_model,suffix=suffix)
                list_layer_index_to_print_base_model = []
                list_layer_index_to_print = []
                for key in dict_layers_argsort.keys():
                    for k in range(num_top):
                        topk = dict_layers_argsort[key][k]
                        list_layer_index_to_print += [[key,topk]]
                        list_layer_index_to_print_base_model += [[key,topk]]
                
                dict_list_layer_index_to_print_base_model[model_name+suffix] = list_layer_index_to_print_base_model
                
                lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
                        ,path_output=output_path_with_model,prexif_name=model_name+suffix,input_name=input_name_lucid,Net=constrNet)
                
                print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path=output_path_with_model,\
                                    constrNet=constrNet)
                    
        else:
            # Random model 
            net_finetuned = get_random_net(constrNet)
            dict_layers_relative_diff,dict_layers_argsort = get_gap_between_weights(list_name_layers,\
                                                                            list_weights,net_finetuned)
            
            name_pb = convert_finetuned_modelToFrozenGraph(model_name,constrNet=constrNet,path=path_lucid_model)
            
            list_layer_index_to_print_base_model = []
            list_layer_index_to_print = []
            for key in dict_layers_argsort.keys():
                for k in range(num_top):
                    topk = dict_layers_argsort[key][k]
                    list_layer_index_to_print += [[key,topk]]
                    list_layer_index_to_print_base_model += [[key,topk]]
            
            output_path_with_model = os.path.join(output_path,model_name)
            pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
            lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
                         ,path_output=output_path_with_model,prexif_name=model_name,input_name='input_1',Net=constrNet)
             
            print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path=output_path_with_model,\
                                    constrNet=constrNet)
    
    
if __name__ == '__main__': 
    Comparaison_of_FineTunedModel(constrNet='InceptionV1')    
        
        
        