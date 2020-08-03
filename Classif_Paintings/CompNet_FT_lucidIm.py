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
# from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.python.keras.layers import Conv2D

# from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index,numeral_layers_index_bitsVersion,\
#     Precompute_Cumulated_Hist_4Moments,load_Cumulated_Hist_4Moments,get_list_im
#import Stats_Fcts 
# import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
#     load_resize_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model,\
#     MLP_model,Perceptron_model,vgg_adaDBN,ResNet_AdaIn,ResNet_BNRefinements_Feat_extractor,\
#     ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction,ResNet_cut,vgg_suffleInStats,\
#     get_ResNet_ROWD_meanX_meanX2_features,get_BaseNorm_meanX_meanX2_features,\
#     get_VGGmodel_meanX_meanX2_features,add_head_and_trainable,extract_Norm_stats_of_ResNet,\
#     vgg_FRN,get_those_layers_output
import StatsConstr_ClassifwithTL #import learn_and_eval
from googlenet import inception_v1_oldTF as Inception_V1
from inceptionV1_keras_utils import get_trainable_layers_name
from keras_resnet_utils import getResNet50_trainable_vizualizable_layers_name
from keras_vgg_utils import getVGG_trainable_vizualizable_layers_name

from inception_v1 import InceptionV1_slim,trainable_layers

#import cv2

import pickle
import pathlib
#import itertools

import matplotlib.pyplot as plt
# import matplotlib.cm as mplcm
# import matplotlib.colors as colors
# from matplotlib.backends.backend_pdf import PdfPages
# from keras_resnet_utils import getBNlayersResNet50,getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion,\
#     fit_generator_ForRefineParameters

import lucid_utils
import platform

from shortmodelname import get_list_shortcut_name_model

list_finetuned_models_name = get_list_shortcut_name_model()
#list_finetuned_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
#                        'IconArt_v1_small001_modif_LastEpoch','IconArt_v1_big001_modif_LastEpoch',
#                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
#                        'RASTA_small01_modif','RASTA_big01_modif',
#                        'RASTA_small001_modif','RASTA_big001_modif',
#                        'RASTA_small001_modif_LastEpoch','RASTA_big001_modif_LastEpoch',
#                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
#                        'RASTA_small01_modif_dataAug',
#                        'RASTA_small01_modif_ep120',
#                        'RASTA_small01_modif_dataAug_ep120',
#                        'RASTA_small01_modif_deepSupervision_ep120',
#                        'RASTA_big001_modif_dataAug',
#                        'RASTA_big001_modif_ep120',
#                        'RASTA_big001_modif_dataAug_ep120',
#                        'RASTA_big001_modif_deepSupervision_ep120',
#                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch','RASTA_big001_modif_LastEpoch',
#                        'RMN_small01_modif','RMN_big01_modif',
#                        'RMN_small001_modif','RMN_big001_modif',
#                        'RMN_small001_modif_LastEpoch','RMN_big001_modif_LastEpoch',
#                        'RMN_small001_modif_deepSupervision','RMN_big001_modif_deepSupervision',
#                        'RMN_small01_modif_dataAug',
#                        'RMN_small01_modif_ep120',
#                        'RMN_small01_modif_dataAug_ep120',
#                        'RMN_small01_modif_deepSupervision_ep120',
#                        'RMN_big001_modif_dataAug',
#                        'RMN_big001_modif_ep120',
#                        'RMN_big001_modif_dataAug_ep120',
#                        'RMN_big001_modif_deepSupervision_ep120',
#                        'RMN_small01_modif_LastEpoch','RMN_small001_modif_LastEpoch','RMN_big001_modif_LastEpoch'
#                        ]

def get_random_net(constrNet='VGG'):
    seed = 0
    tf.set_random_seed(seed) # For  tf v1
    if constrNet=='VGG':
        randomNet = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None)
    elif constrNet=='InceptionV1':
        randomNet = Inception_V1(include_top=False, weights=None)
    elif constrNet == 'ResNet50':
        randomNet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)
    return(randomNet)

def get_fine_tuned_model(model_name,constrNet='VGG',suffix='',get_Metrics=False,
                         verbose=True):
    
    opt_option_small=[0.1,0.001] # Car opt_option = multiplier_lrp, lr
    opt_option_small01=[0.1,0.01]
    opt_option_big=[0.001] # 10**-3
    opt_option_big01=[0.01]
    opt_option_big001=[0.0001] # 10**-4 
    
    if not(model_name in list_finetuned_models_name):
        raise(NotImplementedError(model_name+' is unknown.'))
        
    if 'small001' in  model_name:
        opt_option = opt_option_small
    elif 'big001' in model_name:
        opt_option = opt_option_big
    elif 'big0001' in model_name:
        opt_option = opt_option_big001
    elif 'small01' in  model_name:
        opt_option = opt_option_small01
    elif 'big01' in model_name:
        opt_option = opt_option_big01
    else:
        print('Model unknown :',model_name)
        raise(NotImplementedError)
        
    if 'adam' in model_name:
        optimizer='adam'
        SGDmomentum=0.0
        decay=0.0
    elif 'Adadelta' in model_name:
        optimizer='Adadelta'
        SGDmomentum=0.0
        decay=0.0
    elif 'RMSprop' in model_name:
        optimizer='RMSprop'
        SGDmomentum=0.0
        decay=0.0
    else:
        optimizer='SGD'
        SGDmomentum=0.9
        decay=1e-4
    
    if 'cosineloss' in model_name:
        loss = 'cosine_similarity'
    else:
        loss= None
        
    if 'LastEpoch' in model_name:
        return_best_model=False
    else:
        return_best_model=True
        
    if 'dataAug' in model_name:
        dataAug=True
    elif 'SmallDataAug' in model_name:
        dataAug='SmallDataAug' 
    elif 'MediumDataAug' in model_name:
        dataAug='MediumDataAug'
    else:
        dataAug=False
        
    if 'dropout04' in model_name:
        dropout= 0.4
    elif 'dropout070704' in model_name:
        dropout= [0.7,0.7,0.4]
    else:
        dropout = None
        
    if 'ep120' in model_name:
        epochs=120
    elif 'ep200' in model_name:
        epochs=200
    elif 'ep1' in model_name:
        epochs=1 # For testing
    else:
        epochs=20
        
        
    if 'LRschedG' in model_name:
        LR_scheduling_kind='googlenet'
        decay = 0.0
    elif 'RedLROnPlat' in model_name:
        LR_scheduling_kind='ReduceLROnPlateau'
        decay = 0.0
    else:
        LR_scheduling_kind=None
        
    if 'unfreeze' in model_name:
        model_name_split = model_name.split('_')
        for elt in model_name_split:
            if  'unfreeze' in elt:
                num_layer = int(elt.replace('unfreeze',''))
        pretrainingModif = num_layer
    else:
        pretrainingModif = True
        
    if 'deepSupervision' in model_name:
        deepSupervision=True
    else:
        deepSupervision=False
        
    if 'randomCrop' in model_name:
        randomCrop = True
        cropCenter= False
    else:
        randomCrop = False
        cropCenter=True
        
    if 'RASTA' in model_name:
        target_dataset = 'RASTA'
    elif 'RMN' in model_name:
        target_dataset = 'RMN'
    elif 'IconArt_v1' in model_name:
        target_dataset = 'IconArt_v1'
    else:
        raise(NotImplementedError)

    if 'cn' in model_name:
        if 'cn10' in model_name:
            clipnorm = 10
        elif 'cn1' in model_name:
            clipnorm = 1.0
    else:
        clipnorm = False


    if 'RandInit' in model_name:
        weights = None
    elif 'RandForUnfreezed' in model_name:
        weights = 'RandForUnfreezed'
    else:
        weights = 'imagenet'
    SaveInit = True # il faudra corriger cela
    
    if constrNet=='VGG':
        features = 'block5_pool'
        final_clf = 'MLP1'
        transformOnFinalLayer='GlobalAveragePooling2D' 
    elif constrNet=='InceptionV1':
        features = 'avgpool'
        final_clf = 'MLP1'
        transformOnFinalLayer=None
    elif constrNet=='InceptionV1_slim':
        features = 'avgpool'
        final_clf = 'MLP1'
        transformOnFinalLayer=None
    elif constrNet=='ResNet50':
        features = 'conv5_block3_out'
        final_clf = 'MLP1'
        transformOnFinalLayer=None
    else:
        raise(ValueError(constrNet + ' is unknown in this function'))

    normalisation = False
    source_dataset= 'ImageNet'
    kind_method=  'FT'

    if get_Metrics:
        returnStatistics = False
    else:  # To return the network
        returnStatistics = True   
    output = StatsConstr_ClassifwithTL.learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers=[],weights=weights,\
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                           ReDo=False,
                           returnStatistics=returnStatistics,cropCenter=cropCenter,\
                           optimizer=optimizer,opt_option=opt_option,epochs=epochs,\
                           SGDmomentum=SGDmomentum,decay=decay,return_best_model=return_best_model,\
                           pretrainingModif=pretrainingModif,suffix=suffix,deepSupervision=deepSupervision,\
                           dataAug=dataAug,randomCrop=randomCrop,SaveInit=SaveInit,\
                           loss=loss,clipnorm=clipnorm,LR_scheduling_kind=LR_scheduling_kind,\
                           verbose=verbose,dropout=dropout)
    # If returnStatistics with RandInit 
    # output = net_finetuned, init_net
    # If returnStatistics without RandInit
    # output = net_finetuned
    # If Not returnStatistics ie get metrics
    # output= metrics
    return(output)

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
        if 'RandInit' in model_name:
            net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
        else:
            net_finetuned,_  = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)

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

def get_gap_between_weights(list_name_layers,list_weights,net_finetuned,verbose=False):
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
            
            list_weights_j = list_weights[j]
            if len(list_weights)==2:
                o_filters, o_biases = list_weights_j
                f_filters, f_biases = finetuned_layer.get_weights()
            else:
                o_filters = np.array(list_weights_j[0]) # We certainly are in the Inception_V1 case with no biases
                f_filters = np.array(finetuned_layer.get_weights()[0])
            j+=1
            
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
            if verbose:
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
    elif Net == 'InceptionV1_slim':
        imagenet_model = InceptionV1_slim(include_top=False, weights=weights)
    elif Net == 'ResNet50':
        imagenet_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights)
    else:
        raise(NotImplementedError)
        
    return(get_weights_and_name_layers(imagenet_model))
    
def get_weights_and_name_layers(keras_net):
        
    net_layers = keras_net.layers
       
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

def _print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path='',\
                                    constrNet='InceptionV1',input_name='input_1'):
    try:
        if constrNet=='VGG':
            # For the original pretrained imagenet VGG
            model_path = os.path.join('model','tf_vgg19.pb')
            if not(os.path.exists(model_path)):
                lucid_utils.create_pb_model_of_pretrained(constrNet)
            lucid_utils.print_images(model_path=model_path,list_layer_index_to_print=list_layer_index_to_print_base_model\
                          ,path_output=output_path,prexif_name='Imagnet',input_name=input_name,Net=constrNet)
        elif constrNet=='InceptionV1':
            model_path = os.path.join('model','tf_inception_v1.pb')
            if not(os.path.exists(model_path)):
                lucid_utils.create_pb_model_of_pretrained(constrNet)
            # For the original pretrained imagenet InceptionV1 from Lucid to keras to Lucid
            lucid_utils.print_images(model_path=model_path,list_layer_index_to_print=list_layer_index_to_print_base_model\
                          ,path_output=output_path,prexif_name='Imagnet',input_name=input_name,Net=constrNet)
        elif constrNet=='InceptionV1_slim':
            model_path = os.path.join('model','tf_inception_v1_slim.pb')
            if not(os.path.exists(model_path)):
                lucid_utils.create_pb_model_of_pretrained(constrNet)
            # For the original pretrained imagenet InceptionV1 from slim convert to keras
            lucid_utils.print_images(model_path=model_path,list_layer_index_to_print=list_layer_index_to_print_base_model\
                          ,path_output=output_path,prexif_name='Imagnet',input_name=input_name,Net=constrNet)
        elif constrNet=='ResNet50':
            model_path = os.path.join('model','tf_resnet50.pb')
            if not(os.path.exists(model_path)):
                lucid_utils.create_pb_model_of_pretrained(constrNet)
            # ResNet 50 from Keras
            lucid_utils.print_images(model_path=model_path,list_layer_index_to_print=list_layer_index_to_print_base_model\
                          ,path_output=output_path,prexif_name='Imagnet',input_name=input_name,Net=constrNet)
        else:
            raise(NotImplementedError(constrNet+' is unknown here.'))
        return(False,model_path,None)
    except ValueError as e:
        return(True,model_path,e)
    
def print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path='',\
                                    constrNet='InceptionV1'):
       
     error,path,e = _print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,
                                                   output_path,\
                                                       constrNet)
     if error:
        os.remove(path)
        for i in range(1,4):
            input_name = 'input_'+str(i)
            error2,path,e2 = _print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,
                                                   output_path,\
                                                       constrNet,input_name=input_name) 
            if not(error2):
                return(0)
            
        if error2:
            print('When after removing the pb file, we still have a problem')
            raise(e2)

def get_path_pbmodel_pretrainedModel(constrNet='InceptionV1'):
    if constrNet=='VGG':
        # For the original pretrained imagenet VGG
        model_path = os.path.join('model','tf_vgg19.pb')
        if not(os.path.exists(model_path)):
            lucid_utils.create_pb_model_of_pretrained(constrNet)
    elif constrNet=='InceptionV1':
        model_path = os.path.join('model','tf_inception_v1.pb')
        if not(os.path.exists(model_path)):
            lucid_utils.create_pb_model_of_pretrained(constrNet)
    elif constrNet=='InceptionV1_slim':
        model_path = os.path.join('model','tf_inception_v1_slim.pb')
        if not(os.path.exists(model_path)):
            lucid_utils.create_pb_model_of_pretrained(constrNet)
    elif constrNet=='ResNet50':
        model_path = os.path.join('model','tf_resnet50.pb')
        if not(os.path.exists(model_path)):
            lucid_utils.create_pb_model_of_pretrained(constrNet)
    else:
        raise(NotImplementedError(constrNet+' is unknown here.'))
    input_name_lucid = 'input_1'
    return(model_path,input_name_lucid)

def why_white_output():
    """
    The goal of this function is to find why this feature is completely white !
    mixed4b_3x3_pre_reluConv2D_81_RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200
    """
    
    input_name_lucid ='input_1'
    constrNet='InceptionV1'
    model_name = 'RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200'
    suffix =''
    net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')

    name_pb = convert_finetuned_modelToFrozenGraph(model_name,
               constrNet=constrNet,path=path_lucid_model,suffix=suffix)
    output_path_with_model =''
    list_layer_index_to_print = [['mixed4b_3x3_pre_relu',81]]
    output_list = lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
         ,path_output=output_path_with_model,prexif_name=model_name+suffix,input_name=input_name_lucid,Net=constrNet,
         just_return_output=True)
    output_im = output_list[0]
    image01 = np.array(output_im[0][0])
    print('shape',image01.shape)
    print('max',np.max(image01))
    print('min',np.min(image01))
    print('median',np.median(image01))
    num_bins = len(np.unique(np.mean(image01,axis=-1)))
    x = np.ravel(np.mean(image01,axis=-1))
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    from skimage import exposure
    plt.figure()
    plt.imshow(image01)
    # Equalization
    img_eq = exposure.equalize_hist(image01)
    plt.figure()
    plt.imshow(img_eq)
    # Contrast stretching
    p2, p98 = np.percentile(image01, (2, 98))
    img_rescale = exposure.rescale_intensity(image01, in_range=(p2, p98))
    plt.figure()
    plt.imshow(img_rescale)
    

def Comparaison_of_FineTunedModel(list_models_name,constrNet = 'VGG',doAlsoImagesOfOtherModel_feature = False,
                                  testMode=False):
    """
    This function will load the two models (deep nets) before and after fine-tuning 
    and then compute the difference between the weights and finally run a 
    deep dream on the feature maps of the weights that have the most change
    @param testMode : if true we only print the two first feature
    """
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
        trainable_layers_name = getVGG_trainable_vizualizable_layers_name()
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
        trainable_layers_name = get_trainable_layers_name()
    elif constrNet=='InceptionV1_slim':
        input_name_lucid ='input_1'
        trainable_layers_name = trainable_layers()
    elif constrNet=='ResNet50':
        trainable_layers_name = getResNet50_trainable_vizualizable_layers_name()
        input_name_lucid ='input_1'
        #raise(NotImplementedError('Not implemented yet with ResNet for print_images'))
    else:
        raise(NotImplementedError(constrNet + ' is not implemented sorry.'))
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    list_weights,list_name_layers = get_imageNet_weights(Net=constrNet)

    #list_models_name = ['random']
    #opt_option_tab = [opt_option_small,opt_option_big,opt_option_small,opt_option_big,None]
    
    suffix_tab = ['','1'] # In order to have more than once the model fine-tuned with some given hyperparameters
    suffix_tab = [''] 
    
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
                
                net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
                
                # if constrNet=='ResNet50' or constrNet=='VGG':
                #     print('cela ne marche pas pour VGG et ResNet pour l instant')
                #     break
                
                if 'unfreeze' in model_name:
                    layer_considered_for_print_im = []
                    for layer in net_finetuned.layers:
                        trainable_l = layer.trainable
                        name_l = layer.name
                        #print(name_l,trainable_l)
                        if trainable_l and (name_l in trainable_layers_name):
                            layer_considered_for_print_im += [name_l]

                if not('RandInit' in model_name):
                    num_top = 3
                    # IE in the case of the fine tuning
                    dict_layers_relative_diff,dict_layers_argsort = get_gap_between_weights(list_name_layers,\
                                                                                list_weights,net_finetuned)
                    save_file = os.path.join(output_path_with_model,'dict_layers_relative_diff.pkl')
                    with open(save_file, 'wb') as handle:
                        pickle.dump(dict_layers_relative_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,path=path_lucid_model,suffix=suffix)
                    list_layer_index_to_print_base_model = []
                    list_layer_index_to_print = []
                    #print(layer_considered_for_print_im)
                    for key in dict_layers_argsort.keys():
                        #print(key)
                        if 'unfreeze' in model_name and not(key in layer_considered_for_print_im):
                            continue
                        for k in range(num_top):
                             topk = dict_layers_argsort[key][k]
                             list_layer_index_to_print += [[key,topk]]
                             list_layer_index_to_print_base_model += [[key,topk]]
                             
                    if testMode:
                        list_layer_index_to_print = [list_layer_index_to_print[0],list_layer_index_to_print[1]]
                        list_layer_index_to_print_base_model = [list_layer_index_to_print_base_model[0],list_layer_index_to_print_base_model[1]]
                    
                    dict_list_layer_index_to_print_base_model[model_name+suffix] = list_layer_index_to_print_base_model
                    
                    lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
                         ,path_output=output_path_with_model,prexif_name=model_name+suffix,input_name=input_name_lucid,Net=constrNet)
                    
                    print_imags_for_pretrainedModel(list_layer_index_to_print_base_model,output_path=output_path_with_model,\
                                         constrNet=constrNet)
                        
                     # Do the images for the other models case
                    if doAlsoImagesOfOtherModel_feature:
                         for suffix_local in suffix_tab:
                             if not(suffix_local==suffix):
                                 if model_name+suffix_local in dict_list_layer_index_to_print_base_model.keys():  
                                     print('Do lucid image for other features',suffix_local)
                                     list_layer_index_to_print = dict_list_layer_index_to_print_base_model[model_name+suffix_local]
                                     output_path_with_model_local =  os.path.join(output_path_with_model,'FromOtherTraining'+suffix_local)
                                     pathlib.Path(output_path_with_model_local).mkdir(parents=True, exist_ok=True)
                                     lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
                                                              ,path_output=output_path_with_model_local,prexif_name=model_name+suffix,input_name=input_name_lucid,Net=constrNet)
                    
                else: # In the RandInit case : random initialisation
                    list_weights_initialisation,list_name_layers_init = get_weights_and_name_layers(init_net)
                    dict_layers_relative_diff,dict_layers_argsort = get_gap_between_weights(list_name_layers_init,\
                                                                                list_weights_initialisation,net_finetuned)
                    save_file = os.path.join(output_path_with_model,'dict_layers_relative_diff.pkl')
                    with open(save_file, 'wb') as handle:
                        pickle.dump(dict_layers_relative_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)
                   
                    name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,path=path_lucid_model,suffix=suffix)
                    list_layer_index_to_print_base_model = []
                    list_layer_index_to_print = []
                    num_top = 3
                    for key in dict_layers_argsort.keys():
                        dict_key = dict_layers_argsort[key]
                        for k in range(min(len(dict_key),num_top)):
                            topk = dict_key[k]
                            list_layer_index_to_print += [[key,topk]]
                            list_layer_index_to_print_base_model += [[key,topk]]
                            
                    if testMode:
                        list_layer_index_to_print = [list_layer_index_to_print[0],list_layer_index_to_print[1]]
                        list_layer_index_to_print_base_model = [list_layer_index_to_print_base_model[0],list_layer_index_to_print_base_model[1]]
                    
                    dict_list_layer_index_to_print_base_model[model_name+suffix] = list_layer_index_to_print_base_model
                   
                    lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
                             ,path_output=output_path_with_model,prexif_name=model_name+suffix,input_name=input_name_lucid,Net=constrNet)
        
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
    
def print_DiffRelat_FineTuned_network(constrNet='InceptionV1',list_models_name=None,
                                        suffix_tab=None):
    """
    This function will print the relative difference of the model per layer 
    """    
    
    # Semble diverger dans le cas de InceptionV1  :'RASTA_big01_modif',
    if list_models_name is None:
        list_models_name = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision',
                        'RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch',
                        'RASTA_big001_modif_LastEpoch',
                        'RASTA_small01_modif_dataAug_ep120',
                        'RASTA_small01_modif_deepSupervision_ep120',
                        'RASTA_big001_modif_dataAug',
                        'RASTA_small01_modif_dataAug_ep120_LastEpoch',
                        'RASTA_small01_modif_deepSupervision_ep120_LastEpoch',
                        'RMN_small01_modif',
                        'RMN_small001_modif','RMN_big001_modif',
                        'RMN_small001_modif_deepSupervision_LastEpoch',
                        'RMN_small01_modif_LastEpoch',
                        'RMN_small001_modif','RMN_big001_modif_LastEpoch',
                        'RMN_small001_modif_deepSupervision_LastEpoch',
                        'RASTA_big001_modif_RandInit_ep120']
        
    if suffix_tab is None:
        suffix_tab = ['','1']
        
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet)
    
    for model_name in list_models_name:
        for suffix in suffix_tab:
            output_path_with_model = os.path.join(output_path,model_name+suffix)
            save_file = os.path.join(output_path_with_model,'dict_layers_relative_diff.pkl')
            with open(save_file, 'rb') as handle:
                dict_layers_relative_diff = pickle.load(handle)
            
            
def print_RASTA_performance():
    list_models_name = ['RASTA_small01_modif',
                        'RASTA_small001_modif',
                        'RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision',
                        'RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif_dataAug_ep120',
                        'RASTA_small01_modif_deepSupervision_ep120',
                        'RASTA_big001_modif_dataAug',
                        'RASTA_big001_modif_RandInit_ep120',
                        'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200',
                        'RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200',
                        'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                        'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
                        'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200_cn1',
                        #'RASTA_small01_modif_adam_unfreeze50_SmallDataAug_ep200', # NotImplemented
                        #'RASTA_small01_modif_adam_unfreeze50_SmallDataAug_ep200_cn1'
                        'RASTA_big001_modif_adam_unfreeze50_randomCrop_ep200_cn1',
                        'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200_cn1',
                        'RASTA_big0001_modif_RandInit_deepSupervision_ep200_LRschedG',
                        'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                        'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                        'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
                         'RASTA_big001_modif_Adadelta_unfreeze50_MediumDataAug_ep200'
                        ]
    print_performance_FineTuned_network(constrNet='InceptionV1',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=True)
    
#    list_models_name=['RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                      'RASTA_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
#    print_performance_FineTuned_network(constrNet='InceptionV1_slim',
#                                        list_models_name=list_models_name,
#                                        suffix_tab=[''],latexOutput=True)
#    list_models_name=['RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200']
#    print_performance_FineTuned_network(constrNet='ResNet50',
#                                        list_models_name=list_models_name,
#                                        suffix_tab=[''],latexOutput=True)
def print_IconArtv1_performance():

    list_models_name = ['IconArt_v1_big001_modif_adam_randomCrop_deepSupervision_ep200',
                        'IconArt_v1_big001_modif_Adadelta_unfreeze50_MediumDataAug_ep200',
                        'IconArt_v1_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
                        'IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
                        'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',
                        'IconArt_v1_big001_modif_adam_MediumDataAug_ep200',
                        'IconArt_v1_big001_modif_adam_randomCrop_ep200',
                        'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200',
                         'IconArt_v1_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'IconArt_v1_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'IconArt_v1_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
                        ]

    print_performance_FineTuned_network(constrNet='InceptionV1',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=True)
    
    list_models_name=['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
                      'IconArt_v1_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
    print_performance_FineTuned_network(constrNet='InceptionV1_slim',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=True)
    
def print_performance_FineTuned_network(constrNet='InceptionV1',
                                        list_models_name=None,
                                        suffix_tab=[''],latexOutput=False):
    """
    This function will return and print the metrics / compute them if needed
    for the different dataset 
    """    
    
    # Semble diverger dans le cas de InceptionV1  :'RASTA_big01_modif',
    if list_models_name is None:
        list_models_name = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision',
                        'RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch',
                        'RASTA_big001_modif_LastEpoch',
                        'RASTA_small01_modif_dataAug_ep120',
                        'RASTA_small01_modif_deepSupervision_ep120',
                        'RASTA_big001_modif_dataAug',
                        'RASTA_small01_modif_dataAug_ep120_LastEpoch',
                        'RASTA_small01_modif_deepSupervision_ep120_LastEpoch',
                        'RMN_small01_modif',
                        'RMN_small001_modif','RMN_big001_modif',
                        'RMN_small001_modif_deepSupervision_LastEpoch',
                        'RMN_small01_modif_LastEpoch',
                        'RMN_small001_modif','RMN_big001_modif_LastEpoch',
                        'RMN_small001_modif_deepSupervision_LastEpoch',
                        'RASTA_big001_modif_RandInit_ep120',
                        'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200_cn1',
                        'RASTA_big001_modif_adam_unfreeze50_randomCrop_ep200_cn1',
                        'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200_cn1',
                        'RASTA_big0001_modif_RandInit_deepSupervision_ep200_LRschedG',
                        'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                        'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                        'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200']
        
    # Pour 'RASTA_big001_modif_RandInit_ep120']    
    #    Top-1 accuracy : 42.91%
    #    Top-3 accuracy : 70.09%
    #    Top-5 accuracy : 82.62%
    
    ####  RASTA_big001_modif_dataAug  suffix  1 manquant semble t il 
    
    if suffix_tab is None:
        print("Warning we will define the suffix such as ['','1']")
        suffix_tab = ['','1'] # In order to have more than once the model fine-tuned with some given hyperparameters
    
    K.set_learning_phase(0)
    #with K.get_session().as_default(): 
    #path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    #dict_list_layer_index_to_print_base_model = {}
    
    num_top = 3
    for model_name in list_models_name:
        
        if not(model_name=='random'):
            for suffix in suffix_tab:
                print('#### ',model_name,' ',suffix)
                #output_path_with_model = os.path.join(output_path,model_name+suffix)
                #pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
                
                metrics = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix,
                                               get_Metrics=True,verbose=not(latexOutput))
                
                if not(latexOutput):
                    if not('RASTA' in model_name):
                        AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
                        print('MAP {0:.2f}'.format(np.mean(AP_per_class)))
                    else:
                        top_k_accs,AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class,acc_per_class = metrics
                        for k,top_k_acc in zip([1,3,5],top_k_accs):
                            print('Top-{0} accuracy : {1:.2f}%'.format(k,top_k_acc*100))
                else:
                    latex_str = constrNet.replace('_','\_')  
                    latex_str += ' & ' + model_name.replace('_','\_')
                    if not('RASTA' in model_name):
                        AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
                        latex_str += ' & ' + '{0:.2f}'.format(np.mean(AP_per_class))
                    else:
                        top_k_accs,AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class,acc_per_class = metrics
                        for k,top_k_acc in zip([1,3,5],top_k_accs):
                            latex_str += ' & ' + '{0:.2f}'.format(np.mean(top_k_acc*100))
                            #print('Top-{0} accuracy : {1:.2f}%'.format(k,top_k_acc*100))
                    latex_str += "\\\\"
                    print(latex_str)

def plotHistory_of_training():
    
    path_folder= os.path.join(os.sep,'Users','gonthier','ownCloud','tmp3','Lucid_outputs','history')

    
    # RASTA_big001_modif_RandInit_ep120 suffix none
    name = 'RASTA_big001_modif_RandInit_ep120'
    history_pkl = 'History_InceptionV1_RASTA__RandInit_SGD_lr0.001_avgpool_CropCenter_FT_120_bs32_SGD_sgdm0.9_dec0.0001_BestOnVal.pkl'
    
    # RASTA_small01_modif_ep120 suffix None
    name = 'RASTA_small01_modif_ep120'
    history_pkl = 'History_InceptionV1_RASTA__SGD_lrp0.1_lr0.01_avgpool_CropCenter_FT_120_bs32_SGD_sgdm0.9_dec0.0001_BestOnVal.pkl'
    
    # RASTA_big001_modif_dataAug
    name = 'RASTA_big001_modif_dataAug'
    history_pkl = 'History_InceptionV1_RASTA__SGD_lr0.001_dataAug_avgpool_CropCenter_FT_20_bs32_SGD_sgdm0.9_dec0.0001_BestOnVal.pkl'
    
    name ='IconArt_v1_big001_modif_adam_randomCrop_ep200'
    history_pkl ='History_InceptionV1_IconArt_v1__lr0.001_avgpool_randomCrop_FT_200_bs32_BestOnVal.pkl'
    name ='IconArt_v1_big001_modif_adam_randomCrop_deepSupervision_ep200'
    history_pkl = 'History_InceptionV1_IconArt_v1__deepSupervision_lr0.001_avgpool_randomCrop_FT_200_bs32_BestOnVal.pkl'

    name = 'RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200'
    history_pkl = 'History_InceptionV1_RASTA__deepSupervision_lr0.001_avgpool_randomCrop_FT_200_bs32_BestOnVal.pkl'
    name = 'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200'
    history_pkl = 'History_InceptionV1_RASTA__RandInit_deepSupervision_lr0.001_avgpool_randomCrop_FT_200_bs32_BestOnVal.pkl'

    name ='IconArt_v1_big001_modif_adam_SmallDataAug_ep200'
    history_pkl ='History_InceptionV1_IconArt_v1__lr0.001_SmallDataAug_avgpool_CropCenter_FT_200_bs32_BestOnVal.pkl'
    name ='IconArt_v1_big001_modif_adam_MediumDataAug_ep200'
    history_pkl = 'History_InceptionV1_IconArt_v1__lr0.001_MediumDataAug_avgpool_CropCenter_FT_200_bs32_BestOnVal.pkl'

    name ='IconArt_v1_big001_modif_adam_unfreeze44_SmallDataAug_ep200'
    history_pkl = 'History_InceptionV1_IconArt_v1__lr0.001_SmallDataAug_unfreeze44_avgpool_CropCenter_FT_200_bs32_BestOnVal.pkl'
                        
    name ='IconArt_v1_big001_modif_Adadelta_unfreeze44_cosineloss_MediumDataAug_ep200'
    history_pkl = 'History_InceptionV1_IconArt_v1__Adadelta_cosine_similarity_lr0.001_MediumDataAug_unfreeze44_avgpool_CropCenter_FT_200_bs32_Adadelta_BestOnVal.pkl'

    name = 'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200'
    history_pkl = 'History_InceptionV1_RASTA__lr0.001_SmallDataAug_unfreeze44_avgpool_CropCenter_FT_200_bs32_BestOnVal.pkl'
    
    history_path = os.path.join(path_folder,history_pkl)
    with open(history_path, 'rb') as handle:
        history = pickle.load(handle)
    print(history)
    plt.ion()
    plt.figure()
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss ' + name)
    plt.legend()
    if 'RASTA' in name:
        plt.figure()
        try:
            plt.plot(history['top_1_categorical_accuracy'], label='train')
            plt.plot(history['val_top_1_categorical_accuracy'], label='val')
        except KeyError as e: # deepSupervision certainement
            plt.plot(history['avgpool_prediction_top_1_categorical_accuracy'], label='train')
            plt.plot(history['val_avgpool_prediction_top_1_categorical_accuracy'], label='val')
        plt.title('Top1-Acc ' + name)
        plt.legend()
    else:
        plt.figure()
        try:
            plt.plot(history['acc'], label='train')
            plt.plot(history['val_acc'], label='val')
        except KeyError as e: # deepSupervision certainement
            plt.plot(history['avgpool_prediction_acc'], label='train')
            plt.plot(history['val_avgpool_prediction_acc'], label='val')
            
        plt.title('Acc ' + name)
        plt.legend()
    plt.draw()
    plt.pause(0.001)
    
 
def testFct_PbTrainThenVizu():
    """
    Le but de cette fonciton est de trouver le bug qu il peut y avoir dans la fct  
    Comparaison_of_FineTunedModel quand je cherche a fine-tune puis a visualiser 
    les filtres
    """
    # list_models_name = ['IconArt_v1_big01_modif_RandInit_deepSupervision_ep1_LRschedG_dropout070704',
    #                     'IconArt_v1_big01_modif_RandInit_randomCrop_deepSupervision_ep1_RedLROnPlat_dropout070704']
    
    # Comparaison_of_FineTunedModel(list_models_name,testMode=True,constrNet='InceptionV1')
    
    list_model_name_5 = ['RASTA_small01_modif'
                        ]
    Comparaison_of_FineTunedModel(list_model_name_5,constrNet='ResNet50',testMode=True) 
    

    
if __name__ == '__main__': 
#    print_performance_FineTuned_network(constrNet='InceptionV1',
#                                        list_models_name = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
#                            'RASTA_big001_modif_dataAug','RMN_small01_modif_randomCrop'],
#    
#                                        suffix_tab = [''])
#    print_performance_FineTuned_network(constrNet='InceptionV1',
#                                        list_models_name = ['RMN_small01_modif',
#                                                            'RMN_small001_modif','RMN_big001_modif',
#                                                            'RMN_small001_modif_deepSupervision',
#                                                            'RMN_small01_modif_LastEpoch',
#                                                            'RMN_small001_modif_LastEpoch','RMN_big001_modif_LastEpoch',
#                                                            'RMN_small001_modif_deepSupervision_LastEpoch',
#                                                            'RMN_small01_modif_randomCrop'],
#    
#                                        suffix_tab = [''])
    
#    list_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
#                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
#                        'RASTA_small01_modif','RASTA_big01_modif',
#                        'RASTA_small001_modif','RASTA_big001_modif',
#                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision']
#    
#    list_models_name = ['IconArt_v1_small001_modif','IconArt_v1_big001_modif',
#                        'IconArt_v1_small001_modif_deepSupervision','IconArt_v1_big001_modif_deepSupervision',
#                        'RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
#                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision']
#    # Semble diverger dans le cas de InceptionV1  :'RASTA_big01_modif',
#    list_models_name = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
#                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision',
#                        'RASTA_small01_modif_LastEpoch','RASTA_small001_modif_LastEpoch','RASTA_big001_modif_LastEpoch']
#    list_models_name = ['RMN_small01_modif',
#                        'RMN_small001_modif','RMN_big001_modif',
#                        'RMN_small001_modif_deepSupervision',
#                        'RASTA_small01_modif_dataAug_ep120',
#                        'RASTA_small01_modif_deepSupervision_ep120',
#                        'RASTA_big001_modif_dataAug',
#                        'RASTA_small01_modif_dataAug_ep120_LastEpoch',
#                        'RASTA_small01_modif_deepSupervision_ep120_LastEpoch',
#                        'RMN_small01_modif_randomCrop',
#                        'RASTA_big001_modif_RandInit_ep120',
#                        'RASTA_big001_modif_RandInit_ep120_LastEpoch',
#                        'IconArt_v1_big001_modif_adam_randomCrop_ep200',
#                        'IconArt_v1_big001_modif_adam_randomCrop_ep200_LastEpoch',
#                        'IconArt_v1_big001_modif_adam_randomCrop_deepSupervision_ep200',
#                        'IconArt_v1_big001_modif_adam_randomCrop_deepSupervision_ep200_LastEpoch',
#                        'RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200',
#                        'RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200_LastEpoch',
#                        'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200',
#                        'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200_LastEpoch'
#                        ]
#    
#    # En fait pour Celebra :
#    # model.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['binary_accuracy'])
#    
#    # Pour Flower :
#    # model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#    
#    #list_models_name = ['IconArt_v1_big001_modif_adam_randomCrop_deepSupervision_ep1']
#    list_models_name = [
#                        'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200',
#                        'RASTA_big001_modif_adam_unfreeze44_SmallDataAug_ep200_LastEpoch',
#                        'RASTA_big001_modif_Adadelta_unfreeze44_cosineloss_MediumDataAug_ep200',
#                        'RASTA_big001_modif_Adadelta_unfreeze44_cosineloss_MediumDataAug_ep200_LastEpoch',
#                        'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200',
#                        'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200_LastEpoch',
#                        'RASTA_big001_modif_Adadelta_RandInit_MediumDataAug_ep200',
#                        'RASTA_big001_modif_Adadelta_RandInit_MediumDataAug_ep200_LastEpoch',
#                        ]
#    # Car on a juste pas converger pour RASTA_big001_modif_dataAug_ep120
#
#    list_models_name_afaireplusTard = [
#                        'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',
#                        'IconArt_v1_big001_modif_adam_SmallDataAug_ep200_LastEpoch',
#                        'IconArt_v1_big001_modif_adam_MediumDataAug_ep200',
#                        'IconArt_v1_big001_modif_adam_MediumDataAug_ep200_LastEpoch',
#                        'IconArt_v1_big001_modif_adam_ep200',
#                        'IconArt_v1_big001_modif_adam_ep200_LastEpoch',
#                        'RASTA_big001_modif_RandInit_deepSupervision_ep120',
#                        'RASTA_big001_modif_RandInit_deepSupervision_ep120_LastEpoch',
#                        'RASTA_big001_modif_dataAug_ep120',
#                        'RASTA_big001_modif_dataAug_ep120_LastEpoch',
#                        'RASTA_big001_modif_RandInit_randomCrop_ep120',
#                        'RASTA_big001_modif_RandInit_randomCrop_ep120_LastEpoch',
#                        'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep120',
#                        'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep120_LastEpoch',
#                        'RASTA_big01_modif_RandInit_ep120',
#                        'RASTA_big01_modif_RandInit_ep120_LastEpoch',
#                        'RASTA_big01_modif_RandInit_randomCrop_ep120',
#                        'RASTA_big01_modif_RandInit_randomCrop_ep120_LastEpoch',
#                        'RMN_small001_modif_randomCrop','RMN_big001_modif_randomCrop',
#                        'RASTA_small01_modif_randomCrop',
#                        'RASTA_small01_modif_randomCrop_ep120'
#                        ]
     
#    list_models_name_slim = ['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep1']
    # Juste pour faire les cas du reseau de départ de ImageNet
    # list_models_name_slim = ['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
    #                          'IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200_LastEpoch',
    #                          'IconArt_v1_big001_modif_Adadelta_unfreeze84_cosineloss_MediumDataAug_ep200',
    #                          'IconArt_v1_big001_modif_Adadelta_unfreeze84_cosineloss_MediumDataAug_ep200_LastEpoch',
    #                          'RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
    #                          'RASTA_big001_modif_adam_unfreeze84_SmallDataAug_ep200_LastEpoch',
    #                          'RASTA_big001_modif_Adadelta_unfreeze84_cosineloss_MediumDataAug_ep200',
    #                          'RASTA_big001_modif_Adadelta_unfreeze84_cosineloss_MediumDataAug_ep200_LastEpoch']
    
    # Comparaison_of_FineTunedModel(list_models_name_slim,constrNet='InceptionV1_slim')    

#    list_model_name_1 = ['IconArt_v1_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                             'IconArt_v1_big001_modif_adam_unfreeze50_SmallDataAug_ep200_LastEpoch',
#                             'IconArt_v1_big001_modif_Adadelta_unfreeze50_cosineloss_MediumDataAug_ep200',
#                             'IconArt_v1_big001_modif_Adadelta_unfreeze50_cosineloss_MediumDataAug_ep200_LastEpoch',
#                             'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                             'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200_LastEpoch',
#                             'RASTA_big001_modif_Adadelta_unfreeze50_cosineloss_MediumDataAug_ep200',
#                             'RASTA_big001_modif_Adadelta_unfreeze50_cosineloss_MediumDataAug_ep200_LastEpoch',
#                             'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200',
#                             'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200_LastEpoch']
#    list_model_name_2 = ['IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200',
#                         'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200_LastEpoch',
#                         'IconArt_v1_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                         'IconArt_v1_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_LastEpoch',
#                         'IconArt_v1_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                         'IconArt_v1_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_LastEpoch',
#                         'IconArt_v1_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                         'IconArt_v1_big0001_modif_adam_unfreeze50_SmallDataAug_ep200_LastEpoch',
#                         'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                         'RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_LastEpoch',
#                         'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                         'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_LastEpoch',
#                         'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200',
#                         'RASTA_big0001_modif_adam_unfreeze50_SmallDataAug_ep200_LastEpoch'
#                         ]
#    # RandForUnfreezed
#
##    list_model_name_1 = ['RASTA_big001_modif_Adadelta_unfreeze44_cosineloss_MediumDataAug_ep200',
##                        'RASTA_big001_modif_Adadelta_unfreeze44_cosineloss_MediumDataAug_ep200_LastEpoch',
##                        'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200',
##                        'IconArt_v1_big001_modif_adam_RandInit_SmallDataAug_ep200_LastEpoch']
#    Comparaison_of_FineTunedModel(list_model_name_2,constrNet='InceptionV1') 
        
        # 'RASTA_big001_modif_adam_unfreeze8_SmallDataAug_ep200',
        # a été entraine mais pas encore de visualisaton de feature 
    
       # Suite aux recherches sur le fine tuning et le training de model : cela a ete il faudra analyser les resultats
#    liste_possible_better = ['RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200_cn1',
#                             'RASTA_small01_modif_adam_unfreeze50_SmallDataAug_ep200',
#                             'RASTA_small01_modif_adam_unfreeze50_SmallDataAug_ep200_cn1'
#                             'RASTA_big001_modif_adam_unfreeze50_randomCrop_ep200_cn1',
#                             'RASTA_big001_modif_adam_RandInit_randomCrop_deepSupervision_ep200_cn1'
#                             ]    
#    Comparaison_of_FineTunedModel(liste_possible_better,constrNet='InceptionV1') 
    
    # Il pour essayer de faire un entrainement depuis zero avec un Inception V1
    # On va devoir relancer ces modeles a cause du soucis de fine tuning avec le bug clipnorm
#     liste_possible_fromScatch = []
    # cela a ete fait 
     # liste_possible_fromScatch = ['RASTA_big0001_modif_RandInit_deepSupervision_ep200_LRschedG',
     #                              'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
     # Ca a plante la a cause de la visualisation et je ne sais pas pourquoi .... 
     # Faut il faire les entrainements puis ensuite les visualisations ??? 
    # Il y a un soucis quand je cherche a visualiser les features et je en sais pas pourquoi !!! 
    #  liste_possible_fromScatch = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG'] 
    #  # visu a faire !!!
    #  liste_possible_fromScatch = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
    #                              'RASTA_big01_modif_RandInit_deepSupervision_ep200_LRschedG_dropout070704',
    #                               'RASTA_big01_modif_RandInit_randomCrop_deepSupervision_ep200_RedLROnPlat_dropout070704',
    #                               'RASTA_big01_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_dropout070704',
    #                               'RASTA_big001_modif_RandInit_deepSupervision_ep200_LRschedG_dropout070704',
    #                               'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_dropout070704']
    #  Comparaison_of_FineTunedModel(liste_possible_fromScatch,constrNet='InceptionV1') 
    
    # # Cela a faire : 
    
    # # list_model_name_I = ['RASTA_small01_modif_ep200',
    # #                      'RASTA_small001_modif_ep200']
    # # list_model_name_I = ['RASTA_big0001_modif_RandInit_deepSupervision_ep200']
    # # Comparaison_of_FineTunedModel(list_model_name_I,constrNet='InceptionV1') 
    
    # #list_model_name_5 = ['RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200']

    #  list_model_name_5 = ['RASTA_small01_modif',
    #                       'RASTA_big001_modif_adam_unfreeze50_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze20_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze20_SmallDataAug_ep200',
    #                     ]
    #  Comparaison_of_FineTunedModel(list_model_name_5,constrNet='ResNet50') 
    # InceptionV1 and ResNet50 models have been trained => need to look at the results ! 
    #Test avec RMSprop non fait !
     list_model_name_4 = ['RASTA_big001_modif_adam_unfreeze8_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze8_ep200',
                         'RASTA_big0001_modif_adam_unfreeze8_SmallDataAug_ep200',
                         'RASTA_big0001_modif_adam_unfreeze8_ep200',
                         'RASTA_big001_modif_RMSprop_unfreeze8_SmallDataAug_ep200',
                         'RASTA_big0001_modif_RMSprop_unfreeze8_SmallDataAug_ep200',
                        ]
     Comparaison_of_FineTunedModel(list_model_name_4,constrNet='VGG') 
#    
    # Test pour voir si la visualisation est possible pour les autres modeles : ResNet, VGG  Ok  
    # With unfreeze and without
#    list_model_name_5 = [
#                     'IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep1'
#                    ]
#    Comparaison_of_FineTunedModel(list_model_name_5,constrNet='InceptionV1_slim') 
#    list_model_name_4 = ['RASTA_big001_modif_adam_unfreeze8_SmallDataAug_ep200'
#                        ]
#    Comparaison_of_FineTunedModel(list_model_name_4,constrNet='VGG') 
    
    
    

    
    
        