# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:39:41 2020

The goal of this script is to compute the mean value of each features maps of 
the whole image from a given training dataset for a given network

Remarques tu peux parfois avoir l'erreur suivante : 
UnknownError: 2 root error(s) found.(0) Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
    
Il te faudra alors peut etre vider le cache qui se trouve a l'endroit suivant : 
    Users gonthier AppData Roaming NVIDIA ComputeCache

@author: gonthier
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D,Activation,Concatenate
from tensorflow.python.keras import Model

import numpy as np
import platform
import pathlib
import os
import pickle
import matplotlib

from StatsConstr_ClassifwithTL import predictionFT_net
import Stats_Fcts
from googlenet import inception_v1_oldTF as Inception_V1
from IMDB import get_database
from plots_utils import plt_multiple_imgs
from CompNet_FT_lucidIm import get_fine_tuned_model

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pandas as pd
import tikzplotlib

CB_color_cycle = ['#377eb8', '#ff7f00','#984ea3', '#4daf4a','#A2C8EC','#e41a1c',
                  '#f781bf', '#a65628', '#dede00','#FFBC79','#999999']

def get_Network(Net):
    weights = 'imagenet'
    
    if Net=='VGG':
        imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
    elif Net == 'InceptionV1':
        imagenet_model = Inception_V1(include_top=False, weights=weights)
    else:
        raise(NotImplementedError)
        
    return(imagenet_model)

def get_Model_that_output_StatsOnActivation_forGivenLayers(model,
                                                           list_layers,
                                                           stats_on_layer='mean',
                                                           list_means=None):
    """
    Provide a keras model which outputs the stats_on_layer == mean or max of each 
    features maps
    """
    if stats_on_layer=='cov_global_mean':
        assert(not(list_means is None))
        assert(len(list_means)==len(list_layers))
    list_outputs = []
    
    i= 0
    for layer in model.layers:
        if  layer.name in list_layers :
            layer_output = layer.output
            if stats_on_layer=='mean':
                stats_each_feature = tf.keras.backend.mean(layer_output, axis=[1,2], keepdims=False)
            elif stats_on_layer=='meanAfterRelu':
                stats_each_feature = tf.keras.backend.mean(tf.keras.activations.relu(layer_output), axis=[1,2], keepdims=False)
            elif stats_on_layer=='max':
                stats_each_feature = tf.keras.backend.max(layer_output, axis=[1,2], keepdims=False)
            elif stats_on_layer=='min':
                stats_each_feature = tf.keras.backend.min(layer_output, axis=[1,2], keepdims=False)
            elif stats_on_layer=='meanFirePos':
                stats_each_feature = tf.keras.backend.mean(fct01(layer_output), axis=[1,2], keepdims=False)
            elif stats_on_layer=='meanFirePos_minusMean':
                means = list_means[i]
                i+=1
                stats_each_feature = tf.keras.backend.mean(fct01(layer_output-means), axis=[1,2], keepdims=False)
            elif stats_on_layer=='max&min':
                maxl = tf.keras.backend.max(layer_output, axis=[1,2], keepdims=False)
                minl = tf.keras.backend.min(layer_output, axis=[1,2], keepdims=False)
                stats_each_feature = [maxl,minl]
            elif stats_on_layer== 'cov_instance_mean':
                stats_each_feature = Stats_Fcts.covariance_mean_matrix_only(layer_output)[0]
            elif stats_on_layer=='cov_global_mean':
                means = list_means[i]
                i+=1
                stats_each_feature = Stats_Fcts.covariance_matrix_only(layer_output,means)
            elif stats_on_layer== 'gram':
                stats_each_feature = Stats_Fcts.gram_matrix_only(layer_output)
            else:
                raise(ValueError(stats_on_layer+' is unknown'))
            list_outputs += [stats_each_feature]
            
    new_model = Model(model.input,list_outputs)
    
    return(new_model)
 
def fct01(x):
    """
    This function return 0 if x is inferior or equal to 0 and 1 otherwise
    """
    # fct01(0) currently returns 0.
    sign = tf.sign(x)
    step_func = tf.maximum(0.0, sign)
    return step_func
    
def get_Model_that_output_StatsOnActivation(model,stats_on_layer='mean'):
    """
    Provide a keras model which outputs the stats_on_layer == mean or max of each 
    features maps
    """
    
    list_outputs = []
    list_outputs_name = []
    
    for layer in model.layers:
        if  isinstance(layer, Conv2D) or isinstance(layer,Concatenate) or isinstance(layer,Activation):
            layer_output = layer.output
            if stats_on_layer=='mean':
                stats_each_feature = tf.keras.backend.mean(layer_output, axis=[1,2], keepdims=False)
            elif stats_on_layer=='meanAfterRelu':
                stats_each_feature = tf.keras.backend.mean(tf.keras.activations.relu(layer_output), axis=[1,2], keepdims=False)
            elif stats_on_layer=='meanFirePos':
                stats_each_feature = tf.keras.backend.mean(fct01(layer_output), axis=[1,2], keepdims=False)
            elif stats_on_layer=='max':
                stats_each_feature = tf.keras.backend.max(layer_output, axis=[1,2], keepdims=False)
            elif stats_on_layer=='min':
                stats_each_feature = tf.keras.backend.min(layer_output, axis=[1,2], keepdims=False)
            else:
                raise(ValueError(stats_on_layer+' is unknown'))
            list_outputs += [stats_each_feature]
            list_outputs_name += [layer.name]
            
    new_model = Model(model.input,list_outputs)
    
    return(new_model,list_outputs_name)
    
def get_Model_that_output_Activation(model,list_layers=None):
    """
    Provide a keras model which outputs the stats_on_layer == mean or max of each 
    features maps
    """
    
    list_outputs = []
    list_outputs_name = []
    
    for layer in model.layers:
        if list_layers is None:
            if  isinstance(layer, Conv2D) or isinstance(layer,Concatenate) or isinstance(layer,Activation):
                layer_output = layer.output
                list_outputs += [layer_output]
                list_outputs_name += [layer.name]
        else:
            if layer.name in list_layers:
                layer_output = layer.output
                list_outputs += [layer_output]
                list_outputs_name += [layer.name]
            
    new_model = Model(model.input,list_outputs)
    
    return(new_model,list_outputs_name)
    
    
def compute_OneValue_Per_Feature(dataset,model_name,constrNet,stats_on_layer='mean',
                                 suffix='',cropCenter = True,FTmodel=True):
    """
    This function will compute the mean activation of each features maps for all
    the convolutionnal layers 
    @param : FTmodel : in the case of finetuned from scratch if False use the initialisation
    networks
    """
    K.set_learning_phase(0) #IE no training
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']

    extra_str = ''
    
    if 'XX' in model_name:
        splittedXX = model_name.split('XX')
        weights = splittedXX[1]
        model_name_wo_oldModel = model_name.replace('_XX'+weights+'XX','')
    else:
        model_name_wo_oldModel = model_name
    if dataset in model_name_wo_oldModel:
        dataset_str = ''
    else:
        dataset_str = dataset
        
    if model_name=='pretrained':
        base_model = get_Network(constrNet)
    else:
        # Pour ton windows il va falloir copier les model .h5 finetuné dans ce dossier la 
        # C:\media\gonthier\HDD2\output_exp\Covdata\RASTA\model
        if 'RandInit' in model_name:
            FT_model,init_model = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if FTmodel:
                base_model = FT_model
            else:
                extra_str = '_InitModel'
                base_model = init_model 
        else:
            output = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if len(output)==2:
                base_model, init_model = output
            else:
                base_model = output
    model,list_outputs_name = get_Model_that_output_StatsOnActivation(base_model,stats_on_layer=stats_on_layer)
    #print(model.summary())
    activations = predictionFT_net(model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter,verbose_predict=1)
    print('activations len and shape of first element',len(activations),activations[0].shape)
    
    folder_name = model_name+suffix
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,folder_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,folder_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    act_plus_layer = [list_outputs_name,activations]
    if stats_on_layer=='mean':
        save_file = os.path.join(output_path,'activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='meanAfterRelu':
        save_file = os.path.join(output_path,'meanAfterRelu_activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='max':
        save_file = os.path.join(output_path,'max_activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='min':
        save_file = os.path.join(output_path,'min_activations_per_img'+extra_str+dataset_str+'.pkl')
    else:
        raise(ValueError(stats_on_layer+' is unknown'))
    with open(save_file, 'wb') as handle:
        pickle.dump(act_plus_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return(list_outputs_name,activations)
  
def dead_kernel_QuestionMark(dataset,model_name,constrNet,fraction = 1.0,suffix=''):
    """
    This function will see if some of the kernel are fired (positive activation)
    by none of the training images 
    AND
    if some images don't fire (positive activation) none of the filters of a 
    given layer
    """
    cropCenter = True
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    name_images = df_train[item_name].values
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name+suffix)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+suffix)
    # For images
    output_path_for_img = os.path.join(output_path,'ActivationsImages')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_img).mkdir(parents=True, exist_ok=True) 
        
    save_file = os.path.join(output_path,'max_activations_per_img.pkl')
    
    if os.path.exists(save_file):
        # The file exist
        with open(save_file, 'rb') as handle:
            act_plus_layer = pickle.load(handle)
            [list_outputs_name,activations] = act_plus_layer
    else:
        list_outputs_name,activations = compute_OneValue_Per_Feature(dataset,
                                            model_name,constrNet,stats_on_layer='max',
                                            cropCenter=cropCenter)
    
    
    
    # Loop on the layers
    for layer_name_inlist,activations_l in zip(list_outputs_name,activations):
        #print('Layer',layer_name_inlist)
        # activations_l line = training image / column = feature
        num_training_ex,num_features = activations_l.shape
        
        # Loop on the features : Je sais mon code est sous optimial
        list_dead_features = []
        for num_feature in range(num_features):
            activations_l_f = activations_l[:,num_feature]
            where_max_activation_is_neg = np.where(activations_l_f<=0)[0]
            if len(where_max_activation_is_neg) >= fraction*num_training_ex:
                list_dead_features += [num_feature]
        if len(list_dead_features) >0:
            print('==>',layer_name_inlist,list_dead_features,' are negative for ',fraction*100,' % of the images of the training set of',dataset)
            print(len(list_dead_features),'on ',num_features,'features')
        else:
            print('No dead kernel for layer :',layer_name_inlist)
        noFire = False
        image_that_dontfire = 0
        for number_img in range(num_training_ex):
            activations_l_i = activations_l[number_img,:]
            where_max_activation_is_neg = np.where(activations_l_i<=0)[0]
            if len(where_max_activation_is_neg) == num_features:
#                print('==>',name_images[number_img],'has all is activation negative at the layer',layer_name_inlist)
                noFire = True
                image_that_dontfire += 1
        if not(noFire):
            print('No image that doesn t fire for this layer :',layer_name_inlist)
            
 
def get_list_activations(dataset,output_path,stats_on_layer,
                         model_name,constrNet,suffix,cropCenter,FTmodel,
                         model_name_wo_oldModel):
    
    if dataset in model_name_wo_oldModel:
        dataset_str = ''
    else:
        dataset_str = dataset
    
    if not(FTmodel):
        extra_str = '_InitModel'
    else:
        extra_str = ''
    
    # Load the activations for the main model :
    if stats_on_layer=='mean':
        save_file = os.path.join(output_path,'activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='meanAfterRelu':
        save_file = os.path.join(output_path,'meanAfterRelu_activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='max':
        save_file = os.path.join(output_path,'max_activations_per_img'+extra_str+dataset_str+'.pkl')
    elif stats_on_layer=='min':
        save_file = os.path.join(output_path,'min_activations_per_img'+extra_str+dataset_str+'.pkl')
    else:
        raise(ValueError(stats_on_layer+' is unknown'))
        
    print(save_file)
        
    if os.path.exists(save_file):
        # The file exist
        with open(save_file, 'rb') as handle:
            act_plus_layer = pickle.load(handle)
            [list_outputs_name,activations] = act_plus_layer
    else:
        list_outputs_name,activations = compute_OneValue_Per_Feature(dataset,
                                            model_name,constrNet,suffix=suffix,
                                            stats_on_layer=stats_on_layer,
                                            cropCenter=cropCenter,
                                            FTmodel=FTmodel)
    return(list_outputs_name,activations)
 
def proportion_labels(list_most_pos_images,dataset,verbose=True):
    """
    The goal of this fct is to provide the labels proportion in the list of image 
    provide
    """
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    
    number_im = len(list_most_pos_images)
    list_labels = [0]*num_classes
    for im_name in list_most_pos_images:
        labels_image_i =  df_train[df_train[item_name]==im_name][classes].values
        list_labels += np.ravel(labels_image_i)
    argsort_from_max_to_min = np.argsort(list_labels)[::-1]
    dico_c = {}
    for index_c in argsort_from_max_to_min:
        classe = classes[index_c]
        number_im_c = list_labels[index_c]
        if number_im_c >0:
            per_c = number_im_c/number_im*100.0
            if verbose: print(classe,per_c)
            dico_c[classe] = per_c
    return(dico_c)
    
def plot_images_Pos_Images(dataset,model_name,constrNet,
                            layer_name='mixed4d_3x3_bottleneck_pre_relu',
                            num_feature=64,
                            numberIm=9,stats_on_layer='mean',suffix='',
                            FTmodel=True,
                            output_path_for_img=None,
                            cropCenter = True,
                            alreadyAtInit=False,
                            ReDo=False):
    """
    This function will plot k image a given layer with a given features number
    @param : in the case of a trained (FT) model from scratch FTmodel == False will lead to 
        use the initialization model
    """
    
    printNearZero = False

    model_name_wo_oldModel,weights = get_model_name_wo_oldModel(model_name)
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    name_images = df_train[item_name].values
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name+suffix)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+suffix)
    # For images
    if output_path_for_img is None:
        output_path_for_img = os.path.join(output_path,'ActivationsImages')
    else:
        output_path_for_img = os.path.join(output_path_for_img,'ActivationsImages')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_img).mkdir(parents=True, exist_ok=True) 
        
    list_outputs_name,activations = get_list_activations(dataset,
                                                         output_path,stats_on_layer,
                                                         model_name,constrNet,
                                                         suffix,cropCenter,FTmodel,
                                                         model_name_wo_oldModel)
        
    if alreadyAtInit: # Load the activation on the initialisation model
        if weights == 'pretrained':
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,'pretrained')
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'pretrained')
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path_init,stats_on_layer,
                                                        'pretrained',constrNet,
                                                        '',cropCenter,FTmodel,
                                                        'pretrained')
        elif weights == 'random':
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path,stats_on_layer,
                                                        model_name,constrNet,
                                                        suffix,cropCenter,False,
                                                        model_name) # FTmodel = False to get the initialisation model
        else:
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,weights)
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,weights)
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path,stats_on_layer,
                                                        weights,constrNet,
                                                        '',cropCenter,FTmodel,
                                                        weights)
            
    
    for layer_name_inlist,activations_l in zip(list_outputs_name,activations):
        if layer_name==layer_name_inlist:
            print('===',layer_name,num_feature,'===')
            activations_l_f = activations_l[:,num_feature]
            where_activations_l_f_pos = np.where(activations_l_f>0)[0]
            if len(where_activations_l_f_pos)==0:
                print('No activation positive for this layer')
                print(activations_l_f)
                continue
            activations_l_f_pos = activations_l_f[where_activations_l_f_pos]
            name_images_l_f_pos = name_images[where_activations_l_f_pos]
            argsort = np.argsort(activations_l_f_pos)[::-1]
            # Most positive images
            list_most_pos_images = name_images_l_f_pos[argsort[0:numberIm]]
            act_most_pos_images = activations_l_f_pos[argsort[0:numberIm]]
            
            if alreadyAtInit:
                activations_init_l = activations_init[list_outputs_name_init.index(layer_name)]
                activations_init_l_f = activations_init_l[:,num_feature]
                where_activations_init_l_f_pos = np.where(activations_init_l_f>0)[0]
                if len(where_activations_init_l_f_pos)==0:
                    print('No activation positive for this layer')
                    print(activations_l_f)
                    continue
                activations_init_l_f_pos = activations_init_l_f[where_activations_init_l_f_pos]
                name_images_init_l_f_pos = name_images[where_activations_init_l_f_pos]
                argsort_init = np.argsort(activations_init_l_f_pos)[::-1]
                # Most positive images
                list_most_pos_images_init = list(name_images_init_l_f_pos[argsort_init[0:numberIm]])
            else:
                list_most_pos_images_init = []
                
#            print(len(list_most_pos_images))
#            print(len(list_most_pos_images_init))
#            print('!!! intersection len',len(list(set(list_most_pos_images) & set(list_most_pos_images_init))))                
            # Plot figures
            
            if alreadyAtInit:
                number_img_intersec = len(list(set(list_most_pos_images) & set(list_most_pos_images_init)))
                percentage_intersec = number_img_intersec/numberIm*100.0
            
            title_imgs = []
            for act in act_most_pos_images:
                str_act = '{:.02f}'.format(act)
                title_imgs += [str_act]
            name_fig = dataset+'_'+layer_name+'_'+str(num_feature)+'_Most_Pos_Images_NumberIm'+str(numberIm)
            if not(stats_on_layer=='mean'):
                name_fig += '_'+stats_on_layer
            if not(FTmodel):
                name_fig += '_InitModel'
            if alreadyAtInit:
                name_fig += '_GreenIfInInit'
            
            name_output = os.path.join(output_path_for_img,name_fig+'.png')
            if not(os.path.isfile(name_output)) or ReDo:
                plt_multiple_imgs(list_images=list_most_pos_images,path_output=output_path_for_img,\
                              path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
                              Net=None,title_imgs=title_imgs,roundColor=list_most_pos_images_init)
            if alreadyAtInit:
                print(output_path_for_img,name_fig,'Percentage intersection : ',percentage_intersec)
                
            proportion_labels(list_most_pos_images,dataset)
            
#            # Slightly positive images : TODO
#            list_slightly_pos_images = name_images_l_f_pos[argsort[-numberIm:]]
#            act_slightly_pos_images = activations_l_f_pos[argsort[-numberIm:]]
#            title_imgs = []
#            for act in act_slightly_pos_images:
#                str_act = '{:.02f}'.format(act)
#                title_imgs += [str_act]
#            name_fig = dataset+'_'+layer_name+'_'+str(num_feature) +'_Slightly_Pos_Images_NumberIm'+str(numberIm)
#            plt_multiple_imgs(list_images=list_slightly_pos_images,path_output=output_path_for_img,\
#                              path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
#                              Net=None,title_imgs=title_imgs)
            
            # Positive near zero images
            if printNearZero:
                list_nearZero_pos_images = name_images_l_f_pos[argsort[-numberIm:]]
                act_nearZero_pos_images = activations_l_f_pos[argsort[-numberIm:]]
                title_imgs = []
                for act in act_nearZero_pos_images:
                    str_act = '{:.02f}'.format(act)
                    title_imgs += [str_act]
                name_fig = dataset+'_'+layer_name+'_'+str(num_feature) +'_Near_Zero_Pos_Images_NumberIm'+str(numberIm)
                if not(stats_on_layer=='mean'):
                    name_fig += '_'+stats_on_layer
                if not(FTmodel):
                    name_fig += '_InitModel'
                plt_multiple_imgs(list_images=list_nearZero_pos_images,path_output=output_path_for_img,\
                                  path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
                                  Net=None,title_imgs=title_imgs)
 
def get_model_name_wo_oldModel(model_name):
    if 'XX' in model_name:
        splittedXX = model_name.split('XX')
        weights = splittedXX[1] # original model
        model_name_wo_oldModel = model_name.replace('_XX'+weights+'XX','')
    else:
        model_name_wo_oldModel = model_name
        if 'RandForUnfreezed' in model_name or 'RandInit' in model_name:
            weights = 'random'
        else:
            weights = 'pretrained'
            
    return(model_name_wo_oldModel,weights)

def doing_overlaps_for_paper():
    
    numberIm_list = [100,1000,-1]
    model_list = ['RASTA_small01_modif']
    model_list = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    for model_name in model_list:
        for numberIm in numberIm_list:
            overlapping_rate_print(dataset='RASTA',model_name=model_name,
                               constrNet='InceptionV1',numberIm=numberIm,
                               stats_on_layer='meanAfterRelu')# because meanAfterRelu already computed
               
def overlapping_rate_print(dataset,model_name,constrNet='InceptionV1',
                      list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b'],
                            numberIm=100,stats_on_layer='mean',suffix='',
                            FTmodel=True,
                            output_path_for_dico=None,
                            cropCenter = True,
                            ReDo=False):
    """
    This function will compute the overlapping ratio between the top k images with 
    positive activation before and after the trainingon train set images
    @param : numberIm = top k images used : numberIm = -1 if you want to take all the images
    @param : in the case of a trained (FT) model from scratch FTmodel == False will lead to 
        use the initialization model
    """
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")


    model_name_wo_oldModel,weights = get_model_name_wo_oldModel(model_name)
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    name_images = df_train[item_name].values
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name+suffix)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+suffix)
    # For images
    if output_path_for_dico is None:
        output_path_for_dico = os.path.join(output_path,'Overlapping')
    else:
        output_path_for_dico = os.path.join(output_path_for_dico,'Overlapping')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_dico).mkdir(parents=True, exist_ok=True) 

    dico_percentage_intersec_list = {}
    
    name_dico = 'OverRatio_'+str(numberIm)
    name_path_dico = os.path.join(output_path_for_dico,name_dico+'.pkl')
    
    
    if os.path.exists(name_path_dico) and not(ReDo):
        # The file exist
        with open(name_path_dico, 'rb') as handle:
            dico_percentage_intersec_list = pickle.load(handle)
    else:
    
        list_outputs_name,activations = get_list_activations(dataset,
                                                         output_path,stats_on_layer,
                                                         model_name,constrNet,
                                                         suffix,cropCenter,FTmodel,
                                                         model_name_wo_oldModel)
        
        # Load the activation on the initialisation model # 
        if weights == 'pretrained':
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,'pretrained')
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'pretrained')
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path_init,stats_on_layer,
                                                        'pretrained',constrNet,
                                                        '',cropCenter,FTmodel,
                                                        'pretrained')
        elif weights == 'random':
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path,stats_on_layer,
                                                        model_name,constrNet,
                                                        suffix,cropCenter,False,
                                                        model_name) # FTmodel = False to get the initialisation model
        else:
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,weights)
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,weights)
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            list_outputs_name_init,activations_init= get_list_activations(dataset,
                                                        output_path,stats_on_layer,
                                                        weights,constrNet,
                                                        '',cropCenter,FTmodel,
                                                        weights)
        
        for layer_name_inlist,activations_l in zip(list_outputs_name,activations):
            if layer_name_inlist in list_layers:
                print('===',layer_name_inlist,'===')
                num_im, number_of_features = activations_l.shape
                
                percentage_intersec_list = []
                
                for num_feature in range(number_of_features):
                    activations_l_f = activations_l[:,num_feature]
                    where_activations_l_f_pos = np.where(activations_l_f>0)[0]
                    if len(where_activations_l_f_pos)==0:
                        print('No activation positive for this layer')
                        #print(activations_l_f)
                        percentage_intersec_f = 0
                        percentage_intersec_list += [percentage_intersec_f]
                        continue
                    activations_l_f_pos = activations_l_f[where_activations_l_f_pos]
                    name_images_l_f_pos = name_images[where_activations_l_f_pos]
                    argsort = np.argsort(activations_l_f_pos)[::-1]
                    # Most positive images
                    list_most_pos_images = name_images_l_f_pos[argsort[0:numberIm]]
                    #act_most_pos_images = activations_l_f_pos[argsort[0:numberIm]]
                
                    activations_init_l = activations_init[list_outputs_name_init.index(layer_name_inlist)]
                    activations_init_l_f = activations_init_l[:,num_feature]
                    where_activations_init_l_f_pos = np.where(activations_init_l_f>0)[0]
                    if len(where_activations_init_l_f_pos)==0:
                        print('No activation positive for this layer')
                        #print(activations_l_f)
                        percentage_intersec_f = 0
                        percentage_intersec_list += [percentage_intersec_f]
                        continue
    
                    activations_init_l_f_pos = activations_init_l_f[where_activations_init_l_f_pos]
                    name_images_init_l_f_pos = name_images[where_activations_init_l_f_pos]
                    argsort_init = np.argsort(activations_init_l_f_pos)[::-1]
                    # Most positive images
                    list_most_pos_images_init = list(name_images_init_l_f_pos[argsort_init[0:numberIm]])
                    
                    number_img_intersec = len(list(set(list_most_pos_images) & set(list_most_pos_images_init)))
                    if numberIm==-1:
                        percentage_intersec_f = number_img_intersec/max(len(list_most_pos_images),len(list_most_pos_images_init))*100.0
                    else:
                        percentage_intersec_f = number_img_intersec/numberIm*100.0
                    percentage_intersec_list += [percentage_intersec_f]
                    
                dico_percentage_intersec_list[layer_name_inlist] = percentage_intersec_list
                
                with open(name_path_dico, 'wb') as handle:
                    pickle.dump(dico_percentage_intersec_list,handle)
                    
    # Print the boxplot per layer
    list_percentage = []
    for layer_name_inlist in list_layers:
        percentage_intersec_list = dico_percentage_intersec_list[layer_name_inlist]
        list_percentage += [percentage_intersec_list]
        
    save_or_show = True
    
    if save_or_show:
        matplotlib.use('Agg')
        plt.switch_backend('agg')
        
    output_img = 'png'
    case_str = str(numberIm)
    ext_name = 'OverLap_'
    
    if output_img=='png':
        fig, ax1 = plt.subplots(figsize=(10, 6))
    elif output_img=='tikz':
        fig, ax1 = plt.subplots()
        
    fig.canvas.set_window_title('Boxplots of the Overlapping percentage.')
    bp = ax1.boxplot(list_percentage, notch=0, sym='+')
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    #ax1.set_title('Comparison of '+leg_str+' score for different methods')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Overlapping (\%)')
    
    medians = np.empty(len(list_layers))
    for i in range(len(list_layers)):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Color of the box
        ax1.add_patch(Polygon(box_coords, facecolor=CB_color_cycle[i % (len(CB_color_cycle))],alpha=0.5))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        if output_img=='png':
            ax1.plot(np.average(med.get_xdata()), np.average(list_percentage[i]),
                 color='w', marker='*', markeredgecolor='k', markersize=8)
        elif output_img=='tikz':
            ax1.plot(np.average(med.get_xdata()), np.average(list_percentage[i]),
                 color='w', marker='h', markeredgecolor='k', markersize=6)
    # X labels
    if output_img=='png':
        ax1.set_xticklabels(list_layers,
                    rotation=45, fontsize=8) 
    elif output_img=='tikz':
        ax1.set_xticklabels(list_layers,
                    rotation=75, fontsize=8)    
    if save_or_show:
        if output_img=='png':
            plt.tight_layout()
            path_fig = os.path.join(output_path_for_dico,ext_name+case_str+'_Boxplots_per_layer.png')
            plt.savefig(path_fig,bbox_inches='tight')
            plt.close()
        if output_img=='tikz':
            path_fig = os.path.join(output_path_for_dico,ext_name+case_str+'_Boxplots_per_layer.tex')
            tikzplotlib.save(path_fig)
            # From from DataForPerceptual_Evaluation import modify_underscore,modify_labels,modify_fontsizeByInput
            # si besoin
#            modify_underscore(path_fig)
#            modify_labels(path_fig)
#            modify_fontsizeByInput(path_fig)
    else:
        plt.show()
        input('Enter to close.')
        plt.close()
    
if __name__ == '__main__': 
    # Petit test 
    #compute_OneValue_Per_Feature(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1')
    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
                                                num_feature=64,
                                                numberIm=9)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
#                                                num_feature=64,
#                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
                                                layer_name='mixed4d_3x3_pre_relu',
                                                num_feature=52,
                                                numberIm=81)
#    # mixed4d_pool_reduce_pre_reluConv2D_63_RASTA_small01_modif.png	
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4d_pool_reduce_pre_relu',
#                                                num_feature=63,
#                                                numberIm=81)
#    #Nom de fichier	mixed4b_3x3_bottleneck_pre_reluConv2D_35_RASTA_small01_modif.png	
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
#                                                num_feature=35,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
#                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
#                                                num_feature=35,
#                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
                                                layer_name='mixed4d_pool_reduce_pre_relu',
                                                num_feature=63,
                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
                                                layer_name='mixed4d_3x3_pre_relu',
                                                num_feature=80,
                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
                                                num_feature=21,
                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
                                                layer_name='mixed5a_pool_reduce_pre_relu',
                                                num_feature=120,
                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
                                                layer_name='mixed5b_5x5_bottleneck_pre_relu',
                                                num_feature=41,
                                                numberIm=81)
    
    # Pour IconArt
    plot_images_Pos_Images(dataset='IconArt_v1',model_name='IconArt_v1_big001_modif_adam_randomCrop_ep200',constrNet='InceptionV1',
                                                layer_name='mixed4c_pool_reduce_pre_relu',
                                                num_feature=13,
                                                numberIm=81)
    plot_images_Pos_Images(dataset='IconArt_v1',model_name='IconArt_v1_big001_modif_adam_randomCrop_ep200',constrNet='InceptionV1',
                                                layer_name='mixed4c_pool_reduce_pre_relu',
                                                num_feature=13,
                                                numberIm=81)
    
    # Nom de fichier	mixed3a_5x5_bottleneck_pre_reluConv2D_8_RASTA_small01_modif.png	
    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1')

    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1')

    #you are not on the Nicolas PC, so I think you have the data in the data folder
    #mixed5a_5x5_pre_relu [116]  are negative for  100.0  % of the images of the training set of RASTA
    #mixed5b_5x5_pre_relu [15]  are negative for  100.0  % of the images of the training set of RASTA
    #mixed5b_pool_reduce_pre_relu [15, 16, 28, 87]  are negative for  100.0  % of the images of the training set of RASTA
    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200',constrNet='InceptionV1')
    
    for num_feature in [60,14,106,50,56,46]:
        plot_images_Pos_Images(dataset='RASTA',
                               model_name='RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                               constrNet='InceptionV1',
                                layer_name='mixed4d',
                                num_feature=num_feature,
                                numberIm=81,
                                stats_on_layer='mean')
        
    # Pour le model from scratch trained on RASTA 
    # RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG a faire aussi
    # A faire tourner 
    for num_feature in [469,103,16,66,57,8]:
        plot_images_Pos_Images(dataset='RASTA',
                               model_name='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                               constrNet='InceptionV1',
                                layer_name='mixed4d',
                                num_feature=num_feature,
                                numberIm=100,
                                stats_on_layer='mean')
        plot_images_Pos_Images(dataset='RASTA',
                               model_name='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                               constrNet='InceptionV1',
                                layer_name='mixed4d',
                                num_feature=num_feature,
                                numberIm=100,
                                stats_on_layer='mean',
                                FTmodel=False)
    for num_feature in [469,103,16,66,57,8]:
        plot_images_Pos_Images(dataset='RASTA',
                               model_name='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                               constrNet='InceptionV1',
                                layer_name='mixed4d',
                                num_feature=num_feature,
                                numberIm=100,
                                stats_on_layer='mean')
        plot_images_Pos_Images(dataset='RASTA',
                               model_name='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                               constrNet='InceptionV1',
                                layer_name='mixed4d',
                                num_feature=num_feature,
                                numberIm=100,
                                stats_on_layer='mean',
                                FTmodel=False)
        
    
        
    