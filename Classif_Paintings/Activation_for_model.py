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
import lucid_utils

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pandas as pd
import tikzplotlib
from matplotlib.colors import LinearSegmentedColormap

from CompNet_FT_lucidIm import get_fine_tuned_model,convert_finetuned_modelToFrozenGraph,\
    get_path_pbmodel_pretrainedModel

CB_color_cycle = ['#377eb8', '#ff7f00','#984ea3', '#4daf4a','#A2C8EC','#e41a1c',
                  '#f781bf', '#a65628', '#dede00','#FFBC79','#999999','#747fba']

list_modified_in_unfreeze50 = ['mixed4a',
          'mixed4b','mixed4c',
          'mixed4d','mixed4e',
          'mixed5a','mixed5b']

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
        if 'RandInit' in model_name or 'RandForUnfreezed' in model_name:
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
    @return : dico of percentage and list of proba 
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
    list_c = []
    for index_c in argsort_from_max_to_min:
        classe = classes[index_c]
        number_im_c = list_labels[index_c]
        if number_im_c >0:
            proba_c = number_im_c/number_im
            per_c = number_im_c/number_im*100.0
            if verbose: print(classe,per_c)
            dico_c[classe] = per_c
            list_c += [proba_c]
        else:
            list_c += [0.]
    return(dico_c,list_c)
    
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
        
    # TODO ici il doit y avoir un pb avec le load des trucs
    
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
                print(output_path_for_img,name_fig,'Percentage intersection (overlaping) between before and after training : ',percentage_intersec,' in %')
                
            # This function will print the proportion of the different classes in the input set
            # for a given dataset
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
    numberIm_list = [100]
    model_list = ['RASTA_small01_modif']
    model_list = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    model_list = ['RASTA_small01_modif',
                  'RASTA_big001_modif_deepSupervision',
                  'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                  'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    model_list = ['RASTA_small01_modif']
    for model_name in model_list:
        for numberIm in numberIm_list:
            overlapping_rate_boxplots(dataset='RASTA',model_name=model_name,
                               constrNet='InceptionV1',numberIm=numberIm,
                               stats_on_layer='meanAfterRelu',output_img='tikz')# because meanAfterRelu already computed
 
def doing_impurity_for_paper():
    
    numberIm_list = [100,1000,-1]
    numberIm_list = [100]
    model_list = ['RASTA_small01_modif',
                  'pretrained',
                  'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                  'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    model_list = ['RASTA_small01_modif']
    model_list = ['RASTA_small01_modif','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                  'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    model_list = ['RASTA_small01_modif']
    kind_purity_tab = ['gini','entropy']
    kind_purity_tab = ['entropy']
    #model_list = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    #model_list = ['RASTA_big001_modif_deepSupervision',
    #              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    for model_name in model_list:
        for numberIm in numberIm_list:
            for kind_purity in kind_purity_tab:
                class_purity_boxplots(dataset='RASTA',model_name=model_name,
                               constrNet='InceptionV1',numberIm=numberIm,
                               stats_on_layer='meanAfterRelu',
                               kind_purity=kind_purity,output_img='tikz')# because meanAfterRelu already computed
                
def doing_scatter_for_paper():
    
    numberIm_list = [100,1000,-1]
    numberIm_list = [100]
    model_list = ['RASTA_small01_modif',
                  'pretrained',
                  'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                  'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    model_list = ['RASTA_small01_modif']
    kind_purity_tab = ['gini','entropy']
    kind_purity_tab = ['entropy']
    #model_list = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    #model_list = ['RASTA_big001_modif_deepSupervision',
    #              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    for model_name in model_list:
        for numberIm in numberIm_list:
            for kind_purity in kind_purity_tab:
                scatter_plot_impurity_overlapping(dataset='RASTA',model_name=model_name,
                               constrNet='InceptionV1',numberIm=numberIm,
                               stats_on_layer='meanAfterRelu',
                               kind_purity=kind_purity)# because meanAfterRelu already computed
 
def get_overlapping_dico(dataset,model_name,constrNet='InceptionV1',
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
    if not(stats_on_layer=='meanAfterRelu'):
        name_dico+= '_' + stats_on_layer
    if not(dataset=='RASTA'):
        name_dico+= '_' + dataset
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
            
    return(dico_percentage_intersec_list)
    
def get_deadkernel_dico(dataset,model_name,constrNet='InceptionV1',
                      list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b'],
                            numberIm=100,stats_on_layer='mean',suffix='',
                            output_path_for_dico=None,
                            cropCenter = True,
                            ReDo=False):
    """
    This fct return the dico of channel that have a positive response for at least
    one image on the dataset
    """
    FTmodel = True
    model_name_wo_oldModel,weights = get_model_name_wo_oldModel(model_name)
    
#    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
#    path_data,Not_on_NicolasPC = get_database(dataset)
#    df_train = df_label[df_label['set']=='train']
    #name_images = df_train[item_name].values
    
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

    
    name_dico = 'NotDeadKernel_'+str(numberIm)
    if not(stats_on_layer=='meanAfterRelu'):
        name_dico+= '_' + stats_on_layer
    if not(dataset=='RASTA'):
        name_dico+= '_' + dataset
    name_path_dico = os.path.join(output_path_for_dico,name_dico+'.pkl')
    
    
    if os.path.exists(name_path_dico) and not(ReDo):
        # The file exist
        with open(name_path_dico, 'rb') as handle:
            dico_index_list = pickle.load(handle)
    else:
    
        dico_index_list = {}
        
        list_outputs_name,activations = get_list_activations(dataset,
                                                         output_path,stats_on_layer,
                                                         model_name,constrNet,
                                                         suffix,cropCenter,FTmodel,
                                                         model_name_wo_oldModel)
        
       
        for layer_name_inlist,activations_l in zip(list_outputs_name,activations):
            if layer_name_inlist in list_layers:
                print('===',layer_name_inlist,'===')
                num_im, number_of_features = activations_l.shape
                
                index_list = []
                
                for num_feature in range(number_of_features):
                    activations_l_f = activations_l[:,num_feature]
                    where_activations_l_f_pos = np.where(activations_l_f>0)[0]
                    if len(where_activations_l_f_pos)>0:
                        index_list += [num_feature]
                    
                dico_index_list[layer_name_inlist] = index_list
                
        with open(name_path_dico, 'wb') as handle:
            pickle.dump(dico_index_list,handle)
            
    return(dico_index_list)
    
    
def get_purity_dico(dataset,model_name,constrNet='InceptionV1',
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
                            ReDo=False,
                            kind_purity='entropy'):
    """
    @param kind_purity='entropy' or 'gini'
    """
    
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
    
    name_dico = 'Purity_'+kind_purity+'_inTop'+str(numberIm)
    if not(stats_on_layer=='meanAfterRelu'):
        name_dico+= '_' + stats_on_layer
    if not(dataset=='RASTA'):
        name_dico+= '_' + dataset
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
                
                purity_score_list = []
                
                for num_feature in range(number_of_features):
                    activations_l_f = activations_l[:,num_feature]
                    where_activations_l_f_pos = np.where(activations_l_f>0)[0]
                    if len(where_activations_l_f_pos)==0:
                        print('No activation positive for this layer')
                        continue
                    activations_l_f_pos = activations_l_f[where_activations_l_f_pos]
                    name_images_l_f_pos = name_images[where_activations_l_f_pos]
                    argsort = np.argsort(activations_l_f_pos)[::-1]
                    # Most positive images
                    list_most_pos_images = name_images_l_f_pos[argsort[0:numberIm]]
                    
                    
                    dico_num_im_per_class,list_c = proportion_labels(list_most_pos_images,dataset,verbose=False)
                    np_l = np.array(list_c)
                    
                    if kind_purity=='entropy':
                        np_l = np_l[np.where(np_l>0.)]
                        purity_score_f = np.sum(-np_l*np.log2(np_l)) # Information gain per class
                        # avec 20 classes
                        # The highest value possible is 4.321928094887363 
                        # loest 0.
                    elif kind_purity=='gini':
                        purity_score_f = np.sum(np_l*(1.-np_l))
                        # A Gini Impurity of 0 is the lowest and best possible impurity
                        # avec 20 class the highest is 0.95
                        # lowest 0.
                    purity_score_list += [purity_score_f]
                    
                dico_percentage_intersec_list[layer_name_inlist] = purity_score_list
                
        with open(name_path_dico, 'wb') as handle:
            pickle.dump(dico_percentage_intersec_list,handle)
                    
    return(dico_percentage_intersec_list)
    
              
def do_featVizu_for_Extrem_points(dataset,model_name_base,constrNet='InceptionV1',
                      list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b'],
                            numberIm=100,
                            numb_points=1,stats_on_layer='mean',suffix='',
                            FTmodel=True,
                            output_path_for_dico=None,
                            cropCenter = True,
                            ReDo=False,
                            owncloud_mode=True):
    """
    Do the feature visualization for the extrem points and also the one nearest 
    to the median according to the overlap ratio metric
    @param : numb_points = we will draw 3*numb_points images so (for the fine-tuned model and 
    the original one so 2*3*numb_points)
    @param : owncloud_mode we will use the synchronise owncloud folder tmp3 as output

    """
    if not(FTmodel):
        raise(NotImplementedError)
    assert(numb_points>0)
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name_base+suffix)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name_base+suffix)
    # For dico
    if output_path_for_dico is None:
        output_path_for_dico = os.path.join(output_path,'Overlapping')
    else:
        output_path_for_dico = os.path.join(output_path_for_dico,'Overlapping')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_dico).mkdir(parents=True, exist_ok=True) 
    
    dico_percentage_intersec_list = get_overlapping_dico(dataset,model_name_base,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            FTmodel=FTmodel,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo)
      

    
    
    ROBUSTNESS = True
    DECORRELATE = True
    # Print the boxplot per layer
    list_percentage = []
    
    if 'RandInit' in model_name_base or 'RandForUnfreezed' in model_name_base:
        raise(NotImplementedError)
        list_models = [model_name_base,model_name_base]
        list_suffix = [suffix,'']
    else:
        list_models = [model_name_base,'pretrained']
        list_suffix = [suffix,'']
        
    if DECORRELATE:
        ext='_Deco'
    else:
        ext=''
    
    if ROBUSTNESS:
      ext+= ''
    else:
      ext+= '_noRob'
        
    firstTime = True
     
    for model_name,suffix_str in zip(list_models,list_suffix):
        prexif_name=model_name+suffix
        if model_name=='pretrained':
            prexif_name = 'Imagnet'
        # TODO et le cas init ????
        if not(model_name=='pretrained'):
            net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
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
            path_lucid_model = os.path.join('')
            name_pb,input_name_lucid = get_path_pbmodel_pretrainedModel(constrNet=constrNet)

        add_end_folder_name = suffix_str
        if platform.system()=='Windows': 
            if owncloud_mode:
                path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
            else:
                path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
        else:
            path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
    
        if firstTime:
            dico_overlapping_index_feat = {}

        for layer in list_layers:
            
            path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
            
            pathlib.Path(path_output_lucid_im_l).mkdir(parents=True, exist_ok=True)

            #pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True)
            percentage_intersec_list = dico_percentage_intersec_list[layer]
            percentage_intersec_np = np.array(percentage_intersec_list)
            #print(percentage_intersec_list)
            median_l = np.median(percentage_intersec_list)
            argsort_overlap_ratio = np.argsort(percentage_intersec_list)
            #print(argsort_overlap_ratio)
            smallest_pt = argsort_overlap_ratio[0:numb_points]
            highest_pt = argsort_overlap_ratio[-numb_points:]
            per_minus_med = np.abs(percentage_intersec_np - median_l)
            med_pt = np.argsort(per_minus_med)[0:numb_points]
#            print('==',layer,'==')
#            print('Smallest  : ',layer,smallest_pt)
#            print('values  : ',percentage_intersec_np[smallest_pt])
#            print('Highest  : ',layer,highest_pt)
#            print('values  : ',percentage_intersec_np[highest_pt])
#            print('Mediane  : ',layer,med_pt)
#            print('values  : ',percentage_intersec_np[med_pt])
            
            if firstTime:
                dico_overlapping_index_feat[layer] = [smallest_pt,med_pt,highest_pt,percentage_intersec_np[smallest_pt],percentage_intersec_np[med_pt],percentage_intersec_np[highest_pt]]
            
            all_features_index = list(smallest_pt)
            all_features_index.extend(list(highest_pt))
            all_features_index.extend(list(med_pt))
            #print(all_features_index)
            for index_feature in all_features_index:
                
                obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                        ,Net=constrNet)
    
                #layer
                name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
                
                full_name=os.path.join(path_output_lucid_im_l,name_base)
                print(full_name)
                
                if not(os.path.isfile(full_name)): # If the image do not exist, we 
                    # will create it
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
                    # Ici il peut y avoir un problem si par le passe il y a eu un bug lors de l execution du code, 
                    # le fichier .pb doit etre supprime et recreer
                    #print('name_pb',os.path.join(path_lucid_model,name_pb))
                    lucid_utils.print_images(model_path=os.path.join(path_lucid_model,name_pb),
                                             list_layer_index_to_print=[[layer,index_feature]],
                                             path_output=path_output_lucid_im_l,prexif_name=prexif_name,\
                                             input_name=input_name_lucid,Net=constrNet,sizeIm=224,
                                             ROBUSTNESS=ROBUSTNESS,
                                             DECORRELATE=DECORRELATE)   
                    
                    # Il faut aussi faire le cas pretrained ou init model !
                    #TODO
        firstTime = False
                    
       
    # Now we deal with the figure of the different feat vizu !             
    
    for layer in list_layers:
        i_line = 0 
        figw, figh = numb_points*2, 3
        smallest_pt,med_pt,highest_pt,sm_val,med_val,high_val = dico_overlapping_index_feat[layer]
        plt.rcParams["figure.figsize"] = [figw, figh]
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 6
        
        fig = plt.figure()
        gs0 = gridspec.GridSpec(3, numb_points, figure=fig)
        #gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
        #fig, axes = plt.subplots(3, numb_points*2) 
        # squeeze=False for the case of one figure only
        fig.suptitle(layer)
        for index_feature_list,value_list,case in zip([smallest_pt,med_pt,highest_pt],[sm_val,med_val,high_val],['Smallest','Median','Highest']):
            for j,(index_feature,value) in enumerate(zip(index_feature_list,value_list)):
                
                ## FT model
                #Name of the image 
                model_name = list_models[0]
                suffix_str = list_suffix[0]
                prexif_name=model_name+suffix
                if model_name=='pretrained':
                    prexif_name = 'Imagnet'
                add_end_folder_name = suffix_str
                if platform.system()=='Windows': 
                    if owncloud_mode:
                        path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
                    else:
                        path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
                else:
                    path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
                path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
                obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                        ,Net=constrNet)
                name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
                full_name=os.path.join(path_output_lucid_im_l,name_base)
                
                # Get the subplot
                gsij = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs0[i_line,j],wspace = 0.0)
                #ax = fig.add_subplot(1,2,gs0[i_line,j],wspace = 0.0)
                ax = fig.add_subplot(gsij[0])
                #ax = axes[i_line,j*2]
                img = plt.imread(full_name)
                #print(img)
                ax.imshow(img, interpolation='none')
                title_l_j = 'FT' + ' ' + str(index_feature) + ' : ' + "{:.2f}.".format(value) #+ ' ' + title_beginning
                #gsij.set_title(title_l_j, fontsize=8)
                ax.set_title(title_l_j, fontsize=6,pad=3.0)
                if j==0:
                    ax.set_ylabel(case, fontsize=6)
                ax.tick_params(axis='both', which='both', length=0)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                #fig = ax.get_figure()
                #fig.tight_layout()
                #fig.subplots_adjust(top=1.05)
                
                ## Initialisation model
                #Name of the image 
                model_name = list_models[1]
                suffix_str = list_suffix[1]
                prexif_name=model_name+suffix
                if model_name=='pretrained':
                    prexif_name = 'Imagnet'
                add_end_folder_name = suffix_str
                if platform.system()=='Windows': 
                    if owncloud_mode:
                        path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
                    else:
                        path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
                else:
                    path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
                path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
                obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                        ,Net=constrNet)
                name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
                full_name=os.path.join(path_output_lucid_im_l,name_base)
                
                # Get the subplot
                #ax = axes[i_line,j*2+1]
                ax = fig.add_subplot(gsij[1])
                img = plt.imread(full_name)
                #print(img)
                ax.imshow(img, interpolation='none')
                if model_name=='pretrained':
                    title_beginning = 'Pretrained'
                else:
                    title_beginning = 'Init'
                title_l_j = title_beginning #+' ' + case + ' ' + str(index_feature) + ' : ' + str(value)
                ax.set_title(title_l_j, fontsize=6,pad=3.0)
#                ax.tick_params(axis='both', which='both', length=0)
#                plt.setp(ax.get_xticklabels(), visible=False)
#                plt.setp(ax.get_yticklabels(), visible=False)
                
                #title_l_j = 'FT' + ' ' + str(index_feature) + ' : ' + str(value) + ' ' + title_beginning
                #gsij.set_title(title_l_j, fontsize=8)
                ax.tick_params(axis='both', which='both', length=0)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                #fig = ax.get_figure()
                #fig.tight_layout()
                #fig.subplots_adjust(top=1.05)
                
            i_line += 1
        #plt.subplots_adjust(top=1-1/figh)         
        # name figure
        name_fig = 'FeatVizu with_Caract_OverRatio_'+str(numberIm)
        if not(stats_on_layer=='meanAfterRelu'):
            name_fig+= '_' + stats_on_layer
        if not(dataset=='RASTA'):
            name_fig+= '_' + dataset
            
        name_fig += '_'+str(numb_points) +'_Feat_per_case.png'
        name_fig = layer + '_' + name_fig
        path_fig = os.path.join(output_path_for_dico,name_fig)
        plt.savefig(path_fig,dpi=300,bbox_inches='tight')
        plt.close()
        
def create_pairs_of_feat_vizu(model_name_base,constrNet='InceptionV1',
                              list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b']
                                 ,suffix='',
                                FTmodel=True,
                                output_path_for_pair=None,
                                cropCenter = True,
                                ReDo=False,
                                owncloud_mode=True):
    """
    This fct will create images with pairs of feat vizu pairs for the layers in 
    list_layers
    @param : owncloud_mode we will use the synchronise owncloud folder tmp3 as output

    """
    
    matplotlib.use('Agg')
    plt.switch_backend('agg')
    
    if not(FTmodel):
        raise(NotImplementedError)

    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name_base+suffix)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name_base+suffix)
    # For output images
    if output_path_for_pair is None:
        output_path_for_pair = os.path.join(output_path,'Pairs')
    else:
        output_path_for_pair = os.path.join(output_path_for_pair,'Pairs')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_pair).mkdir(parents=True, exist_ok=True) 
    
    
    ROBUSTNESS = True
    DECORRELATE = True
    
    if 'RandInit' in model_name_base or 'RandForUnfreezed' in model_name_base:
        raise(NotImplementedError)
        list_models = [model_name_base,model_name_base]
        list_suffix = [suffix,'']
    else:
        list_models = [model_name_base,'pretrained']
        list_suffix = [suffix,'']
        
    if DECORRELATE:
        ext='_Deco'
    else:
        ext=''
    
    if ROBUSTNESS:
      ext+= ''
    else:
      ext+= '_noRob'
        
    firstTime = True
     
    list_num_feat = {}
    
    for model_name,suffix_str in zip(list_models,list_suffix):
        prexif_name=model_name+suffix
        if model_name=='pretrained':
            prexif_name = 'Imagnet'
        # TODO et le cas init ????
        if not(model_name=='pretrained'):
            net_finetuned, init_net = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)

            path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
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
            path_lucid_model = os.path.join('')
            name_pb,input_name_lucid = get_path_pbmodel_pretrainedModel(constrNet=constrNet)

        add_end_folder_name = suffix_str
        if platform.system()=='Windows': 
            if owncloud_mode:
                path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
            else:
                path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
        else:
            path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
    
        if firstTime:
            dico_overlapping_index_feat = {}

        for layer in list_layers:
            
            # output folder of the images
            path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
            pathlib.Path(path_output_lucid_im_l).mkdir(parents=True, exist_ok=True)
            
            net_layer = net_finetuned.get_layer(layer)
            shape_layer = net_layer.output_shape
            num_feat = shape_layer[-1]
            if firstTime:
                list_num_feat[layer] = num_feat
            
            all_features_index = [i for i in range(num_feat)]
            
            for index_feature in all_features_index:
                
                obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                        ,Net=constrNet)
    
                #layer
                name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
                
                full_name=os.path.join(path_output_lucid_im_l,name_base)
                #print(full_name)
                
                if not(os.path.isfile(full_name)): # If the image do not exist, we 
                    # will create it
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
                    # Ici il peut y avoir un problem si par le passe il y a eu un bug lors de l execution du code, 
                    # le fichier .pb doit etre supprime et recreer
                    #print('name_pb',os.path.join(path_lucid_model,name_pb))
                    lucid_utils.print_images(model_path=os.path.join(path_lucid_model,name_pb),
                                             list_layer_index_to_print=[[layer,index_feature]],
                                             path_output=path_output_lucid_im_l,prexif_name=prexif_name,\
                                             input_name=input_name_lucid,Net=constrNet,sizeIm=224,
                                             ROBUSTNESS=ROBUSTNESS,
                                             DECORRELATE=DECORRELATE)   
                    
                    # Il faut aussi faire le cas pretrained ou init model !
                    #TODO
        firstTime = False
                    
       
    # Now we deal with the figure of the different feat vizu !             
    
    for layer in list_layers:
        print('===',layer,'===')
        
        output_path_for_pair_l =os.path.join(output_path_for_pair,layer)

        pathlib.Path(output_path_for_pair_l).mkdir(parents=True, exist_ok=True)
        
        #i_line = 0 
        figw, figh = 2,1.1
        num_feat = list_num_feat[layer]
        
        
        #smallest_pt,med_pt,highest_pt,sm_val,med_val,high_val = dico_overlapping_index_feat[layer]
        plt.rcParams["figure.figsize"] = [figw, figh]
        plt.rcParams["axes.titlesize"] = 12
        #plt.rcParams["axes.labelsize"] = 6
        
        
        #gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
        #fig, axes = plt.subplots(3, numb_points*2) 
        # squeeze=False for the case of one figure only
        #fig.suptitle(layer)
        for index_feature in range(0,num_feat):
            fig = plt.figure()
            gs0 = gridspec.GridSpec(1, 1, figure=fig)
                
            ## FT model
            #Name of the image 
            model_name = list_models[0]
            suffix_str = list_suffix[0]
            prexif_name=model_name+suffix
            if model_name=='pretrained':
                prexif_name = 'Imagnet'
            add_end_folder_name = suffix_str
            if platform.system()=='Windows': 
                if owncloud_mode:
                    path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
                else:
                    path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
            else:
                path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
            path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
            obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                    ,Net=constrNet)
            name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
            full_name=os.path.join(path_output_lucid_im_l,name_base)
            
            # Get the subplot
            gsij = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs0[0],wspace = 0.0)
            #ax = fig.add_subplot(1,2,gs0[i_line,j],wspace = 0.0)
            ax = fig.add_subplot(gsij[0])
            #ax = axes[i_line,j*2]
            img = plt.imread(full_name)
            #print(img)
            ax.imshow(img, interpolation='none')
            title_l_j = str(index_feature) + ' : FT'
            #gsij.set_title(title_l_j, fontsize=8)
            ax.set_title(title_l_j, fontsize=6,pad=3.0)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            #fig = ax.get_figure()
            #fig.tight_layout()
            #fig.subplots_adjust(top=1.05)
            
            ## Initialisation model
            #Name of the image 
            model_name = list_models[1]
            suffix_str = list_suffix[1]
            prexif_name=model_name+suffix
            if model_name=='pretrained':
                prexif_name = 'Imagnet'
            add_end_folder_name = suffix_str
            if platform.system()=='Windows': 
                if owncloud_mode:
                    path_output_lucid_im = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,model_name+add_end_folder_name)
                else:
                    path_output_lucid_im = os.path.join('CompModifModel',constrNet,model_name+add_end_folder_name)
            else:
                path_output_lucid_im = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name+add_end_folder_name)
            path_output_lucid_im_l = os.path.join(path_output_lucid_im,layer)
            obj_str,kind_layer = lucid_utils.get_obj_and_kind_layer(layer_to_print=layer
                                                                    ,Net=constrNet)
            name_base = layer + kind_layer +'_'+str(index_feature)+'_'+prexif_name+ext+'_toRGB.png'
            full_name=os.path.join(path_output_lucid_im_l,name_base)
            
            # Get the subplot
            #ax = axes[i_line,j*2+1]
            ax = fig.add_subplot(gsij[1])
            img = plt.imread(full_name)
            #print(img)
            ax.imshow(img, interpolation='none')
            if model_name=='pretrained':
                title_beginning = 'Pretrained'
            else:
                title_beginning = 'Init'
            title_l_j = title_beginning #+' ' + case + ' ' + str(index_feature) + ' : ' + str(value)
            ax.set_title(title_l_j, fontsize=6,pad=3.0)
#                ax.tick_params(axis='both', which='both', length=0)
#                plt.setp(ax.get_xticklabels(), visible=False)
#                plt.setp(ax.get_yticklabels(), visible=False)
            
            #title_l_j = 'FT' + ' ' + str(index_feature) + ' : ' + str(value) + ' ' + title_beginning
            #gsij.set_title(title_l_j, fontsize=8)
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            #fig = ax.get_figure()
            #fig.tight_layout()
                #fig.subplots_adjust(top=1.05)

        #plt.subplots_adjust(top=1-1/figh)         
        # name figure
            name_fig = 'FeatVizu_Pair_'+str(index_feature)
            
            name_fig = layer + '_' + name_fig
            path_fig = os.path.join(output_path_for_pair_l,name_fig)
            plt.savefig(path_fig,dpi=300,bbox_inches='tight')
            plt.close()
                    
                    
                    
        
def overlapping_rate_boxplots(dataset,model_name,constrNet='InceptionV1',
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
                            ReDo=False,
                            output_img = 'png'):
    """
    This function will plot in boxplot overlapping ratio between the top k images 
    @param : numberIm = top k images used : numberIm = -1 if you want to take all the images
    """
    
    if 'RandForUnfreezed' in model_name:
        if not('unfreeze50' in  model_name):
           raise(NotImplementedError)
        list_layers_new = []
        index_start_color =0
        for layer in list_layers:
            if layer in list_modified_in_unfreeze50:
                list_layers_new += [layer]
            else:
                index_start_color+=1
        list_layers = list_layers_new
    else:
        index_start_color= 0
    
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")

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
    
    dico_percentage_intersec_list = get_overlapping_dico(dataset,model_name,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            FTmodel=FTmodel,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo)
      
    # Print the boxplot per layer
    list_percentage = []
    for layer_name_inlist in list_layers:
        percentage_intersec_list = dico_percentage_intersec_list[layer_name_inlist]
        list_percentage += [percentage_intersec_list]
        
    save_or_show = True
    
    if save_or_show:
        matplotlib.use('Agg')
        plt.switch_backend('agg')

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
        ax1.add_patch(Polygon(box_coords, facecolor=CB_color_cycle[index_start_color+i % (len(CB_color_cycle))],alpha=0.5))
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
                    rotation=45, fontsize=8)    
    if save_or_show:
        if output_img=='png':
            plt.tight_layout()
            path_fig = os.path.join(output_path_for_dico,ext_name+case_str+'_Boxplots_per_layer.png')
            plt.savefig(path_fig,bbox_inches='tight')
            plt.close()
        if output_img=='tikz':
            path_fig = os.path.join(output_path_for_dico,ext_name+case_str+'_Boxplots_per_layer.tex')
            print('save at :',path_fig)
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
        
def scatter_plot_impurity_overlapping(dataset,model_name,constrNet='InceptionV1',
                      list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b'],
#                      list_layers=['mixed5b'],
                            numberIm=100,stats_on_layer='mean',suffix='',
                            FTmodel=True,
                            output_path_for_dico=None,
                            cropCenter = True,
                            ReDo=False,
                            kind_purity='entropy'):
    """
    This function will plot in boxplot class purity in the top k images 
    @param : numberIm = top k images used : numberIm = -1 if you want to take all the images
    """
    
    #scatter_plot_impurity_overlapping('RASTA','RASTA_small01_modif',stats_on_layer=)
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")

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
    
    dico_not_deadkernel_list = get_deadkernel_dico(dataset,model_name,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo)
      
    
    dico_percentage_intersec_list = get_overlapping_dico(dataset,model_name,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            FTmodel=FTmodel,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo)
      
    # Print the boxplot per layer
    list_percentage = [] # Overlapping
    for layer_name_inlist in list_layers:
        percentage_intersec_list = dico_percentage_intersec_list[layer_name_inlist]
        list_percentage += [percentage_intersec_list]
    
    
    dico_score_list = get_purity_dico(dataset,model_name,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            FTmodel=FTmodel,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo,
                            kind_purity=kind_purity)
      
    if kind_purity=='entropy':
        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset)
        np_l = np.array([1./num_classes]*num_classes)
        max_entropy = np.sum(-np_l*np.log2(np_l))
    
    # Print the boxplot per layer
    list_impurity = []
    for layer_name_inlist in list_layers:
        impurity_score = dico_score_list[layer_name_inlist]
        if kind_purity=='entropy':
            # we will normalize the entropy by the maximum entropy possible
            impurity_score /= max_entropy 
        list_impurity += [impurity_score]
        
    save_or_show = False
    
    if save_or_show:
        matplotlib.use('Agg')
        plt.switch_backend('agg')
        
    output_img = 'png'
    case_str = str(numberIm)

    ext_name =  'Scatter_'+kind_purity
    if kind_purity=='entropy':
       str_kind = "Entropy"
       leg_str = 'Entropy over classes'
    elif kind_purity=='gini':
        str_kind = 'Gini Impurity'
        leg_str = 'Gini Impurity over classes'
                
    output_img = 'png'
    ext_name += '_OverLap_'
    
    if output_img=='png':
        fig, ax = plt.subplots(figsize=(10, 6))
    elif output_img=='tikz':
        fig, ax = plt.subplots()
        
    fig.canvas.set_window_title('Overlapping percentage versus ' +str_kind) 
        
#    cmap_name = 'plasma'
#    n_bin = len(list_layers)
#    cm = LinearSegmentedColormap.from_list(
#        cmap_name, colors, N=n_bin)
    
    lim_impurity = 0.2
    
    extrem_pts = []
    dict_extrem_pts = {}
    dict_extrem_keys = []
    
    for i,layer_name in enumerate(list_layers):
        x = list_percentage[i]
        y = list_impurity[i]
        
        if not(len(x)==len(y)):
            not_dead_kernel = dico_not_deadkernel_list[layer_name]
            x = np.array(x)[not_dead_kernel]
        else:
            not_dead_kernel = np.arange(0,len(x),1)
            
        where_low_impurity = np.where(y<lim_impurity)[0]
        y_where = y[where_low_impurity]
        x_where = np.array(x)[where_low_impurity]
        feat_indexes = np.array(not_dead_kernel)[where_low_impurity]
        for per,imp,feat_index in zip(x_where,y_where,feat_indexes):
            if not(per in dict_extrem_keys):     
                dict_extrem_pts[per] = [layer_name,per,imp,feat_index] 
                dict_extrem_keys += [per]
            else:
                _,per_old,imp_old,_ = dict_extrem_pts[per]
                if imp <= imp_old:
                    dict_extrem_pts[per] = [layer_name,per,imp,feat_index] 
            
        print(i,layer_name,len(x),len(y))
        color_i = CB_color_cycle[i % (len(CB_color_cycle))]
        plt.scatter(x, y,c=color_i,label=layer_name,alpha=0.7)
        
    offset_x = 0.1
    offset_y = -0.2
    for key in dict_extrem_keys :
        [layer_name,per,imp,feat_index] = dict_extrem_pts[key]
        i =  list_layers.index(layer_name)
        color_i = CB_color_cycle[i % (len(CB_color_cycle))]
        ax.annotate(feat_index, (per+offset_x,imp+offset_y),c=color_i)
        
    plt.grid(True)
    plt.ylim((-0.01,1.01))
    plt.xlim((-1.,101.))
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
        
    ax.set_xlabel("Overlapping (\%)",fontsize=20)
    ax.set_ylabel(leg_str,fontsize=20)
    
    ncol= 1
    bbox_to_anchor=(1.01, 0.5)
    loc='center left'
    ax.legend(loc=loc,  bbox_to_anchor=bbox_to_anchor, # under  bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=ncol,fontsize=18)

    

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
        #input('Enter to close.')
        #plt.close()    
        
def class_purity_boxplots(dataset,model_name,constrNet='InceptionV1',
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
                            ReDo=False,
                            kind_purity='gini',
                            output_img = 'png'):
    """
    This function will plot in boxplot class purity in the top k images 
    @param : numberIm = top k images used : numberIm = -1 if you want to take all the images
    """
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")
    
    if 'RandForUnfreezed' in model_name:
        if not('unfreeze50' in  model_name):
           raise(NotImplementedError)
        list_layers_new = []
        index_start_color =0
        for layer in list_layers:
            if layer in list_modified_in_unfreeze50:
                list_layers_new += [layer]
            else:
                index_start_color+=1
        list_layers = list_layers_new
    else:
        index_start_color= 0
            
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
    
    dico_score_list = get_purity_dico(dataset,model_name,constrNet=constrNet,
                      list_layers=list_layers,
                            numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                            FTmodel=FTmodel,
                            output_path_for_dico=None,
                            cropCenter = cropCenter,
                            ReDo=ReDo,
                            kind_purity=kind_purity)
      
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    np_l = np.array([1./num_classes]*num_classes)
    max_score = np.sum(-np_l*np.log2(np_l))
    
    # Print the boxplot per layer
    list_percentage = []
    for layer_name_inlist in list_layers:
        percentage_intersec_list = dico_score_list[layer_name_inlist]
        # we will normalize the entropy by the maximum entropy possible or gini index by it max
        percentage_intersec_list /= max_score 
        list_percentage += [percentage_intersec_list]
        
    save_or_show = True
    
    if save_or_show:
        matplotlib.use('Agg')
        plt.switch_backend('agg')
        
    
    case_str = str(numberIm)

    ext_name =  'Purity_'+kind_purity
    if kind_purity=='entropy':
       str_kind = "Entropy"
       leg_str = 'Entropy over classes'
    elif kind_purity=='gini':
        str_kind = 'Gini Impurity'
        leg_str = 'Gini Impurity over classes'
    
    if output_img=='png':
        fig, ax1 = plt.subplots(figsize=(10, 6))
    elif output_img=='tikz':
        fig, ax1 = plt.subplots()
        
    fig.canvas.set_window_title('Boxplots of the Impurity computed with '+str_kind+'.')
    bp = ax1.boxplot(list_percentage, notch=0, sym='+')
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    #ax1.set_title('Comparison of '+leg_str+' score for different methods')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel(leg_str)
    
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
        ax1.add_patch(Polygon(box_coords, facecolor=CB_color_cycle[index_start_color+i % (len(CB_color_cycle))],alpha=0.5))
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
                    rotation=45, fontsize=8)    
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
#    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
#                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
#                                                num_feature=64,
#                                                numberIm=9)
##    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
##                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
##                                                num_feature=64,
##                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4d_3x3_pre_relu',
#                                                num_feature=52,
#                                                numberIm=81)
##    # mixed4d_pool_reduce_pre_reluConv2D_63_RASTA_small01_modif.png	
##    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
##                                                layer_name='mixed4d_pool_reduce_pre_relu',
##                                                num_feature=63,
##                                                numberIm=81)
##    #Nom de fichier	mixed4b_3x3_bottleneck_pre_reluConv2D_35_RASTA_small01_modif.png	
##    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
##                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
##                                                num_feature=35,
##                                                numberIm=81)
##    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
##                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
##                                                num_feature=35,
##                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
#                                                layer_name='mixed4d_pool_reduce_pre_relu',
#                                                num_feature=63,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
#                                                layer_name='mixed4d_3x3_pre_relu',
#                                                num_feature=80,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
#                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
#                                                num_feature=21,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
#                                                layer_name='mixed5a_pool_reduce_pre_relu',
#                                                num_feature=120,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big001_modif_RandInit_ep120',constrNet='InceptionV1',
#                                                layer_name='mixed5b_5x5_bottleneck_pre_relu',
#                                                num_feature=41,
#                                                numberIm=81)
#    
#    # Pour IconArt
#    plot_images_Pos_Images(dataset='IconArt_v1',model_name='IconArt_v1_big001_modif_adam_randomCrop_ep200',constrNet='InceptionV1',
#                                                layer_name='mixed4c_pool_reduce_pre_relu',
#                                                num_feature=13,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='IconArt_v1',model_name='IconArt_v1_big001_modif_adam_randomCrop_ep200',constrNet='InceptionV1',
#                                                layer_name='mixed4c_pool_reduce_pre_relu',
#                                                num_feature=13,
#                                                numberIm=81)
    

#    list_features = [['mixed4c_3x3_bottleneck_pre_relu',78],
#                     ['mixed4c_pool_reduce_pre_relu',2],
#                     ['mixed4d_5x5_pre_relu',49]
#                     ]
#    for layer_name,num_feature in list_features:
#        plot_images_Pos_Images(dataset='IconArt_v1',model_name='IconArt_v1_small01_modif',
#                               constrNet='InceptionV1',
#                            layer_name=layer_name,
#                            num_feature=num_feature,
#                            numberIm=100)

#    
#    # Nom de fichier	mixed3a_5x5_bottleneck_pre_reluConv2D_8_RASTA_small01_modif.png	
#    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1')
#
#    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1')
#
#    #you are not on the Nicolas PC, so I think you have the data in the data folder
#    #mixed5a_5x5_pre_relu [116]  are negative for  100.0  % of the images of the training set of RASTA
#    #mixed5b_5x5_pre_relu [15]  are negative for  100.0  % of the images of the training set of RASTA
#    #mixed5b_pool_reduce_pre_relu [15, 16, 28, 87]  are negative for  100.0  % of the images of the training set of RASTA
#    dead_kernel_QuestionMark(dataset='RASTA',model_name='RASTA_big001_modif_adam_randomCrop_deepSupervision_ep200',constrNet='InceptionV1')
#    
#    for num_feature in [60,14,106,50,56,46]:
#        plot_images_Pos_Images(dataset='RASTA',
#                               model_name='RASTA_big001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                               constrNet='InceptionV1',
#                                layer_name='mixed4d',
#                                num_feature=num_feature,
#                                numberIm=81,
#                                stats_on_layer='mean')
#        
#    # Pour le model from scratch trained on RASTA 
#    # RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG a faire aussi
#    # A faire tourner 
#    for num_feature in [469,103,16,66,57,8]:
#        plot_images_Pos_Images(dataset='RASTA',
#                               model_name='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
#                               constrNet='InceptionV1',
#                                layer_name='mixed4d',
#                                num_feature=num_feature,
#                                numberIm=100,
#                                stats_on_layer='mean')
#        plot_images_Pos_Images(dataset='RASTA',
#                               model_name='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
#                               constrNet='InceptionV1',
#                                layer_name='mixed4d',
#                                num_feature=num_feature,
#                                numberIm=100,
#                                stats_on_layer='mean',
#                                FTmodel=False)
#    for num_feature in [469,103,16,66,57,8]:
#        plot_images_Pos_Images(dataset='RASTA',
#                               model_name='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
#                               constrNet='InceptionV1',
#                                layer_name='mixed4d',
#                                num_feature=num_feature,
#                                numberIm=100,
#                                stats_on_layer='mean')
#        plot_images_Pos_Images(dataset='RASTA',
#                               model_name='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
#                               constrNet='InceptionV1',
#                                layer_name='mixed4d',
#                                num_feature=num_feature,
#                                numberIm=100,
#                                stats_on_layer='mean',
#                                FTmodel=False)
#                                          list_layers=['conv2d0','conv2d1',
#                                              'conv2d2','mixed3a',
#                                              'mixed3b','mixed4a',
#                                              'mixed4b','mixed4c',
#                                              'mixed4d','mixed4e',
#                                              'mixed5a','mixed5b'],
#    do_featVizu_for_Extrem_points(dataset='RASTA',model_name_base='RASTA_small01_modif',
#                                  constrNet='InceptionV1',
#                                  list_layers=['conv2d0','conv2d1',
#                                              'conv2d2','mixed3a',
#                                              'mixed3b','mixed4a',
#                                              'mixed4b','mixed4c',
#                                              'mixed4d','mixed4e',
#                                              'mixed5a','mixed5b'],
#                                    numberIm=100,
#                                    numb_points=5,stats_on_layer='meanAfterRelu',suffix='',
#                                    FTmodel=True,
#                                    output_path_for_dico=None,
#                                    cropCenter = True,
#                                    ReDo=False,
#                                    owncloud_mode=True)
#    do_featVizu_for_Extrem_points(dataset='RASTA',model_name_base='RASTA_small01_modif',
#                                  constrNet='InceptionV1',
#                                  list_layers=['mixed4c'],
#                                    numberIm=100,
#                                    numb_points=10,stats_on_layer='meanAfterRelu',suffix='',
#                                    FTmodel=True,
#                                    output_path_for_dico=None,
#                                    cropCenter = True,
#                                    ReDo=False,
#                                    owncloud_mode=True)
    
    # Model RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200 : faire l'image Top 100 pour certaines couches
# Random Initialisation
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                           constrNet='InceptionV1',
                            layer_name='mixed4d_5x5_pre_relu',
                            num_feature=50,
                            numberIm=100,alreadyAtInit=False,
                            FTmodel=False)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                           constrNet='InceptionV1',
                            layer_name='mixed5a_3x3_bottleneck_pre_relu',
                            num_feature=1,
                            numberIm=100,alreadyAtInit=False,
                            FTmodel=False)
    # Trained model 
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                           constrNet='InceptionV1',
#                            layer_name='mixed4d_5x5_pre_relu',
#                            num_feature=50,
#                            numberIm=9,alreadyAtInit=False)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                           constrNet='InceptionV1',
#                            layer_name='mixed5a_3x3_bottleneck_pre_relu',
#                            num_feature=1,
#                            numberIm=9,alreadyAtInit=False)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                           constrNet='InceptionV1',
#                            layer_name='mixed4d_5x5_pre_relu',
#                            num_feature=50,
#                            numberIm=100,alreadyAtInit=False)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                           constrNet='InceptionV1',
#                            layer_name='mixed5a_3x3_bottleneck_pre_relu',
#                            num_feature=1,
#                            numberIm=100,alreadyAtInit=False)
    
    create_pairs_of_feat_vizu(model_name_base='RASTA_small01_modif',constrNet='InceptionV1',
                              list_layers=['conv2d0','conv2d1',
                                  'conv2d2','mixed3a',
                                  'mixed3b','mixed4a',
                                  'mixed4b','mixed4c',
                                  'mixed4d','mixed4e',
                                  'mixed5a','mixed5b']
                                 ,suffix='',
                                FTmodel=True,
                                output_path_for_pair=None,
                                cropCenter = True,
                                ReDo=False,
                                owncloud_mode=True)
        
    