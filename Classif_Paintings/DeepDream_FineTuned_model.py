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


def DeepDream_withFinedModel():
    """
    This function will load the two models (deep nets) before and after fine-tuning 
    and then compute the difference between the weights and finally run a 
    deep dream on the feature maps of the weights that have the most change
    """
    
    target_dataset = 'IconArt_v1'
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','DeepDream',target_dataset)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    features = 'block5_pool'
    normalisation = False
    getBeforeReLU = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset= 'ImageNet'
    kind_method=  'FT'
    transformOnFinalLayer='GlobalAveragePooling2D'
       
    
    Model_dict = {}
    list_markers = ['o','s','X','*']
    alpha = 0.7
    
    dict_of_dict_hist = {}
    dict_of_dict = {}
    constrNet = 'VGG'
    
    weights = 'imagenet'
    
    if 'VGG' in constrNet:
        imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
        net_layers = imagenet_model.layers
    else:
        raise(NotImplementedError)
       
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
    
    final_clf = 'MLP2'
    
    computeGlobalVariance = False
    optimizer='SGD'
    opt_option=[0.1,0.001]
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
    finetuned_layers = net_finetuned.layers
        
    dict_layers_argsort = {}
    dict_layers_mean_squared = {}
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
        
        dict_layers_mean_squared[layer_name] = mean_squared
        argsort = np.argsort(mean_squared)[::-1]
        dict_layers_argsort[layer_name] = argsort
        
    K.set_learning_phase(0)
    
    # /!\ Attention en fait tu es en train de travailler apres ReLU ! il va falloir changer cela peut etre
    
    #run_VisualisationOnLotImages_kin(output_path,net_layers,net_finetuned,dict_layers_argsort)
    #run_Visualisation_PosAndNegFeatures(output_path,net_layers,net_finetuned,dict_layers_argsort)
    
def run_VisualisationOnLotImages_kin(output_path,net_layers,net_finetuned,dict_layers_argsort):
    
    step = 0.01  # Gradient ascent step size
    iterations = 100  # Number of ascent steps per scale
    
    init_images = []
    init_images_name = []
    init_rand_im = np.random.normal(loc=125.,scale=3.,size=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['smallRand']
    init_rand_im = np.random.normal(loc=125.,scale=15.,size=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['mediumRand']
    init_rand_im = np.random.uniform(low=0.,high=255.,size=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['unifRand']
    init_rand_im = np.zeros(shape=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['black']
    init_rand_im = 255.*np.ones(shape=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['white']
    
    output_path_f = os.path.join(output_path,'Feature')
    pathlib.Path(output_path_f).mkdir(parents=True, exist_ok=True) 
    output_path_wl = os.path.join(output_path,'WholeLayer')
    pathlib.Path(output_path_wl).mkdir(parents=True, exist_ok=True) 
    output_path_abs = os.path.join(output_path,'AbsScalarProduct')
    pathlib.Path(output_path_abs).mkdir(parents=True, exist_ok=True) 
    output_path_mean = os.path.join(output_path,'MeanSquaredProduct')
    pathlib.Path(output_path_mean).mkdir(parents=True, exist_ok=True) 
        
    for layer in net_layers:
        layer_name = layer.name
        if not('conv' in layer_name):
            continue
        
        # On a specific feature of the given layer
        argsort = dict_layers_argsort[layer_name]
        number_kernel_considered = int(0.03*len(argsort))
        for i in range(number_kernel_considered):
            index_feature = argsort[i]
            print('Start deep dreaming for ',layer_name,index_feature)
            #init_rand_im = np.random.normal(loc=125.,scale=3.,size=(1,224,224,3))
            deepdream_model = DeepDream_on_one_specific_featureMap(net_finetuned,layer_name,index_feature)
            
            for init_rand_im,init_rand_name in zip(init_images,init_images_name):
                output_image = deepdream_model.gradient_ascent(np.copy(init_rand_im),iterations,step)
                deprocess_output = deprocess_image(np.copy(output_image))
                
                result_prefix = 'VGG_finetuned_'+layer_name+'_'+str(index_feature)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'.png'
                name_saved_im = os.path.join(output_path_f,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output))
                
                deprocess_output_equ = cv2.equalizeHist(cv2.cvtColor(np.copy(deprocess_output),cv2.COLOR_RGB2GRAY))
                deprocess_output_equ = cv2.cvtColor(deprocess_output_equ,cv2.COLOR_GRAY2RGB)
                result_prefix = 'VGG_finetuned_'+layer_name+'_'+str(index_feature)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'_eq.png'
                name_saved_im = os.path.join(output_path_f,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output_equ))
        
            del deepdream_model
            
        # On the whole layer
        deepdream_model =  DeepDream_on_one_specific_layer(net_finetuned,layer_name)
        print('Start Deep Dream on the whole layer')
        for init_rand_im,init_rand_name in zip(init_images,init_images_name):
                output_image = deepdream_model.gradient_ascent(np.copy(init_rand_im),iterations,step)
                deprocess_output = deprocess_image(np.copy(output_image))
                
                result_prefix = 'VGG_finetuned_'+layer_name+'_wholeLayer_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'.png'
                name_saved_im = os.path.join(output_path_wl,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output))
                
                deprocess_output_equ = cv2.equalizeHist(cv2.cvtColor(np.copy(deprocess_output),cv2.COLOR_RGB2GRAY))
                deprocess_output_equ = cv2.cvtColor(deprocess_output_equ,cv2.COLOR_GRAY2RGB)
                result_prefix = 'VGG_finetuned_'+layer_name+'_wholeLayer_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'_eq.png'
                name_saved_im = os.path.join(output_path_wl,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output_equ))
        
        del deepdream_model
        
        # # On the square of dot of two feature map
        print('Start deep dream on the abs of scalar product of two features')
        argsort_top5 = argsort[0:5]
        pairs_of_index = itertools.combinations(argsort_top5,r=2)
        for index_feature1,index_feature2 in pairs_of_index:
            deepdream_model = DeepDream_omeanPointWise_of_2features(net_finetuned,layer_name,index_feature1,index_feature2)
            
            for init_rand_im,init_rand_name in zip(init_images,init_images_name):
                output_image = deepdream_model.gradient_ascent(np.copy(init_rand_im),iterations,step)
                deprocess_output = deprocess_image(np.copy(output_image))
                
                result_prefix = 'VGG_finetuned_'+layer_name+'_AbsScalarProduct_'+str(index_feature1)+'_'+str(index_feature2)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'.png'
                name_saved_im = os.path.join(output_path_abs,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output))
                
                deprocess_output_equ = cv2.equalizeHist(cv2.cvtColor(np.copy(deprocess_output),cv2.COLOR_RGB2GRAY))
                deprocess_output_equ = cv2.cvtColor(deprocess_output_equ,cv2.COLOR_GRAY2RGB)
                result_prefix = 'VGG_finetuned_'+layer_name+'_AbsScalarProduct_'+str(index_feature1)+'_'+str(index_feature2)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'_eq.png'
                name_saved_im = os.path.join(output_path_abs,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output_equ))
        
            del deepdream_model
            
        print('Start deep dream on sum of square of pointwise multiplication of two features')
        pairs_of_index = itertools.combinations(argsort_top5,r=2)
        for index_feature1,index_feature2 in pairs_of_index:
            deepdream_model = DeepDream_on_squared_pointwiseproduct_of_2features(net_finetuned,layer_name,index_feature1,index_feature2)
            
            for init_rand_im,init_rand_name in zip(init_images,init_images_name):
                output_image = deepdream_model.gradient_ascent(np.copy(init_rand_im),iterations,step)
                
                deprocess_output = deprocess_image(np.copy(output_image))
                result_prefix = 'VGG_finetuned_'+layer_name+'_MeanSquaredProduct_'+str(index_feature1)+'_'+str(index_feature2)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'.png'
                name_saved_im = os.path.join(output_path_mean,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output))
                
                deprocess_output_equ = cv2.equalizeHist(cv2.cvtColor(np.copy(deprocess_output),cv2.COLOR_RGB2GRAY))
                deprocess_output_equ = cv2.cvtColor(deprocess_output_equ,cv2.COLOR_GRAY2RGB)
                result_prefix = 'VGG_finetuned_'+layer_name+'_MeanSquaredProduct_'+str(index_feature1)+'_'+str(index_feature2)+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'_eq.png'
                name_saved_im = os.path.join(output_path_mean,result_prefix)
                save_img(name_saved_im, np.copy(deprocess_output_equ))
        
            del deepdream_model
            
def run_Visualisation_PosAndNegFeatures(output_path,net_layers,net_finetuned,dict_layers_argsort):
    
    step = 0.05  # Gradient ascent step size
    iterations = 2048  # Number of ascent steps per scale
    
    init_images = []
    init_images_name = []
    init_rand_im = np.random.uniform(low=0.,high=255.,size=(1,224,224,3))
    init_images += [init_rand_im]
    init_images_name += ['unifRand']
    
    output_path_f = os.path.join(output_path,'Feature_Neg_Pos')
    pathlib.Path(output_path_f).mkdir(parents=True, exist_ok=True) 
    
    kind_of_optim_tab = ['pos','neg']
        
    for layer in net_layers:
        layer_name = layer.name
        if not('conv' in layer_name):
            continue
        
        # On a specific feature of the given layer
        argsort = dict_layers_argsort[layer_name]
        number_kernel_considered = int(0.05*len(argsort))
        for i in range(number_kernel_considered):
            index_feature = argsort[i]
            print('Start deep dreaming for ',layer_name,index_feature)
            #init_rand_im = np.random.normal(loc=125.,scale=3.,size=(1,224,224,3))
            for kind_of_optim in kind_of_optim_tab:
                deepdream_model = DeepDream_on_one_specific_featureMap(net_finetuned,layer_name,index_feature,kind_of_optim=kind_of_optim)
                
                for init_rand_im,init_rand_name in zip(init_images,init_images_name):
                    output_image = deepdream_model.gradient_ascent(np.copy(init_rand_im),iterations,step)
                    deprocess_output = deprocess_image(np.copy(output_image))
                    
                    result_prefix = 'VGG_finetuned_'+layer_name+'_'+str(index_feature)+'_'+kind_of_optim+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'.png'
                    name_saved_im = os.path.join(output_path_f,result_prefix)
                    save_img(name_saved_im, np.copy(deprocess_output))
                    
                    deprocess_output_equ = cv2.equalizeHist(cv2.cvtColor(np.copy(deprocess_output),cv2.COLOR_RGB2GRAY))
                    deprocess_output_equ = cv2.cvtColor(deprocess_output_equ,cv2.COLOR_GRAY2RGB)
                    result_prefix = 'VGG_finetuned_'+layer_name+'_'+str(index_feature)+'_'+kind_of_optim+'_iter'+str(iterations)+'_s'+str(step)+'_'+init_rand_name+'_eq.png'
                    name_saved_im = os.path.join(output_path_f,result_prefix)
                    save_img(name_saved_im, np.copy(deprocess_output_equ))
            
                del deepdream_model
    
def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    # x /= 2.
    # x += 0.5
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x          
    
def print_stats_on_diff(np_list,k=3):
    print('Max :',np.max(np_list),'Median :',np.median(np_list),'last decile :',np.percentile(np_list, 90),'Min :',np.min(np_list))
    argsort = np.argsort(np_list)[::-1]
    for i in range(k):
        print('Top ',i,': index =',argsort[i],' value :',np_list[argsort[i]])

class DeepDream_on_one_specific_featureMap(object):
    """
    Deep Dream on one specific feature number index_feature of a given layer
    """
    def __init__(self,model,layer_name, index_feature,kind_of_optim='squared'):
        """
        Initialisation function of the Optimization of the feature map
        
        Parameters
        ----------
        model : keras model 
        layer_name : string : the layer you want to use
        index_feature : string : the index of the feature in the given layer
        kind_of_optim : string  
            The kind of optimisation maded on the given feature. The default is 'squared'.
            Use 'pos' for the positive feature maximisation and 'neg' for the negative one

        Returns
        -------
        None

        """
        self.model = model
        self.layer_name = layer_name
        self.index_feature = index_feature
        self.kind_of_optim = kind_of_optim

        dream = model.input
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layers_all = [layer.name for layer in model.layers]
        if layer_name not in layers_all:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
           
        # Define the loss.
        loss = K.variable(0.)
        for layer_local in  model.layers:
            if layer_local.name==layer_name:
                x = layer_local.output
                
                # We avoid border artifacts by only involving non-border pixels in the loss.
                if K.image_data_format() == 'channels_first':
                    raise(NotImplementedError)
                    x_index_feature = x[:,index_feature,:,:]
                    x_index_feature = K.expand_dims(x_index_feature,axis=1)
                    scaling = K.prod(K.cast(K.shape(x_index_feature), 'float32'))
                    loss = loss + K.sum(K.square(x_index_feature[:, :, 2: -2, 2: -2])) / scaling
                else:
                    x_index_feature = x[:,:,:,index_feature]
                    x_index_feature = K.expand_dims(x_index_feature,axis=-1)
                    scaling = K.prod(K.cast(K.shape(x_index_feature), 'float32'))
                    if self.kind_of_optim=='squared':
                        loss = loss + K.sum(K.square(x_index_feature[:, 2: -2, 2: -2, :])) / scaling
                    elif self.kind_of_optim=='pos':
                        loss = loss + K.sum(x_index_feature[:, 2: -2, 2: -2, :]) / scaling
                    elif self.kind_of_optim=='neg':
                        loss = loss - K.sum(x_index_feature[:, 2: -2, 2: -2, :]) / scaling
        
        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
        
        # Set up function to retrieve the value
        # of the loss and gradients given an input image.
        outputs = [loss, grads]
        self.fetch_loss_and_grads = K.function([dream], outputs)      
        
    def gradient_ascent(self,x, iterations, step, max_loss=None,Net='VGG'):
        self.Net = Net
        if 'VGG' in self.Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet' in self.Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        x =  preprocessing_function(x)
        
        for i in range(iterations):
            loss_value, grad_values = self.fetch_loss_and_grads([x])
            if max_loss is not None and loss_value > max_loss:
                break
            #print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x
    
class DeepDream_on_one_specific_layer(object):
    """
    Deep dream on one given layer of the net
    """
    def __init__(self,model,layer_name):
        self.model = model
        self.layer_name = layer_name

        dream = model.input
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layers_all = [layer.name for layer in model.layers]
        if layer_name not in layers_all:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
           
        # Define the loss.
        loss = K.variable(0.)
        for layer_local in  model.layers:
            if layer_local.name==layer_name:
                x = layer_local.output
                
                # We avoid border artifacts by only involving non-border pixels in the loss.
                if K.image_data_format() == 'channels_first':
                    scaling = K.prod(K.cast(K.shape(x), 'float32'))
                    loss = loss + K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
                else:
                    scaling = K.prod(K.cast(K.shape(x), 'float32'))
                    loss = loss + K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
        
        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
        
        # Set up function to retrieve the value
        # of the loss and gradients given an input image.
        outputs = [loss, grads]
        self.fetch_loss_and_grads = K.function([dream], outputs)      
        
    def gradient_ascent(self,x, iterations, step, max_loss=None,Net='VGG'):
        self.Net = Net
        if 'VGG' in self.Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet' in self.Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        x =  preprocessing_function(x)
        
        for i in range(iterations):
            loss_value, grad_values = self.fetch_loss_and_grads([x])
            if max_loss is not None and loss_value > max_loss:
                break
            #print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x
    
#class DeepDream_on_correlation_of_2features(object):
class DeepDream_omeanPointWise_of_2features(object):
    """
    Deep dream on the correlation of two given features maps of a given layer
    """
    def __init__(self,model,layer_name,index_feature1,index_feature2):
        self.model = model
        self.layer_name = layer_name
        self.index_feature1 = index_feature1
        self.index_feature2 = index_feature2
        dream = model.input
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layers_all = [layer.name for layer in model.layers]
        if layer_name not in layers_all:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
           
        # Define the loss.
        loss = K.variable(0.)
        for layer_local in  model.layers:
            if layer_local.name==layer_name:
                x = layer_local.output

                # We avoid border artifacts by only involving non-border pixels in the loss.
                if K.image_data_format() == 'channels_first':
                    raise(NotImplementedError)
                    scaling = K.prod(K.cast(K.shape(x), 'float32'))
                    loss = loss + K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
                else:
                    x_index_feature1 = x[:, 2: -2, 2: -2,index_feature1]
                    x_index_feature2 = x[:, 2: -2, 2: -2,index_feature2]
                    x_index_feature1_flatten = K.flatten(x_index_feature1) 
                    x_index_feature2_flatten = K.flatten(x_index_feature2) 
                    score12 = tf.math.abs(tf.reduce_mean(tf.math.multiply(x_index_feature1_flatten,x_index_feature2_flatten),axis=0))
                    loss = loss + score12
        
        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
        
        # Set up function to retrieve the value
        # of the loss and gradients given an input image.
        outputs = [loss, grads]
        self.fetch_loss_and_grads = K.function([dream], outputs)      
        
    def gradient_ascent(self,x, iterations, step, max_loss=None,Net='VGG'):
        self.Net = Net
        if 'VGG' in self.Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet' in self.Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        x =  preprocessing_function(x)
        
        for i in range(iterations):
            loss_value, grad_values = self.fetch_loss_and_grads([x])
            if max_loss is not None and loss_value > max_loss:
                break
            #print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x
    
class DeepDream_on_squared_pointwiseproduct_of_2features(object):
    """
    Deep dream on the correlation of two given features maps of a given layer
    """
    def __init__(self,model,layer_name,index_feature1,index_feature2):
        self.model = model
        self.layer_name = layer_name
        self.index_feature1 = index_feature1
        self.index_feature2 = index_feature2
        dream = model.input
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layers_all = [layer.name for layer in model.layers]
        if layer_name not in layers_all:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
           
        # Define the loss.
        loss = K.variable(0.)
        for layer_local in  model.layers:
            if layer_local.name==layer_name:
                x = layer_local.output

                # We avoid border artifacts by only involving non-border pixels in the loss.
                if K.image_data_format() == 'channels_first':
                    raise(NotImplementedError)
                    scaling = K.prod(K.cast(K.shape(x), 'float32'))
                    loss = loss + K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
                else:
                    x_index_feature1 = x[:, 2: -2, 2: -2,index_feature1]
                    x_index_feature2 = x[:, 2: -2, 2: -2,index_feature2]
                    x_index_feature1_flatten = K.flatten(x_index_feature1) 
                    x_index_feature2_flatten = K.flatten(x_index_feature2) 
                    sum_squared_12 = tf.reduce_mean(K.square(tf.multiply(x_index_feature1_flatten,x_index_feature2_flatten)),axis=0)
                    
                    loss = loss + sum_squared_12
        
        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
        
        # Set up function to retrieve the value
        # of the loss and gradients given an input image.
        outputs = [loss, grads]
        self.fetch_loss_and_grads = K.function([dream], outputs)      
        
    def gradient_ascent(self,x, iterations, step, max_loss=None,Net='VGG'):
        self.Net = Net
        if 'VGG' in self.Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet' in self.Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        x =  preprocessing_function(x)
        
        for i in range(iterations):
            loss_value, grad_values = self.fetch_loss_and_grads([x])
            if max_loss is not None and loss_value > max_loss:
                break
            #print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x
        
def define_deepDream_model_layerBased(model):
    dream = model.input
    print('Model loaded.')
    
    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    # Define the loss.
    loss = K.variable(0.)
    for layer_name in settings['features']:
        # Add the L2 norm of the features of a layer to the loss.
        if layer_name not in layer_dict:
            raise ValueError('Layer ' + layer_name + ' not found in model.')
        coeff = settings['features'][layer_name]
        x = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
    
    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    
    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

def deepDreamProcess_fromKerasTuto():
    """Process:
    
    - Load the original image.
    - Define a number of processing scales (i.e. image shapes),
        from smallest to largest.
    - Resize the original image to the smallest scale.
    - For every scale, starting with the smallest (i.e. current one):
        - Run gradient ascent
        - Upscale image to the next scale
        - Reinject the detail that was lost at upscaling time
    - Stop when we are back to the original size.
    
    To obtain the detail lost during upscaling, we simply
    take the original image, shrink it down, upscale it,
    and compare the result to the (resized) original image.
    """
    
    
    # Playing with these hyperparameters will also allow you to achieve new effects
    step = 0.01  # Gradient ascent step size
    num_octave = 3  # Number of scales at which to run gradient ascent
    octave_scale = 1.4  # Size ratio between scales
    iterations = 20  # Number of ascent steps per scale
    max_loss = 10.
    
    img = preprocess_image(base_image_path)
    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
    
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
    
    save_img(result_prefix + '.png', deprocess_image(np.copy(img)))       
        
if __name__ == '__main__': 
    DeepDream_withFinedModel()    
        
        
        