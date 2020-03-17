#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:24:07 2020

In this script we will load the Model before and after fine-tuning and do 
some deep-dream of the features maps of the weights that change the most

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


import pickle
import pathlib

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
        diff_squared = diff_filters**2
        mean_squared = np.mean(diff_squared,axis=(0,1,2))
        print('For layer :',layer_name)
        print('Min :',np.min(mean_squared),'Max :',np.max(mean_squared),'Median :',np.median(mean_squared),'last decile :',np.percentile(mean_squared, 90))
        dict_layers_mean_squared[layer_name] = mean_squared
        argsort = np.argsort(mean_squared)[::-1]
        dict_layers_argsort[layer_name] = argsort
        for i in range(3):
            print('Top ',i,':',mean_squared[argsort[i]])
        
    K.set_learning_phase(0)
    
    step = 0.01  # Gradient ascent step size
    iterations = 100  # Number of ascent steps per scale
    
    for layer in net_layers:
        layer_name = layer.name
        if not('conv' in layer_name):
            continue
        argsort = dict_layers_argsort[layer_name]
        for i in range(3):
            index_feature = argsort[i]
            print('Start deep dreaming for ',layer_name,index_feature)
            init_rand_im = np.random.normal(loc=125.,scale=3.,size=(1,224,224,3))
            deepdream_model = DeepDream_on_one_specific_featureMap(net_finetuned,layer_name,index_feature)
            output_image = deepdream_model.gradient_ascent(init_rand_im,iterations,step)
            deprocess_output = deprocess_image(output_image)
            
            result_prefix = 'VGG_finetuned_'+layer_name+'_'+str(index_feature)+'.png'
            name_saved_im = os.path.join(output_path,result_prefix)
            save_img(name_saved_im, np.copy(deprocess_output))
        
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
    
class DeepDream_on_one_specific_featureMap(object):
    def __init__(self,model,layer_name, index_feature):
        self.model = model
        self.layer_name = layer_name
        self.index_feature = index_feature

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
                x_index_feature = x[:,:,:,index_feature]
                # We avoid border artifacts by only involving non-border pixels in the loss.
                scaling = K.prod(K.cast(K.shape(x), 'float32'))
                if K.image_data_format() == 'channels_first':
                    loss = loss + K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
                else:
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
            print('..Loss value at', i, ':', loss_value)
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
        
        
        