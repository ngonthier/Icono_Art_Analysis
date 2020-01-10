#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:23:41 2019

In this script 

@author: gonthier
"""

# Tensorflow functions
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import Activation,Dense,Flatten,Input,Dropout
from tensorflow.python.keras import layers
from tensorflow.python.keras import utils
import utils_keras
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import BatchNormalization,Conv2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Layer,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.backend import expand_dims
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.python.ops import math_ops,array_ops
from tensorflow import dtypes

#from custom_pooling import GlobalMinPooling2D
from lr_multiplier import LearningRateMultiplier
from common.layers import DecorrelatedBN
from keras_resnet_utils import getBNlayersResNet50,getResNet50layersName

# Others libraries
import numpy as np
from PIL import Image
import os
import os.path
import re
from functools import partial
#import time
#import functools

### Multi Layer perceptron
def MLP_model(num_of_classes=10,optimizer='SGD',lr=0.01,verbose=False,num_layers=2,\
              regulOnNewLayer=None,regulOnNewLayerParam=[],dropout=None,\
              nesterov=False,SGDmomentum=0.9,decay=0.0):
  """ Return a MLP model ready to fit
  @param : dropout if None : not dropout otherwise must be a value between 0 and 1
  """
    
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)
    
  if optimizer=='SGD':
      opt = SGD(learning_rate=lr,nesterov=nesterov,momentum=SGDmomentum,decay=decay)
  elif optimizer=='adam':
      opt= Adam(learning_rate=lr,decay=decay)
  elif optimizer=='RMSprop':
      opt= RMSprop(learning_rate=lr,decay=decay,momentum=SGDmomentum)
  model =  tf.keras.Sequential()
  if num_layers==2:
      model.add(Dense(256, activation='relu',kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      model.add(Dense(num_of_classes, activation='sigmoid',kernel_regularizer=regularizers))
  elif num_layers==3:
      model.add(Dense(256, activation='relu', kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      model.add(Dense(128, activation='relu', kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      model.add(Dense(num_of_classes, activation='sigmoid', kernel_regularizer=regularizers))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  
  return(model)
  
### one Layer perceptron
def Perceptron_model(num_of_classes=10,optimizer='SGD',lr=0.01,verbose=False,\
                     regulOnNewLayer=None,regulOnNewLayerParam=[],dropout=None,\
                     nesterov=False,SGDmomentum=0.9,decay=0.0):
    
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)

  if optimizer=='SGD':
      opt = SGD(learning_rate=lr,momentum=SGDmomentum,decay=decay,nesterov=nesterov)
  elif optimizer=='adam': 
      opt = Adam(learning_rate=lr,decay=decay)
  elif optimizer=='RMSprop':
      opt= RMSprop(learning_rate=lr,decay=decay,momentum=SGDmomentum)
  model =  tf.keras.Sequential()
  if not(dropout is None): model.add(Dropout(dropout))
  model.add(Dense(num_of_classes, activation='sigmoid', kernel_regularizer=regularizers))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

  return(model)

def get_regularizers(regulOnNewLayer=None,regulOnNewLayerParam=[]):
  if not(regulOnNewLayer is None):
      if regulOnNewLayer=='l1':
          if len(regulOnNewLayerParam)==0: 
              regularizers = tf.keras.regularizers.l1(0.01)
          else:
              regularizers = tf.keras.regularizers.l1(regulOnNewLayerParam[0])
      elif regulOnNewLayer=='l2':
          if len(regulOnNewLayerParam)==0: 
              regularizers = tf.keras.regularizers.l2(0.01)
          else:
              regularizers = tf.keras.regularizers.l2(regulOnNewLayerParam[0])
      elif regulOnNewLayer=='l1_l2':
          if len(regulOnNewLayerParam)==0:
              regularizers = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
          elif len(regulOnNewLayerParam)==1:
              regularizers = tf.keras.regularizers.l1_l2(l1=regulOnNewLayerParam[0], l2=regulOnNewLayerParam[0])
          else:
              regularizers = tf.keras.regularizers.l1_l2(l1=regulOnNewLayerParam[0], l2=regulOnNewLayerParam[1])
  else:
      regularizers = None
  return(regularizers)

### To fine Tune a VGG
  
def new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,regularizers,dropout):
  if final_clf=='MLP3':
      model.add(Dense(256, activation='relu',kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      model.add(Dense(128, activation='relu',kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      if lr_multiple:
          multipliers[model.layers[-1].name] = None
  elif final_clf=='MLP2':
      model.add(Dense(256, activation='relu',kernel_regularizer=regularizers))
      if not(dropout is None): model.add(Dropout(dropout))
      if lr_multiple:
          multipliers[model.layers[-1].name] = None
  if final_clf in ['MLP3','MLP2','MLP1']:
      if final_clf=='MLP1':
          if not(dropout is None): model.add(Dropout(dropout))
      model.add(Dense(num_of_classes, activation='sigmoid',kernel_regularizer=regularizers))
      if lr_multiple:
          multipliers[model.layers[-1].name] = None
          opt = LearningRateMultiplier(opt, lr_multipliers=multipliers, learning_rate=lr)  
      else:
          opt = opt(learning_rate=lr)
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])  
  return(model)     
  
def VGG_baseline_model(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                       pretrainingModif=True,verbose=False,weights='imagenet',optimizer='adam',\
                       opt_option=[0.01],freezingType='FromTop',final_clf='MLP2',\
                       final_layer='block5_pool',regulOnNewLayer=None,regulOnNewLayerParam=[],\
                       dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0): 
  """
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  @param : regulOnNewLayer used on kernel_regularizer 
  """
  # create model
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)
  
  model =  tf.keras.Sequential()
  pre_model = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  SomePartFreezed = False
  
  list_name_layers = []
  for layer in pre_model.layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
  
  if type(pretrainingModif)==bool:
      pre_model.trainable = pretrainingModif
  else:
      SomePartFreezed = True # We will unfreeze pretrainingModif==int layers from the end of the net
      number_of_trainable_layers =  16
      assert(number_of_trainable_layers >= pretrainingModif)
  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt = optimizer

  ilayer = 0
  for layer in pre_model.layers:
     if SomePartFreezed and 'conv' in layer.name:
         if freezingType=='FromTop':
             if ilayer >= number_of_trainable_layers - pretrainingModif:
                 layer.trainable = True
             else:
                 layer.trainable = False
         elif freezingType=='FromBottom':
             if ilayer < pretrainingModif:
                 layer.trainable = True
             else:
                 layer.trainable = False
         elif freezingType=='Alter':
             pretrainingModif_bottom = pretrainingModif//2
             pretrainingModif_top = pretrainingModif//2 + pretrainingModif%2
             if (ilayer < pretrainingModif_bottom) or\
                 (ilayer >= number_of_trainable_layers - pretrainingModif_top):
                 layer.trainable = True
             else:
                 layer.trainable = False
         ilayer += 1
         model.add(layer)
     else:
         model.add(layer)
     if lr_multiple and layer.trainable:
         multipliers[layer.name] = multiply_lrp
     if layer.name==final_layer:
         if not(final_layer in  ['fc2','fc1','flatten']):
              if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
                  model.add(GlobalMaxPooling2D()) 
              elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
                  model.add(GlobalAveragePooling2D())
              elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
                  model.add(Flatten())
         break
  
  #model.add(tf.keras.layers.Lambda(lambda x :  tf.Print(x, [x,tf.shape(x)])))
    
  model = new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,regularizers,dropout)

  if verbose: print(model.summary())
  return model

### VGG adaptation for new dataset

def vgg_FRN(style_layers,num_of_classes=10,\
              transformOnFinalLayer='GlobalMaxPooling2D',getBeforeReLU=True,verbose=False,\
              weights='imagenet',final_clf='MLP2',final_layer='block5_pool',\
              optimizer='adam',opt_option=[0.01],regulOnNewLayer=None,regulOnNewLayerParam=[],\
              dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0):
  """
  VGG with an  Filter  Response  Normalization  with  Thresh-olded Activation layer 
  are the only learnable parameters
  with the last x dense layer 
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  Idea from https://arxiv.org/pdf/1911.09737v1.pdf
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False

  custom_objects = {}
  custom_objects['FRN_Layer']= FRN_Layer
  
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)

  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  else:
      opt = optimizer
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
  
  i = 0  
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      new_layer = FRN_Layer() # This layer include a activation function
      new_layer.trainable = True
      model.add(new_layer) 

      i += 1
    else:
      model.add(layer)
      
    if layer.name==final_layer:
      if not(final_layer in  ['fc2','fc1','flatten']):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
          elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
              model.add(Flatten())
      break

  model = new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,
                           regularizers,dropout)

  if getBeforeReLU:# refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=True,needFix = True,
                                              custom_objects=custom_objects)

  if verbose: print(model.summary())
  return model

def vgg_AdaIn(style_layers,num_of_classes=10,\
              transformOnFinalLayer='GlobalMaxPooling2D',getBeforeReLU=True,verbose=False,\
              weights='imagenet',final_clf='MLP2',final_layer='block5_pool',\
              optimizer='adam',opt_option=[0.01],regulOnNewLayer=None,regulOnNewLayerParam=[],\
              dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0,\
              list_mean_and_std_source=None,list_mean_and_std_target=None):
  """
  VGG with an Instance normalisation learn only those are the only learnable parameters
  with the last x dense layer 
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  
  if list_mean_and_std_source is None or list_mean_and_std_target is None:
      ParamInitialision = False
  else:
      ParamInitialision = True
  
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)

  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt = optimizer
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
      
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      if not(ParamInitialision):
          bn_layer = layers.BatchNormalization(axis=-1, center=True, scale=True)
      else:
          # Scaling parameters
          beta_initializer = tf.constant_initializer(list_mean_and_std_source[i][0])
          gamma_initializer = tf.constant_initializer(list_mean_and_std_source[i][1])
          # Normalisation parameters
          moving_mean_initializer = tf.constant_initializer(list_mean_and_std_target[i][0])
          moving_variance_initializer = tf.constant_initializer(np.square(list_mean_and_std_target[i][1]))
          bn_layer = layers.BatchNormalization(axis=-1, center=True, scale=True,\
                                               beta_initializer=beta_initializer,\
                                               gamma_initializer=gamma_initializer,\
                                               moving_mean_initializer=moving_mean_initializer,\
                                               moving_variance_initializer=moving_variance_initializer)
      model.add(bn_layer)
        
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)
    if layer.name==final_layer:
      if not(final_layer in  ['fc2','fc1','flatten']):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
          elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
              model.add(Flatten())
      break
  
  model = new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,regularizers,dropout)


  if getBeforeReLU:# refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=True,needFix = True)
  
  if verbose: print(model.summary())
  return model

def vgg_adaDBN(style_layers,num_of_classes=10,\
              transformOnFinalLayer='GlobalMaxPooling2D',getBeforeReLU=True,verbose=False,\
              weights='imagenet',final_clf='MLP2',final_layer='block5_pool',\
              optimizer='adam',opt_option=[0.01],regulOnNewLayer=None,regulOnNewLayerParam=[],\
              dbn_affine=True,m_per_group=16,dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0):
  """
  VGG with some decorrelated  learn only those are the only learnable parameters
  with a 2 dense layer MLP or one layer MLP according to the final_clf parameters
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)

  custom_objects = {}
  custom_objects['DecorrelatedBN']= DecorrelatedBN

  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      multipliers = {}
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt = optimizer
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)

  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
      
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      model.add(DecorrelatedBN(m_per_group=m_per_group, affine=dbn_affine,name='decorrelated_bn'+str(i)))
        
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)
    if layer.name==final_layer:
      if not(final_layer in  ['fc2','fc1','flatten']):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
          elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
              model.add(Flatten())
      break
  
  model = new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,regularizers,dropout)

  if getBeforeReLU:# refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=True,needFix = True,\
                                              custom_objects=custom_objects)
  
  if verbose: print(model.summary())
  return model

def vgg_suffleInStats(style_layers,num_of_classes=10,\
              transformOnFinalLayer='GlobalMaxPooling2D',getBeforeReLU=True,verbose=False,\
              weights='imagenet',final_clf='MLP2',final_layer='block5_pool',\
              optimizer='adam',opt_option=[0.01],regulOnNewLayer=None,regulOnNewLayerParam=[],\
              dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0,kind_of_shuffling='shuffle'):
  """
  VGG with a shuffling the mean and standard deviation of the features maps between
  instance
  In this case the statistics are shuffle the statistics of features maps at each layer concerned 
  there is not sharing in the shuffling between layers
  """
  # TODO : faire une multiplication par du bruit des statistics
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  regularizers=get_regularizers(regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam)

  custom_objects = {}
  if kind_of_shuffling=='shuffle':
      custom_objects['Shuffle_MeanAndVar']= Shuffle_MeanAndVar
  if kind_of_shuffling=='roll':
      custom_objects['Roll_MeanAndVar']= Roll_MeanAndVar

  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      multipliers = {}
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt = optimizer
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)    
    
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      if kind_of_shuffling=='shuffle':
          model.add(Shuffle_MeanAndVar())
      if kind_of_shuffling=='roll':
          model.add(Roll_MeanAndVar())
        
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)
    if layer.name==final_layer:
      if not(final_layer in  ['fc2','fc1','flatten']):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
          elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
              model.add(Flatten())
      break
  
  model = new_head_VGGcase(model,num_of_classes,final_clf,lr,lr_multiple,multipliers,opt,regularizers,dropout)

  if getBeforeReLU:# refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=True,needFix = True,\
                                              custom_objects=custom_objects)
  
  if verbose: print(model.summary())
  return model


### ResNet baseline
  
def new_head_ResNet(pre_model,x,final_clf,num_of_classes,multipliers,lr_multiple,lr,opt,dropout):
  if final_clf=='MLP2' or final_clf=='MLP3' :
      x = Dense(256, activation='relu')(x)
      if not(dropout is None): x = Dropout(dropout)(x)
  if final_clf=='MLP3' :
      x = Dense(128, activation='relu')(x)
      if not(dropout is None): x = Dropout(dropout)(x)
  if final_clf=='MLP2' or final_clf=='MLP1' or final_clf=='MLP3':
      predictions = Dense(num_of_classes, activation='sigmoid')(x)
      if final_clf=='MLP1':
          if not(dropout is None): x = Dropout(dropout)(x)
  model = Model(inputs=pre_model.input, outputs=predictions)
  if lr_multiple:
      if final_clf=='MLP3': multipliers[model.layers[-3].name] = None
      if final_clf=='MLP3' or final_clf=='MLP2': multipliers[model.layers[-2].name] = None
      if final_clf=='MLP3' or final_clf=='MLP2' or final_clf=='MLP1': multipliers[model.layers[-1].name] = None
      opt = LearningRateMultiplier(opt, lr_multipliers=multipliers, learning_rate=lr)
  else:
      opt = opt(learning_rate=lr)
      
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return(model)

def ResNet_baseline_model(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             pretrainingModif=True,verbose=True,weights='imagenet',res_num_layers=50,\
                             optimizer='adam',opt_option=[0.01],freezingType='FromTop',final_clf='MLP2',\
                             dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0): 
  """
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
  SomePartFreezed = False
  if type(pretrainingModif)==bool:
      pre_model.trainable = pretrainingModif
  else:
      SomePartFreezed = True # We will unfreeze pretrainingModif==int layers from the end of the net
      assert(number_of_trainable_layers >= pretrainingModif)
  
  lr_multiple = False
  multipliers = {}
  multiply_lrp, lr = None,None
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt =  optimizer
      
  ilayer = 0
  for layer in pre_model.layers:
      if SomePartFreezed and (layer.count_params() > 0):
         
         if freezingType=='FromTop':
             if ilayer >= number_of_trainable_layers - pretrainingModif:
                 layer.trainable = True
                 #print('FromTop',layer.name,ilayer,pretrainingModif,number_of_trainable_layers - pretrainingModif)
             else:
                 layer.trainable = False
         elif freezingType=='FromBottom':
             if ilayer < pretrainingModif:
                 layer.trainable = True
                 #print('FromBottom',layer.name,ilayer,pretrainingModif,number_of_trainable_layers - pretrainingModif)
             else:
                 layer.trainable = False
         elif freezingType=='Alter':
             pretrainingModif_bottom = pretrainingModif//2
             pretrainingModif_top = pretrainingModif//2 + pretrainingModif%2
             if (ilayer < pretrainingModif_bottom) or\
                 (ilayer >= number_of_trainable_layers - pretrainingModif_top):
                 layer.trainable = True
                 #print('Alter',layer.name,ilayer,pretrainingModif_bottom,pretrainingModif_top)
             else:
                 layer.trainable = False
                 
      ilayer += 1
      # Only if the layer have some trainable parameters
      if lr_multiple and layer.trainable: 
          multipliers[layer.name] = multiply_lrp

  x = pre_model.output
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
  
  model = new_head_ResNet(pre_model,x,final_clf,num_of_classes,multipliers,lr_multiple,lr,opt,dropout)
 
  if verbose: print(model.summary())
  return model

# For Feature extraction
def ResNet_cut(final_layer='activation_48',transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=False,weights='imagenet',res_num_layers=50): 
  """
  To use the Resnet as a features extractor
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
 
  Return a keras model that can be use for features extraction  
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
      
  if not(final_layer in getResNet50layersName()):
      print(final_layer,'is not in ResNet')
      raise(NotImplementedError)
  if not(transformOnFinalLayer in [None,'','GlobalMaxPooling2D','GlobalAveragePooling2D']):
      print(transformOnFinalLayer,'is unknwon')
      raise(NotImplementedError)
  
  pre_model .trainable = False
  for layer in pre_model.layers:
      if layer.name==final_layer:
          x = layer.output

  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
  
  model = Model(inputs=pre_model.input, outputs=x)
 
  if verbose: print(model.summary())
  return model

def set_momentum_BN(model,momentum):
    for layer in model.layers:
      if 'bn' in layer.name:
          layer.momentum = momentum
    return(model)

### Resnet adaptative layers 

def ResNet_AdaIn(style_layers,final_layer='activation_48',num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=True,weights='imagenet',\
                             res_num_layers=50,\
                             optimizer='adam',opt_option=[0.01],\
                             final_clf='MLP2',dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0): 
  """
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  We only allow to train the layer in the style_layers listt
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
  
  if not(final_layer in getResNet50layersName()):
      print(final_layer,'is not in ResNet')
      raise(NotImplementedError)
  if not(transformOnFinalLayer in [None,'','GlobalMaxPooling2D','GlobalAveragePooling2D']):
      print(transformOnFinalLayer,'is unknwon')
      raise(NotImplementedError)
    
  lr_multiple = False
  multiply_lrp, lr  = None,None
  multipliers = {}
  if len(opt_option)==2:
      multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
      multipliers = {}
      lr_multiple = True
  elif len(opt_option)==1:
      lr = opt_option[-1]
  else:
      lr = 0.01
  if optimizer=='SGD': 
      opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
  elif optimizer=='adam': 
      opt = partial(Adam,decay=decay)
  elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
  else:
      opt = optimizer

  for layer in pre_model.layers:
      if layer.name in style_layers:
          layer.trainable = True
          if lr_multiple: 
              multipliers[layer.name] = multiply_lrp
      else:
          layer.trainable = False
      if final_layer==layer.name:
          x = layer.output

#  x = pre_model.output
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
  
  model = new_head_ResNet(pre_model,x,final_clf,num_of_classes,multipliers,lr_multiple,lr,opt,dropout)
 
  if verbose: print(model.summary())
  return model

def ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(style_layers,list_mean_and_std_target,\
                                                         final_layer,\
                                   transformOnFinalLayer=None,res_num_layers=50,\
                                   weights='imagenet',verbose=True):
  """
  VGG with an Instance normalisation : we impose the mean and std of reference 
  instance per instance
  @param : final_layer final layer provide for feature extraction
  @param : transformOnFinalLayer : on va modifier la derniere couche du réseau
  @param : getBeforeReLU if True we modify the features before the ReLU
  TODO : find why this function fail first time run
  """
  if style_layers is None :
      if not(list_mean_and_std_target is None):
          assert(len(list_mean_and_std_target)==0)
  elif list_mean_and_std_target is None:
      assert(len(style_layers)==0)
  else:
      assert(len(style_layers)==len(list_mean_and_std_target))
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=weights,\
                                                          input_shape= (224, 224, 3))
#      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
      
  if not(final_layer in getResNet50layersName()):
      print(final_layer,'is not in ResNet')
      raise(NotImplementedError)
  if not(transformOnFinalLayer in [None,'','GlobalMaxPooling2D','GlobalAveragePooling2D']):
      print(transformOnFinalLayer,'is unknwon')
      raise(NotImplementedError)
      
  i = 0
  sess = K.get_session()
  for layer in pre_model.layers:
      name_layer = layer.name
      if i < len(style_layers) and name_layer==style_layers[i]:
          
          mean_src = list_mean_and_std_target[i][0] 
          std_src = list_mean_and_std_target[i][1] 
          # replace the statictics used for normalisation in the batch normalisation
          sess.run(layer.moving_mean.assign(mean_src))
          sess.run(layer.moving_variance.assign(std_src**2))

          i += 1

      if name_layer==final_layer:
          x = layer.output
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
             x = GlobalMaxPooling2D()(x)
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              x = GlobalAveragePooling2D()(x)
          elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
              x= Flatten()(x)
          break
      
  pre_model.trainable = False

  try:
      model = Model(inputs=pre_model.input, outputs=x)
  except UnboundLocalError as e:
      print('UnboundLocalError')
      print(name_layer,final_layer)
      for l in pre_model.layers:
          print(l.name)
      raise(e)
 
  return model
  
def add_head_and_trainable(pre_model,num_of_classes,optimizer='adam',opt_option=[0.01],\
                             final_clf='MLP2',dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0,\
                             verbose=False,AdaIn_mode=False,style_layers=[]):
    """
    This function makes the model trainable and add it a head (MLP at 1 2 or 3 layers)
    @param AdaIn_mode : if True means that only batch normalisation are trainable
    """
    
    lr_multiple = False
    multiply_lrp, lr  = None,None
    multipliers = {}
    if len(opt_option)==2:
        multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
        multipliers = {}
        lr_multiple = True
    elif len(opt_option)==1:
        lr = opt_option[-1]
    else:
        lr = 0.01
    if optimizer=='SGD': 
        opt = partial(SGD,momentum=SGDmomentum, nesterov=nesterov,decay=decay)# SGD
    elif optimizer=='adam': 
        opt = partial(Adam,decay=decay)
    elif optimizer=='RMSprop':
      opt= partial(RMSprop,learning_rate=lr,decay=decay,momentum=SGDmomentum)
    else:
        opt = optimizer
        
    if AdaIn_mode:
      for layer in pre_model.layers:
          if layer.name in style_layers:
              layer.trainable = True
              if lr_multiple: 
                  multipliers[layer.name] = multiply_lrp
          else:
              layer.trainable = False
    else:
        pre_model.trainable = True
        for layer in pre_model.layers:
          if lr_multiple: 
              multipliers[layer.name] = multiply_lrp
    
    x= pre_model.output
    model = new_head_ResNet(pre_model,x,final_clf,num_of_classes,multipliers,lr_multiple,lr,opt,dropout)
    if verbose: print(model.summary())
    return(model)

### Resnet Refinement of the batch normalisation statistics 

class HomeMade_BatchNormalisation_Refinement(Layer):
    """
    HomeMade Batch Normalisation function that only update the whitening / 
    normalizing parameters
    """

    def __init__(self,batchnorm_layer,momentum, **kwargs):
        #self.moving_mean = batchnorm_layer.non_trainable_variables[0]
        self.moving_mean_init = tf.keras.backend.eval(batchnorm_layer.moving_mean)
        self.gamma = tf.keras.backend.eval(batchnorm_layer.gamma)
        #self.moving_variance = batchnorm_layer.non_trainable_variables[1]
        self.moving_variance_init = tf.keras.backend.eval(batchnorm_layer.moving_variance)
        self.beta = tf.keras.backend.eval(batchnorm_layer.beta)
        self.epsilon = tf.keras.backend.eval(batchnorm_layer.epsilon)
        #batchnorm_layer.trainable = False
        #self.batchnorm_layer=batchnorm_layer
        self.momentum = momentum 
        super(HomeMade_BatchNormalisation_Refinement, self).__init__(**kwargs)

    def build(self,input_shape):
        self.moving_mean = self.add_weight(name='moving_mean',
                                  shape=self.moving_mean_init.shape,
                                  initializer=tf.constant_initializer(value=self.moving_mean_init),
                                  #UNTRAINABLE 
                                  trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                  shape=self.moving_variance_init.shape,
                                  initializer=tf.constant_initializer(value=self.moving_variance_init),
                                  #UNTRAINABLE 
                                  trainable=False)
        super(HomeMade_BatchNormalisation_Refinement, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        #output = self.batchnorm_layer(x)
        mean,variance = tf.nn.moments(x,axes=(0,1,2),keep_dims=False)
        batch_norm = (x - mean) * math_ops.rsqrt(variance + self.epsilon)
        batch_norm *= self.gamma
        output =  batch_norm + self.beta
        # Version sous optimisee bien sur
        update_moving_mean = tf.keras.backend.moving_average_update(x=self.moving_mean,\
                                               value=mean,\
                                               momentum=self.momentum) # Returns An Operation to update the variable.
        update_moving_variance = tf.keras.backend.moving_average_update(x=self.moving_variance,\
                                               value=variance,\
                                               momentum=self.momentum) # Returns An Operation to update the variable.
        #self.add_update([update_moving_mean,update_moving_variance],x)
        self.add_update(update_moving_mean)
        self.add_update(update_moving_variance)
        #self.add_update((self.moving_mean, update_moving_mean), x)
        #self.add_update((self.moving_variance, update_moving_variance), x)
        
        return(output)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return(input_shape)
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(HomeMade_BatchNormalisation_Refinement, self).get_config()
        config['moving_mean_init'] = self.moving_mean_init
        config['moving_mean'] = self.moving_mean
        config['gamma'] = self.gamma
        config['beta'] = self.beta
        config['moving_variance_init'] = self.moving_variance_init
        config['moving_variance'] = self.moving_variance
        config['batchnorm_layer'] = self.batchnorm_layer
        config['momentum'] = self.momentum
        config['epsilon'] = self.epsilon
        return(config) 

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

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

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
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]

        if len(layer_input) >1:
            layer_input = layers_unique(layer_input)
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory
            # new_layer = insert_layer_factory() 
            # Change by Nicolas Gonthier
#            if insert_layer_name:
#                new_layer.name = insert_layer_name
#            else:
#                new_layer.name = '{}_{}'.format(layer.name, 
#                                                new_layer.name)
            x = new_layer(x)
#            print('Layer {} inserted after layer {}'.format(new_layer.name,
#                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)

def ResNet_BNRefinements_Feat_extractor_old(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=True,weights='imagenet',\
                             res_num_layers=50,momentum=0.9,kind_method='TL'): 
  """
  ResNet with BN statistics refinement and then features extraction TL
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  @param : momentum in the updating of the statistics of BN 
  BNRF
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  
  print('ResNet_BNRefinements_Feat_extractor does not work sorry, you have to use the ResNet50 base model')
#  raise(NotImplementedError)
  
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
      bn_layers = getBNlayersResNet50()
      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)

  print('beginning',len(pre_model.updates))
  for layer in pre_model.layers:
      layer.trainable = False
      if layer.name in bn_layers:
          
          new_bn = HomeMade_BatchNormalisation_Refinement(layer,momentum)
          layer_regex = layer.name
          insert_layer_name =  layer.name +'_rf'
          pre_model = insert_layer_nonseq(pre_model, layer_regex, new_bn,
                        insert_layer_name=insert_layer_name, position='replace')   
          print(layer.name,len(pre_model.updates))
          #print(layer.name,pre_model.updates)
          # print(pre_model.summary())
          #input('wait')
#   
#      
#      if layer.name in style_layers:
#          layer.trainable = True
#          if lr_multiple: 
#              multipliers[layer.name] = multiply_lrp
#      else:
#          layer.trainable = False
#
  x = pre_model.output
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
  predictions = x 
#  
#  if final_clf=='MLP2':
#      x = Dense(256, activation='relu')(x)
#  if final_clf=='MLP2' or final_clf=='MLP1':
#      predictions = Dense(num_of_classes, activation='sigmoid')(x)
  model = Model(inputs=pre_model.input, outputs=predictions)
  print('Finale updates lits',len(model.updates))
#  if lr_multiple:
#      multipliers[model.layers[-2].name] = None
#      multipliers[model.layers[-1].name] = None
#      opt = LearningRateMultiplier(opt, lr_multipliers=multipliers, learning_rate=lr)
#  else:
#      opt = opt(learning_rate=lr)
  # Compile model
  #pre_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  if verbose: print(model.summary())
  
  return model

def ResNet_BNRefinements_Feat_extractor(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=True,weights='imagenet',\
                             res_num_layers=50,momentum=0.9,kind_method='TL'): 
  """
  ResNet with BN statistics refinement and then features extraction TL
  Solution adopte remplacer les batch normalisation par de nouvelles 
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  @param : momentum in the updating of the statistics of BN 
  BNRF
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  
  print('ResNet_BNRefinements_Feat_extractor does not work sorry, you have to use the ResNet50 base model')
#  raise(NotImplementedError)
  
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
      bn_layers = getBNlayersResNet50()
      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)

  print('beginning new fct',len(pre_model.updates))
  for layer in pre_model.layers:
      layer.trainable = False
      if layer.name in bn_layers:
          gamma_initializer = tf.constant_initializer(value=tf.keras.backend.eval(layer.gamma))
          beta_initializer = tf.constant_initializer(value=tf.keras.backend.eval(layer.beta))
          moving_variance_initializer = tf.constant_initializer(value=tf.keras.backend.eval(layer.moving_variance))
          moving_mean_initializer = tf.constant_initializer(value=tf.keras.backend.eval(layer.moving_mean))
          new_bn =  BatchNormalization(axis=-1,
                                        momentum=momentum,
                                        epsilon=layer.epsilon,
                                        center=layer.center,
                                        scale=layer.scale,
                                        beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer,
                                        moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer,
                                        beta_regularizer=layer.beta_regularizer,
                                        gamma_regularizer=layer.gamma_regularizer,
                                        beta_constraint=layer.beta_constraint,
                                        gamma_constraint=layer.gamma_constraint,
                                        renorm=layer.renorm,
                                        fused=layer.fused,
                                        trainable=True,
                                        virtual_batch_size=layer.virtual_batch_size,
                                        adjustment=layer.adjustment,
                                        name=layer.name)
          layer_regex = layer.name
          insert_layer_name =  layer.name
          pre_model = insert_layer_nonseq(pre_model, layer_regex, new_bn,
                        insert_layer_name=insert_layer_name, position='replace')   
          print(layer.name,len(pre_model.updates))
          pre_model=utils_keras.apply_modifications(pre_model,include_optimizer=False, custom_objects=None,needFix = False)
          print(layer.name,len(pre_model.updates))
          #print(layer.name,pre_model.updates)
          # print(pre_model.summary())
          #input('wait')
#   
#      
#      if layer.name in style_layers:
#          layer.trainable = True
#          if lr_multiple: 
#              multipliers[layer.name] = multiply_lrp
#      else:
#          layer.trainable = False
#
  x = pre_model.output
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
  predictions = x 
#  
#  if final_clf=='MLP2':
#      x = Dense(256, activation='relu')(x)
#  if final_clf=='MLP2' or final_clf=='MLP1':
#      predictions = Dense(num_of_classes, activation='sigmoid')(x)
  model = Model(inputs=pre_model.input, outputs=predictions)
  print('Finale updates lits',len(model.updates))
#  if lr_multiple:
#      multipliers[model.layers[-2].name] = None
#      multipliers[model.layers[-1].name] = None
#      opt = LearningRateMultiplier(opt, lr_multipliers=multipliers, learning_rate=lr)
#  else:
#      opt = opt(learning_rate=lr)
  # Compile model
  #pre_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  if verbose: print(model.summary())
  
  return model

### ResNet with Batch normalisation splitted 
def ResNet_Split_batchNormalisation(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=True,weights='imagenet',\
                             res_num_layers=50,momentum=0.9,kind_method='TL'): 
  """
  ResNet with BN statistics refinement and then features extraction TL
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  @param : momentum in the updating of the statistics of BN 
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
      bn_layers = getBNlayersResNet50()
      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)

  for layer in pre_model.layers:
      layer.trainable = False
      if layer.name in bn_layers:
          new_bn = HomeMade_BatchNormalisation_Refinement(layer,momentum)
          layer_regex = layer.name
          insert_layer_name =  layer.name +'_rf'
          pre_model = insert_layer_nonseq(pre_model, layer_regex, new_bn,
                        insert_layer_name=insert_layer_name, position='replace')          
#          
#      
#      if layer.name in style_layers:
#          layer.trainable = True
#          if lr_multiple: 
#              multipliers[layer.name] = multiply_lrp
#      else:
#          layer.trainable = False
#
  x = pre_model.output
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      x= Flatten()(x)
#  
#  if final_clf=='MLP2':
#      x = Dense(256, activation='relu')(x)
#  if final_clf=='MLP2' or final_clf=='MLP1':
#      predictions = Dense(num_of_classes, activation='sigmoid')(x)
#  model = Model(inputs=pre_model.input, outputs=predictions)
#  if lr_multiple:
#      multipliers[model.layers[-2].name] = None
#      multipliers[model.layers[-1].name] = None
#      opt = LearningRateMultiplier(opt, lr_multipliers=multipliers, learning_rate=lr)
#  else:
#      opt = opt(learning_rate=lr)
  # Compile model

  #pre_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  if verbose: print(pre_model.summary())
  return pre_model

### Get features gram matrix and mean of features before the layer in the style layers list 
def get_ResNetmodel_gram_mean_features_layerBefore(style_layers,res_num_layers=50,weights='imagenet'):
  """Helper function to compute the Gram matrix and mean of feature representations 
  from resnet50.
  
  Get features gram matrix and mean of features before the layer in the style layers list 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the model that return the features !
  """
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights,\
                                                          input_shape= (224, 224, 3))
#      bn_layers = getBNlayersResNet50()
#      number_of_trainable_layers = 106
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)

  list_stats = []
  last_layer = None
  for layer in pre_model.layers:
      if layer.name in style_layers:
          if not(last_layer is None):
              output = last_layer.output
          else: 
              output = pre_model.input
          mean_layer = Mean_Matrix_Layer()(output)
          cov_layer = Cov_Matrix_Layer()([output,mean_layer])
          list_stats += [cov_layer,mean_layer]
      else:
          last_layer = layer
  
  model = models.Model(pre_model.input,list_stats)
  model.trainable = False
  return(model)
  
def get_ResNet_ROWD_gram_mean_features(style_layers_exported,style_layers_imposed,\
                                    list_mean_and_std_target,transformOnFinalLayer=None,res_num_layers=50,
                                    weights='imagenet'):
  """Helper function to compute the Gram matrix and mean of feature representations 
  from a modified resnet50. that have the features maps modified
  
  The gram matrices are computed 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the model that return the features ! 
  """
  
  pre_model = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(style_layers=style_layers_imposed,\
                                   list_mean_and_std_target=list_mean_and_std_target,\
                                   final_layer=style_layers_exported[-1],\
                                   transformOnFinalLayer=transformOnFinalLayer,res_num_layers=res_num_layers,\
                                   weights=weights)
  pre_model.trainable = False

  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  last_layer = None
  for layer in pre_model.layers:
      name_layer = layer.name
      print(name_layer)
      if name_layer==style_layers_exported[i]:
#          if not(last_layer is None):
#              output = last_layer.output
#          else: 
#              output = pre_model.input
          output = layer.input
          #print(output)
          mean_layer = Mean_Matrix_Layer()(output)
          cov_layer = Cov_Matrix_Layer()([output,mean_layer])
          list_stats += [cov_layer,mean_layer]
          i+= 1
      else:
          last_layer = layer
      
      if i==len(style_layers_exported): # No need to compute further
          break
  
  model = models.Model(pre_model.input,list_stats)
  model.trainable = False
  return(model)
  
def get_ResNet_ROWD_meanX_meanX2_features(style_layers_exported,style_layers_imposed,\
                                    list_mean_and_std_target,transformOnFinalLayer=None,res_num_layers=50,
                                    weights='imagenet',useFloat32=False):
  """Helper function to compute the Mean of feature and features square representations 
  from a modified resnet50. that have the features maps modified
  
  The gram matrices are computed 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the model that return the features ! 
  """
  
  
  
  pre_model = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(style_layers=style_layers_imposed,\
                                   list_mean_and_std_target=list_mean_and_std_target,\
                                   final_layer=style_layers_exported[-1],\
                                   transformOnFinalLayer=transformOnFinalLayer,\
                                   res_num_layers=res_num_layers,\
                                   weights=weights)
  pre_model.trainable = False

  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  last_layer = None
  for layer in pre_model.layers:
      name_layer = layer.name
      #print(name_layer)
      if name_layer==style_layers_exported[i]:
#          if not(last_layer is None):
#              output = last_layer.output
#          else: 
#              output = pre_model.input
          output = layer.input
          mean_and_meanOfSquared_layer = Mean_and_MeanSquare_Layer(useFloat32=useFloat32)(output)
          list_stats += [mean_and_meanOfSquared_layer]
          i+= 1
      else:
          last_layer = layer
      
      if i==len(style_layers_exported): # No need to compute further
          break
  
  model = models.Model(pre_model.input,list_stats)
  model.trainable = False
  return(model)

def extract_Norm_stats_of_ResNet(model,res_num_layers=50,model_type='normal'):
    """
    Return the normalisation statistics of a ResNet model
    it return the mean and the std
    """
    if res_num_layers==50:
      number_of_trainable_layers = 106
      if model_type=='normal' or model_type=='ResNet50' or model_type=='BNRF' or model_type=='ResNet50_BNRF' or model_type=='ResNet50_ROWD_CUMUL_AdaIn' or model_type=='ResNet50_ROWD_CUMUL':
          list_bn_layers = getBNlayersResNet50()
          list_bn_layers_name_for_dict = list_bn_layers
#      elif model_type=='BNRF' or model_type=='ResNet50_BNRF':
#          list_bn_layers_name_for_dict = getBNlayersResNet50()
#          list_bn_layers = ['home_made__batch_normalisation__refinement']
#          for i in range(len(list_bn_layers_name_for_dict)-1):
#              list_bn_layers += ['home_made__batch_normalisation__refinement_'+str(i+1)]
#          assert(len(list_bn_layers_name_for_dict)==len(list_bn_layers))
      else:
          raise(NotImplementedError)
      correspondance_names = {}
      for layer_bn,layer_dict in zip(list_bn_layers,list_bn_layers_name_for_dict):
          correspondance_names[layer_bn] = layer_dict
    else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
    current_list_mean_and_std_target = []
    dict_stats_coherent = {}
    for layer in model.layers:
      name_layer = layer.name
      
      if name_layer in list_bn_layers:
          batchnorm_layer = model.get_layer(name_layer)
          moving_mean = batchnorm_layer.moving_mean
          moving_variance = batchnorm_layer.moving_variance
          moving_mean_eval = tf.keras.backend.eval(moving_mean)
          moving_std_eval = np.sqrt(tf.keras.backend.eval(moving_variance))
          mean_and_std_layer = moving_mean_eval,moving_std_eval
          dict_stats_coherent[correspondance_names[name_layer]] = mean_and_std_layer
          current_list_mean_and_std_target += [mean_and_std_layer]
    return(dict_stats_coherent,current_list_mean_and_std_target)
  
### Preprocessing functions 

def load_resize(path_to_img,max_dim = 224):
  img = Image.open(path_to_img)
  img = img.resize((max_dim, max_dim), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def load_resize_and_process_img(path_to_img,Net,max_dim = 224):
 # img = load_resize(path_to_img,max_dim=max_dim)
 
  img = tf.keras.preprocessing.image.load_img(
    path_to_img,
    grayscale=False,
    target_size=(max_dim, max_dim),
    interpolation='nearest')
  # This way to load the images and to convert them to numpy array before preprocessing
  # are certainly suboptimal
  img = kp_image.img_to_array(img)
  img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
  if 'VGG' in Net:
    preprocessing_function = tf.keras.applications.vgg19.preprocess_input
  elif 'ResNet' in Net:
    preprocessing_function = tf.keras.applications.resnet50.preprocess_input
  else:
    print(Net,'is unknwon')
    raise(NotImplementedError)
  img =  preprocessing_function(img)
  return img

def load_and_process_img(path_to_img,Net):
  img = load_img(path_to_img)
  if 'VGG' in Net:
    preprocessing_function = tf.keras.applications.vgg19.preprocess_input
  elif 'ResNet' in Net:
    preprocessing_function = tf.keras.applications.resnet50.preprocess_input
  else:
    print(Net,'is unknwon')
    raise(NotImplementedError)
  img =  preprocessing_function(img)
  return img

def load_img(path_to_img,max_dim = 512):
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def covariance_matrix(features_map, eps=1e-8):
  """
  Compute the covariance matric of a specific features map with shape 1xHxWxC
  Return : covariance matrix and means 
  """
  bs, H, W, C = tf.unstack(tf.shape(features_map))
  # Remove batch dim and reorder to CxHxW
  features_map_squeezed = tf.reshape(features_map,[H, W, C])
  features_map_reshaped = tf.transpose(features_map_squeezed, (2, 0, 1))
  # CxHxW -> CxH*W
  features_map_flat = tf.reshape(features_map_reshaped, (C, H*W))
  mean = tf.cast(tf.reduce_mean(features_map_flat, axis=1, keepdims=True), tf.float32)
  f = features_map_flat - mean
  # Covariance
  cov = tf.matmul(f, f, transpose_b=True) / (tf.cast(H*W, tf.float32) - 1.) + tf.eye(C)*eps                          
  return(cov,mean,f,C,H,W)
  
def covariance_matrix2(features_map, eps=1e-8):
  """
  Compute the covariance matric of a specific features map with shape 1xHxWxC
  Return : covariance matrix and means 
  """
  bs, H, W, C = tf.unstack(tf.shape(features_map))
  # Remove batch dim and reorder to CxHxW
  features_map_squeezed = tf.reshape(features_map,[H, W, C])
  features_map_reshaped = tf.transpose(features_map_squeezed, (2, 0, 1))
  # CxHxW -> CxH*W
  features_map_flat = tf.reshape(features_map_reshaped, (C, H*W))
  mean = tf.cast(tf.reduce_mean(features_map_flat, axis=1, keepdims=True), tf.float32)
  f = features_map_flat - mean
  # Covariance
  cov = tf.matmul(f, f, transpose_b=True) / (tf.cast(H*W, tf.float32)) # + tf.eye(C)*eps                          
  return(cov,mean,f,C,H,W)
  
def covariance_mean_matrix_only(features_map, eps=1e-8):
  """
  Compute the covariance matric of a specific features map with shape 1xHxWxC
  Return : covariance matrix and means 
  """
  bs, H, W, C = tf.unstack(tf.shape(features_map))
  # Remove batch dim and reorder to CxHxW
  features_map_squeezed = tf.reshape(features_map,[H, W, C])
  features_map_reshaped = tf.transpose(features_map_squeezed, (2, 0, 1))
  # CxHxW -> CxH*W
  features_map_flat = tf.reshape(features_map_reshaped, (C, H*W))
  mean = tf.cast(tf.reduce_mean(features_map_flat, axis=1, keepdims=True), tf.float32)
  f = features_map_flat - mean
  # Covariance
  cov = tf.matmul(f, f, transpose_b=True) / (tf.cast(H*W, tf.float32)) # + tf.eye(C)*eps                          
  return(cov,mean)
  
def covariance_matrix_only(features_map,mean, eps=1e-8):
  """
  Compute the covariance matric of a specific features map with shape 1xHxWxC
  Return : covariance matrix and means 
  """
  bs, H, W, C = tf.unstack(tf.shape(features_map))
  # Remove batch dim 
  #features_map_squeezed = tf.reshape(features_map,[H, W, C])
  # reorder to bsxCxHxW
  features_map_reshaped = tf.transpose(features_map, (0,3, 1, 2))
  # bsxCxHxW -> bsxCxH*W
  features_map_flat = tf.reshape(features_map_reshaped, (bs,C, H*W))
  f = features_map_flat - tf.reshape(mean, (bs,C, 1))
  # Covariance
  cov = tf.matmul(f, f, transpose_b=True) / (tf.cast(H*W, tf.float32)) # + tf.eye(C)*eps                          
  return(cov)
                                 
### Whiten-Color Transform ops ###

def wct_tf(new_img, cov_ref,m_ref, eps=1e-8):
    '''TensorFlow version of Whiten-Color Transform
       Assume that content encodings have shape 1xHxWxC
       See p.4 of the Universal Style Transfer paper for corresponding equations:
       https://arxiv.org/pdf/1705.08086.pdf
    '''
    # covariance
    new_img = tf.cast(new_img, tf.float32)
    ff, m, f, C,H,W = covariance_matrix(new_img,eps=eps)
    cov_ref = tf.convert_to_tensor(cov_ref)
    m = tf.convert_to_tensor(m, tf.float32)
    # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
    # TODO : il faudra peut etre meme passer a du numpy car c est plus rapide
    with tf.device('/cpu:0'):  
        S, U, _ = tf.linalg.svd(ff) # Ou tf.svd pour les versions plus anciennes de TF
        S_ref, U_ref, _ = tf.linalg.svd(cov_ref) # Precompute that before !
        
    U_ref = tf.cast(U_ref, tf.float32)

    ## Uncomment to perform SVD for content/style with np in one call
    ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
    # @jit
    # def np_svd(content, style):
    #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    #     Uc, Sc, _ = np.linalg.svd(content)
    #     Us, Ss, _ = np.linalg.svd(style)
    #     return Uc, Sc, Us, Ss
    # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

    # Filter small singular values
    k = tf.reduce_sum(tf.cast(tf.greater(S, 1e-5), tf.int32))
    k_ref = tf.reduce_sum(tf.cast(tf.greater(S_ref, 1e-5), tf.int32))

    # Whiten input feature
    #D = tf.diag(tf.pow(S[:k], -0.5))
    D = tf.linalg.tensor_diag(tf.pow(S[:k], -0.5))
    f_hat = tf.matmul(tf.matmul(tf.matmul(U[:,:k], D), U[:,:k], transpose_b=True), f)

    # Color content with reference
    #D_ref = tf.diag(tf.pow(S_ref[:k_ref], 0.5)) # Pour les codes plus anciens
    D_ref = tf.linalg.tensor_diag(tf.pow(S_ref[:k_ref], 0.5)) 
    D_ref = tf.cast(D_ref, tf.float32)
    fnew_ref_hat = tf.matmul(tf.matmul(tf.matmul(U_ref[:,:k_ref], D_ref), U_ref[:,:k_ref], transpose_b=True), f_hat)
    
    # Re-center with mean of reference
    fnew_ref_hat = tf.add(fnew_ref_hat,tf.reshape(m_ref,(C,1)))

    # CxH*W -> CxHxW
    fnew_ref_hat = tf.reshape(fnew_ref_hat, (C,H,W))
    # CxHxW -> 1xHxWxC
    fnew_ref_hat = tf.expand_dims(tf.transpose(fnew_ref_hat, (1,2,0)), 0)

    return fnew_ref_hat

def get_intermediate_layers_vgg(style_layers,getBeforeReLU=False):
  """ Creates our model with access to intermediate layers. 
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
  Returns:
    returns a keras model that takes image inputs and outputs the style and 
      content intermediate layers. 
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  if not(getBeforeReLU):
      # Get output layers corresponding to style and content layers 
      style_outputs = [vgg.get_layer(name).output for name in style_layers]
      model_outputs = style_outputs
      # Build model 
      model =  models.Model(vgg.input, model_outputs)
  else: # Before ReLU ! 
      model = tf.keras.Sequential()
      i = 0
      vgg_layers = vgg.layers
      style_outputs = []
      for layer in vgg_layers:
          name_layer = layer.name
          if i < len(style_layers) and name_layer==style_layers[i]:
              # remove the non linearity
              layer.activation = activations.linear # i.e. identity
              style_outputs += [layer.output]
              model.add(layer)
              # add back the non linearity
              model.add(Activation('relu'))
              i += 1
          else:
             model.add(layer)
             
      model =  models.Model(vgg.input, style_outputs)     
      # refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=False,needFix = True)
  return(model)
  
def get_those_layers_output(model,list_of_concern_layers):
  """ Creates our model with access to intermediate layers. 
  
  This function will return a model with a list as output, this list will contain
  the output of the considered layer 
  
  You need to provide the specific name of each layer
  """
  model.trainable = False
  model_outputs = [model.get_layer(name).output for name in list_of_concern_layers]

  model =  models.Model(model.input, model_outputs)     
  return(model)

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)


def get_gram_mean_features(model,img_path):
  """Helper function to compute the Gram matrix and mean of feature representations 
  from vgg.
  
  The gram matrices are computed 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the features. 
  """
  # Load our images in 
  image = load_resize_and_process_img(img_path)
  
  # Todo modification here needed !!!! 
  # if TF2.0  you can comment the following line 
  image  = tf.convert_to_tensor(image)
  
  # batch compute content and style features
  outputs = model(image)
  list_stats = []
  for output in outputs:
    cov,mean,f,C,H,W =covariance_matrix2(output, eps=1e-8) 
    list_stats += [[cov,mean]]
  return list_stats

class Cov_Mean_Matrix_Layer(Layer):

    def __init__(self, **kwargs):

        super(Cov_Mean_Matrix_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
#        input_shape = input_shape.as_list()
#        assert isinstance(input_shape, list)
        super(Cov_Mean_Matrix_Layer, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        cov,mean =covariance_mean_matrix_only(x, eps=1e-8) 
        output = concatenate([cov,mean])
        output = expand_dims(output,axis=0)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        b,k1,k2,c = input_shape
#        print('in compute output shape',(b,c, c+1))
        return (b,c, c+1)
    
class Global_Prescrib_Mean_Std(Layer):
    """
    This Layer modify the mean and std  of the features  computed on the full dataset 
        with a precomputed mean and std of an other dataset
    """

    def __init__(self,mean_src,std_src,mean_tgt,std_tgt,epsilon = 0.00001, **kwargs):
        self.mean_src= mean_src # offset ie new mean
        self.mean_tgt= mean_tgt
        self.std_src = std_src # New std
        self.std_tgt = std_tgt
        self.epsilon = epsilon # Small float added to variance to avoid dividing by zero.
        super(Global_Prescrib_Mean_Std, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Global_Prescrib_Mean_Std, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        std_tgt_plus_eps = tf.add(self.std_tgt,self.epsilon) 
        output =  tf.add(tf.divide(tf.multiply(tf.add(x,tf.multiply(self.mean_tgt,-1.0)),self.std_src),(std_tgt_plus_eps)), self.mean_src)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(Global_Prescrib_Mean_Std, self).get_config()
        config['mean_src'] = self.mean_src
        config['mean_tgt'] = self.mean_tgt
        config['std_src'] = self.std_src
        config['std_tgt'] = self.std_tgt
        config['epsilon'] = self.epsilon
        return(config)
        
class Roll_MeanAndVar(Layer):
    """
    In this case we only roll the mean and variance the the other element of the 
    batch
    """

    def __init__(self,epsilon = 0.00001, **kwargs):
        self.epsilon = epsilon # Small float added to variance to avoid dividing by zero.
        super(Roll_MeanAndVar, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Roll_MeanAndVar, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        mean,variance = tf.nn.moments(x,axes=(1,2),keep_dims=True)
        std = math_ops.sqrt(variance)
        new_mean= tf.roll(mean,shift=1,axis=0)
        new_std= tf.roll(std,shift=1,axis=0)
        output =  (((x - mean) * new_std )/ (std+self.epsilon))  + new_mean
        return K.in_train_phase(output,x)


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(Roll_MeanAndVar, self).get_config()
        config['epsilon'] = self.epsilon
        return(config)
        
class Shuffle_MeanAndVar(Layer):

    def __init__(self,epsilon = 0.00001, **kwargs):
        self.epsilon = epsilon # Small float added to variance to avoid dividing by zero.
        super(Shuffle_MeanAndVar, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Shuffle_MeanAndVar, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        mean,variance = tf.nn.moments(x,axes=(1,2),keep_dims=True)
        std = math_ops.sqrt(variance)
        mean_and_std = tf.stack([mean,std])
        rand_mean_and_std = tf.random.shuffle(mean_and_std) #  The tensor is shuffled along dimension 0, such that each value[j] is mapped to one and only one output[i]
        new_mean = rand_mean_and_std[0,:]
        new_std = rand_mean_and_std[1,:]
        output =  (((x - mean) * new_std )/ (std+self.epsilon))  + new_mean
        return K.in_train_phase(output,x)


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(Shuffle_MeanAndVar, self).get_config()
        config['epsilon'] = self.epsilon
        return(config)
        
class FRN_Layer(Layer):

    def __init__(self,epsilon =1e-6, **kwargs):
        self.epsilon = epsilon # Small float added to variance to avoid dividing by zero.
        super(FRN_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.tau = self.add_weight(name='tau', 
                                      shape=(1,1,1,input_shape[3].value),
                                      initializer='uniform',
                                      trainable=True)
        self.beta = self.add_weight(name='beta', 
                                      shape=(1,1,1,input_shape[3].value),
                                      initializer='zeros',
                                      trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,1,1,input_shape[3].value),
                                      initializer='ones',
                                      trainable=True)
        
        super(FRN_Layer, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        # x: Input tensor of shape [BxHxWxC].
        # alpha, beta, gamma: Variables of shape [1, 1, 1, C].
        # eps: A scalar constant or learnable variable.
        # Compute the mean norm of activations per channel.
        nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2],keepdims=True)
        # Perform FRN.
        x = x*tf.math.rsqrt(nu2 + tf.abs(self.epsilon))
        # Return after applying the Offset-ReLU non-linearity.
        return tf.maximum(self.gamma*x + self.beta, self.tau)


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(FRN_Layer, self).get_config()
#        config['tau'] = self.tau
#        config['beta'] = self.beta
#        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        return(config)
        
class HomeMade_BatchNormalisation(Layer):

    def __init__(self,beta,gamma,epsilon = 0.00001, **kwargs):
        self.beta=beta # offset ie new mean
        self.gamma = gamma # New std
        self.epsilon = epsilon # Small float added to variance to avoid dividing by zero.
        super(HomeMade_BatchNormalisation, self).__init__(**kwargs)

    def build(self,input_shape):
        super(HomeMade_BatchNormalisation, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        mean,variance = tf.nn.moments(x,axes=(1,2),keep_dims=True)
        std = math_ops.sqrt(variance)
        output =  (((x - mean) * self.gamma  )/ (std+self.epsilon))  + self.beta 
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(HomeMade_BatchNormalisation, self).get_config()
        config['beta'] = self.beta
        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        return(config)
    
class HomeMade_adapt_BatchNormalisation(Layer):

    def __init__(self,beta,gamma,epsilon = 0.00001, **kwargs):
        self.beta=beta # offset ie new mean
        self.gamma = gamma # New std
        self.epsilon = epsilon
        super(HomeMade_adapt_BatchNormalisation, self).__init__(**kwargs)

    def build(self,input_shape):
        super(HomeMade_adapt_BatchNormalisation, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        mean,variance = tf.nn.moments(x,axes=(1,2),keep_dims=True)
        std = math_ops.sqrt(variance)
        comparison = tf.greater_equal(std,tf.cast(self.gamma,tf.float32)) # Returns the truth value of (x > y) element-wise.
        float_comp = tf.cast(comparison,tf.float32)
        feature_maps_modified =  (((x - mean) * self.gamma  )/ (std+self.epsilon))  + self.beta
        output = tf.multiply(float_comp,feature_maps_modified) + tf.multiply(1. - float_comp,x)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(HomeMade_adapt_BatchNormalisation, self).get_config()
        config['beta'] = self.beta
        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        return(config)
    
class Mean_Matrix_Layer(Layer):

    def __init__(self, **kwargs):
        super(Mean_Matrix_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Mean_Matrix_Layer, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        # x size bs,H,W,C
        mean = tf.reduce_mean(x,axis=[1,2],keepdims=False)
        return mean

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        b,k1,k2,c = input_shape
        return (b,1,1,c)
    
class Four_Param_Layer(Layer):
    """
    This layer return the 4 first parameters of each of the feature of a features maps
    mean, variance, skewness and kurtosis
    """

    def __init__(self, **kwargs):
        super(Four_Param_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Four_Param_Layer, self).build(input_shape)  
        # Be sure to call this at the end
    
    def call(self, x):
        # x size bs,H,W,C
        axes=[1,2]
        # The dynamic range of fp16 is too limited to support the collection of
        # sufficient statistics. As a workaround we simply perform the operations
        # on 32-bit floats before converting the mean and variance back to fp16
        y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
        # Compute true mean while keeping the dims for proper broadcasting.
        mean = math_ops.reduce_mean(y, axes, keepdims=True, name="mean")
        # sample variance, not unbiased variance
        # Note: stop_gradient does not change the gradient that gets
        #       backpropagated to the mean from the variance calculation,
        #       because that gradient is zero
        variance = math_ops.reduce_mean(
            math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
            axes,
            keepdims=True,
            name="variance")
        std = math_ops.sqrt(variance,name='std')
        y_centered = array_ops.stop_gradient(y - mean)
        y_centered_norm = array_ops.stop_gradient(math_ops.divide(y_centered,std))
        skewness = math_ops.reduce_mean(math_ops.pow(y_centered_norm,3),axes,keepdims=True,name='skewness')
        kurtosis = math_ops.reduce_mean(math_ops.pow(y_centered_norm,4),axes,keepdims=True,name='kurtosis')

        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
        skewness = array_ops.squeeze(skewness, axes)
        kurtosis = array_ops.squeeze(kurtosis, axes)
        if x.dtype == dtypes.float16:
          return (math_ops.cast(mean, dtypes.float16),
                  math_ops.cast(variance, dtypes.float16),
                  math_ops.cast(skewness, dtypes.float16),
                  math_ops.cast(kurtosis, dtypes.float16))
        else:
          return (mean, variance, skewness, kurtosis)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        b,k1,k2,c = input_shape
        return (4,b,c)
    
class Mean_and_MeanSquare_Layer(Layer):

    def __init__(self,useFloat32=False, **kwargs):
        self.useFloat32 = useFloat32
        super(Mean_and_MeanSquare_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Mean_and_MeanSquare_Layer, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        # x size bs,H,W,C
        if self.useFloat32:
            x = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
        mean_of_features = tf.reduce_mean(x,axis=[1,2],keepdims=False)
        mean_of_squared_features = tf.reduce_mean(tf.pow(x,2),axis=[1,2],keepdims=False)
        return(mean_of_features,mean_of_squared_features)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        b,k1,k2,c = input_shape
        return (2,b,c)
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(Mean_and_MeanSquare_Layer, self).get_config()
        config['useFloat32'] = self.useFloat32
        return(config)
    
class Cov_Matrix_Layer(Layer):

    def __init__(self, **kwargs):
        super(Cov_Matrix_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Cov_Matrix_Layer, self).build(input_shape)  
        # Be sure to call this at the end

    def call(self, x):
        f,mean= x
        # f size bs,H,W,C
        cov = covariance_matrix_only(f,mean)
        return cov

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        b,k1,k2,c = input_shape[0]
        return (b,c,c)
    
    
    
### To get the Covariance matrices    
def get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=False):
  """Helper function to compute the Gram matrix and mean of feature representations 
  from vgg.
  
  The gram matrices are computed 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain the mean and gram matrix of 
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the keras model that return the features . 
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  vgg_layers = vgg.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Mean_Matrix_Layer']= Mean_Matrix_Layer
      custom_objects['Cov_Matrix_Layer']= Cov_Matrix_Layer
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          mean_layer = Mean_Matrix_Layer()(output)
          cov_layer = Cov_Matrix_Layer()([output,mean_layer])
          list_stats += [cov_layer,mean_layer]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,custom_objects=custom_objects,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)
  
def get_meanX_meanX2_of_InputImages(useFloat32=True):
  """Helper function to compute the mean of feature and mean of squared of the 
  input images in order to compute the mean on the total sets
    
  Returns:
    returns the keras model that return the elements . 
  """
  model = tf.keras.Sequential()
  mean_and_meanSquared_layer = [Mean_and_MeanSquare_Layer(useFloat32=useFloat32)(model.input)]
  model = models.Model(model.input,mean_and_meanSquared_layer)
  model.trainable = False 
  return(model)
  
def get_cov_mean_of_InputImages():
  """Helper function to compute the covariance and the mean of the input  RGB images
  input images in order to compute the mean on the total sets
    
  Returns:
    returns the keras model that return the elements . 
  """
  model = tf.keras.Sequential()
  mean_and_meanSquared_layer = [Cov_Mean_Matrix_Layer()(model.input)]
  model = models.Model(model.input,mean_and_meanSquared_layer)
  model.trainable = False 
  return(model)
  
def get_VGGmodel_meanX_meanX2_features(style_layers,getBeforeReLU=False,useFloat32=False):
  """Helper function to compute the mean of feature and mean of squared features representations 
  from vgg.
  
  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain the mean of 
  the outputs of the intermediate layers and the mean of the square of it
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the keras model that return the features . 
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  vgg_layers = vgg.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Mean_and_MeanSquare_Layer']= Mean_and_MeanSquare_Layer
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          mean_and_meanSquared_layer = Mean_and_MeanSquare_Layer(useFloat32=useFloat32)(output)
          list_stats += [mean_and_meanSquared_layer]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,custom_objects=custom_objects,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)
  
def get_VGGmodel_4Param_features(style_layers,getBeforeReLU=False):
  """Helper function to compute the 4 first moments of the features representations 
  from vgg.
  
  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain the mean, variance, 
  skewness and kurtosis of the outputs of the intermediate layers. 
  
  Arguments:
    the layer considered
    
  Returns:
    returns the keras model that return the features . 
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  vgg_layers = vgg.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Four_Param_Layer']= Four_Param_Layer
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          mean_and_meanSquared_layer = Four_Param_Layer()(output)
          list_stats += [mean_and_meanSquared_layer]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,custom_objects=custom_objects,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)
  
def get_VGGmodel_features(style_layers,getBeforeReLU=False):
  """Helper function to compute the features representations of the input image
  
  This function will simply load and preprocess the images from their path. 
  
  Arguments:
    the layer considered
    
  Returns:
    returns the keras model that return the features . 
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  vgg_layers = vgg.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          list_stats += [output]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)

def get_BaseNorm_meanX_meanX2_features(style_layers_exported,style_layers_imposed,list_mean_and_std_source,\
                                    list_mean_and_std_target,getBeforeReLU=False,useFloat32=False):
  """Helper function to compute the mean of feature and squared features representations 
  from a modified VGG that have the features maps modified
  

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    the layer considered
    
  Returns:
    returns the model that return the features !
  """
  model = tf.keras.Sequential()
  VGGBaseNorm = vgg_BaseNorm(style_layers_imposed,list_mean_and_std_source,\
                                   list_mean_and_std_target,final_layer='fc2',
                                   getBeforeReLU=getBeforeReLU)
  VGGBaseNorm.trainable = False
  vgg_layers = VGGBaseNorm.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Mean_and_MeanSquare_Layer']= Mean_and_MeanSquare_Layer
      custom_objects['Global_Prescrib_Mean_Std']= Global_Prescrib_Mean_Std
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers_exported[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          mean_and_meanSquared_layer = Mean_and_MeanSquare_Layer(useFloat32=useFloat32)(output)
          list_stats += [mean_and_meanSquared_layer]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers_exported): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,custom_objects=custom_objects,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)
  
def get_BaseNorm_gram_mean_features(style_layers_exported,style_layers_imposed,list_mean_and_std_source,\
                                    list_mean_and_std_target,getBeforeReLU=False):
  """Helper function to compute the Gram matrix and mean of feature representations 
  from a modified VGG that have the features maps modified
  
  The gram matrices are computed 

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the model that return the features !
  """
  model = tf.keras.Sequential()
  VGGBaseNorm = vgg_BaseNorm(style_layers_imposed,list_mean_and_std_source,\
                                   list_mean_and_std_target,final_layer='fc2',
                                   getBeforeReLU=getBeforeReLU)
  VGGBaseNorm.trainable = False
  vgg_layers = VGGBaseNorm.layers
  # Get output layers corresponding to style and content layers 
  list_stats = []
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Mean_Matrix_Layer']= Mean_Matrix_Layer
      custom_objects['Cov_Matrix_Layer']= Cov_Matrix_Layer
      custom_objects['Global_Prescrib_Mean_Std']= Global_Prescrib_Mean_Std
  
  for layer in vgg_layers:
      name_layer = layer.name
      if name_layer==style_layers_exported[i]:
          if getBeforeReLU:
              layer.activation = activations.linear # i.e. identity
          model.add(layer)
            
          output = model.output
          mean_layer = Mean_Matrix_Layer()(output)
          cov_layer = Cov_Matrix_Layer()([output,mean_layer])
          list_stats += [cov_layer,mean_layer]
          
          if getBeforeReLU:
              model.add(Activation('relu'))
          
          i+= 1
          if i==len(style_layers_exported): # No need to compute further
              break
      else:
         model.add(layer)
  
  model = models.Model(model.input,list_stats)
  model.trainable = False
  if getBeforeReLU:
      model = utils_keras.apply_modifications(model,custom_objects=custom_objects,include_optimizer=False) # TODO trouver comme faire cela avec tf keras  
  return(model)


def vgg_cut(final_layer,transformOnFinalLayer=None,weights='imagenet'):
  """
  Return VGG output up to final_layer
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False

  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D','GlobalMinPooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)

  for layer in vgg_layers:
    name_layer = layer.name
    model.add(layer)
#    print(model.output)
    if name_layer==final_layer:
      if not(transformOnFinalLayer is None or transformOnFinalLayer==''):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
#          elif transformOnFinalLayer =='GlobalMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
#          elif transformOnFinalLayer =='GlobalMaxMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
      break
  model_outputs = model.output
  model.trainable = False
 
  return(models.Model(model.input, model_outputs)) 
  
### VGG with features modifed
def vgg_InNorm(style_layers,list_mean_and_std,final_layer='fc2',HomeMadeBatchNorm=True,
              transformOnFinalLayer=None,getBeforeReLU=False):
  """
  VGG with an Instance normalisation : we impose the mean and std of reference 
  instance per instance
  @param : final_layer final layer provide for feature extraction
  @param : transformOnFinalLayer : on va modifier la derniere couche du réseau
  @param : getBeforeReLU if True we modify the features before the ReLU
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['HomeMade_BatchNormalisation']= HomeMade_BatchNormalisation
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D','GlobalMinPooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
  if not(final_layer in ['fc1','fc2','flatten']) and not(transformOnFinalLayer in ['GlobalMaxPooling2D','GlobalAveragePooling2D']):
      print('Are you sure of what you are doing ? You will get a matrix and not a global spatial pooling')
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
      
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      betas = list_mean_and_std[i][0] 
      # Offset of beta => beta = mean 
      gammas = list_mean_and_std[i][1]
      # multiply by gamma => gamma = std
      if HomeMadeBatchNorm:
          model.add(HomeMade_BatchNormalisation(betas,gammas))
      else:
          # Don t work
          raise(NotImplementedError)
          betas = tf.keras.initializers.Constant(value=betas)
          gammas = gammas**2 # from std to var
          gammas = tf.keras.initializers.Constant(value=gammas)
          model.add(layers.BatchNormalization(axis=-1, center=True, scale=True,\
                                       beta_initializer=betas,\
                                       gamma_initializer=gammas,\
                                       fused = False))
          # fused = True accelerated version
          
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)
    if name_layer==final_layer:
      if not(transformOnFinalLayer is None or transformOnFinalLayer==''):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D())
#          elif transformOnFinalLayer =='GlobalMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
#          elif transformOnFinalLayer =='GlobalMaxMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
      break
  
  model_outputs = model.output
  Model_final = models.Model(model.input, model_outputs)
  if getBeforeReLU:# refresh the non linearity 
       Model_final = utils_keras.apply_modifications(Model_final,custom_objects=custom_objects,include_optimizer=False)
  
  Model_final.trainable = False
 
  return(Model_final) 
  
def vgg_InNorm_adaptative(style_layers,list_mean_and_std,final_layer='fc2',
                         HomeMadeBatchNorm=True,transformOnFinalLayer=None,
                         getBeforeReLU=False):
  """
  VGG with an Instance normalisation but we only modify the features 
      that have a std lower than the reference one= Adaptative
  @param : final_layer final layer provide for feature extraction
  @param : transformOnFinalLayer : on va modifier la derniere couche du réseau
  @param : getBeforeReLU if True we modify the features before the ReLU
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['HomeMade_adapt_BatchNormalisation']= HomeMade_adapt_BatchNormalisation
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D','GlobalMinPooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
  
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU: # remove the non linearity
          layer.activation = activations.linear # i.e. identity
        
      model.add(layer)
      betas = list_mean_and_std[i][0] 
      # Offset of beta => beta = mean 
      gammas = list_mean_and_std[i][1]
      # multiply by gamma => gamma = std
      if HomeMadeBatchNorm:
          model.add(HomeMade_adapt_BatchNormalisation(betas,gammas))
      else:
          # Don t work
          raise(NotImplementedError)
          betas = tf.keras.initializers.Constant(value=betas)
          gammas = gammas**2 # from std to var
          gammas = tf.keras.initializers.Constant(value=gammas)
          model.add(layers.BatchNormalization(axis=-1, center=True, scale=True,\
                                       beta_initializer=betas,\
                                       gamma_initializer=gammas,\
                                       fused = False))
          # fused = True accelerated version
     
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
          
      i += 1
    else:
      model.add(layer)
    if name_layer==final_layer:
      if not(transformOnFinalLayer is None or transformOnFinalLayer==''):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D())
#          elif transformOnFinalLayer =='GlobalMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
#          elif transformOnFinalLayer =='GlobalMaxMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
      break
  
  model_outputs = model.output
  Model_final = models.Model(model.input, model_outputs)
  if getBeforeReLU:# refresh the non linearity 
      Model_final = utils_keras.apply_modifications(Model_final,custom_objects=custom_objects,include_optimizer=False)
  
  Model_final.trainable = False
 
  return(Model_final) 

def vgg_BaseNorm(style_layers,list_mean_and_std_source,list_mean_and_std_target,\
                 final_layer='fc2',transformOnFinalLayer=None,getBeforeReLU=False):
  """
  VGG with an whole Base normalisation : we impose the mean and std of reference 
  instance per instance
  @param : final_layer final layer provide for feature extraction
  @param : transformOnFinalLayer : on va modifier la derniere couche du réseau
  @param : getBeforeReLU if True we modify the features before the ReLU
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['Global_Prescrib_Mean_Std']= Global_Prescrib_Mean_Std
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D','GlobalMinPooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  list_name_layers = []
  for layer in vgg_layers:
    list_name_layers += [layer.name]
  if not(final_layer in list_name_layers):
    print(final_layer,'is not in VGG net')
    raise(NotImplementedError)
      
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      mean_src = list_mean_and_std_source[i][0] 
      mean_tgt = list_mean_and_std_target[i][0] 
      std_src = list_mean_and_std_source[i][1] 
      std_tgt = list_mean_and_std_target[i][1] 
      model.add(Global_Prescrib_Mean_Std(mean_src,std_src,mean_tgt,std_tgt))
      
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)
    if name_layer==final_layer:
      if not(transformOnFinalLayer is None or transformOnFinalLayer==''):
          if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
              model.add(GlobalMaxPooling2D()) 
#          elif transformOnFinalLayer =='GlobalMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
#          elif transformOnFinalLayer =='GlobalMaxMinPooling2D': # IE spatial max pooling
#              model.add(GlobalMinPooling2D)
          elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
              model.add(GlobalAveragePooling2D())
      break
  
  model_outputs = model.output
  Model_final = models.Model(model.input, model_outputs)
  if getBeforeReLU:# refresh the non linearity 
       Model_final = utils_keras.apply_modifications(Model_final,custom_objects=custom_objects,include_optimizer=False)
  
  Model_final.trainable = False
 
  return(Model_final)   
  



# Define an keras layer that impose the Gram Matrix
class GramImposed_Layer(Layer):

    def __init__(self, cov_matrix,mean,manner, **kwargs):
        self.cov_matrix = cov_matrix
        self.mean = mean
        self.manner = manner
        super(GramImposed_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Besoin de creer une fonction partielle ?
        super(GramImposed_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.manner=='WCT':
          output = wct_tf(x, self.cov_matrix,self.mean)
        return output

    def compute_output_shape(self, input_shape):
        return(input_shape)

def vgg_GramImposed(style_layers,list_Gram_matrices,manner):
  """
  VGG with Gram matrices imposed according to different method (manner)
  @param manner : 
   - WCT : Whitening coloring methods [Li 2017]
   - CF : Close Form [Lu 2019]
   - CDTO : displacement cost [Li 2019]
  """
  list_implemented_manner = ['WCT']
  if not(manner in list_implemented_manner):
    print(manner,'is unknown, chooce among :')
    print(list_implemented_manner)
    raise(NotImplementedError)
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      model.add(layer)
      cov = list_Gram_matrices[i][0] 
      m = list_Gram_matrices[i][1]
      gram_layer = GramImposed_Layer(cov,m,manner)
      model.add(gram_layer)
      i += 1
    else:
      model.add(layer)
    if name_layer=='fc2':
      model_outputs = model.output
  model.trainable = False
 
  return(models.Model(model.input, model_outputs))
  
def get_1st_2nd_moments_features(model,img_path):
  """Helper function to compute the mean and variance of feature representations 
  from vgg.

  This function will simply load and preprocess the images from their path. 
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the features. 
  """
  # Load our images in 
  image = load_and_process_img(img_path)
  
  # batch compute content and style features
  outputs = model(image)
  list_moments = []
  for output in outputs:
    moments = tf.nn.moments(output,axes=[0, 1, 2]) # Moments computed on the batch, height, width dimension
    # Get the style and content feature representations from our model  
    list_moments += [moments]
  return list_moments

def preprocessingDataet():
    
    # Dans ResNet de Keras la fonction preprocessing est la suivante : 
#    def preprocess_input(x, data_format=None):
#      return imagenet_utils.preprocess_input(
#      x, data_format=data_format, mode='caffe')
#    Sachant que  :
#        x: Input Numpy or symbolic tensor, 3D or 4D.
#      The preprocessed data is written over the input data
#      if the data types are compatible. To avoid this
#      behaviour, `numpy.copy(x)` can be used.
#    data_format: Data format of the image tensor/array.
#    mode: One of "caffe", "tf" or "torch".
#      - caffe: will convert the images from RGB to BGR,
#          then will zero-center each color channel with
#          respect to the ImageNet dataset,
#          without scaling.
#      - tf: will scale pixels between -1 and 1,
#          sample-wise.
#      - torch: will scale pixels between 0 and 1 and then
#          will normalize each channel with respect to the
#          ImageNet datase
    
    return(0)
    
    

def unity_test_of_vgg_InNorm(adapt=False):
    """ In this function we compare the fc2 for VGG19 of a given image
    and the fc2 of the vgg_InNorm with the parameters computed on the same image
    @param : if adapt == True we will test the adaptative model
    """
    
#    images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
#    image_path =  os.path.join(images_path,'abd_aag_002276_624x544.jpg')
    image_path =  os.path.join('data','Q23898.jpg')
    #init_image = tf.convert_to_tensor(load_resize_and_process_img(image_path))
    init_image = load_resize_and_process_img(image_path)
    # Fc2 of VGG
    # Add a layer where input is the output of the  second last layer 
    vgg_full = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
#    print(vgg_full.layers)
    fc2_layer = vgg_full.layers[-2].output
    
    #Then create the corresponding model 
    vgg_fc2_model = tf.keras.Model(inputs=vgg_full.input, outputs=fc2_layer)
    vgg_fc2_model.trainable=False
    fc2_original_vgg19 = vgg_fc2_model.predict(init_image)
    print('fc2 shape',fc2_original_vgg19.shape)
    
    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
    
    for getBeforeReLU in [False,True]:
        print('=== getBeforeReLU :',getBeforeReLU,' ===')
        # Load the model that compute cov matrices and mean
        vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=getBeforeReLU)
        vgg_cov_mean = vgg_get_cov.predict(init_image, batch_size=1)
        vgg_mean_vars_values = []
        for l,layer in enumerate(style_layers):
            cov = vgg_cov_mean[2*l]
            cov = np.squeeze(cov,axis=0) 
            mean = vgg_cov_mean[2*l+1]
            mean = mean.reshape((mean.shape[1],))
            stds = np.sqrt(np.diag(cov))
            vgg_mean_vars_values += [[mean,stds]]
        
        if adapt:
            VGGInNorm = vgg_InNorm_adaptative(style_layers,vgg_mean_vars_values,final_layer='fc2',
                             HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
        else:
            VGGInNorm = vgg_InNorm(style_layers,vgg_mean_vars_values,final_layer='fc2',
                             HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
        #print(VGGInNorm.summary())
        #sess.run(tf.global_variables_initializer())
    #    sess.run(tf.local_variables_initializer())
        for i in range(2):
            output_VGGInNorm = VGGInNorm.predict(init_image)
            print(i,'First elts AdaIn run',output_VGGInNorm[0,0:5])
    #    output_VGGInNorm = sess.run(vgg_fc2_model(init_image))
        print('output_VGGInNorm shape',output_VGGInNorm.shape)
        
        print('Equal fc2vgg  fc2VGGInNorm ? :',np.all(np.equal(output_VGGInNorm, fc2_original_vgg19)))
        print('max abs(a-b)',np.max(np.abs(output_VGGInNorm - fc2_original_vgg19)))
        print('max abs(a-b)/abs(a)',np.max(np.abs(output_VGGInNorm - fc2_original_vgg19)/np.abs(fc2_original_vgg19+10**(-16))))
        print(' norm2(a-b)/norm2(a)',np.linalg.norm(output_VGGInNorm - fc2_original_vgg19)/np.linalg.norm(fc2_original_vgg19+10**(-16)))
        print('First elts of VGGInNorm and original net',output_VGGInNorm[0,0:5],fc2_original_vgg19[0,0:5])
    
def unity_test_of_vgg_BaseNorm(getBeforeReLU=False):
    """ In this function we compare the fc2 for VGG19 of a given image
    and the fc2 of the vgg_BaseNorm with the parameters computed on the same image
    @param : if adapt == True we will test the adaptative model
    """
    image_path =  os.path.join('data','Q23898.jpg')
    #init_image = tf.convert_to_tensor(load_resize_and_process_img(image_path))
    init_image = load_resize_and_process_img(image_path)
    # Fc2 of VGG
    # Add a layer where input is the output of the  second last layer 
    vgg_full = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
#    print(vgg_full.layers)
    fc2_layer = vgg_full.layers[-2].output
    
    #Then create the corresponding model 
    vgg_fc2_model = tf.keras.Model(inputs=vgg_full.input, outputs=fc2_layer)
    vgg_fc2_model.trainable=False
    fc2_original_vgg19 = vgg_fc2_model.predict(init_image)
    print('fc2 shape',fc2_original_vgg19.shape)
    
    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
    
    for getBeforeReLU in [True]:
        print('=== getBeforeReLU :',getBeforeReLU,' ===')
        # Load the model that compute cov matrices and mean
        vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=getBeforeReLU)
        vgg_cov_mean = vgg_get_cov.predict(init_image, batch_size=1)
        vgg_mean_stds_values = []
        for l,layer in enumerate(style_layers):
            cov = vgg_cov_mean[2*l]
            cov = np.squeeze(cov,axis=0) 
            mean = vgg_cov_mean[2*l+1]
            mean = mean.reshape((mean.shape[1],))
            stds = np.sqrt(np.diag(cov))
            vgg_mean_stds_values += [[mean,stds]]
        

        VGGBaseNorm = vgg_BaseNorm(style_layers,vgg_mean_stds_values,vgg_mean_stds_values,final_layer='fc2',
                                   getBeforeReLU=getBeforeReLU)
        #print(VGGInNorm.summary())
        #sess.run(tf.global_variables_initializer())
    #    sess.run(tf.local_variables_initializer())
        for i in range(2):
            output_VGGBaseNorm = VGGBaseNorm.predict(init_image)
            print(i,'First elts AdaIn run',output_VGGBaseNorm[0,0:5])
    #    output_VGGInNorm = sess.run(vgg_fc2_model(init_image))
        print('output_VGGInNorm shape',output_VGGBaseNorm.shape)
        
        print('Equal fc2vgg  fc2VGGInNorm ? :',np.all(np.equal(output_VGGBaseNorm, fc2_original_vgg19)))
        print('max abs(a-b)',np.max(np.abs(output_VGGBaseNorm - fc2_original_vgg19)))
        print('max abs(a-b)/abs(a)',np.max(np.abs(output_VGGBaseNorm - fc2_original_vgg19)/np.abs(fc2_original_vgg19+10**(-16))))
        print(' norm2(a-b)/norm2(a)',np.linalg.norm(output_VGGBaseNorm - fc2_original_vgg19)/np.linalg.norm(fc2_original_vgg19+10**(-16)))
        print('First elts of VGGInNorm and original net',output_VGGBaseNorm[0,0:5],fc2_original_vgg19[0,0:5])
    
def topn(x,n=5):
    return(x[np.argsort(x)[-n:]])
    
def test_change_mean_std(adapt=False):
    """ In this fct we test to use the mean and std of a givent image to an other one
    """
    print('Adapt =',adapt)
#    images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
#    image_path =  os.path.join(images_path,'abd_aag_002276_624x544.jpg')
    cat_path =  os.path.join('data','cat.jpg')
#    cat_path =  os.path.join('data','ny_yag_yorag_156_624x544.jpg')
    #init_image = tf.convert_to_tensor(load_resize_and_process_img(image_path))
    cat_image = load_resize_and_process_img(cat_path)
    # Fc2 of VGG
    # Add a layer where input is the output of the  second last layer 
    vgg_full = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
    predictions_cat = vgg_full.predict(cat_image)
#    print('Top k index')
#    print(topn(predictions_cat))
    cat_decode = decode_predictions(predictions_cat)
    print(cat_decode)
    
    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

    num_style_layers = len(style_layers)
    # Todo a finir le vgg_InNorm_adaptative avec gamma==std a l infini on modife 
    # encore le reseau alors qu'on devrait pas
    list_imgs = ['bus','dog','flower','cat2']
#    list_imgs = ['dog']
    style_layers_tab = [style_layers,[style_layers[0]],[style_layers[-1]]]
#    style_layers_tab = [style_layers]
    for style_layers in style_layers_tab:
        print('Layers :',style_layers)
        # Load the model that compute cov matrices and mean
        vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
        for j,elt in enumerate(list_imgs):
            print('====',elt,'===')
            object_path =  os.path.join('data',elt+'.jpg')
            object_image = load_resize_and_process_img(object_path)
            vgg_cov_mean = vgg_get_cov.predict(object_image, batch_size=1)
            vgg_mean_stds_values = []
            vgg_mean_stds_values_0_1 = []
            for l,layer in enumerate(style_layers):
                cov = vgg_cov_mean[2*l]
                cov = np.squeeze(cov,axis=0) 
                mean = vgg_cov_mean[2*l+1]
                mean = mean.reshape((mean.shape[1],))
                stds = np.sqrt(np.diag(cov))
                vgg_mean_stds_values += [[mean,stds]]
                if j ==len(list_imgs)-1:
                    vgg_mean_stds_values_0_1 += [[np.zeros_like(mean),np.ones_like(stds)]]
            if adapt:
                VGGInNorm = vgg_InNorm_adaptative(style_layers,vgg_mean_stds_values,final_layer='predictions',
                             HomeMadeBatchNorm=True)
            else:
                VGGInNorm = vgg_InNorm(style_layers,vgg_mean_stds_values,final_layer='predictions',
                             HomeMadeBatchNorm=True)
            object_predction = VGGInNorm.predict(cat_image)
            object_decode = decode_predictions(object_predction)
            print(object_decode)
            if j ==len(list_imgs)-1:
                print('=== means=0 and stds=1 ===')
                if adapt:
                    VGGInNorm = vgg_InNorm_adaptative(style_layers,vgg_mean_stds_values_0_1,final_layer='predictions',
                             HomeMadeBatchNorm=True)
                else:
                    VGGInNorm = vgg_InNorm(style_layers,vgg_mean_stds_values_0_1,final_layer='predictions',
                             HomeMadeBatchNorm=True)
                object_predction = VGGInNorm.predict(cat_image)
                object_decode = decode_predictions(object_predction)
                print(object_decode) 
        K.clear_session() # To clean memory
            
def unity_test_StatsCompute():
    """ In this function we try to se if we have a stable and consistent computation
    of mean and variance of the features maps"""
    image_path =  os.path.join('data','Q23898.jpg')
    init_image_original = load_resize_and_process_img(image_path)
    init_image = np.tile(init_image_original,(2,1,1,1))

    style_layers = ['block1_conv1','block2_conv1']

    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
    print(vgg_get_cov.summary())
#    print('vgg_get_cov',vgg_get_cov)
#    vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
    
    # First we will test that we have the same output between 2 times the same images
    # and between two run of the same model !
    old_mean = []
    old_cov = []
    for i in range(2):
        print('Restart',i)
#        print('init_image',init_image.shape)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
    
        vgg_cov_mean = vgg_get_cov.predict(init_image, batch_size=1)
#        print(vgg_cov_mean)
        vgg_mean_vars_values = []
        for l,layer in enumerate(style_layers):
            cov = vgg_cov_mean[2*l]
#            print(cov.shape)
            #cov = np.squeeze(cov,axis=0) # If image bs ==1
#            print(cov.shape)
            mean = vgg_cov_mean[2*l+1]
            assert((mean[0,:]==mean[1,:]).all())
            assert((cov[0,:,:]==cov[1,:,:]).all())
            if i==0:
                old_mean += [mean]
                old_cov += [cov]
            elif i==1:
                assert((old_mean[l]==mean).all())
                assert((old_cov[l]==cov).all())
    
    # Do we obtain the save mean and var than computed by numpy ?
    vgg_inter = get_intermediate_layers_vgg(style_layers) 
    features_maps = vgg_inter.predict(init_image_original)
    for l,layer in enumerate(style_layers):
        features_maps_l = features_maps[l]
        features_maps_l = features_maps[l]
        np_mean = np.mean(features_maps_l,axis=(0,1,2))
        np_var = np.var(features_maps_l-np_mean,axis=(0,1,2))
        keras_mean = old_mean[l][0,:]
        keras_cov= old_cov[l][0,:,:]
        keras_var = np.diag(keras_cov)
        try:
            assert((keras_mean==np_mean).all())
        except AssertionError:
            print(layer,'mean')
            print('max abs(a-b)',np.max(np.abs(np_mean - keras_mean)))
            print('max abs(a-b)/abs(a)',np.max(np.abs(np_mean - keras_mean)/np.abs(np_mean+10**(-16))))
            print(' norm2(a-b)/norm2(a)',np.linalg.norm(np_mean - keras_mean)/np.linalg.norm(np_mean+10**(-16)))
            print('First elts',np_mean[0:5],keras_mean[0:5])
        try:
            assert((keras_var==np_var).all())
        except AssertionError:
            print(layer,'var')
            print('max abs(a-b)',np.max(np.abs(np_var - keras_var)))
            print('max abs(a-b)/abs(a)',np.max(np.abs(np_var - keras_var)/np.abs(np_var+10**(-16))))
            print(' norm2(a-b)/norm2(a)',np.linalg.norm(np_var - keras_var)/np.linalg.norm(np_var+10**(-16)))
            print('First elts',np_var[0:5],keras_var[0:5])

    sess.close()   
            
def Test_BatchNormRefinement():
    """
    Pour tester l'ancienne méthode de BNRF 
    """
    
    ### Peut etre que cela ne ùarche pas a cause de ca : 
    # https://pgaleone.eu/tensorflow/keras/2019/01/19/keras-not-yet-interface-to-tensorflow/
    
    bs = 64
    features_size = 24
    a = Input(shape=(features_size,features_size,3))
    b = Conv2D(filters=features_size,kernel_size=3,padding='same')(a)
    c = BatchNormalization(axis=1,name='bn')(b)
    pre_model = Model(inputs=a, outputs=c)  
#    pre_model = Sequential([b,c])
#    pre_model = Model(inputs=b, outputs=c) 
#    pre_model.build(input_shape=(bs,features_size,features_size,3))
    bn_layers = ['bn']
    momentum = 0.9
    print("Model before anything")
    print(pre_model.summary())    
    batchnorm_layer = pre_model.get_layer('bn')
    moving_mean = batchnorm_layer.non_trainable_variables[0]
    moving_variance = batchnorm_layer.non_trainable_variables[1]
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))
    
    for layer in pre_model.layers:
      layer.trainable = False
      if layer.name in bn_layers:
          new_bn = HomeMade_BatchNormalisation_Refinement(layer,momentum)
          layer_regex = layer.name
          insert_layer_name =  layer.name +'_rf'
          pre_model = insert_layer_nonseq(pre_model, layer_regex, new_bn,
                        insert_layer_name=insert_layer_name, position='replace')   
    
    for layer in pre_model.layers:
        print(layer.name)
    #tf.keras.backend.set_learning_phase(True)
    print("Model after modification")
    print(pre_model.summary())    
    print('pre_model.updates',pre_model.updates)
    batchnorm_layer = pre_model.get_layer('home_made__batch_normalisation__refinement')
    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))
      
    epochs = 5
    sess = K.get_session()
    train_fn = K.function(inputs=[pre_model.input], \
            outputs=[pre_model.output], updates=pre_model.updates)
    init = tf.global_variables_initializer()
    sess.run(init)
    for n  in range(epochs):
        x = 1.+np.random.rand(bs,features_size,features_size,3)
        x = x.astype(np.float32)
        #pre_model.predict(x)
        #print(pre_model.updates)
        train_fn(x)
        print('pre_model.updates',pre_model.updates)
        batchnorm_layer = pre_model.get_layer('home_made__batch_normalisation__refinement')
        moving_mean = batchnorm_layer.moving_mean
        moving_variance = batchnorm_layer.moving_variance
        print('moving_mean',tf.keras.backend.eval(moving_mean))
        print('moving_variance',tf.keras.backend.eval(moving_variance))  
        #train_fn(tf.convert_to_tensor(x))
        #sess.run(train_fn(tf.convert_to_tensor(x)))
#        trainable_weights = pre_model.trainable_weights
#        print(tf.keras.backend.eval(trainable_weights))
#        trainable_variables = pre_model.trainable_variables
#        print(tf.keras.backend.eval(trainable_variables))
    
    print("Model after epochs")   
    batchnorm_layer = pre_model.get_layer('home_made__batch_normalisation__refinement')
    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))  
    print("End the moving average application")
    
    # Now we want to predict without  changing the parameters
    for n  in range(epochs):
        x = 10.+np.random.rand(bs,features_size,features_size,3)
        x = x.astype(np.float32)
        pre_model.predict(x)
    print("Model predictions epochs")   
    batchnorm_layer = pre_model.get_layer('home_made__batch_normalisation__refinement')
    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))  
    print("End") 
    
    sess.close()

def Test_BatchNormRefinement_V2():
    """
    Pour tester le refinement des Batch normalisation parametres sans avoir a 
    remplacer des modules (bloc)
    """
    
    ### Peut etre que cela ne ùarche pas a cause de ca : 
    # https://stackoverflow.com/questions/55421984/batch-normalization-in-tf-keras-does-not-calculate-average-mean-and-average-vari
    # https://pgaleone.eu/tensorflow/keras/2019/01/19/keras-not-yet-interface-to-tensorflow/
    # https://github.com/tensorflow/tensorflow/blob/5eb81ea161515e18a039241b3e4eb8efbfbaa354/tensorflow/python/layers/normalization_test.py
    
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    tf.keras.backend.set_learning_phase(True)
    
    bs = 64
    features_size = 24
    training = array_ops.placeholder(dtypes.bool)
    a = Input(shape=(features_size,features_size,3))
    xt = array_ops.placeholder(dtypes.float32, shape=(None,features_size,features_size,3))
    #b = Conv2D(filters=features_size,kernel_size=3,padding='same')(a)
    momentum =0.5
    gamma_initializer = tf.constant_initializer(value=np.ones((3,)))
    beta_initializer = tf.constant_initializer(value=-np.ones((3,)))
    moving_variance_initializer = tf.constant_initializer(value=10.*np.ones((3,)))
    moving_mean_initializer = tf.constant_initializer(value=-0.1*np.ones((3,)))
    c = BatchNormalization(axis=-1,name='bn',momentum=momentum,moving_mean_initializer=moving_mean_initializer,
                           moving_variance_initializer=moving_variance_initializer,
                           beta_initializer=beta_initializer,
                           gamma_initializer=gamma_initializer)(a)
    pre_model = Model(inputs=a, outputs=c)  
    pre_model.trainable = True
#    pre_model = Sequential([b,c])
#    pre_model = Model(inputs=b, outputs=c) 
#    pre_model.build(input_shape=(bs,features_size,features_size,3))
    bn_layers = ['bn']
    
    print("Model before anything")
    pre_model = set_momentum_BN(pre_model,momentum) # Ne marche pas
    batchnorm_layer = pre_model.get_layer('bn')

    # batchnorm_layer.moving_mean.trainable = True
    # batchnorm_layer.moving_variance.trainable = True
    print(pre_model.summary())   

    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('momentum',batchnorm_layer.momentum)
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))
      
    epochs = 5
    print('pre_model.updates',pre_model.updates)
    sess = K.get_session()
    # train_fn = K.function(inputs=[pre_model.input], \
    #         outputs=[pre_model.output], updates=pre_model.updates)
        # updates: Additional update ops to be run at function call.
    tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, pre_model.updates)
    
    #tf.control_dependencies(updates_op)
    #updates_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

    from tensorflow.python.ops import variables
    all_vars = dict([(v.name, v) for v in variables.global_variables()])
    print('all variables :',all_vars)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    #train_fn = K.function(inputs=[pre_model.input], \
    #        outputs=[pre_model.output], updates=updates_ops)
    running_mean = 0.0
    running_var =1.0
    for n  in range(epochs):
        #x = 20.+2.*np.random.rand(bs,features_size,features_size,3)
        x = np.random.normal(loc=500.,scale=200.,size=(bs,features_size,features_size,3))
        print(np.mean(x,axis=(0,1,2)))
        print(np.var(x,axis=(0,1,2)))
        x = x.astype(np.float32)
        
        #train_fn(x)
        updates_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        sess.run(updates_ops,  {pre_model.input : x})
        #sess.run(c, {a : x})
        #aa = sess.run(pre_model.updates,feed_dict={a : x, training: True})
        # print(aa)
        # with tf.control_dependencies(pre_model.updates):
        #     sess.run(updates_ops,feed_dict={a : x})
            # train_fn(x)
        #pre_model.predict(x)
        #print(pre_model.updates)
        #tf.keras.backend.eval(train_fn(x))
        #sess.run(updates_ops,feed_dict={a : x})
        #sess.run(train_fn,feed_dict={a : x})
        #sess.run(updates_ops)
        print('pre_model.updates',pre_model.updates)
        print('updates_ops',updates_ops)
        batchnorm_layer = pre_model.get_layer('bn')
        moving_mean = batchnorm_layer.moving_mean
        moving_variance = batchnorm_layer.moving_variance
        print('moving_mean',tf.keras.backend.eval(moving_mean))
        print('moving_variance',tf.keras.backend.eval(moving_variance))  
        
        # Verify that the statistics are updated during training.
        np_inputs = x
        np_mean = np.mean(np_inputs, axis=(0, 1, 2))
        np_std = np.std(np_inputs, axis=(0, 1, 2))
        np_variance = np.square(np_std)
        running_mean = momentum * running_mean + (1 - momentum) * np_mean
        running_var = momentum * running_var + (1 - momentum) * np_variance
        print('running_mean',running_mean)
        print('running_var',running_var)
        
        #train_fn(tf.convert_to_tensor(x))
        #sess.run(train_fn(tf.convert_to_tensor(x)))
#        trainable_weights = pre_model.trainable_weights
#        print(tf.keras.backend.eval(trainable_weights))
#        trainable_variables = pre_model.trainable_variables
#        print(tf.keras.backend.eval(trainable_variables))
    
    print("Model after epochs")   
    batchnorm_layer = pre_model.get_layer('bn')
    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))  
    print("End the moving average application")
    
    # Now we want to predict without  changing the parameters
    for n  in range(epochs):
        x = 10.+np.random.rand(bs,features_size,features_size,3)
        x = x.astype(np.float32)
        pre_model.predict(x)
    print("Model predictions epochs")   
    batchnorm_layer = pre_model.get_layer('bn')
    moving_mean = batchnorm_layer.moving_mean
    moving_variance = batchnorm_layer.moving_variance
    print('moving_mean',tf.keras.backend.eval(moving_mean))
    print('moving_variance',tf.keras.backend.eval(moving_variance))  
    print("End") 
    
    sess.close()

       
        
if __name__ == '__main__':
#  unity_test_StatsCompute()
#  unity_test_of_vgg_InNorm()
    Test_BatchNormRefinement_V2()
    #Test_BatchNormRefinement()
