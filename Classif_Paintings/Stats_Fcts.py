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
from tensorflow.python.keras.layers import Activation,Dense,Flatten,Input
from tensorflow.python.keras import layers
from tensorflow.python.keras import utils
import utils_keras
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.backend import expand_dims
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.optimizers import SGD

#from custom_pooling import GlobalMinPooling2D
from lr_multiplier import LearningRateMultiplier

# Others libraries
import numpy as np
from PIL import Image
import os
import os.path
#import time
#import functools

### To fine Tune a VGG
def VGG_baseline_model(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                       pretrainingModif=True,verbose=False,weights='imagenet',optimizer='adam',\
                       opt_option=[0.01],freezingType='FromTop'): 
  """
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  # create model
  model =  tf.keras.Sequential()
  pre_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
  SomePartFreezed = False
  if type(pretrainingModif)==bool:
      pre_model.trainable = pretrainingModif
  else:
      SomePartFreezed = True # We will unfreeze pretrainingModif==int layers from the end of the net
      number_of_trainable_layers =  16
      assert(number_of_trainable_layers >= pretrainingModif)
  lr_multiple = False
  if optimizer=='SGD':
      if len(opt_option)==2:
          multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
          multipliers = {}
          lr_multiple = True
      if len(opt_option)==1:
          lr = opt_option[0]
          opt = SGD(learning_rate=lr,momentum=0.9)
  else:
      opt=optimizer
  
#  last = pre_model.output
#
#  x = Flatten()(last)
#  x = Dense(256, input_shape=(25088,), activation='relu')(x)
#  preds = Dense(num_of_classes, activation='sigmoid')(x)
#
#  model = Model(pre_model.input, preds)
  
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
     if lr_multiple:
         multipliers[layer.name] = multiply_lrp
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
      model.add(GlobalMaxPooling2D()) 
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      model.add(GlobalAveragePooling2D())
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      model.add(Flatten())
  
  model.add(Dense(256, activation='relu'))
  if lr_multiple:
      multipliers[model.layers[-1].name] = None
  model.add(Dense(num_of_classes, activation='sigmoid'))
  if lr_multiple:
      multipliers[model.layers[-1].name] = None
      opt = LearningRateMultiplier(SGD, lr_multipliers=multipliers, lr=lr, momentum=0.9)
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  if verbose: print(model.summary())
  return model

def ResNet_baseline_model(num_of_classes=10,transformOnFinalLayer ='GlobalMaxPooling2D',\
                             pretrainingModif=True,verbose=True,weights='imagenet',res_num_layers=50,\
                             optimizer='adam',opt_option=[0.01]): 
  """
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  # create model
#  input_tensor = Input(shape=(224, 224, 3)) 
  model =  tf.keras.Sequential()
  if res_num_layers==50:
      pre_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=weights)
  else:
      print('Not implemented yet the resnet 101 or 152 need to update to tf 2.0')
      raise(NotImplementedError)
  pre_model.trainable = pretrainingModif
  
  lr_multiple = False
  if optimizer=='SGD' and len(opt_option)==2:
      if len(opt_option)==2:
          multiply_lrp, lr = opt_option # lrp : learning rate pretrained and lr : learning rate
          multipliers = {}
          lr_multiple = True
      if len(opt_option)==1:
          lr = opt_option[0]
          opt = SGD(learning_rate=lr,momentum=0.9)
  else:
      opt=optimizer
  
  x = pre_model.output
  if lr_multiple:
      for layer in pre_model.layers:  
         multipliers[layer.name] = multiply_lrp
#  for layer in pre_model.layers:
#     print(layer)
#     model.add(layer)
  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
      #model.add(GlobalMaxPooling2D()) 
     x = GlobalMaxPooling2D()(x)
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      #model.add(GlobalAveragePooling2D())
      x = GlobalAveragePooling2D()(x)
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
#      model.add(Flatten())
      x= Flatten()(x)
  
  x = Dense(256, activation='relu')(x)
  predictions = Dense(num_of_classes, activation='sigmoid')(x)
  if lr_multiple:
      multipliers[layer.name] = lr
  model = Model(inputs=pre_model.input, outputs=predictions)
  if lr_multiple:
      multipliers[model.layers[-2].name] = None
      multipliers[model.layers[-1].name] = None
      opt = LearningRateMultiplier(SGD, lr_multipliers=multipliers, lr=lr, momentum=0.9)
  #model.add(Dense(num_of_classes, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  if verbose: print(model.summary())
  return model

def vgg_AdaIn(style_layers,num_of_classes=10,
              transformOnFinalLayer='GlobalMaxPooling2D',getBeforeReLU=True,verbose=False,\
              weights='imagenet'):
  """
  VGG with an Instance normalisation learn only those are the only learnable parameters
  with the last 2 dense layer 
  @param : weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  
  if getBeforeReLU: 
      custom_objects = {}
      custom_objects['HomeMade_BatchNormalisation']= HomeMade_BatchNormalisation
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
      raise(NotImplementedError)
      
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      if getBeforeReLU:# remove the non linearity
          layer.activation = activations.linear # i.e. identity
      model.add(layer)
      model.add(layers.BatchNormalization(axis=-1, center=True, scale=True))
        
      if getBeforeReLU: # add back the non linearity
          model.add(Activation('relu'))
      i += 1
    else:
      model.add(layer)

  if transformOnFinalLayer =='GlobalMaxPooling2D': # IE spatial max pooling
      model.add(GlobalMaxPooling2D())
  elif transformOnFinalLayer =='GlobalAveragePooling2D': # IE spatial max pooling
      model.add(GlobalAveragePooling2D())
  elif transformOnFinalLayer is None or transformOnFinalLayer=='' :
      model.add(Flatten())
  
  model.add(Dense(256, activation='relu'))
  model.add(Dense(num_of_classes, activation='sigmoid'))

  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  if getBeforeReLU:# refresh the non linearity 
      model = utils_keras.apply_modifications(model,include_optimizer=True,needFix = True)
  
  if verbose: print(model.summary())
  return model


def load_crop(path_to_img,max_dim = 224):
  img = Image.open(path_to_img)
  img = img.resize((max_dim, max_dim), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def load_crop_and_process_img(path_to_img,max_dim = 224):
 # img = load_crop(path_to_img,max_dim=max_dim)
 
  img = tf.keras.preprocessing.image.load_img(
    path_to_img,
    grayscale=False,
    target_size=(max_dim, max_dim),
    interpolation='nearest')
  # This way to load the images and to convert them to numpy array before preprocessing
  # are certainly suboptimal
  img = kp_image.img_to_array(img)
  img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
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

def get_intermediate_layers_vgg(style_layers):
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
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  model_outputs = style_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)

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
  image = load_crop_and_process_img(img_path)
  
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
  Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    img_path: The path to the image.
    
  Returns:
    returns the features. 
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
    returns the features. 
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


def vgg_cut(final_layer,transformOnFinalLayer=None):
  """
  Return VGG output up to final_layer
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
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
      custom_objects['Global_Prescrib_Mean_Std']= Global_Prescrib_Mean_Std
  
  otherOutputPorposed = ['GlobalMaxPooling2D','',None,'GlobalAveragePooling2D','GlobalMinPooling2D']
  if not(transformOnFinalLayer in otherOutputPorposed):
      print(transformOnFinalLayer,'is unknown in the transformation of the last layer')
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

def unity_test_of_vgg_InNorm(adapt=False):
    """ In this function we compare the fc2 for VGG19 of a given image
    and the fc2 of the vgg_InNorm with the parameters computed on the same image
    @param : if adapt == True we will test the adaptative model
    """
    
#    images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
#    image_path =  os.path.join(images_path,'abd_aag_002276_624x544.jpg')
    image_path =  os.path.join('data','Q23898.jpg')
    #init_image = tf.convert_to_tensor(load_crop_and_process_img(image_path))
    init_image = load_crop_and_process_img(image_path)
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
    #init_image = tf.convert_to_tensor(load_crop_and_process_img(image_path))
    init_image = load_crop_and_process_img(image_path)
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
    #init_image = tf.convert_to_tensor(load_crop_and_process_img(image_path))
    cat_image = load_crop_and_process_img(cat_path)
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
            object_image = load_crop_and_process_img(object_path)
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
    init_image_original = load_crop_and_process_img(image_path)
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
            
            
    
if __name__ == '__main__':
#  unity_test_StatsCompute()
  unity_test_of_vgg_InNorm()