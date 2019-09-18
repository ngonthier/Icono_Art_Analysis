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
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

# Others libraries
import numpy as np
from PIL import Image
import os
import os.path
#import time
#import functools

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

def get_VGGmodel_gram_mean_features(style_layers):
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
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]

  list_stats = []
  for output in style_outputs:
    cov,mean,f,C,H,W =covariance_matrix2(output, eps=1e-8) 
    list_stats += [[cov,mean]]
  return models.Model(vgg.input,list_stats)


### VGG with features modifed
  
#def local_batch_normalization()

def vgg_AdaIn(style_layers,list_mean_and_std,final_layer='fc2'):
  """
  VGG with an AdaIn : Instance normalisation
  @param : final_layer final layer provide for feature extraction
  """
  model = tf.keras.Sequential()
  vgg = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
  vgg_layers = vgg.layers
  vgg.trainable = False
  i = 0
  for layer in vgg_layers:
    name_layer = layer.name
    if i < len(style_layers) and name_layer==style_layers[i]:
      model.add(layer)
      betas = tf.keras.initializers.Constant(value=list_mean_and_std[i][0]) 
      # Offset of beta => beta = mean
      gammas = tf.keras.initializers.Constant(value=list_mean_and_std[i][1]) 
      # multiply by gamma => gamma = std
      model.add(tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True,\
                                                   beta_initializer=betas, 
                                                   gamma_initializer=gammas,
                                                   epsilon=np.finfo(np.float32).eps,
                                                   fused=False,
                                                   trainable=True)) # fused = True accelerated version
      i += 1
    else:
      model.add(layer)
    if name_layer==final_layer:
      model_outputs = model.output
      break
      
  model.trainable = False
 
  return(models.Model(model.input, model_outputs)) 



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

def unity_test_of_VGG_AdaIn():
    """ In this function we compare the fc2 for VGG19 of a given image
    and the fc2 of the VGG_adaIn with the parameters computed on the same image"""
    
    images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
    image_path =  os.path.join(images_path,'abd_aag_002276_624x544.jpg')
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

    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    
    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
    vgg_cov_mean = vgg_get_cov.predict(init_image)
    vgg_mean_vars_values = []
    for l,layer in enumerate(style_layers):
        [cov,mean] = vgg_cov_mean[l]
        mean = mean.reshape((-1,))
        vgg_mean_vars_values += [mean,np.sqrt(np.diag(cov))]
        print('mean05',layer,mean[0:5])
#    sess.close()     
    
    vggAdaIn = vgg_AdaIn(style_layers,vgg_mean_vars_values,final_layer='fc2')
    #print(vggAdaIn.summary())
    #sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    for i in range(1):
        output_vggAdaIn = vggAdaIn.predict(init_image)
        print('First elts',output_vggAdaIn[0,0:5])
#    output_vggAdaIn = sess.run(vgg_fc2_model(init_image))
    print('output_vggAdaIn shape',output_vggAdaIn.shape)
    
    print('Equal fc2vgg  fc2vggAdaIn ? :',np.all(np.equal(output_vggAdaIn, fc2_original_vgg19)))
    print('max abs(a-b)',np.max(np.abs(output_vggAdaIn - fc2_original_vgg19)))
    print('max abs(a-b)/abs(a)',np.max(np.abs(output_vggAdaIn - fc2_original_vgg19)/np.abs(fc2_original_vgg19+10**(-16))))
    print(' norm2(a-b)/norm2(a)',np.linalg.norm(output_vggAdaIn - fc2_original_vgg19)/np.linalg.norm(fc2_original_vgg19+10**(-16)))
    print('First elts',output_vggAdaIn[0,0:5],fc2_original_vgg19[0,0:5])
    
    
def unity_test_StatsCompute():
    """ In this function we try to se if we have a stable and consistent computation
    of mean and variance of the features maps"""
    
    images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
    image_path =  os.path.join(images_path,'abd_aag_002276_624x544.jpg')
    #init_image = tf.convert_to_tensor(load_crop_and_process_img(image_path))
    init_image = load_crop_and_process_img(image_path)

    style_layers = ['block1_conv1']

    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    
    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
    for i in range(5):
        vgg_cov_mean = vgg_get_cov.predict(init_image)
        vgg_mean_vars_values = []
        for l,layer in enumerate(style_layers):
            [cov,mean] = vgg_cov_mean[l]
            mean = mean.reshape((-1,))
            vgg_mean_vars_values += [mean,np.sqrt(np.diag(cov))]
            print('mean05',layer,mean[0:5])
#    sess.close()     
    
if __name__ == '__main__':
  unity_test_StatsCompute()
#  unity_test_of_VGG_AdaIn()