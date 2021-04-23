#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:51:03 2019

The goal of this script is to evalaute a new way to transfer network for a 
classification task by modifying the data to a new artistic domain : 
    
    - Impose the 
    
For the Gram Matrix, you can learn use :
    - identity
    - diagonal matrix learned
    - full matrix learned
    
The Gram Matrix can be learn from :
    - the dataset
    - a big artistic dataset
    - a photography dataset : ImageNet


@author: gonthier
"""

import tensorflow as tf
from trouver_classes_parmi_K import TrainClassif

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def vgg_layers_stats(layer_names,stats='MeanVar'):
  """ Creates a vgg model that returns a list of statistics of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  if stats=='MeanVar':
    outputs = [mean_and_var(vgg.get_layer(name).output) for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  """
    This function compute the Gram Matrix Covariance matrix of an input tensor
  """
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def mean_and_var(input_tensor):
  mean_and_var = tf.nn.moments(input_tensor,axes=1,shift=None,name=None,keep_dims=None)
  return(mean_and_var)
  
def get_dataset(dataset):
  ext = '.txt'
  path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
  if database=='Paintings':
      item_name = 'name_img'
      path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/' 
  elif database=='VOC12':
      item_name = 'name_img'
      path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
  elif(database=='WikiTenLabels'):
      ext = '.csv'
      item_name='item'
      path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/MiniSet10c_Qname/'
  elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset'):
      item_name = 'image'
      if not(augmentation):
          path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/224/'
      else:
          path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/256/'
  else:
      item_name = 'image'
      path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/224/'
  databasetxt = path_data + database + ext
  df_label = pd.read_csv(databasetxt,sep=",")
  return(df_label,item_name)
  
def get_training_set_list(dataset):
  df_label,item_name = get_dataset(dataset)
  df_test = df_label[df_label['set']=='test']
  list_names = df_test[item_name].values
  return(list_names)
  
  
    if augmentation:
        N = 50
    else: 
        N=1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    if(kind=='fuco8'):
        size_output = 1000
    elif(kind in ['fuco6','relu7','relu6','fuco7']):
        size_output = 4096    
    
    name_img = df_label[item_name][0]
    i = 0
    itera = 1000
        
    name_network = 'VGG'+VGG+'_'
    name_pkl_values = path_data+'Values_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    if not(concate):
        features_resnet = None
    else:
        features_resnet = pickle.load(open(name_pkl_values, 'rb'))
        im_resnet = pickle.load(open(name_pkl_im, 'rb'))

    with tf.Graph().as_default():
      # The Inception networks expect the input image to have color channels scaled from [-1, 1]
      #Load the model
      sess = tf.Session()
      if VGG=='19':
          vgg_layers = vgg19.get_vgg_layers()
      elif VGG=='16':
          vgg_layers = vgg16.get_vgg_layers()
      input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
      if VGG=='19':
          net = vgg19.net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME')
      elif VGG=='16':
          net = vgg16.net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME')

      sess.run(tf.global_variables_initializer())

      for i,name_img in  enumerate(df_label[item_name]):
            if not(concate) or not(name_img in  im_resnet):
                if i%itera==0:
                    print(i,name_img)
                
                if database=='VOC12' or database=='Paintings':
                    complet_name = path_to_img + name_img + '.jpg'
                    im = cv2.imread(complet_name)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def get_statistics(dataset,stats,layers):
  stats_extractor = vgg_layers_stats(layers,stats)
  
  list_names = get_training_set_list(dataset)
  
  stats_dict = {}
  for im_name in list_names:
    style_image = load_img(style_path)
    stats_outputs = stats_extractor(style_image*255)
    #Look at the statistics of each layer's output
    for name, output in zip(style_layers, style_outputs):
      #stats_dict[name] = ? # TODO finir ici
#  x = tf.keras.applications.vgg19.preprocess_input(content_image*255)      
  
def TL_With_Imposed_Stats(dataset,dataset_stats=None,stats='MeanVar'):
  """ In this function we will precompute some statistics on some train set then
    extract features and imposing those statistics and after learn a classifier
    per class on the features 
  @param stats : type of statistics used : 'MeanVar' or 'Gram' or None
  @param dataset_stats : dataset on with we compute the statistics if None we take 
      0 and 1 for mean and variance and Identity for Gram Matrixes
  """
  
  # Style layer of interest
  style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
  
  # Precompute the statistics on dataset_stats
  if dataset_stats is None or dataset_stats=='':
      if stats=='MeanVar':
          return(0)
  else:
      get_statistics(dataset,stats,style_layers)
      
      if stats=='MeanVar':
          
          
      
  

    TrainClassif(X,y,clf='LinearSVC',class_weight=None,gridSearch=True,n_jobs=-1,C_finalSVM=1,cskind=None)