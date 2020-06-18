#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:33:15 2020

The goal of this script is to try to use the lucid package 
https://github.com/tensorflow/lucid

For importing model : 
    https://github.com/tensorflow/lucid/wiki/Importing-Models-into-Lucid

@author: gonthier
"""

import os

from lucid.modelzoo.vision_models import Model # Need to install lucid doesn't support tf 2.x
# from lucid.modelzoo.vision_base import Model as base_Model # Need to install lucid doesn't support tf 2.x
# Cf ici : https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/vision_base.py
import lucid.optvis.render as render
from lucid.misc.io import show
import lucid.optvis.param as param
import lucid.modelzoo.vision_models as lucid_model
import lucid.optvis.transform as transform
import lucid.optvis.objectives as objectives

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import naive, fft_image

import scipy.ndimage as nd
import tensorflow as tf 
from tensorflow.python.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

from inception_v1 import InceptionV1_slim
from inceptionV1_keras_utils import get_dico_layers_type

from googlenet import create_googlenet as InceptionV1
from googlenet import inception_v1_oldTF

from tensorflow.python.framework.graph_util import convert_variables_to_constants

from lucid.misc.gradient_override import gradient_override_map
from lucid.misc.redirected_relu_grad import redirected_relu_grad

from tensorflow.python.platform import gfile

from lucid.modelzoo.vision_base import IMAGENET_MEAN_BGR

from ImageProcUtils import change_from_BRG_to_RGB

import pathlib

from new_objectif_lucid import autocorr

from infere_layers_info import get_dico_layers_type_all_layers

from lucid.optvis import objectives, param, transform
from lucid.misc.io import show
from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad
from lucid.misc.gradient_override import gradient_override_map

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Function from https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    
    # Il est possible qu il soit necesaire de remplacer par with K.get_session().as_default():
    # car Keras doesn't register it's session as default. As such, you'll want to 
    # do something like this:
    
    #with graph.as_default():
    with K.get_session().as_default():
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph




class Lucid_ResNet(Model):
    
    def __init__(self,model_path = 'model/tf_resnet50.pb',image_shape = [224, 224, 3],\
                 image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       # Il semblerait que cela ne soit pas pris en compte !
       self.input_name = input_name
       super(Lucid_ResNet, self).__init__(**kwargs)
       
class Lucid_VGGNet(Model):
    
    def __init__(self,model_path = 'model/tf_vgg19.pb',image_shape = [224, 224, 3],\
                 image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       # Il semblerait que cela ne soit pas pris en compte !
       self.input_name = input_name
       super(Lucid_VGGNet, self).__init__(**kwargs)
       
class Lucid_InceptionV1_caffe(Model):
    
    def __init__(self,model_path = 'model/tf_inception_v1_caffe.pb',image_shape = [224, 224, 3],\
                 image_value_range = (-IMAGENET_MEAN_BGR, 255-IMAGENET_MEAN_BGR),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       self.input_name = input_name
       super(Lucid_InceptionV1_caffe, self).__init__(**kwargs)
       
class Lucid_GenericFeatureMaps(Model):
    
    def __init__(self,model_path,image_shape = [224, 224, 3],\
                 image_value_range =  (0., np.inf),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       self.input_name = input_name
       super(Lucid_GenericFeatureMaps, self).__init__(**kwargs)
       
class Lucid_InceptionV1(Model):
    
    def __init__(self,model_path = 'model/tf_inception_v1.pb',image_shape = [224, 224, 3],\
                 image_value_range =  (-117, 255-117),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       self.input_name = input_name
       super(Lucid_InceptionV1, self).__init__(**kwargs)
       
class Lucid_Inception_v1_slim(Model):
    
    def __init__(self,model_path = 'model/tf_inception_v1_slim.pb',image_shape = [224, 224, 3],\
                 image_value_range = (-1, 1),input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       self.input_name = input_name
       super(Lucid_Inception_v1_slim, self).__init__(**kwargs)

def create_pb_model_of_pretrained(Net):
    K.set_learning_phase(0)
    with K.get_session().as_default():
        if Net=='InceptionV1_slim':
            model = InceptionV1_slim(include_top=True, weights='imagenet')
            name= "tf_inception_v1_slim.pb"
        elif Net=='InceptionV1':
            model = inception_v1_oldTF(weights='imagenet',include_top=True)
            name= "tf_inception_v1.pb"
        elif Net=='VGG':
            model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
            name= "tf_vgg19.pb"
        elif Net=='ResNet50':
            model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',\
                                                          input_shape= (224, 224, 3))
            name= "tf_resnet50.pb"
        else:
            raise(ValueError(Net+' is unknown'))
            
        os.makedirs('./model', exist_ok=True)
        
        frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs],
                                  clear_devices=True)
        # Save the pb model 
        tf.io.write_graph(frozen_graph,logdir= "model",name=name, as_text=False)
        

def test_render_Inception_v1_slim():
    
    K.set_learning_phase(0)
    with K.get_session().as_default():
        model = InceptionV1_slim(include_top=True, weights='imagenet')
        os.makedirs('./model', exist_ok=True)
        
        #model.save('./model/inception_v1_keras_model.h5')
        frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs],
                                  clear_devices=True)
        # Save the pb model 
        tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_inception_v1_slim.pb", as_text=False)
        
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        
        # f = gfile.FastGFile("/model/tf_inception_v1.pb", 'rb')
        # graph_def = tf.GraphDef()
        # # Parses a serialized binary message into the current message.
        # graph_def.ParseFromString(f.read())
        # f.close()
        
        # sess.graph.as_default()
        # # Import a serialized TensorFlow `GraphDef` protocol buffer
        # # and place into the current default `Graph`.
        # tf.import_graph_def(graph_def)
        
        # nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #print(nodes_tab)
        with gradient_override_map({'Relu': redirected_relu_grad,'ReLU': redirected_relu_grad}):
            # cela ne semble pas apporter quoique ce soit de particulier 
            lucid_inception_v1 = Lucid_Inception_v1_slim()
            lucid_inception_v1.load_graphdef()
        
        neuron1 = ('mixed4b_pre_relu', 111)     # large fluffy
        C = lambda neuron: objectives.channel(*neuron)
        out = render.render_vis(lucid_inception_v1, 'Mixed_4b_Concatenated/concat:452',\
                                relu_gradient_override=True,use_fixed_seed=True)
        plt.imshow(out[0][0])

        JITTER = 1
        ROTATE = 5
        SCALE  = 1.1
        
        transforms = [
            transform.pad(2*JITTER),
            transform.jitter(JITTER),
            transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
            transform.random_rotate(range(-ROTATE, ROTATE+1))
        ]
        # https://github.com/tensorflow/lucid/issues/82
        with gradient_override_map({'Relu': redirected_relu_grad,'ReLU': redirected_relu_grad}):
            out = render.render_vis(lucid_inception_v1, "Mixed_4b_Concatenated/concat:452", transforms=transforms,
                                     param_f=lambda: param.image(64), 
                                     thresholds=[2048], verbose=False,\
                                     relu_gradient_override=True,use_fixed_seed=True)
        # out = render.render_vis(lucid_inception_v1, "Mixed_4d_Branch_2_b_3x3_act/Relu:452", transforms=transforms,
        #                          param_f=lambda: param.image(64), 
        #                          thresholds=[2048], verbose=False) # Cela ne marche pas !
        plt.imshow(out[0][0])
        
        out = render.render_vis(lucid_inception_v1, "Mixed_3c_Concatenated/concat:479", transforms=transforms,
                                 param_f=lambda: param.image(64), 
                                 thresholds=[2048], verbose=False,\
                                 relu_gradient_override=True,use_fixed_seed=True)
        plt.imshow(out[0][0])
        
def test_render_Inception_v1():
    
    tf.reset_default_graph()
    K.set_learning_phase(0)
    if not(os.path.isfile("model/tf_inception_v1.pb")):
        with K.get_session().as_default():
            model = inception_v1_oldTF(weights='imagenet',include_top=True) #include_top=True, weights='imagenet')
            print(model.input)
            os.makedirs('./model', exist_ok=True)
            
            frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.outputs],
                                      clear_devices=True)
            # Save the pb model 
            tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_inception_v1.pb", as_text=False)
            nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print(nodes_tab)
        
    #with tf.Graph().as_default() as graph, tf.Session() as sess:
    with gradient_override_map({'Relu': redirected_relu_grad,'ReLU': redirected_relu_grad}):
        lucid_inception_v1 = Lucid_InceptionV1()
        lucid_inception_v1.load_graphdef()
        
    obj = lambda couple_layer_id: autocorr(*couple_layer_id)
    
    out = render.render_vis(lucid_inception_v1, obj('mixed4a_1x1_pre_relu/Conv2D',0),\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])
    
    
    out = render.render_vis(lucid_inception_v1, obj('mixed4b_pre_relu/concat',452),\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])

    
    JITTER = 1
    ROTATE = 5
    SCALE  = 1.1
    
    transforms = [
        transform.pad(2*JITTER),
        transform.jitter(JITTER),
        transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
        transform.random_rotate(range(-ROTATE, ROTATE+1))
    ]
    
    imgs = render.render_vis(lucid_inception_v1,obj('mixed4b_pre_relu/concat',452), 
                             transforms=transforms,
                             param_f=lambda: param.image(64), 
                             thresholds=[2048], verbose=False,
                             relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(imgs[0][0])
    
def test_autocorr_render_Inception_v1():
    
    tf.reset_default_graph()
    K.set_learning_phase(0)
    if not(os.path.isfile("model/tf_inception_v1.pb")):
        with K.get_session().as_default():
            model = inception_v1_oldTF(weights='imagenet',include_top=True) #include_top=True, weights='imagenet')
            print(model.input)
            os.makedirs('./model', exist_ok=True)
            
            frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.outputs],
                                      clear_devices=True)
            # Save the pb model 
            tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_inception_v1.pb", as_text=False)
            nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print(nodes_tab)
        
    #with tf.Graph().as_default() as graph, tf.Session() as sess:
    with gradient_override_map({'Relu': redirected_relu_grad,'ReLU': redirected_relu_grad}):
        lucid_inception_v1 = Lucid_InceptionV1()
        lucid_inception_v1.load_graphdef()
        
    
    out = render.render_vis(lucid_inception_v1, 'mixed4a_1x1_pre_relu/Conv2D:0',\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])
    
    
    out = render.render_vis(lucid_inception_v1, 'mixed4b_pre_relu/concat:452',\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])

    
    JITTER = 1
    ROTATE = 5
    SCALE  = 1.1
    
    transforms = [
        transform.pad(2*JITTER),
        transform.jitter(JITTER),
        transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
        transform.random_rotate(range(-ROTATE, ROTATE+1))
    ]
    
    imgs = render.render_vis(lucid_inception_v1, 'mixed4b_pre_relu/concat:452', transforms=transforms,
                             param_f=lambda: param.image(64), 
                             thresholds=[2048], verbose=False,
                             relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(imgs[0][0])
    #input('Enter to close')
    #plt.close()
    
def test_render_ResNet50():
    
    tf.reset_default_graph()
    K.set_learning_phase(0)
    if not(os.path.isfile("model/tf_resnet50.pb")):
        with K.get_session().as_default():
            model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',\
                                                          input_shape= (224, 224, 3))
            print(model.input)
            os.makedirs('./model', exist_ok=True)
            
            frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in model.outputs],
                                      clear_devices=True)
            # Save the pb model 
            tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_resnet50.pb", as_text=False)
            nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print(nodes_tab)

    lucid_resnet50 = Lucid_ResNet()
    lucid_resnet50.load_graphdef()
        
    
    out = render.render_vis(lucid_resnet50, 'conv4_block6_2_conv/Conv2D:0',\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])
    
    
    out = render.render_vis(lucid_resnet50, 'conv2_block1_2_conv/Conv2D:32',\
                            relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(out[0][0])

    
    JITTER = 1
    ROTATE = 5
    SCALE  = 1.1
    
    transforms = [
        transform.pad(2*JITTER),
        transform.jitter(JITTER),
        transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
        transform.random_rotate(range(-ROTATE, ROTATE+1))
    ]
    
    imgs = render.render_vis(lucid_resnet50,'conv4_block4_2_conv/Conv2D:0', transforms=transforms,
                             param_f=lambda: param.image(64), 
                             thresholds=[2048], verbose=False,
                             relu_gradient_override=True,use_fixed_seed=True)
    plt.figure()
    plt.imshow(imgs[0][0])



def feature_block(channels,w, h=None, batch=None, sd=None, fft=True):
  #decorrelate=True
  h = h or w
  batch = batch or 1
  shape = [batch, w, h, channels]
  #param_f = fft_image if fft else naive
  #t = param_f(shape, sd=sd)
  #rgb = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
  sd = sd or 0.01
  init_val = sd*np.random.randn(*shape).astype("float32")
  t = tf.Variable(init_val)
  #t = tf.nn.sigmoid(t) # To cast in 0-1 but maybe need to be cast in -1/1 or something else
  #t = tf.math.tanh(t) # To cast in -1/1 avec tanh or other => divergence
  #return(t)
  return tf.convert_to_tensor(t)

def feature_block_var(channels,w, h=None, batch=None, sd=None, fft=True):
  #decorrelate=True
  h = h or w
  batch = batch or 1
  shape = [batch, w, h, channels]
  #param_f = fft_image if fft else naive
  #t = param_f(shape, sd=sd)
  #rgb = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
  sd = sd or 0.00001
  init_val = sd*np.random.randn(*shape).astype("float32")
  print('init_val',np.max(init_val),np.min(init_val))
  t = tf.Variable(init_val)
  #t = tf.nn.sigmoid(t) # To cast in 0-1 but maybe need to be cast in -1/1 or something else
  #t = tf.math.tanh(t) # To cast in -1/1 avec tanh or other => divergence
  #return(t)
  return t
  
def print_images(model_path,list_layer_index_to_print,path_output='',prexif_name='',\
                 input_name='block1_conv1_input',Net='VGG',sizeIm=256,\
                 DECORRELATE = True,ROBUSTNESS  = True,just_return_output=False,
                 dico=None,image_shape=None,inverseAndSave=True):
    """
    This fct will run the feature visualisation for the layer and feature in the
    list_layer_index_to_print list 
    """
    #,printOnlyRGB=True
    
    if not(os.path.isfile(os.path.join(model_path))):
        raise(ValueError(model_path + ' does not exist !'))
    
    if Net=='VGG':
        lucid_net = Lucid_VGGNet(model_path=model_path,input_name=input_name)
    elif Net=='InceptionV1':
        lucid_net = Lucid_InceptionV1(model_path=model_path,input_name=input_name)
    elif Net=='InceptionV1_slim':
        lucid_net = Lucid_Inception_v1_slim(model_path=model_path,input_name=input_name)
    elif 'ResNet' in Net:
        lucid_net = Lucid_ResNet(model_path=model_path,input_name=input_name)
    else:
        raise(ValueError(Net+ 'is unkonwn'))
    lucid_net.load_graphdef()
    nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
    assert(input_name in nodes_tab)
    #print(nodes_tab)
    
    # `fft` parameter controls spatial decorrelation
    # `decorrelate` parameter controls channel decorrelation
    param_f = lambda: param.image(sizeIm, fft=DECORRELATE, decorrelate=DECORRELATE)
    
    if DECORRELATE:
        ext='_Deco'
    else:
        ext=''
    
    if ROBUSTNESS:
      transforms = transform.standard_transforms
      ext+= ''
    else:
      transforms = []
      ext+= '_noRob'

    verbose = True

#    LEARNING_RATE = 0.005 # Valeur par default
#    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    output_im_list = []
    for layer_index_to_print in list_layer_index_to_print:
        layer, i = layer_index_to_print
    
        obj_str, type_layer= get_obj_and_kind_layer(layer,Net)
        obj = obj_str+':'+str(i)
        name_base = layer  + type_layer+'_'+str(i)+'_'+prexif_name+ext+'.png'
              
        output_im = render.render_vis(lucid_net,obj ,
                                      transforms=transforms,
                                      thresholds=[2048],
                                      param_f=param_f,
#                                      optimizer=optimizer,
                                      use_fixed_seed=True,
                                      verbose=verbose)
        if just_return_output:
            output_im_list += [output_im]
        else:
            image = np.array(output_im[0][0]*255) # car une seule image dans le batch
            name_output = os.path.join(path_output,name_base)
            new_output_path = os.path.join(path_output,'RGB')
            if inverseAndSave:
                name_output = name_output.replace('.png','_toRGB.png')
                image =image[:,:,[2,1,0]]
                tf.keras.preprocessing.image.save_img(name_output, image)
            
            else:
                tf.keras.preprocessing.image.save_img(name_output, image)
                pathlib.Path(new_output_path).mkdir(parents=True, exist_ok=True) 
                change_from_BRG_to_RGB(img_name_path=name_output,output_path=new_output_path,
                                   ext_name='toRGB')
    return(output_im_list)
    
def get_feature_block_that_maximizeGivenOutput(model_path,list_layer_index_to_print,
                                         input_name='block1_conv1_input',sizeIm=256,\
                                         DECORRELATE = True,ROBUSTNESS  = True,
                                         dico=None,image_shape=None):

    
    if not(os.path.isfile(os.path.join(model_path))):
        raise(ValueError(model_path + ' does not exist !'))
    
    assert(not(image_shape is None))
    lucid_net = Lucid_GenericFeatureMaps(model_path=model_path,
                                         image_shape=image_shape,
                                         input_name=input_name)
    lucid_net.load_graphdef()
    nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
    assert(input_name in nodes_tab)
    #print(nodes_tab)
    
    # `fft` parameter controls spatial decorrelation
    # `decorrelate` parameter controls channel decorrelation
    print('image_shape[2],image_shape[0], h=image_shape[1]')
    print(image_shape[2],image_shape[0],image_shape[1])
    param_f = lambda: feature_block_var(image_shape[2],image_shape[0], h=image_shape[1], batch=1, sd=None, fft=DECORRELATE)
      
    #print(feature_block(image_shape[2],image_shape[0], h=image_shape[1], batch=1, sd=None, fft=DECORRELATE))
    
#    if DECORRELATE:
#        ext='_Deco'
#    else:
#        ext=''
    
    if ROBUSTNESS:
        JITTER = 1
        #ROTATE = 1
        SCALE  = 1.1
        
        transforms = [
            transform.pad(2*JITTER),
            transform.jitter(JITTER),
            transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
            #transform.random_rotate(range(-ROTATE, ROTATE+1))
        ]
    else:
      transforms = []

    verbose = False
    # You need to  provide a dico of the correspondance between the layer name
    # an op node !
    assert(not(dico is None))
    
    LEARNING_RATE = 0.0005 # Valeur par default
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    output_im_list = []
    for layer_index_to_print in list_layer_index_to_print:
        layer, i = layer_index_to_print
        
        layer_str = dico[layer]
        obj = layer  + '/'+layer_str+':'+str(i)
        print('obj',obj)
        
#        output_im = render.render_vis(lucid_net,obj ,
#                                      transforms=transforms,
#                                      thresholds=[0],
#                                      param_f=param_f,
#                                      optimizer=optimizer,
#                                      use_fixed_seed=True,
#                                      verbose=verbose)
        output_im = lbfgs_min(lucid_net,obj ,
                                      transforms=transforms,
                                      thresholds=[2048],
                                      param_f=param_f,
                                      optimizer=optimizer,
                                      use_fixed_seed=True,
                                      verbose=verbose)
        output_im_list += [output_im]
        
    return(output_im_list)
    
def lbfgs_min(model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=True, relu_gradient_override=True, use_fixed_seed=False):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
    
        if use_fixed_seed:  # does not mean results are reproducible, see Args doc
          tf.set_random_seed(0)
    
#        T = render.make_vis_T(model, objective_f, param_f, optimizer, transforms,
#                       relu_gradient_override)
        
        t_image = param_f()
#        placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
#        placeholder_clip = tf.placeholder(tf.float32, shape=init_img.shape)
#
#        assign_op = net['input'].assign(placeholder)
        #assert(isinstance(t_image, tf.Tensor))
        objective_f = objectives.as_objective(objective_f)
        transform_f = render.make_transform_f(transforms)
        #optimizer = make_optimizer(optimizer, [])
        
        #global_step = tf.train.get_or_create_global_step()
        #init_global_step = tf.variables_initializer([global_step])
        #init_global_step.run()
        
        if relu_gradient_override:
            with gradient_override_map({'Relu': redirected_relu_grad,
                                        'Relu6': redirected_relu6_grad}):
                T = render.import_model(model, transform_f(t_image), t_image)
        else:
            T = render.import_model(model, transform_f(t_image), t_image)
        loss = objective_f(T)
        t_image= T("input")
        
        #print_objective_func = render.make_print_objective_func(print_objectives, T)
        #loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        
        gradient = tf.gradients(loss,t_image)
        
        i = 0
        def callback(loss,var,grad):
          nonlocal i
          print('Loss evaluation #', i, ', loss:', loss,'var max',np.max(var),'grad max',np.max(grad))
          i += 1
        maxcor = 30
        print_disp = 1
        optimizer_kwargs = {'maxiter':max(thresholds),'maxcor': maxcor, \
                    'disp': print_disp}
          #bnds = get_lbfgs_bnds(init_img,clip_value_min,clip_value_max,BGR)
        trainable_variables = tf.trainable_variables()[0]
        print(trainable_variables)
#        var_eval = trainable_variables.eval()
#        print('initialization before variable init',np.max(var_eval),np.min(var_eval))
          #var_to_bounds = {trainable_variables: bnds}
#        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,var_to_bounds=var_to_bounds,
#                        method='L-BFGS-B',options=optimizer_kwargs)
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(-loss,
                        method='L-BFGS-B',options=optimizer_kwargs)
        
        tf.global_variables_initializer().run()
        
        var_eval = trainable_variables.eval()
        print('initialization after variable init',np.max(var_eval),np.min(var_eval))
        var_eval = t_image.eval()
        print('initialization',np.max(var_eval),np.min(var_eval))
        images = []
        loss_ = sess.run([loss])
        print("beginning loss :", loss_)
        optimizer.minimize(sess, fetches=[loss,t_image,gradient], loss_callback=callback)
        vis = t_image.eval()
        images.append(vis)
        loss_ = sess.run([loss])
        print("End loss :", loss_)
#        try:
#          #sess.run(assign_op, {placeholder: init_img})
#          optimizer.minimize(sess,step_callback=callback)
#          for i in range(max(thresholds)+1):
#            loss_, _ = sess.run([loss, vis_op])
#            if i in thresholds:
#              vis = t_image.eval()
#              images.append(vis)
#              if verbose:
#                print(i, loss_)
#                print_objective_func(sess)
#                show(np.hstack(vis))
#        except KeyboardInterrupt:
#          log.warning("Interrupted optimization at step {:d}.".format(i+1))
#          vis = t_image.eval()
#          show(np.hstack(vis))
    
        return images

@objectives.wrap_objective
def direction_neuron_cossim_S(layer_name, vec, batch=None, x=None, y=None, cossim_pow=1, S=None):
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
          raise RuntimeError("requires batch")

        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None: vec_ = tf.matmul(vec_[None], S)[0]
        mag = tf.sqrt(tf.reduce_sum(acts**2))
        dot = tf.reduce_mean(acts * vec_)
        cossim = dot/(1e-4 + mag)
        cossim = tf.maximum(0.1, cossim)
        return dot * cossim ** cossim_pow
    return inner  
 
def get_obj_and_kind_layer(layer_to_print,Net):
    if Net=='VGG':
        if '_conv' in layer_to_print:
            type_layer = 'Relu' # Cas les couches convolutionnels sont avec relu en fait
        elif 'fc' in  layer_to_print:
            type_layer = 'Relu'
        elif 'predictions' in  layer_to_print:
            type_layer = 'Softmax'
        else:
            raise(ValueError(layer_to_print+' for '+Net))
        obj_str = layer_to_print  + '/' + type_layer
        kind_layer = type_layer
    elif 'ResNet' in Net:
        if '_conv' in layer_to_print:
            type_layer = 'Conv2D'
        elif '_relu' in layer_to_print:
            type_layer = 'Activation' # Activation we hope it is a max
        elif '_bn' in layer_to_print:
            type_layer = 'BatchNormalization' # I never try that
        else:
            raise(ValueError(layer_to_print+' for '+Net))
        obj_str = layer_to_print  + '/' + type_layer
        kind_layer = type_layer
        
    elif Net=='InceptionV1':
        dico = get_dico_layers_type()
        type_layer = dico[layer_to_print]
        obj_str = layer_to_print  + '/'+type_layer # It could also be BiasAdd or concat
        kind_layer = type_layer
#        if not(obj_str in nodes_tab):
#            obj_str = layer_to_print  + '/Concat'
#            kind_layer = 'Concat'
#            if not(obj_str in nodes_tab):
#                obj_str = layer_to_print  + '/Relu'
#                kind_layer = 'Relu'
#                if not(obj_str in nodes_tab):
#                    print(nodes_tab)
#                    raise(KeyError(obj_str +' not in the graph'))
    elif Net=='InceptionV1_slim':
        if '_Concatenated' in layer_to_print:
            type_layer = 'concat'
        elif '_conv' in layer_to_print:
            type_layer = 'Conv2D'
        elif '_act' in layer_to_print:
            type_layer = 'Max' # Activation we hope it is a max
        elif '_bn' in layer_to_print:
            type_layer = 'BatchNormalization' # I never try that
        elif 'MaxPool' in layer_to_print:
            type_layer = 'Max' # I never try that
        else:
            raise(ValueError(layer_to_print+' for '+Net))
            
        obj_str = layer_to_print  + '/'+type_layer # It could also be BiasAdd or concat
        kind_layer = type_layer

    else:
        raise(NotImplementedError)    
        
    return(obj_str,kind_layer)


def print_PCA_images(model_path,layer_to_print,weights,index_features_withinLayer,\
                     path_output='',prexif_name='',\
                     input_name='block1_conv1_input',Net='VGG',sizeIm=256,\
                     DECORRELATE=True,ROBUSTNESS=True,\
                     inverseAndSave=True,cossim=False,dot_vector = False,
                     num_features=None):
#    ,printOnlyRGB=True
    
    if not(os.path.isfile(os.path.join(model_path))):
        raise(ValueError(model_path + ' does not exist !'))
         
    if Net=='VGG':
        lucid_net = Lucid_VGGNet(model_path=model_path,input_name=input_name)
    elif Net=='InceptionV1':
        lucid_net = Lucid_InceptionV1(model_path=model_path,input_name=input_name)
    elif Net=='InceptionV1_slim':
        lucid_net = Lucid_Inception_v1_slim(model_path=model_path,input_name=input_name)
    elif Net=='ResNet':
        lucid_net = Lucid_ResNet(model_path=model_path,input_name=input_name)
    else:
        raise(ValueError(Net+ 'is unkonwn'))
    lucid_net.load_graphdef()
    #nodes_tab = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    
    obj_str, kind_layer= get_obj_and_kind_layer(layer_to_print,Net)
        
    param_f = lambda: param.image(sizeIm, fft=DECORRELATE, decorrelate=DECORRELATE)
    
    if DECORRELATE:
        ext='_Deco'
    else:
        ext=''
    
    if ROBUSTNESS:
      transforms = transform.standard_transforms
      ext+= ''
    else:
      transforms = []
      ext+= '_noRob'

#    LEARNING_RATE = 0.005 # Valeur par default
#    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
      
    if Net=='VGG':
        name_base = layer_to_print  + kind_layer+'_'+prexif_name+ext+'.png'
    elif Net=='InceptionV1':
        name_base = layer_to_print  + kind_layer+'_'+prexif_name+ext+'.png'
    elif Net=='InceptionV1_slim':
        name_base = layer_to_print  + kind_layer+'_'+prexif_name+ext+'.png'
    else:
        raise(NotImplementedError)
            
    
    # input = couple of Two arguments : first one name of the layer, 
    # second one number of the features must be an integer
    
    if(cossim is True):
        raise(NotImplementedError)
        # Pas fini ici il faut le S 
        # cette fonction objectif est le produit entre le produit scalaire de v et des couches ainsi que 
        # multiplier par le cosinus a une certaine valeur
        obj_list = ([
                direction_neuron_cossim_S(layer, v, batch=n, S=S, cossim_pow=4) for n,v in enumerate(directions)
                ])
        total_obj = objectives.Objective.sum(obj_list)
    elif dot_vector:
        assert(not(num_features is None))
        def inner(T): 
            layer = T(obj_str)
            #print('num_features',num_features)
            if len(weights)==num_features:
                total_obj = tf.reduce_mean(layer * weights)
            else:
                weights_total = np.zeros((num_features,))
                weights_total[index_features_withinLayer] = weights
                total_obj = tf.reduce_mean(layer * weights_total)

            return(total_obj)
            
        total_obj = inner
    else:
        C = lambda layer_i: objectives.channel(*layer_i)
        total_obj = None
        for i,weight_i in zip(index_features_withinLayer,weights):
            
            if total_obj is None:
                total_obj = weight_i*C((obj_str,i))
            else: 
                total_obj += weight_i*C((obj_str,i))

    output_im = render.render_vis(lucid_net,total_obj ,
                                  transforms=transforms,
                                  thresholds=[2048],
                                  param_f=param_f,
#                                      optimizer=optimizer,
                                  use_fixed_seed=True)
    image = np.array(output_im[0][0]*255) # car une seule image dans le batch
    name_output = os.path.join(path_output,name_base)
    new_output_path = os.path.join(path_output,'RGB')
    
    if inverseAndSave:
        name_output = name_output.replace('.png','_toRGB.png')
        image =image[:,:,[2,1,0]]
        tf.keras.preprocessing.image.save_img(name_output, image)
    
    else:
        tf.keras.preprocessing.image.save_img(name_output, image)
        pathlib.Path(new_output_path).mkdir(parents=True, exist_ok=True) 
        change_from_BRG_to_RGB(img_name_path=name_output,output_path=new_output_path,
                           ext_name='toRGB')

def test_render_VGG19():
    
    #with tf.Graph().as_default() as graph, tf.Session() as sess:
    K.set_learning_phase(0)
    with K.get_session().as_default(): 
        #images = tf.placeholder("float32", [None, 224, 224, 3], name="input")
    
        # <Code to construct & load your model inference graph goes here>
        model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
        
        #  ! il va falloir ajouter des noeuds / node pre_relu !
        
        os.makedirs('./model', exist_ok=True)
        #model.save('./model/keras_model.h5')
        frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
        
        # Show current session graph with TensorBoard in Jupyter Notebook.
        #show_graph(tf.get_default_graph().as_graph_def())
        
        tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_vgg19.pb", as_text=False)
        
        nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
        
        # base_Model_instance = base_Model()
        # base_Model_instance.suggest_save_args()
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        
        lucid_vgg = Lucid_VGGNet()
        lucid_vgg.load_graphdef()
        # for node in lucid_vgg.graph_def.node:
        #     if 'conv' in node.op:
        #         print(node.name)

        #Model.suggest_save_args()
        #lucid_model = Model.load_graphdef("tf_model.pb")
    
        LEARNING_RATE = 0.05 # Valeur par default

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Conv2D:0",use_fixed_seed=0)
        # Il semblerait que par default cela renvoit un crop de l image genere en 224*224 de taille 128*128
        plt.imshow(output_im[0][0])
        print(np.max(output_im),np.min(output_im))
        
        output_im = render.render_vis(lucid_vgg, "block1_conv1/BiasAdd:1",use_fixed_seed=0)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Relu:0",use_fixed_seed=True)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Relu:1",use_fixed_seed=True)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Relu:2",use_fixed_seed=True)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Relu:3",use_fixed_seed=True)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block5_conv4/Relu:0")
        plt.imshow(output_im[0][0])
        
        # <IPython.core.display.HTML object> only plot in jupyter Notebook
        param_f = lambda: param.image(128)
        output_im = render.render_vis(lucid_vgg, "block5_conv4/BiasAdd:100", param_f,thresholds=[256])
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block5_conv4/Relu:100", param_f,thresholds=[256])
        plt.imshow(output_im[0][0])
        
        # Using alternate parameterizations is one of the primary ingredients for
        # effective visualization
        param_f = lambda: param.image(224, fft=True, decorrelate=True)
        output_im = render.render_vis(lucid_vgg,"block1_conv1/Relu:0", param_f)
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg,"block5_conv4/Relu:0", param_f)
        plt.imshow(output_im[0][0])
        
        JITTER = 1
        ROTATE = 5
        SCALE  = 1.1
        
        transforms = [
            transform.pad(2*JITTER),
            transform.jitter(JITTER),
            transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
            transform.random_rotate(range(-ROTATE, ROTATE+1))
        ]
        
        image_full_tricks = render.render_vis(lucid_vgg, "block5_conv4/Relu:0", transforms=transforms,
                                 param_f=lambda: param.image(64, fft=True, decorrelate=True),
                                 optimizer=optimizer,
                                 thresholds=[2048], verbose=False)
        plt.imshow(image_full_tricks[0][0])
        image_full_tricks = render.render_vis(lucid_vgg, "block3_conv1/Relu:0", transforms=transforms,
                                 param_f=lambda: param.image(64, fft=True, decorrelate=True),
                                 optimizer=optimizer,
                                 thresholds=[2048], verbose=False)
        plt.imshow(image_full_tricks[0][0])
        image_full_tricks = render.render_vis(lucid_vgg, "block3_conv1/Relu:1", transforms=transforms,
                                 param_f=lambda: param.image(64, fft=False, decorrelate=False),
                                 optimizer=optimizer,
                                 thresholds=[2048], verbose=False)
        plt.imshow(image_full_tricks[0][0])

        # Note that we're doubling the image scale to make artifacts more obvious
        #image_mde = show([nd.zoom(img[0], [2,2,1], order=0) for img in imgs])
        #plt.imshow(image_mde)
        
        ### TODO :
        #  https://github.com/totti0223/lucid4keras/blob/master/examples/ex3.ipynb
        # Verifier que tu as les meme sortie pour le VGG pretrained avec le code ci dessus !
        # Tester vec un VGG random ce que tu obtient pour la meme features !
        # Pour cela il va falloir essayer de figer la seed  : 
        # Essayer d'autres generation d images avec la couche 1 du reseau et voir si on a autre chose
        
        # Verifier si tu nas pas un problem avec le range de limage de sortie autoriser. J'ai l'impression que tu as que du 0 1 au lieu du -mean imagenet 255 - imagnenet mean
        # https://github.com/tensorflow/lucid/wiki/Importing-Models-into-Lucid
        
        # Verifier que les con2d ou bias sont bien la sortie des couches juste avant le ReLu ! 

if __name__ == '__main__':
    test_render_Inception_v1()

