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

import scipy.ndimage as nd
import tensorflow as tf 
from tensorflow.python.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

from show_graph import show_graph

from inception_v1 import InceptionV1

from tensorflow.python.framework.graph_util import convert_variables_to_constants

from lucid.misc.gradient_override import gradient_override_map
from lucid.misc.redirected_relu_grad import redirected_relu_grad

from tensorflow.python.platform import gfile

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




class Lucid_VGGNet(Model):
    
    def __init__(self,model_path = 'model/tf_vgg19.pb',image_shape = [224, 224, 3],\
                 image_value_range = [-125., 125.],input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       # Il semblerait que cela ne soit pas pris en compte !
       self.input_name = input_name
       super(Lucid_VGGNet, self).__init__(**kwargs)
       
class Lucid_Inception_v1(Model):
    
    def __init__(self,model_path = 'model/tf_inception_v1.pb',image_shape = [224, 224, 3],\
                 image_value_range = [-1., 1.],input_name = 'input_1', **kwargs):
       self.model_path = model_path
       self.image_shape = image_shape
       self.image_value_range = image_value_range
       # Il semblerait que cela ne soit pas pris en compte !
       self.input_name = input_name
       super(Lucid_Inception_v1, self).__init__(**kwargs)

def test_render_Inception_v1():
    
    K.set_learning_phase(0)
    with K.get_session().as_default():
        model = InceptionV1(include_top=True, weights='imagenet')
        os.makedirs('./model', exist_ok=True)
        
        #model.save('./model/inception_v1_keras_model.h5')
        frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs],
                                  clear_devices=True)
        # Save the pb model 
        tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_inception_v1.pb", as_text=False)
        
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
            lucid_inception_v1 = Lucid_Inception_v1()
            lucid_inception_v1.load_graphdef()
        
        neuron1 = ('mixed4b_pre_relu', 111)     # large fluffy
        C = lambda neuron: objectives.channel(*neuron)
        out = render.render_vis(lucid_inception_v1, 'Mixed_4b_Concatenated/concat:111',\
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
            out = render.render_vis(lucid_inception_v1, "Mixed_3a_Concatenated/concat:0", transforms=transforms,
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
  
def print_images(model_path,list_layer_index_to_print):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        
        lucid_vgg = Lucid_VGGNet(model_path=model_path)
        lucid_vgg.load_graphdef()
    
        JITTER = 1
        ROTATE = 5
        SCALE  = 1.1
        
        transforms = [
            transform.pad(2*JITTER),
            transform.jitter(JITTER),
            transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
            transform.random_rotate(range(-ROTATE, ROTATE+1))
        ]
        
    
        LEARNING_RATE = 0.005 # Valeur par default

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    
        for layer_index_to_print in list_layer_index_to_print:
            layer, i = layer_index_to_print
            obj = layer  + '/Relu:'+str(i)
            name_base = layer  + '/Relu:'+str(i)
            # "block1_conv1/Conv2D:0"
            output_im = render.render_vis(lucid_vgg,obj ,
                                          transforms=transforms,
                                          thresholds=[4096],
                                          optimizer=optimizer,
                                          use_fixed_seed=True)
      
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

