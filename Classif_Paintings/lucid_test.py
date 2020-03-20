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
from lucid.misc.io import show,image
import lucid.optvis.param as param
import lucid.modelzoo.vision_models as lucid_model
import lucid.optvis.transform as transform

import scipy.ndimage as nd
import tensorflow as tf 
from tensorflow.python.keras import backend as K

import matplotlib.pyplot as plt

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
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    # Il est possible qu il soit necesaire de remplacer par with K.get_session().as_default():  car Keras doesn't register it's session as default. As such, you'll want to do something like this:
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
  model_path = 'model/tf_model.pb'
  image_shape = [224, 224, 3]
  image_value_range = (-125., 125.) # Il semblerait que cela ne soit pas pris en compte !
  input_name = 'input_1'

with tf.Graph().as_default() as graph, tf.Session() as sess:
    images = tf.placeholder("float32", [None, 224, 224, 3], name="input")

    # <Code to construct & load your model inference graph goes here>
    weights = 'imagenet'
    imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights,input_shape=(224,224,3))


    Model.suggest_save_args()

def test_render_VGG19():
    
    #with tf.Graph().as_default() as graph, tf.Session() as sess:
    with K.get_session().as_default(): 
        #images = tf.placeholder("float32", [None, 224, 224, 3], name="input")
    
        # <Code to construct & load your model inference graph goes here>
        model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
        
        #  ! il va falloir ajouter des noeuds / node pre_relu !
        
        os.makedirs('./model', exist_ok=True)
        model.save('./model/keras_model.h5')
        frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
        tf.io.write_graph(frozen_graph,logdir= "model",name= "tf_model.pb", as_text=False)
        
        nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
        
        # base_Model_instance = base_Model()
        # base_Model_instance.suggest_save_args()
        
        lucid_vgg = Lucid_VGGNet()
        lucid_vgg.load_graphdef()
        for node in lucid_vgg.graph_def.node:
            if 'conv' in node.op:
                print(node.name)

        #Model.suggest_save_args()
        #lucid_model = Model.load_graphdef("tf_model.pb")
    
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Conv2D:0")
        # Il semblerait que par default cela renvoit un crop de l image genere en 224*224 de taille 128*128
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/BiasAdd:0")
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block1_conv1/Relu:0")
        plt.imshow(output_im[0][0])
        output_im = render.render_vis(lucid_vgg, "block5_conv4/Relu:0")
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
        
        imgs = render.render_vis(lucid_vgg, "block5_conv4/Relu:0", transforms=transforms,
                                 param_f=lambda: param.image(64), 
                                 thresholds=(1, 32, 128, 256, 2048), verbose=False)


        # Note that we're doubling the image scale to make artifacts more obvious
        image_mde = show([nd.zoom(img[0], [2,2,1], order=0) for img in imgs])
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
