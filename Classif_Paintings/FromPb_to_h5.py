# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:07:14 2020

@author: gonthier
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from inception_v1 import InceptionV1_slim
from tensorflow.python.keras.layers import Input, Dense,Lambda, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.python.keras import Model
from tensorflow.python.framework import tensor_util
from lucid.misc.io.loading import load
from googlenet import create_googlenet,inception_v1_oldTF
from googlenet import LRN_keras

#GRAPH_PB_PATH = 'gs://modelzoo/vision/caffe_models/InceptionV1.pb' #path to your .pb file 
GRAPH_PB_PATH = 'gs://modelzoo/vision/other_models/InceptionV1.pb' #path to your .pb file 


with tf.Session() as sess:
    if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
        raise(NotImplementedError)
    print("load graph")
    graph_def = load(GRAPH_PB_PATH)
#    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
    sess =tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    
    list_conv = []
    list_localresponsenorm = []
    for n in graph_nodes:
        if n.op =='Conv2D':
            list_conv += [n]
        print('name',n.name,'input',n.input)
        if 'localresponsenorm' in n.name:
            list_localresponsenorm +=[n]
    for conv in list_conv:
        print('name',conv.name,'input',conv.input)
        print(conv)
    
    wts = [n for n in graph_nodes if n.op=='Const']
    
    weight_dict = {}
    for i, n in enumerate(wts):
        weight_dict[n.name] = i
        
    model = inception_v1_oldTF(include_top=True)
    model.summary()
    
    for layer in model.layers:
        layer_weight = layer.get_weights()
        name = layer.name
        print('name layer :',name)
        if len(layer_weight) ==0:
            continue

        if  isinstance(layer, Conv2D) :
            print('instance of Conv2D')
            if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
                kname = name + '/weights'
                bname = name + '/biases'
            elif GRAPH_PB_PATH == 'gs://modelzoo/vision/other_models/InceptionV1.pb':
                name = name.replace('_pre_relu','')
                kname = name + '_w'
                bname = name + '_b'
            if kname not in weight_dict or bname not in weight_dict:
                print('Not in dict !!!!')
                print(kname, bname)
            else:
                weights = []
            idx = weight_dict[kname]
            wtensor = wts[idx].attr['value'].tensor
            weight = tensor_util.MakeNdarray(wtensor)
            weights.append(weight)
    
            idx = weight_dict[bname]
            wtensor = wts[idx].attr['value'].tensor
            weight = tensor_util.MakeNdarray(wtensor)
            weights.append(weight)
            layer.set_weights(weights)
            continue
       
        if isinstance(layer,LRN_keras):
            
            layer_lrn = None
            for pb_layer in list_localresponsenorm:
                if pb_layer.name == name:
                    layer_lrn = pb_layer
            if layer_lrn is None:
                print('Thats not normal !!!')
                raise(NotImplementedError)

            
            print('instance of LRN')

            weights = []

            wtensor =layer_lrn.attr['depth_radius'].tensor
            weights.append(wtensor)
            wtensor =layer_lrn.attr['bias'].tensor
            weights.append(wtensor)
            wtensor =layer_lrn.attr['alpha'].tensor
            weights.append(wtensor)
            wtensor =layer_lrn.attr['beta'].tensor
            weights.append(wtensor)
            layer.set_weights(weights)
            continue
        
        if isinstance(layer, Dense):
            print('instance of Dense')
            if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
                kname = name + '/weights'
                bname = name + '/biases'
            elif GRAPH_PB_PATH == 'gs://modelzoo/vision/other_models/InceptionV1.pb':
                name = name.replace('_pre_relu','')
                name = name.replace('_pre_activation','')
                kname = name + '_w'
                bname = name + '_b'
            if kname not in weight_dict or bname not in weight_dict:
                print('Not in dict !!!!')
                print(kname, bname)
            else:
                weights = []
                idx = weight_dict[kname]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                print('weight.shape',weight.shape)
                weights.append(weight)
        
                idx = weight_dict[bname]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                weights.append(weight)
                print('bias weight.shape',weight.shape)
                layer.set_weights(weights)
                continue
            
    if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
        model.save('model/InceptionV1_Caffe.h5')
    elif GRAPH_PB_PATH == 'gs://modelzoo/vision/other_models/InceptionV1.pb':
        model.save('model/InceptionV1_FromLucid.h5')
