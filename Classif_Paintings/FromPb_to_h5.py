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
from googlenet import LRN

#GRAPH_PB_PATH = 'gs://modelzoo/vision/caffe_models/InceptionV1.pb' #path to your .pb file 
GRAPH_PB_PATH = 'gs://modelzoo/vision/other_models/InceptionV1.pb' #path to your .pb file 


with tf.Session() as sess:
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
    for n in graph_nodes:
        if n.op =='Conv2D':
            list_conv += [n]
        print('name',n.name,'input',n.input)
    for conv in list_conv:
        print('name',conv.name,'input',conv.input)
    
    wts = [n for n in graph_nodes if n.op=='Const']
    
    weight_dict = {}
    for i, n in enumerate(wts):
        weight_dict[n.name] = i
        
    model = inception_v1_oldTF(include_top=True)
    model.summary()
    
    for layer in model.layers:
        layer_weight = layer.get_weights()
        name = layer.name
        print(name)
        if len(layer_weight) == 0:
            continue
        if  isinstance(layer, Conv2D) :
            print('Conv2D')
            if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
                kname = name + '/weights'
                bname = name + '/biases'
            elif GRAPH_PB_PATH == 'gs://modelzoo/vision/other_models/InceptionV1.pb':
                name = name.replace('_pre_relu','')
                kname = name + '_w'
                bname = name + '_b'
            if kname not in weight_dict or bname not in weight_dict:
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
       
        if isinstance(layer,LRN):
            print('LRN')
            beta_name = name + '/beta'
            gamma_name = name + '/gamma'
            mmean_name = name + '/n'
            mvar_name = name + '/k'
    
            if beta_name not in weight_dict or gamma_name not in weight_dict or\
                    mmean_name not in weight_dict or mvar_name not in weight_dict:
                print( beta_name, gamma_name, mmean_name, mvar_name)
            else:
                weights = []
                idx = weight_dict[gamma_name]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                weights.append(weight)
        
                idx = weight_dict[beta_name]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                weights.append(weight)
        
                idx = weight_dict[mmean_name]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                weights.append(weight)
        
                idx = weight_dict[mvar_name]
                wtensor = wts[idx].attr['value'].tensor
                weight = tensor_util.MakeNdarray(wtensor)
                weights.append(weight)
                layer.set_weights(weights)
                continue
        if isinstance(layer, Dense):
            print('Dense')
            kname = name + '/kernel'
            bname = name + '/bias'
            if kname not in weight_dict or bname not in weight_dict:
                print( kname, bname)
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
            
    if GRAPH_PB_PATH=='gs://modelzoo/vision/caffe_models/InceptionV1.pb':
        model.save('model/InceptionV1_Caffe.h5')
    elif GRAPH_PB_PATH == 'gs://modelzoo/vision/other_models/InceptionV1.pb':
        model.save('model/InceptionV1_FromLucid.h5')
