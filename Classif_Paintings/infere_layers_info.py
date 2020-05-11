# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:39:51 2020

The goal of this script is to infere the layer information such as the type of 
the layer

@author: gonthier
"""

from tensorflow.python.keras.layers import Input, Dense,Lambda, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation

import tensorflow as tf

from inception_v1 import InceptionV1_slim
from googlenet import inception_v1_oldTF

def get_dico_layers_type_all_layers(Net):
    if Net=='InceptionV1_slim':
        model = InceptionV1_slim(include_top=True, weights='imagenet')
    elif Net=='InceptionV1':
        model = inception_v1_oldTF(weights='imagenet',include_top=True)
    elif Net=='VGG':
        model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
    elif Net=='ResNet50':
        model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',\
                                                      input_shape= (224, 224, 3))
    else:
        raise(ValueError(Net+' is unknown'))
        
    dico = get_dico_layers_type_all_layers_fromNet(model)
    
    return(dico)
    
def get_dico_layers_type_all_layers_fromNet(net):
    dico = {}
    for layer in net.layers:
        if  isinstance(layer, Conv2D) :
            dico[layer.name] = 'Conv2D'
        elif isinstance(layer, ZeroPadding2D) :
            dico[layer.name] = 'Pad'
        elif isinstance(layer, Activation) :
            config=layer.get_config()
            activation = config['activation']
            if activation=='relu':     
                dico[layer.name] = 'Relu'
            elif activation=='softmax':     
                dico[layer.name] = 'Softmax'
            elif activation=='sigmoid':     
                dico[layer.name] = 'Sigmoid'
            else:     
                raise(NotImplementedError(activation+'is unknown'))
        elif isinstance(layer, MaxPooling2D) :
            dico[layer.name] = 'MaxPool'
        elif isinstance(layer, AveragePooling2D) :
            dico[layer.name] = 'AvgPool'
        elif isinstance(layer, Concatenate) :
            dico[layer.name] = 'concat'
        elif isinstance(layer, Dense) :
            config=layer.get_config()
            activation = config['activation']
            if activation=='relu':     
                dico[layer.name] = 'Relu'
            elif activation=='softmax':     
                dico[layer.name] = 'Softmax'
            elif activation=='sigmoid':     
                dico[layer.name] = 'Sigmoid'
            elif activation=='linear':     
                dico[layer.name] = 'MatMul'
            else:     
                raise(NotImplementedError(activation+'is unknown'))
            #dico[layer.name] = 'Softmax' # To hit the last activation !
#        elif isinstance(layer, Dropout) :
#            dico[layer.name] = 'dropout'
        elif isinstance(layer, Flatten) :
            dico[layer.name] = 'Reshape'
            
            
    return(dico)
        
