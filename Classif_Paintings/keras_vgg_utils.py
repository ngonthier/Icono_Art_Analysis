# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:21:36 2020

Some useful fct for using tf.keras VGG model

@author: gonthier
"""

    
def getVGG_trainable_vizualizable_layers_name():
    """
    It is VGG19 layers
    """
    liste = ['input_1',
     'block1_conv1',
     'block1_conv2',
     'block2_conv1',
     'block2_conv2',
     'block3_conv1',
     'block3_conv2',
     'block3_conv3',
     'block3_conv4',
     'block4_conv1',
     'block4_conv2',
     'block4_conv3',
     'block4_conv4',
     'block5_conv1',
     'block5_conv2',
     'block5_conv3',
     'block5_conv4',
     'fc1',
     'fc2',
     'predictions']
    return(liste)
