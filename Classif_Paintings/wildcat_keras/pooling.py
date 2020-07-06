# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:46:43 2020

Wildcat poolings repimplementation in tf.keras 
Based on https://github.com/durandtibo/wildcat.pytorch/blob/master/wildcat/pooling.py

@author: gonthier
"""

import sys
from tensorflow.python.keras.layers import Layer

import tensorflow as tf



class WildcatPool2d(Layer):
    def __init__(self, kmax=1, kmin=None, alpha=1, **kwargs):
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha
        super(WildcatPool2d, self).__init__(**kwargs)

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def build(self, input_shape):
        #batch_size = self.input_shape[0]
        self.num_channels = input_shape[3]
        self.h = input_shape[1]
        self.w = input_shape[2]
        self.n = self.h * self.w  # number of regions / pixels of the feature maps
        
    def call(self, input):

        kmax = self.get_positive_k(self.kmax, self.n)
        kmin = self.get_positive_k(self.kmin, self.n)
        
        # As topk Finds values and indices of the k largest entries for the last dimension.
        # We need to reshape the maps
        
        x = tf.reshape(input, shape=(-1,self.n,self.num_channels))
        x = tf.transpose(x,[0,2,1]) # To put the pixel in the last dimension
        topk_x,topk_x_indices = tf.math.top_k(x, k=kmax)
        output = tf.math.reduce_mean(topk_x,axis=-1)

        if kmin > 0 and (self.alpha != 0):
            mink_x,mink_x_indices = tf.math.top_k(tf.multiply(x,-1.0), k=kmin) # The kmin lowest values
            output = tf.add(output,tf.math.reduce_mean(tf.multiply(mink_x,-1.0),axis=-1))
        return output


    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_channels = input_shape[1]
        output_shape = (batch_size,num_channels)
        return(output_shape)
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(WildcatPool2d, self).get_config()
        config['kmax'] = self.kmax
        config['kmin'] = self.kmin
        config['alpha'] = self.alpha
        config['num_channels'] = self.num_channels
        config['h'] = self.h
        config['w'] = self.w
        config['n'] = self.n
        return(config) 



class ClassWisePool(Layer):
    """
    Class wise pooling 
    """
    def __init__(self, num_maps, **kwargs):
        self.num_maps = num_maps
        super(ClassWisePool, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #batch_size = self.input_shape[0]
        self.num_channels = input_shape[3]
        self.h = input_shape[1]
        self.w = input_shape[2]

        if self.num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        self.num_outputs = int(self.num_channels // self.num_maps)
        # IE must be the number of classes

    def call(self, input):

        x = tf.reshape(input, shape=(-1,self.h,self.w,self.num_maps,self.num_outputs))
        output_x = tf.math.reduce_mean(x,axis=-2)
        return output_x
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_channels = input_shape[1]
        h = input_shape[2]
        w = input_shape[3]
        num_outputs = int(num_channels / self.num_maps)
        output_shape = (batch_size,num_outputs,h,w)
        return(output_shape)
    
    def get_config(self): # Need this to save correctly the model with this kind of layer
        config = super(ClassWisePool, self).get_config()
        config['num_maps'] = self.num_maps
        #config['num_channels'] = self.num_channels
        config['num_outputs'] = self.num_outputs
        config['h'] = self.h
        config['w'] = self.w
        return(config) 

