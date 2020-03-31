#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:56:36 2020

The goal of this script is to load a GoogLeNet (a.k.a Inception V1) Szegedy 2015
Converted from Caffe or Lucid pb

Code from https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14 
adapted to keras in tensorflow

Problem ! need image with channel first ! C'est pour cela que de nombre variantes
ont été faite par Nicolas, la plus utiles est inception_v1_oldTF mais 
nécessite les poids de Lucid : cf FromPb_to_h5.py pour convertir les poids en h5

@author: gonthier and  joelouismarino
"""

from __future__ import print_function
import imageio
from PIL import Image
import numpy as np

from tensorflow.python.keras.layers import Input, Dense,Lambda, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers import Layer


from tensorflow.python.keras import backend as K
import tensorflow as tf
from keras.utils.conv_utils import convert_kernel

class PoolHelper(Layer):
    """PoolHelper : remove the first row and column in image dimensions"""

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if K.image_data_format()=='channels_last':
            return x[:,1:,1:,:]
        elif K.image_data_format()=='channels_first':
            return x[:,:,1:,1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class LRN_keras(Lambda):
    """ wraps up the tf.nn.local_response_normalisation
     into the keras.layers.Lambda so
     we can have a custom keras layer as
     a class that will perform LRN ops  """
    def __init__(self,  depth_radius=5, bias=1, alpha=1., beta=0.75 , **kwargs):

        params = {
            "alpha": alpha,
            "beta": beta,
            "bias" :bias,
            "depth_radius": depth_radius
        }
        # construct a function for use with Keras Lambda
        lrn_fn = lambda inputs: tf.nn.local_response_normalization(inputs, **params)

        # pass the function to Keras Lambda
        return super().__init__(lrn_fn, **kwargs)

class LRN(Layer):
    
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if K.image_data_format()=='channels_last':
            b, r, c, ch = x.shape
            half_n = self.n // 2 # half the local region
            input_sqr = K.square(x) # square the input
            input_sqr = tf.pad(input_sqr, [[0, 0], [0, 0], [0, 0], [half_n, half_n]])
            scale = self.k # offset for the scale
            norm_alpha = self.alpha / self.n # normalized alpha
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, :, :,  i:i+ch]
        elif K.image_data_format()=='channels_first':
            b, ch, r, c = x.shape
            half_n = self.n // 2 # half the local region
            input_sqr = K.square(x) # square the input
            input_sqr = tf.pad(input_sqr, [[0, 0], [half_n, half_n], [0, 0], [0, 0]])
            scale = self.k # offset for the scale
            norm_alpha = self.alpha / self.n # normalized alpha
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def create_googlenet_channel_first(weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    input = Input(shape=(3, 224, 224))
    K.set_image_data_format('channels_first')

    input_pad = ZeroPadding2D(padding=(3, 3))(input)
    conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool2/3x3_s2')(pool2_helper)

    inception_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
    inception_3a_3x3 = Conv2D(128, (3,3), padding='valid', activation='relu', name='inception_3a/3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
    inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
    inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='inception_3a/5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
    inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a/pool')(pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = Concatenate(axis=1, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

    inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
    inception_3b_3x3 = Conv2D(192, (3,3), padding='valid', activation='relu', name='inception_3b/3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
    inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
    inception_3b_5x5 = Conv2D(96, (5,5), padding='valid', activation='relu', name='inception_3b/5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
    inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b/pool')(inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = Concatenate(axis=1, name='inception_3b/output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3/3x3_s2')(pool3_helper)

    inception_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_4a/1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
    inception_4a_3x3 = Conv2D(208, (3,3), padding='valid', activation='relu', name='inception_4a/3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
    inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
    inception_4a_5x5 = Conv2D(48, (5,5), padding='valid', activation='relu', name='inception_4a/5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
    inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a/pool')(pool3_3x3_s2)
    inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_output = Concatenate(axis=1, name='inception_4a/output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

    loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool')(inception_4a_output)
    loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
    loss1_flat = Flatten()(loss1_conv)
    loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
    loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
    loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation('softmax')(loss1_classifier)

    inception_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4b/1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
    inception_4b_3x3 = Conv2D(224, (3,3), padding='valid', activation='relu', name='inception_4b/3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
    inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
    inception_4b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4b/5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4b/pool')(inception_4a_output)
    inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = Concatenate(axis=1, name='inception_4b/output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

    inception_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
    inception_4c_3x3 = Conv2D(256, (3,3), padding='valid', activation='relu', name='inception_4c/3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
    inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
    inception_4c_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4c/5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4c/pool')(inception_4b_output)
    inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = Concatenate(axis=1, name='inception_4c/output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

    inception_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4d/1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
    inception_4d_3x3 = Conv2D(288, (3,3), padding='valid', activation='relu', name='inception_4d/3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
    inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
    inception_4d_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4d/5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4d/pool')(inception_4c_output)
    inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = Concatenate(axis=1, name='inception_4d/output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

    loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2/ave_pool')(inception_4d_output)
    loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2/conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
    loss2_flat = Flatten()(loss2_conv)
    loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)
    loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
    loss2_classifier = Dense(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation('softmax')(loss2_classifier)

    inception_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_4e/1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
    inception_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_4e/3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
    inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
    inception_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_4e/5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
    inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4e/pool')(inception_4d_output)
    inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_output = Concatenate(axis=1, name='inception_4e/output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4/3x3_s2')(pool4_helper)

    inception_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_5a/1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
    inception_5a_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_5a/3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
    inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
    inception_5a_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5a/5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
    inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5a/pool')(pool4_3x3_s2)
    inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
    inception_5a_output = Concatenate(axis=1, name='inception_5a/output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

    inception_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='inception_5b/1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
    inception_5b_3x3 = Conv2D(384, (3,3), padding='valid', activation='relu', name='inception_5b/3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
    inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
    inception_5b_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5b/5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
    inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5b/pool')(inception_5a_output)
    inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
    inception_5b_output = Concatenate(axis=1, name='inception_5b/output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5/7x7_s2')(inception_5b_output)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
    loss3_classifier = Dense(1000, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    googlenet = Model(inputs=input, outputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])

    if weights_path:
        googlenet.load_weights(weights_path)

    # convert the convolutional kernels for tensorflow
    ops = []
    for layer in googlenet.layers:
        if layer.__class__.__name__ == 'Conv2D':
            original_w = K.get_value(layer.kernel)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.kernel, converted_w).op)
    K.get_session().run(ops)

    return googlenet

def create_googlenet(weights=None,include_top=True):
    """
    Tentative de faire un Inception V1 avec les poids de Theano mais channels_last
    Cela ne marche pas : on n'a pas les meme sorties que le modèle au dessus ! 
    """
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    K.set_image_data_format('channels_last')
    img_input = Input(shape=(224, 224, 3))
    
    axis_concat = -1

    input_pad = ZeroPadding2D(padding=(3, 3))(img_input)
    conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='relu', name='conv1_7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1_3x3_s2')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2_3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2_3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool2_3x3_s2')(pool2_helper)

    inception_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3a_1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
    inception_3a_3x3 = Conv2D(128, (3,3), padding='valid', activation='relu', name='inception_3a_3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
    inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
    inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='inception_3a_5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
    inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a_pool')(pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a_pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = Concatenate(axis=axis_concat, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

    inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b_1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
    inception_3b_3x3 = Conv2D(192, (3,3), padding='valid', activation='relu', name='inception_3b_3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
    inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
    inception_3b_5x5 = Conv2D(96, (5,5), padding='valid', activation='relu', name='inception_3b_5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
    inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b_pool')(inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b_pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = Concatenate(axis=axis_concat, name='inception_3b_output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3_3x3_s2')(pool3_helper)

    inception_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_4a_1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
    inception_4a_3x3 = Conv2D(208, (3,3), padding='valid', activation='relu', name='inception_4a_3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
    inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
    inception_4a_5x5 = Conv2D(48, (5,5), padding='valid', activation='relu', name='inception_4a_5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
    inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a_pool')(pool3_3x3_s2)
    inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a_pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_output = Concatenate(axis=axis_concat, name='inception_4a/output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

    loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1_ave_pool')(inception_4a_output)
    loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1_conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
    
    
    if include_top:
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1_fc', kernel_regularizer=l2(0.0002))(loss1_flat)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_classifier = Dense(1000, name='loss1_classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

    inception_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4b_1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
    inception_4b_3x3 = Conv2D(224, (3,3), padding='valid', activation='relu', name='inception_4b_3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
    inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
    inception_4b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4b_5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4b_pool')(inception_4a_output)
    inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4b_pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = Concatenate(axis=axis_concat, name='inception_4b_output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

    inception_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c_1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
    inception_4c_3x3 = Conv2D(256, (3,3), padding='valid', activation='relu', name='inception_4c_3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
    inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4c_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
    inception_4c_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4c_5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4c_pool')(inception_4b_output)
    inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4c_pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = Concatenate(axis=axis_concat, name='inception_4c_output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

    inception_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4d_1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='inception_4d_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
    inception_4d_3x3 = Conv2D(288, (3,3), padding='valid', activation='relu', name='inception_4d_3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
    inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4d_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
    inception_4d_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4d_5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4d_pool')(inception_4c_output)
    inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4d_pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = Concatenate(axis=axis_concat, name='inception_4d_output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

    loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2_ave_pool')(inception_4d_output)
    loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2_conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
    
    
    if include_top:
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc = Dense(1024, activation='relu', name='loss2_fc', kernel_regularizer=l2(0.0002))(loss2_flat)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_classifier = Dense(1000, name='loss2_classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax')(loss2_classifier)

    inception_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_4e_1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4e_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
    inception_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_4e_3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
    inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4e_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
    inception_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_4e_5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
    inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4e_pool')(inception_4d_output)
    inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4e_pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_output = Concatenate(axis=axis_concat, name='inception_4e_output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4_3x3_s2')(pool4_helper)

    inception_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_5a_1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_5a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
    inception_5a_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_5a_3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
    inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_5a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
    inception_5a_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5a_5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
    inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5a_pool')(pool4_3x3_s2)
    inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5a_pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
    inception_5a_output = Concatenate(axis=axis_concat, name='inception_5a_output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

    inception_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='inception_5b_1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_5b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
    inception_5b_3x3 = Conv2D(384, (3,3), padding='valid', activation='relu', name='inception_5b_3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
    inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='inception_5b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
    inception_5b_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5b_5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
    inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5b_pool')(inception_5a_output)
    inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5b_pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
    inception_5b_output = Concatenate(axis=axis_concat, name='inception_5b_output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5_7x7_s2')(inception_5b_output)
    
    
    if include_top:
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
        loss3_classifier = Dense(1000, name='loss3_classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        googlenet = Model(inputs=img_input, outputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act], name='inception_v1')
   
    else: 
        googlenet = Model(inputs=img_input, outputs=[loss1_conv,loss2_conv,pool5_7x7_s1], name='inception_v1')

    if weights=='theano_imagenet':
        weights_path = 'model/googlenet_weights.h5'
        googlenet.load_weights(weights_path)
        
        # convert the convolutional kernels for tensorflow
        ops = []
        for layer in googlenet.layers:
            if layer.__class__.__name__ in ['Conv2D','Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
                original_w = K.get_value(layer.kernel)
                converted_w = convert_kernel(original_w) # Because we have correlation in TF and convolution in Theano
                converted_w = np.swapaxes(converted_w,3,2)
                converted_w = np.rot90(converted_w,k=-1, axes=(0,1)) # The first two dimensions are rotated; therefore, the array must be at least 2-D.
                ops.append(tf.compat.v1.assign(layer.kernel, converted_w).op)
        K.get_session().run(ops)
        
        import warnings
        warnings.simplefilter("The way to load the filter is not good at all for the moment sorry.")

    elif weights=='imagenet':
        if include_top:
            raise(NotImplementedError)
        else:
            weights_path = 'model/InceptionV1_fromLucid.h5'
            googlenet.load_weights(weights_path)

    return googlenet

def inception_v1_oldTF(weights='imagenet',include_top=True,input_shape= (224, 224, 3)):
    """
    Inception V1 avec les poids qui viennent de Lucid  ! 
    https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py
    
    Parameters
    ----------
    weights : string
        one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet). default is imagenet 
    include_top : bool
        whether to include the fully-connected layer at the top of the network.
        If False return model with last layers equal to head0_bottleneck
        head1_bottleneck and avgpool
    
    """
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    
    if not(input_shape==(224, 224, 3)) and include_top:
        raise ValueError('If using `include_top` as true, `input_shape` should be (224, 224, 3) due to the fully connected layers.')
    
    kernel_regularizer = l2(0.0002)
    K.set_image_data_format('channels_last')
    img_input = Input(shape=(224, 224, 3),name='input_1')
    
    axis_concat = -1
    output_classes_num = 1008
    
    # Ces valeurs ont été lu dans le graph pb de Lucid
    beta = 0.5
    alpha = 1e-4
    bias = 2
    depth_radius = 5
    input_pad = ZeroPadding2D(padding=(3, 3))(img_input)
    conv1_7x7_s2_pre_relu = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='linear', name='conv2d0_pre_relu', kernel_regularizer=kernel_regularizer)(input_pad)
    conv1_7x7_s2 = Activation('relu',name='conv2d0')(conv1_7x7_s2_pre_relu)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool0')(pool1_helper)
    pool1_norm1 = LRN(beta=beta,alpha=alpha,n=depth_radius,k=bias,name='localresponsenorm0')(pool1_3x3_s2)

    conv2_3x3_reduce_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='conv2d1_pre_relu', kernel_regularizer=kernel_regularizer)(pool1_norm1)
    conv2_3x3_reduce = Activation('relu',name='conv2d1')(conv2_3x3_reduce_pre_relu)
    conv2_3x3_pre_relu = Conv2D(192, (3,3), padding='same', activation='linear', name='conv2d2_pre_relu', kernel_regularizer=kernel_regularizer)(conv2_3x3_reduce)
    conv2_3x3 = Activation('relu',name='conv2d2')(conv2_3x3_pre_relu)
    conv2_norm2 = LRN(beta=beta,alpha=alpha,n=depth_radius,k=bias,name='localresponsenorm1')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool1')(pool2_helper)

    inception_3a_1x1_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed3a_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(pool2_3x3_s2)
    #inception_3a_1x1 = Activation('relu',name='mixed3a_1x1')(inception_3a_1x1_pre_relu)
    inception_3a_3x3_reduce_pre_relu = Conv2D(96, (1,1), padding='same', activation='linear', name='mixed3a_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool2_3x3_s2)
    inception_3a_3x3_reduce = Activation('relu',name='mixed3a_3x3_bottleneck')(inception_3a_3x3_reduce_pre_relu)
    inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
    inception_3a_3x3_pre_relu = Conv2D(128, (3,3), padding='valid', activation='linear', name='mixed3a_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_3x3_pad)
    #inception_3a_3x3 = Activation('relu',name='mixed3a_3x3')(inception_3a_3x3_pre_relu)
    inception_3a_5x5_reduce_pre_relu = Conv2D(16, (1,1), padding='same', activation='linear', name='mixed3a_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool2_3x3_s2)
    inception_3a_5x5_reduce = Activation('relu',name='mixed3a_5x5_bottleneck')(inception_3a_5x5_reduce_pre_relu)
    inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
    inception_3a_5x5_pre_relu = Conv2D(32, (5,5), padding='valid', activation='linear', name='mixed3a_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_5x5_pad)
    #inception_3a_5x5 =Activation('relu',name='mixed3a_5x5')(inception_3a_5x5_pre_relu)
    inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed3a_pool_pre_relu')(pool2_3x3_s2)
    inception_3a_pool_proj_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed3a_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_pool)
    #inception_3a_pool_proj = Activation('relu',name='mixed3a_pool_reduce')(inception_3a_pool_proj_pre_relu)
    inception_3a_output_pre_relu = Concatenate(axis=axis_concat, name='mixed3a_pre_relu')([inception_3a_1x1_pre_relu,inception_3a_3x3_pre_relu,inception_3a_5x5_pre_relu,inception_3a_pool_proj_pre_relu])
    inception_3a_output = Activation('relu', name='mixed3a')(inception_3a_output_pre_relu)

    inception_3b_1x1_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed3b_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_output)
    #inception_3b_1x1 = Activation('relu',name='mixed3b_1x1')(inception_3b_1x1_pre_relu)
    inception_3b_3x3_reduce_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed3b_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_output)
    inception_3b_3x3_reduce = Activation('relu',name='mixed3b_3x3_bottleneck')(inception_3b_3x3_reduce_pre_relu)
    inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
    inception_3b_3x3_pre_relu = Conv2D(192, (3,3), padding='valid', activation='linear', name='mixed3b_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3b_3x3_pad)
    #inception_3b_3x3 = Activation('relu',name='mixed3b_3x3')(inception_3b_3x3_pre_relu)
    inception_3b_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed3b_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_output)
    inception_3b_5x5_reduce = Activation('relu',name='mixed3b_5x5_bottleneck')(inception_3b_5x5_reduce_pre_relu)
    inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
    inception_3b_5x5_pre_relu = Conv2D(96, (5,5), padding='valid', activation='linear', name='mixed3b_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3b_5x5_pad)
    #inception_3b_5x5 = Activation('relu',name='mixed3b_5x5')(inception_3b_5x5_pre_relu)
    inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed3b_pool')(inception_3a_output)
    inception_3b_pool_proj_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed3b_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3b_pool)
    #inception_3b_pool_proj = Activation('relu',name='mixed3b_pool_reduce')(inception_3b_pool_proj_pre_relu)
    inception_3b_output_pre_relu = Concatenate(axis=axis_concat, name='mixed3b_pre_relu')([inception_3b_1x1_pre_relu,inception_3b_3x3_pre_relu,inception_3b_5x5_pre_relu,inception_3b_pool_proj_pre_relu])
    inception_3b_output = Activation('relu', name='mixed3b')(inception_3b_output_pre_relu)

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool2')(pool3_helper)

    inception_4a_1x1_pre_relu  = Conv2D(192, (1,1), padding='same', activation='linear', name='mixed4a_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(pool3_3x3_s2)
    #inception_4a_1x1 = Activation('relu',name='mixed4a_1x1')(inception_4a_1x1_pre_relu)
    inception_4a_3x3_reduce_pre_relu  = Conv2D(96, (1,1), padding='same', activation='linear', name='mixed4a_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool3_3x3_s2)
    inception_4a_3x3_reduce = Activation('relu',name='mixed4a_3x3_bottleneck')(inception_4a_3x3_reduce_pre_relu)
    inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
    inception_4a_3x3_pre_relu  = Conv2D(204, (3,3), padding='valid', activation='linear', name='mixed4a_3x3_pre_relu' ,kernel_regularizer=kernel_regularizer)(inception_4a_3x3_pad)
    #inception_4a_3x3 = Activation('relu',name='mixed4a_3x3')(inception_4a_3x3_pre_relu)
    inception_4a_5x5_reduce_pre_relu  = Conv2D(16, (1,1), padding='same', activation='linear', name='mixed4a_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool3_3x3_s2)
    inception_4a_5x5_reduce = Activation('relu',name='mixed4a_5x5_bottleneck')(inception_4a_5x5_reduce_pre_relu)
    inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
    inception_4a_5x5_pre_relu  = Conv2D(48, (5,5), padding='valid', activation='linear', name='mixed4a_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4a_5x5_pad)
    #inception_4a_5x5 = Activation('relu',name='mixed4a_5x5')(inception_4a_5x5_pre_relu)
    inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4a_pool')(pool3_3x3_s2)
    inception_4a_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4a_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4a_pool)
    #inception_4a_pool_proj = Activation('relu',name='mixed4a_pool_reduce')(inception_4a_pool_proj_pre_relu)
    inception_4a_output_pre_relu = Concatenate(axis=axis_concat, name='mixed4a_pre_relu')([inception_4a_1x1_pre_relu,inception_4a_3x3_pre_relu,inception_4a_5x5_pre_relu,inception_4a_pool_proj_pre_relu])
    inception_4a_output =  Activation('relu', name='mixed4a')(inception_4a_output_pre_relu)

    loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='head0_pool')(inception_4a_output)
    loss1_conv_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='head0_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(loss1_ave_pool)
    loss1_conv = Activation('relu',name='head0_bottleneck')(loss1_conv_pre_relu)
    
    if include_top:
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc_pre_relu = Dense(1024, activation='linear', name='nn0_pre_relu', kernel_regularizer=kernel_regularizer)(loss1_flat)
        loss1_fc = Activation('relu',name='nn0')(loss1_fc_pre_relu)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_classifier = Dense(output_classes_num, name='softmax0_pre_activation', kernel_regularizer=kernel_regularizer, activation='linear')(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax',name='softname0')(loss1_classifier)

    inception_4b_1x1_pre_relu  = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed4b_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4a_output)
    #inception_4b_1x1 = Activation('relu',name='mixed4b_1x1')(inception_4b_1x1_pre_relu)
    inception_4b_3x3_reduce_pre_relu  = Conv2D(112, (1,1), padding='same', activation='linear', name='mixed4b_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4a_output)
    inception_4b_3x3_reduce = Activation('relu',name='mixed4b_3x3_bottleneck')(inception_4b_3x3_reduce_pre_relu)
    inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
    inception_4b_3x3_pre_relu  = Conv2D(224, (3,3), padding='valid', activation='linear', name='mixed4b_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_3x3_pad)
    #inception_4b_3x3 = Activation('relu',name='mixed4b_3x3')(inception_4b_3x3_pre_relu)
    inception_4b_5x5_reduce_pre_relu = Conv2D(24, (1,1), padding='same', activation='linear', name='mixed4b_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4a_output)
    inception_4b_5x5_reduce = Activation('relu',name='mixed4b_5x5_bottleneck')(inception_4b_5x5_reduce_pre_relu)
    inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
    inception_4b_5x5_pre_relu  = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4b_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_5x5_pad)
    #inception_4b_5x5 = Activation('relu',name='mixed4b_5x5')(inception_4b_5x5_pre_relu)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4b_pool')(inception_4a_output)
    inception_4b_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4b_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_pool)
    #inception_4b_pool_proj = Activation('relu',name='mixed4b_pool_reduce')(inception_4b_pool_proj_pre_relu)
    inception_4b_output_pre_relu = Concatenate(axis=axis_concat, name='mixed4b_pre_relu')([inception_4b_1x1_pre_relu,inception_4b_3x3_pre_relu,inception_4b_5x5_pre_relu,inception_4b_pool_proj_pre_relu])
    inception_4b_output = Activation('relu', name='mixed4b')(inception_4b_output_pre_relu)

    inception_4c_1x1_pre_relu  = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4c_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_output)
    #inception_4c_1x1 = Activation('relu',name='mixed4c_1x1')(inception_4c_1x1_pre_relu)
    inception_4c_3x3_reduce_pre_relu  = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4c_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_output)
    inception_4c_3x3_reduce = Activation('relu',name='mixed4c_3x3_bottleneck')(inception_4c_3x3_reduce_pre_relu)
    inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
    inception_4c_3x3_pre_relu  = Conv2D(256, (3,3), padding='valid', activation='linear', name='mixed4c_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_3x3_pad)
    #inception_4c_3x3 = Activation('relu',name='mixed4c_3x3')(inception_4c_3x3_pre_relu)
    inception_4c_5x5_reduce_pre_relu  = Conv2D(24, (1,1), padding='same', activation='linear', name='mixed4c_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4b_output)
    inception_4c_5x5_reduce = Activation('relu',name='mixed4c_5x5_bottleneck')(inception_4c_5x5_reduce_pre_relu)
    inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
    inception_4c_5x5_pre_relu  = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4c_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_5x5_pad)
    #inception_4c_5x5 = Activation('relu',name='mixed4c_5x5')(inception_4c_5x5_pre_relu)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4c_pool')(inception_4b_output)
    inception_4c_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4c_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_pool)
    #inception_4c_pool_proj = Activation('relu',name='mixed4c_pool_reduce')(inception_4c_pool_proj_pre_relu)
    inception_4c_output_pre_relu = Concatenate(axis=axis_concat, name='mixed4c_pre_relu')([inception_4c_1x1_pre_relu,inception_4c_3x3_pre_relu,inception_4c_5x5_pre_relu,inception_4c_pool_proj_pre_relu])
    inception_4c_output = Activation('relu',  name='mixed4c')(inception_4c_output_pre_relu)

    inception_4d_1x1_pre_relu = Conv2D(112, (1,1), padding='same', activation='linear', name='mixed4d_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_output)
    #inception_4d_1x1 =  Activation('relu',name='mixed4d_1x1')(inception_4d_1x1_pre_relu)
    inception_4d_3x3_reduce_pre_relu = Conv2D(144, (1,1), padding='same', activation='linear', name='mixed4d_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_output)
    inception_4d_3x3_reduce =  Activation('relu',name='mixed4d_3x3_bottleneck')(inception_4d_3x3_reduce_pre_relu)
    inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
    inception_4d_3x3_pre_relu = Conv2D(288, (3,3), padding='valid', activation='linear', name='mixed4d_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_3x3_pad)
    #inception_4d_3x3 =  Activation('relu',name='mixed4d_3x3')(inception_4d_3x3_pre_relu)
    inception_4d_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed4d_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4c_output)
    inception_4d_5x5_reduce =  Activation('relu',name='mixed4d_5x5_bottleneck')(inception_4d_5x5_reduce_pre_relu)
    inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
    inception_4d_5x5_pre_relu = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4d_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_5x5_pad)
    #inception_4d_5x5 =  Activation('relu',name='mixed4d_5x5')(inception_4d_5x5_pre_relu)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4d_pool')(inception_4c_output)
    inception_4d_pool_proj_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4d_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_pool)
    #inception_4d_pool_proj = Activation('relu',name='mixed4d_pool_reduce')(inception_4d_pool_proj_pre_relu)
    inception_4d_output_pre_relu = Concatenate(axis=axis_concat, name='mixed4d_pre_relu')([inception_4d_1x1_pre_relu,inception_4d_3x3_pre_relu,inception_4d_5x5_pre_relu,inception_4d_pool_proj_pre_relu])
    inception_4d_output = Activation('relu',  name='mixed4d')(inception_4d_output_pre_relu)

    loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='head1_pool')(inception_4d_output)
    loss2_conv_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='head1_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(loss2_ave_pool)
    loss2_conv = Activation('relu',name='head1_bottleneck')(loss2_conv_pre_relu)
    
    if include_top:
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc_pre_relu = Dense(1024, activation='linear', name='nn1_pre_relu', kernel_regularizer=kernel_regularizer)(loss2_flat)
        loss2_fc = Activation('relu',name='nn1')(loss2_fc_pre_relu)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_classifier = Dense(output_classes_num, name='softmax1_pre_activation', kernel_regularizer=kernel_regularizer,activation='linear')(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax',name='softname1')(loss2_classifier)

    inception_4e_1x1_pre_relu = Conv2D(256, (1,1), padding='same', activation='linear', name='mixed4e_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_output)
    #inception_4e_1x1 = Activation('relu',name='mixed4e_1x1')(inception_4e_1x1_pre_relu)
    inception_4e_3x3_reduce_pre_relu = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed4e_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_output)
    inception_4e_3x3_reduce = Activation('relu',name='mixed4e_3x3_bottleneck')(inception_4e_3x3_reduce_pre_relu)
    inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
    inception_4e_3x3_pre_relu = Conv2D(320, (3,3), padding='valid', activation='linear', name='mixed4e_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4e_3x3_pad)
    #inception_4e_3x3 =Activation('relu',name='mixed4e_3x3')(inception_4e_3x3_pre_relu)
    inception_4e_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed4e_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4d_output)
    inception_4e_5x5_reduce =Activation('relu',name='mixed4e_5x5_bottleneck')(inception_4e_5x5_reduce_pre_relu)
    inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
    inception_4e_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed4e_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4e_5x5_pad)
    #inception_4e_5x5 =Activation('relu',name='mixed4e_5x5')(inception_4e_5x5_pre_relu)
    inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4e_pool')(inception_4d_output)
    inception_4e_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4e_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_4e_pool)
    #inception_4e_pool_proj = Activation('relu',name='mixed4e_pool_reduce')(inception_4e_pool_proj_pre_relu)
    inception_4e_output_pre_relu = Concatenate(axis=axis_concat, name='mixed4e_pre_relu')([inception_4e_1x1_pre_relu,inception_4e_3x3_pre_relu,inception_4e_5x5_pre_relu,inception_4e_pool_proj_pre_relu])
    inception_4e_output = Activation('relu', name='mixed4e')(inception_4e_output_pre_relu)

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool3')(pool4_helper)

    inception_5a_1x1_pre_relu = Conv2D(256, (1,1), padding='same', activation='linear', name='mixed5a_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(pool4_3x3_s2)
    #inception_5a_1x1 = Activation('relu',name='mixed5a_1x1')(inception_5a_1x1_pre_relu)
    inception_5a_3x3_reduce_pre_relu = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed5a_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool4_3x3_s2)
    inception_5a_3x3_reduce = Activation('relu',name='mixed5a_3x3_bottleneck')(inception_5a_3x3_reduce_pre_relu)
    inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
    inception_5a_3x3_pre_relu = Conv2D(320, (3,3), padding='valid', activation='linear', name='mixed5a_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_3x3_pad)
    #inception_5a_3x3 = Activation('relu',name='mixed5a_3x3')(inception_5a_3x3_pre_relu)
    inception_5a_5x5_reduce_pre_relu = Conv2D(48, (1,1), padding='same', activation='linear', name='mixed5a_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(pool4_3x3_s2)
    inception_5a_5x5_reduce = Activation('relu',name='mixed5a_5x5_bottleneck')(inception_5a_5x5_reduce_pre_relu)
    inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
    inception_5a_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed5a_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_5x5_pad)
    #inception_5a_5x5 = Activation('relu',name='mixed5a_5x5')(inception_5a_5x5_pre_relu)
    inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed5a_pool')(pool4_3x3_s2)
    inception_5a_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed5a_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_pool)
    #inception_5a_pool_proj = Activation('relu',name='mixed5a_pool_reduce')(inception_5a_pool_proj_pre_relu)
    inception_5a_output_pre_relu = Concatenate(axis=axis_concat, name='mixed5a_pre_relu')([inception_5a_1x1_pre_relu,inception_5a_3x3_pre_relu,inception_5a_5x5_pre_relu,inception_5a_pool_proj_pre_relu])
    inception_5a_output = Activation('relu', name='mixed5a')(inception_5a_output_pre_relu)

    inception_5b_1x1_pre_relu = Conv2D(384, (1,1), padding='same', activation='linear', name='mixed5b_1x1_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_output)
    #inception_5b_1x1 = Activation('relu',name='mixed5b_1x1')(inception_5b_1x1_pre_relu)
    inception_5b_3x3_reduce_pre_relu = Conv2D(192, (1,1), padding='same', activation='linear', name='mixed5b_3x3_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_output)
    inception_5b_3x3_reduce = Activation('relu',name='mixed5b_3x3_bottleneck')(inception_5b_3x3_reduce_pre_relu)
    inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
    inception_5b_3x3_pre_relu = Conv2D(384, (3,3), padding='valid', activation='linear', name='mixed5b_3x3_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5b_3x3_pad)
    #inception_5b_3x3 = Activation('relu',name='mixed5b_3x3')(inception_5b_3x3_pre_relu)
    inception_5b_5x5_reduce_pre_relu = Conv2D(48, (1,1), padding='same', activation='linear', name='mixed5b_5x5_bottleneck_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5a_output)
    inception_5b_5x5_reduce = Activation('relu',name='mixed5b_5x5_bottleneck')(inception_5b_5x5_reduce_pre_relu)
    inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
    inception_5b_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed5b_5x5_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5b_5x5_pad)
    #inception_5b_5x5 = Activation('relu',name='mixed5b_5x5')(inception_5b_5x5_pre_relu)
    inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed5b_pool')(inception_5a_output)
    inception_5b_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed5b_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_5b_pool)
    #inception_5b_pool_proj = Activation('relu',name='mixed5b_pool_reduce')(inception_5b_pool_proj_pre_relu)
    inception_5b_output_pre_relu = Concatenate(axis=axis_concat, name='mixed5b_pre_relu')([inception_5b_1x1_pre_relu,inception_5b_3x3_pre_relu,inception_5b_5x5_pre_relu,inception_5b_pool_proj_pre_relu])
    inception_5b_output = Activation('relu', name='mixed5b')(inception_5b_output_pre_relu)

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='avgpool')(inception_5b_output)
    
    if include_top:
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
        loss3_classifier = Dense(output_classes_num, name='softmax2_pre_activation', kernel_regularizer=kernel_regularizer,activation='linear')(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='softmax2')(loss3_classifier)

        googlenet = Model(inputs=img_input, outputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act], name='inception_v1')
   
    else: 
        googlenet = Model(inputs=img_input, outputs=[loss1_conv,loss2_conv,pool5_7x7_s1], name='inception_v1')

    if weights=='imagenet':
        if not(include_top):
            weights_path = 'model/InceptionV1_fromLucid_without_head.h5'
        else:
            weights_path = 'model/InceptionV1_fromLucid.h5'
        googlenet.load_weights(weights_path)

    return googlenet

def inception_v1_oldTF_all_relu(weights=None,include_top=True):
    """
    Tentative de faire un Inception V1 avec les poids de Theano mais channels_last
    Cela ne marche pas : on n'a pas les meme sorties que le modèle au dessus ! 
    """
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    
    kernel_regularizer = l2(0.0002)
    K.set_image_data_format('channels_last')
    img_input = Input(shape=(224, 224, 3),name='input_1')
    
    output_classes_num = 1008
    axis_concat = -1

    input_pad = ZeroPadding2D(padding=(3, 3))(img_input)
    conv1_7x7_s2_pre_relu = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='linear', name='conv2d0_pre_relu', kernel_regularizer=l2(0.0002))(input_pad)
    conv1_7x7_s2 = Activation('relu',name='conv2d0')(conv1_7x7_s2_pre_relu)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool0')(pool1_helper)
    pool1_norm1 = LRN(name='localresponsenorm0')(pool1_3x3_s2)

    conv2_3x3_reduce_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='conv2d1_pre_relu', kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3_reduce = Activation('relu',name='conv2d1')(conv2_3x3_reduce_pre_relu)
    conv2_3x3_pre_relu = Conv2D(192, (3,3), padding='same', activation='linear', name='conv2d2_pre_relu', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_3x3 = Activation('relu',name='conv2d2')(conv2_3x3_pre_relu)
    conv2_norm2 = LRN(name='localresponsenorm1')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool1')(pool2_helper)

    inception_3a_1x1_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed3a_1x1_pre_relu', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_1x1 = Activation('relu',name='mixed3a_1x1')(inception_3a_1x1_pre_relu)
    inception_3a_3x3_reduce_pre_relu = Conv2D(96, (1,1), padding='same', activation='linear', name='mixed3a_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Activation('relu',name='mixed3a_3x3_bottleneck')(inception_3a_3x3_reduce_pre_relu)
    inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
    inception_3a_3x3_pre_relu = Conv2D(128, (3,3), padding='valid', activation='linear', name='mixed3a_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
    inception_3a_3x3 = Activation('relu',name='mixed3a_3x3')(inception_3a_3x3_pre_relu)
    inception_3a_5x5_reduce_pre_relu = Conv2D(16, (1,1), padding='same', activation='linear', name='mixed3a_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5_reduce = Activation('relu',name='mixed3a_5x5_bottleneck')(inception_3a_5x5_reduce_pre_relu)
    inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
    inception_3a_5x5_pre_relu = Conv2D(32, (5,5), padding='valid', activation='linear', name='mixed3a_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
    inception_3a_5x5 =Activation('relu',name='mixed3a_5x5')(inception_3a_5x5_pre_relu)
    inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed3a_poo')(pool2_3x3_s2)
    inception_3a_pool_proj_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed3a_pool_reduce_pre_relu', kernel_regularizer=kernel_regularizer)(inception_3a_pool)
    inception_3a_pool_proj = Activation('relu',name='mixed3a_pool_reduce')(inception_3a_pool_proj_pre_relu)
    inception_3a_output = Concatenate(axis=axis_concat, name='mixed3a')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

    inception_3b_1x1_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed3b_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_1x1 = Activation('relu',name='mixed3b_1x1')(inception_3b_1x1_pre_relu)
    inception_3b_3x3_reduce_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed3b_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Activation('relu',name='mixed3b_3x3_bottleneck')(inception_3b_3x3_reduce_pre_relu)
    inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
    inception_3b_3x3_pre_relu = Conv2D(192, (3,3), padding='valid', activation='linear', name='mixed3b_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
    inception_3b_3x3 = Activation('relu',name='mixed3b_3x3')(inception_3b_3x3_pre_relu)
    inception_3b_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed3b_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5_reduce = Activation('relu',name='mixed3b_5x5_bottleneck')(inception_3b_5x5_reduce_pre_relu)
    inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
    inception_3b_5x5_pre_relu = Conv2D(96, (5,5), padding='valid', activation='linear', name='mixed3b_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
    inception_3b_5x5 = Activation('relu',name='mixed3b_5x5')(inception_3b_5x5_pre_relu)
    inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed3b_pool')(inception_3a_output)
    inception_3b_pool_proj_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed3b_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_pool_proj = Activation('relu',name='mixed3b_pool_reduce')(inception_3b_pool_proj_pre_relu)
    inception_3b_output = Concatenate(axis=axis_concat, name='mixed3b')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool2')(pool3_helper)

    inception_4a_1x1_pre_relu  = Conv2D(192, (1,1), padding='same', activation='linear', name='mixed4a_1x1_pre_relu', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_1x1 = Activation('relu',name='mixed4a_1x1')(inception_4a_1x1_pre_relu)
    inception_4a_3x3_reduce_pre_relu  = Conv2D(96, (1,1), padding='same', activation='linear', name='mixed4a_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Activation('relu',name='mixed4a_3x3_bottleneck')(inception_4a_3x3_reduce_pre_relu)
    inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
    inception_4a_3x3_pre_relu  = Conv2D(204, (3,3), padding='valid', activation='linear', name='mixed4a_3x3_pre_relu' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
    inception_4a_3x3 = Activation('relu',name='mixed4a_3x3')(inception_4a_3x3_pre_relu)
    inception_4a_5x5_reduce_pre_relu  = Conv2D(16, (1,1), padding='same', activation='linear', name='mixed4a_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5_reduce = Activation('relu',name='mixed4a_5x5_bottleneck')(inception_4a_5x5_reduce_pre_relu)
    inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
    inception_4a_5x5_pre_relu  = Conv2D(48, (5,5), padding='valid', activation='linear', name='mixed4a_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
    inception_4a_5x5 = Activation('relu',name='mixed4a_5x5')(inception_4a_5x5_pre_relu)
    inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4a_pool')(pool3_3x3_s2)
    inception_4a_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4a_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_pool_proj = Activation('relu',name='mixed4a_pool_reduce')(inception_4a_pool_proj_pre_relu)
    inception_4a_output = Concatenate(axis=axis_concat, name='mixed4a')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

    loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='head0_pool')(inception_4a_output)

    if include_top:
        
        loss1_conv_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='head0_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_conv = Activation('relu',name='head0_bottleneck')(loss1_conv_pre_relu)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc_pre_relu = Dense(1024, activation='linear', name='nn0_pre_relu', kernel_regularizer=l2(0.0002))(loss1_flat)
        loss1_fc = Activation('relu',name='nn0')(loss1_fc_pre_relu)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_classifier = Dense(output_classes_num, name='softmax0_pre_activation', kernel_regularizer=l2(0.0002), activation='linear')(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax',name='softname0')(loss1_classifier)

    inception_4b_1x1_pre_relu  = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed4b_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_1x1 = Activation('relu',name='mixed4b_1x1')(inception_4b_1x1_pre_relu)
    inception_4b_3x3_reduce_pre_relu  = Conv2D(112, (1,1), padding='same', activation='linear', name='mixed4b_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Activation('relu',name='mixed4b_3x3_bottleneck')(inception_4b_3x3_reduce_pre_relu)
    inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
    inception_4b_3x3_pre_relu  = Conv2D(224, (3,3), padding='valid', activation='linear', name='mixed4b_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
    inception_4b_3x3 = Activation('relu',name='mixed4b_3x3')(inception_4b_3x3_pre_relu)
    inception_4b_5x5_reduce_pre_relu = Conv2D(24, (1,1), padding='same', activation='linear', name='mixed4b_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5_reduce = Activation('relu',name='mixed4b_5x5_bottleneck')(inception_4b_5x5_reduce_pre_relu)
    inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
    inception_4b_5x5_pre_relu  = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4b_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
    inception_4b_5x5 = Activation('relu',name='mixed4b_5x5')(inception_4b_5x5_pre_relu)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4b_pool')(inception_4a_output)
    inception_4b_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4b_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_pool_proj = Activation('relu',name='mixed4b_pool_reduce')(inception_4b_pool_proj_pre_relu)
    inception_4b_output = Concatenate(axis=axis_concat, name='mixed4b')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

    inception_4c_1x1_pre_relu  = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4c_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_1x1 = Activation('relu',name='mixed4c_1x1')(inception_4c_1x1_pre_relu)
    inception_4c_3x3_reduce_pre_relu  = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4c_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Activation('relu',name='mixed4c_3x3_bottleneck')(inception_4c_3x3_reduce_pre_relu)
    inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
    inception_4c_3x3_pre_relu  = Conv2D(256, (3,3), padding='valid', activation='linear', name='mixed4c_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
    inception_4c_3x3 = Activation('relu',name='mixed4c_3x3')(inception_4c_3x3_pre_relu)
    inception_4c_5x5_reduce_pre_relu  = Conv2D(24, (1,1), padding='same', activation='linear', name='mixed4c_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5_reduce = Activation('relu',name='mixed4c_5x5_bottleneck')(inception_4c_5x5_reduce_pre_relu)
    inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
    inception_4c_5x5_pre_relu  = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4c_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
    inception_4c_5x5 = Activation('relu',name='mixed4c_5x5')(inception_4c_5x5_pre_relu)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4c_pool')(inception_4b_output)
    inception_4c_pool_proj_pre_relu  = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4c_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_pool_proj = Activation('relu',name='mixed4c_pool_reduce')(inception_4c_pool_proj_pre_relu)
    inception_4c_output = Concatenate(axis=axis_concat, name='mixed4c')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

    inception_4d_1x1_pre_relu = Conv2D(112, (1,1), padding='same', activation='linear', name='mixed4d_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_1x1 =  Activation('relu',name='mixed4d_1x1')(inception_4d_1x1_pre_relu)
    inception_4d_3x3_reduce_pre_relu = Conv2D(144, (1,1), padding='same', activation='linear', name='mixed4d_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce =  Activation('relu',name='mixed4d_3x3_bottleneck')(inception_4d_3x3_reduce_pre_relu)
    inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
    inception_4d_3x3_pre_relu = Conv2D(288, (3,3), padding='valid', activation='linear', name='mixed4d_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
    inception_4d_3x3 =  Activation('relu',name='mixed4d_3x3')(inception_4d_3x3_pre_relu)
    inception_4d_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed4d_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5_reduce =  Activation('relu',name='mixed4d_5x5_bottleneck')(inception_4d_5x5_reduce_pre_relu)
    inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
    inception_4d_5x5_pre_relu = Conv2D(64, (5,5), padding='valid', activation='linear', name='mixed4d_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
    inception_4d_5x5 =  Activation('relu',name='mixed4d_5x5')(inception_4d_5x5_pre_relu)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4d_pool')(inception_4c_output)
    inception_4d_pool_proj_pre_relu = Conv2D(64, (1,1), padding='same', activation='linear', name='mixed4d_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_pool_proj = Activation('relu',name='mixed4d_pool_reduce')(inception_4d_pool_proj_pre_relu)
    inception_4d_output = Concatenate(axis=axis_concat, name='mixed4d')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

    loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='head1_pool')(inception_4d_output)
    
    if include_top:
        loss2_conv_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='head1_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
        loss2_conv = Activation('relu',name='head1_bottleneck')(loss2_conv_pre_relu)
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc_pre_relu = Dense(1024, activation='linear', name='nn1_pre_relu', kernel_regularizer=l2(0.0002))(loss2_flat)
        loss2_fc = Activation('relu',name='nn1')(loss2_fc_pre_relu)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_classifier = Dense(output_classes_num, name='softmax1_pre_activation', kernel_regularizer=l2(0.0002),activation='linear')(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax',name='softname1')(loss2_classifier)

    inception_4e_1x1_pre_relu = Conv2D(256, (1,1), padding='same', activation='linear', name='mixed4e_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_1x1 = Activation('relu',name='mixed4e_1x1')(inception_4e_1x1_pre_relu)
    inception_4e_3x3_reduce_pre_relu = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed4e_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Activation('relu',name='mixed4e_3x3_bottleneck')(inception_4e_3x3_reduce_pre_relu)
    inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
    inception_4e_3x3_pre_relu = Conv2D(320, (3,3), padding='valid', activation='linear', name='mixed4e_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
    inception_4e_3x3 =Activation('relu',name='mixed4e_3x3')(inception_4e_3x3_pre_relu)
    inception_4e_5x5_reduce_pre_relu = Conv2D(32, (1,1), padding='same', activation='linear', name='mixed4e_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5_reduce =Activation('relu',name='mixed4e_5x5_bottleneck')(inception_4e_5x5_reduce_pre_relu)
    inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
    inception_4e_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed4e_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
    inception_4e_5x5 =Activation('relu',name='mixed4e_5x5')(inception_4e_5x5_pre_relu)
    inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed4e_pool')(inception_4d_output)
    inception_4e_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed4e_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_pool_proj = Activation('relu',name='mixed4e_pool_reduce')(inception_4e_pool_proj_pre_relu)
    inception_4e_output = Concatenate(axis=axis_concat, name='mixed4e')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool3')(pool4_helper)

    inception_5a_1x1_pre_relu = Conv2D(256, (1,1), padding='same', activation='linear', name='mixed5a_1x1_pre_relu', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_1x1 = Activation('relu',name='mixed5a_1x1')(inception_5a_1x1_pre_relu)
    inception_5a_3x3_reduce_pre_relu = Conv2D(160, (1,1), padding='same', activation='linear', name='mixed5a_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_reduce = Activation('relu',name='mixed5a_3x3_bottleneck')(inception_5a_3x3_reduce_pre_relu)
    inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
    inception_5a_3x3_pre_relu = Conv2D(320, (3,3), padding='valid', activation='linear', name='mixed5a_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
    inception_5a_3x3 = Activation('relu',name='mixed5a_3x3')(inception_5a_3x3_pre_relu)
    inception_5a_5x5_reduce_pre_relu = Conv2D(48, (1,1), padding='same', activation='linear', name='mixed5a_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_5x5_reduce = Activation('relu',name='mixed5a_5x5_bottleneck')(inception_5a_5x5_reduce_pre_relu)
    inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
    inception_5a_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed5a_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
    inception_5a_5x5 = Activation('relu',name='mixed5a_5x5')(inception_5a_5x5_pre_relu)
    inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed5a_pool')(pool4_3x3_s2)
    inception_5a_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed5a_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_pool)
    inception_5a_pool_proj = Activation('relu',name='mixed5a_pool_reduce')(inception_5a_pool_proj_pre_relu)
    inception_5a_output = Concatenate(axis=axis_concat, name='mixed5a')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

    inception_5b_1x1_pre_relu = Conv2D(384, (1,1), padding='same', activation='linear', name='mixed5b_1x1_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_1x1 = Activation('relu',name='mixed5b_1x1')(inception_5b_1x1_pre_relu)
    inception_5b_3x3_reduce_pre_relu = Conv2D(192, (1,1), padding='same', activation='linear', name='mixed5b_3x3_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_reduce = Activation('relu',name='mixed5b_3x3_bottleneck')(inception_5b_3x3_reduce_pre_relu)
    inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
    inception_5b_3x3_pre_relu = Conv2D(384, (3,3), padding='valid', activation='linear', name='mixed5b_3x3_pre_relu', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
    inception_5b_3x3 = Activation('relu',name='mixed5b_3x3')(inception_5b_3x3_pre_relu)
    inception_5b_5x5_reduce_pre_relu = Conv2D(48, (1,1), padding='same', activation='linear', name='mixed5b_5x5_bottleneck_pre_relu', kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_5x5_reduce = Activation('relu',name='mixed5b_5x5_bottleneck')(inception_5b_5x5_reduce_pre_relu)
    inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
    inception_5b_5x5_pre_relu = Conv2D(128, (5,5), padding='valid', activation='linear', name='mixed5b_5x5_pre_relu', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
    inception_5b_5x5 = Activation('relu',name='mixed5b_5x5')(inception_5b_5x5_pre_relu)
    inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='mixed5b_pool')(inception_5a_output)
    inception_5b_pool_proj_pre_relu = Conv2D(128, (1,1), padding='same', activation='linear', name='mixed5b_pool_reduce_pre_relu', kernel_regularizer=l2(0.0002))(inception_5b_pool)
    inception_5b_pool_proj = Activation('relu',name='mixed5b_pool_reduce')(inception_5b_pool_proj_pre_relu)
    inception_5b_output = Concatenate(axis=axis_concat, name='mixed5b')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='avgpool')(inception_5b_output)
    
    if include_top:
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
        loss3_classifier = Dense(output_classes_num, name='softmax2_pre_activation', kernel_regularizer=l2(0.0002),activation='linear')(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='softmax2')(loss3_classifier)

        googlenet = Model(inputs=img_input, outputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act], name='inception_v1')
   
    else: 
        googlenet = Model(inputs=img_input, outputs=[loss1_ave_pool,loss2_ave_pool,pool5_7x7_s1], name='inception_v1')

    if weights=='imagenet':
        if not(include_top):
            raise(NotImplementedError)
        else:
            weights_path = 'model/InceptionV1_FromLucid.h5'
            googlenet.load_weights(weights_path)

    return googlenet

def test_googlenet_channel_first():
    img = imageio.imread('data/cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    model = create_googlenet_channel_first('model/googlenet_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img) # note: the model has three outputs
    labels = np.loadtxt('data/synset_words.txt', str, delimiter='\t')
    predicted_label = np.argmax(out[2])
    predicted_prob = np.max(out[2])
    predicted_class_name = labels[predicted_label]
    print('Channel First ! Predicted Class: ', predicted_label,' prob : ',predicted_prob,' Class Name: ', predicted_class_name)
    # Channel First ! Predicted Class:  282  prob :  0.79697007  Class Name:  n02123159 tiger cat

def test_googlenet_channel_last():
    img = imageio.imread('data/cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]] # BGR
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    #model = create_googlenet('model/googlenet_weights.h5')
    model = create_googlenet('tmp.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img) # note: the model has three outputs
    labels = np.loadtxt('data/synset_words.txt', str, delimiter='\t')
    predicted_label = np.argmax(out[2])
    predicted_prob = np.max(out[2])
    predicted_class_name = labels[predicted_label]
    print('Predicted Class: ', predicted_label,' prob : ',predicted_prob,' Class Name: ', predicted_class_name)

    # Avec juste le numpy transpose : Predicted Class:  282  prob :  0.5057224  Class Name:  n02123159 tiger cat
    # Avec transpose et rot90 : Predicted Class:  282  prob :  0.6221494  Class Name:  n02123159 tiger cat
    # Avec swapaxes : Predicted Class:  282  prob :  0.5057224  Class Name:  n02123159 tiger cat
    
def test_inceptio_v1_fromLucid():
    img = imageio.imread('data/cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]] # BGR
    img = np.expand_dims(img, axis=0)

    # Test pretrained model
    #model = create_googlenet('model/googlenet_weights.h5')
    model = inception_v1_oldTF(weights='imagenet',include_top=True)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img) # note: the model has three outputs
    labels = np.loadtxt('data/synset_words.txt', str, delimiter='\t')
    predicted_label = np.argmax(out[2])
    predicted_prob = np.max(out[2])
    predicted_class_name = labels[predicted_label]
    print('Predicted Class: ', predicted_label,' prob : ',predicted_prob,' Class Name: ', predicted_class_name)

if __name__ == "__main__":
    #test_googlenet_channel_first()
    test_inceptio_v1_fromLucid()
    
    # Can use : 
    #from keras.applications.imagenet_utils import preprocess_input
    #img = preprocess_input(img, mode='caffe')  # also converts RGB to BGR
    