# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:56:59 2019

@author: gonthier
"""

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
import numpy as np


def minimalCase_forIssue():
    max_dim = 224
    path_to_img =  'Q23898.jpg'
    img = tf.keras.preprocessing.image.load_img(
        path_to_img,
        grayscale=False,
        target_size=(max_dim, max_dim),
        interpolation='nearest')
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img_tensor = tf.convert_to_tensor(img)
    
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    output = vgg.get_layer('block1_conv1').output
    model = models.Model(vgg.input, output)
    sess = tf.Session()
    
    for i in range(2):
        sess.run(tf.global_variables_initializer())
        evaluation_by_sess = sess.run(model(img_tensor))
        print('With sess.run :',evaluation_by_sess[0,0,0,:])
        evaluation_by_predict = model.predict(img,batch_size=1)
        print('With .predict() :',evaluation_by_predict[0,0,0,:])
    sess.close()
    
if __name__ == '__main__':
    minimalCase_forIssue()
    
