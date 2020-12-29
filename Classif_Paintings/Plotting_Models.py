# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:49:14 2020

The goal of this script is to print the architecture of the keras model  

@author: gonthier
"""

import tensorflow as tf

import os

from googlenet import inception_v1_oldTF as Inception_V1

from inception_v1 import InceptionV1_slim

def print_model_used_in_FT():
    
    path_to_im = os.path.join('C:\\','Users','gonthier','ownCloud','Mes_Latex','2021_PhD_Thesis','imHD','im')
    path_to_im = ''
    weights = 'imagenet'
    
    Net_list = ['VGG','InceptionV1','ResNet50']
    #Net_list = ['VGG']
    
    for Net in Net_list:
    
        if Net=='VGG':
            imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=True, weights=weights)
        elif Net == 'InceptionV1':
            imagenet_model = Inception_V1(include_top=True, weights=weights)
        elif Net == 'InceptionV1_slim':
            imagenet_model = InceptionV1_slim(include_top=True, weights=weights)
        elif Net == 'ResNet50':
            imagenet_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=weights)
    
        output_file_name = os.path.join(path_to_im,Net+'_keras.png')
    
        print(Net,output_file_name)
        print(imagenet_model.summary())
        
        with open(Net + '_report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            imagenet_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
    
        #Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
    
#        tf.keras.utils.plot_model(
#                imagenet_model,
#                to_file=output_file_name,
#                show_shapes=True,
#                show_layer_names=True,
#                rankdir="TB",
#                expand_nested=False,
#                dpi=96)

if __name__ == '__main__': 
    print_model_used_in_FT()


