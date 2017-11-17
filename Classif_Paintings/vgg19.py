#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:56:04 2017

@author: gonthier
"""
import tensorflow as tf
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4','pool5')
#layers   = [2 5 10 19 28]; for texture generation
style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
# TODO : check if the N value are right for the poolx

def plot_image(path_to_image):
    """
    Function to plot an image
    """
    img = Image.open(path_to_image)
    plt.imshow(img)
    
def get_vgg_layers(VGG19_mat='/media/HDD/models/imagenet-vgg-verydeep-19.mat'):
    """
    Load the VGG 19 layers
    """
    if('imagenet-vgg-verydeep-19.mat' in VGG19_mat):
        # The vgg19 network from http://www.vlfeat.org/matconvnet/pretrained/
        try:
            vgg_rawnet = scipy.io.loadmat(VGG19_mat)
            vgg_layers = vgg_rawnet['layers'][0]
        except(FileNotFoundError):
            print("The path to the VGG19_mat is not right or the .mat is not here")
            print("You can download it here : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat")
            raise
    else:
        print("The path to the VGG19_mat is unknown.")
    return(vgg_layers)

def net_preloaded(vgg_layers, input_image,pooling_type='max',padding='SAME'):
    """
    This function read the vgg layers and create the net architecture
    We need the input image to know the dimension of the input layer of the net
    """
    
    net = {}
    _,height, width, numberChannels = input_image.shape # In order to have the right shape of the input
    current = tf.Variable(np.zeros((1, height, width, numberChannels), dtype=np.float32))
    net['input'] = current
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        #print(name,current.shape)
        if(kind == 'conv'):
            # Only way to get the weight of the kernel of convolution
            # Inspired by http://programtalk.com/vs2/python/2964/facenet/tmp/vggverydeep19.py/
            kernels = vgg_layers[i][0][0][2][0][0] 
            bias = vgg_layers[i][0][0][2][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
            bias = tf.constant(bias.reshape(-1))
            current = conv_layer(current, kernels, bias,name,padding) 
            # Update the  variable named current to have the right size
        elif(kind == 'relu'):
            current = tf.nn.relu(current,name=name)
        elif(kind == 'pool'):
            current = pool_layer(current,name,pooling_type,padding)

        net[name] = current
    
    net['output'] = tf.contrib.layers.flatten(current)
    assert len(net) == len(VGG19_LAYERS) +2 # Test if the length is right 
    return(net)

def conv_layer(input, weights, bias,name,padding='SAME'):
    """
    This function create a conv2d with the already known weight and bias
    
    conv2d :
    Computes a 2-D convolution given 4-D input and filter tensors.
    input: A Tensor. Must be one of the following types: half, float32, float64
    Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
    a filter / kernel tensor of shape 
    [filter_height, filter_width, in_channels, out_channels]
    """
    stride = 1
    if(padding=='SAME'):
        conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
            padding=padding,name=name)
    elif(padding=='VALID'):
        input = get_img_2pixels_more(input)
        conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
            padding='VALID',name=name)
    # We need to impose the weights as constant in order to avoid their modification
    # when we will perform the optimization
    return(tf.nn.bias_add(conv, bias))

def get_img_2pixels_more(input):
    new_input = tf.concat([input,input[:,0:2,:,:]],axis=1)
    new_input = tf.concat([new_input,new_input[:,:,0:2,:]],axis=2)
    return(new_input)

def pool_layer(input,name,pooling_type='avg',padding='SAME'):
    """
    Average pooling on windows 2*2 with stride of 2
    input is a 4D Tensor of shape [batch, height, width, channels]
    Each pooling op uses rectangular windows of size ksize separated by offset 
    strides in the avg_pool function 
    """
    stride_pool = 2
    if(padding== 'VALID'): # TODO Test if paire ou impaire !!! 
        _,h,w,_ = input.shape
        if not(h%2==0):
            input = tf.concat([input,input[:,0:2,:,:]],axis=1)
        if not(w%2==0):
            input = tf.concat([input,input[:,:,0:2,:]],axis=2)
    if pooling_type == 'avg':
        pool = tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
                padding=padding,name=name) 
    elif pooling_type == 'max':
        pool = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, stride_pool, stride_pool, 1),
                padding=padding,name=name) 
    return(pool)

if __name__ == '__main__':
    input_image = np.zeros((1,224,224,3))
    vgg_layers = get_vgg_layers()
    net_preloaded(vgg_layers,input_image,pooling_type='max',padding='SAME')