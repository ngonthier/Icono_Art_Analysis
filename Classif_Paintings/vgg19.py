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
import cv2

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4','pool5','fuco6','relu6','fuco7','relu7','fuco8','prob')
#layers   = [2 5 10 19 28]; for texture generation
style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
# TODO : check if the N value are right for the poolx
    
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
    
    This Net need images in RGB 
    
    """
    net = {}
    #_,height, width, numberChannels = input_image.shape # In order to have the right shape of the input
    #current = tf.Variable(input_image, dtype=np.float32)
    current = tf.cast(input_image, tf.float32)
    net['input'] = current
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        #print(name,current.shape)
        if(kind == 'conv'):
            # Only way to get the weight of the kernel of convolution
            # Inspired by http://programtalk.com/vs2/python/2964/facenet/tmp/vggverydeep19.py/
            kernels = vgg_layers[i][0][0][2][0][0] 
            bias = vgg_layers[i][0][0][2][0][1]
            # F is an array of dimension FW x FH x FC x K where (FH,FW) are the filter height and width and K the number o filters in the bank.
            # http://www.vlfeat.org/matconvnet/mfiles/vl_nnconv/
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
            # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
            # tensorflow: weights are [height, width, in_channels, out_channels]
            #kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
            #kernels = tf.constant(kernels[:,:,::-1,:])
#            if(name=='conv1_1'):
#                kernels = tf.constant(kernels[:,:,::-1,:])
#            else:
#                kernels = tf.constant(kernels)
            kernels = tf.constant(kernels)
            bias = tf.constant(bias.reshape(-1))
            current = conv_layer(current, kernels, bias,name,padding) 
            # Update the  variable named current to have the right size
        elif(kind == 'relu'):
            current = tf.nn.relu(current,name=name)
        elif(kind == 'pool'):
            current = pool_layer(current,name,pooling_type,padding)
        elif(kind=='fuco'):
            kernels = vgg_layers[i][0][0][2][0][0] 
            bias = vgg_layers[i][0][0][2][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            #kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
            kernels = tf.constant(kernels)
            bias = tf.constant(bias.reshape(-1))
            current = fuco_layer(current, kernels, bias)
        elif(kind=='prob'):
            current  = tf.nn.softmax(current, name="prob")
        net[name] = current
        
    assert net['fuco6'].get_shape().as_list()[1:] == [4096]
    #net['output'] = tf.contrib.layers.flatten(current)
    assert len(net) == len(VGG19_LAYERS) +1 # Test if the length is right 
    return(net)
    
    
def fuco_layer(input,weights, bias):  
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(input, [-1, dim])
    weights = tf.reshape(weights,[dim,-1])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
    return(fc) # TODO

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

def pool_layer(input,name,pooling_type='max',padding='SAME'):
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
   import yaml
   with tf.Graph().as_default():
        vgg_layers = get_vgg_layers()
        input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
        net = net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME')
        sess = tf.Session()
        #im = cv2.resize(cv2.imread('dog.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
        #im2 = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
        im =  np.array(Image.open('loulou.jpg').resize((224,224))).astype(np.float32)
        im2 =  np.array(Image.open('cat.jpg').resize((224,224))).astype(np.float32)
        
        im[:,:,2] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,0] -= 123.68 
        #im = im[:,:,::-1] 

        im2[:,:,2] -= 103.939
        im2[:,:,1] -= 116.779
        im2[:,:,0] -= 123.68 
        #im2 = im2[:,:,::-1] 
        im = np.expand_dims(im, axis=0)
        im2 = np.expand_dims(im2, axis=0)
        ims = np.concatenate((im,im2))
        #ims = im
      
        predict_values = sess.run(net['fuco8'], feed_dict={input_tensor: ims})
        print(predict_values.shape)
        print(predict_values)
        dictr = yaml.load(open("imageNet_map.txt").read().replace('\n',''))
        
        numIm = predict_values.shape[0]
        for j in range(numIm):
            print("Im ",j)
            string = "5 first Predicted class : \n"
            out_sort_arg = np.argsort(predict_values[j,:])[::-1]
            #out_sort_arg = np.flip(np.argsort(predict_values[j,:]),axis=1)[0]
            for i in range(5):
                string += str(out_sort_arg[i]) + ' : ' + dictr[out_sort_arg[i]] + ' : ' + str(predict_values[j,out_sort_arg[i]]) + '\n'
            print(string)

# Avec les kernels non retourner sauf le premier
#[[-2.73314285  1.09662354 -4.03027868 ..., -1.74038112  1.51331854
#   4.80882072]
# [-1.92644882 -3.12641644 -0.26798052 ..., -1.03142285  3.46962547
#   2.97795534]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#265 : toy poodle
#154 : Pekinese, Pekingese, Peke
#266 : miniature poodle
#850 : teddy, teddy bear
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#287 : lynx, catamount
#282 : tiger cat
#281 : tabby, tabby cat
#478 : carton


