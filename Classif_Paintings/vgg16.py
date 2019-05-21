#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:36:17 2017

@author: gonthier
"""

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
VGG16_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3','pool5','fuco6','relu6','fuco7','relu7','fuco8','prob')
#layers   = [2 5 10 19 28]; for texture generation
style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
# TODO : check if the N value are right for the poolx
    
def get_vgg_layers(VGG16_mat='/media/gonthier/HDD/models/imagenet-vgg-verydeep-16.mat'):
    """
    Load the VGG 16 layers
    """
    if('imagenet-vgg-verydeep-16.mat' in VGG16_mat):
        # The vgg19 network from http://www.vlfeat.org/matconvnet/pretrained/
        try:
            vgg_rawnet = scipy.io.loadmat(VGG16_mat)
            vgg_layers = vgg_rawnet['layers'][0]
        except(FileNotFoundError):
            print("The path to the VGG16_mat is not right or the .mat is not here")
            print("You can download it here : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat")
            raise
    else:
        print("The path to the VGG16_mat is unknown.")
    return(vgg_layers)

def net_preloaded(vgg_layers, input_image,pooling_type='max',padding='SAME',sess=None):
    """
    This function read the vgg layers and create the net architecture
    We need the input image to know the dimension of the input layer of the net
    """
    net = {}
    #_,height, width, numberChannels = input_image.shape # In order to have the right shape of the input
    #current = tf.Variable(input_image, dtype=np.float32)
    current = tf.cast(input_image, tf.float32)
    net['input'] = current
    for i, name in enumerate(VGG16_LAYERS):
        kind = name[:4]
        #print(name,current.shape)
        if(kind == 'conv'):
            # Only way to get the weight of the kernel of convolution
            # Inspired by http://programtalk.com/vs2/python/2964/facenet/tmp/vggverydeep19.py/
            kernels = vgg_layers[i][0][0][2][0][0] 
            bias = vgg_layers[i][0][0][2][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
#            if(i==0):
#                kernels = tf.constant(kernels[:,:,::-1,:]) # If you want RGB image as input ???
#                #kernels = tf.constant(kernels)
#            else:
#                kernels = tf.constant(kernels)
            #kernels = tf.constant(np.transpose(kernels, (1,0 ,2, 3)))
            kernels = tf.constant(kernels)
#            with sess.as_default():
#                #print(kernels.eval())
#                np.save(name,kernels.eval())
            
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
            current = fuco_layer(current, kernels, bias,sess=sess)
        elif(kind=='prob'):
            current  = tf.nn.softmax(current, name="prob")
        net[name] = current

    #net['output'] = tf.contrib.layers.flatten(current)
    assert len(net) == len(VGG16_LAYERS) +1 # Test if the length is right 
    return(net)
    
    
def fuco_layer(input,weights, bias,sess=None):  
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(input, [-1, dim])
    weights = tf.reshape(weights,[dim,-1])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
#    with sess.as_default():
#        print(weights.eval())
#        print(bias.eval())
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        vgg_layers = get_vgg_layers()
        input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
        net = net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME',sess=sess)

        im = Image.open('loulou.jpg').resize((224,224),Image.NEAREST)
        im = np.array(im).astype(np.float32) 
        im2 = Image.open('cat.jpg').resize((224,224),Image.NEAREST) 
        im2 = np.array(im2).astype(np.float32) 
        
        im[:,:,2] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,0] -= 123.68 
        #im = im[:,:,::-1] 
        im2[:,:,2] -= 103.939
        im2[:,:,1] -= 116.779
        im2[:,:,0] -= 123.68 
        #im2 = im2[:,:,::-1] 
        
#        im = cv2.resize(cv2.imread('dog.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
#        im2 = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
#      # Need to read image in BGR
#      
#      # Remove train image mean
#        im[:,:,0] -= 103.939
#        im[:,:,1] -= 116.779
#        im[:,:,2] -= 123.68
#      
#        im2[:,:,0] -= 103.939
#        im2[:,:,1] -= 116.779
#        im2[:,:,2] -= 123.68
        
        im = im.reshape(-1,224,224,3)
        im2 = im2.reshape(-1,224,224,3)
        ims = np.concatenate((im,im2))
#      
#        im = Image.open('cat.jpg').resize((224,224))
#        im = np.array(im)
#        im = im[:,::-1]        
#        im = im.reshape(-1,224,224,3)
#        im2 = Image.open('dog.jpg').resize((224,224))
#        im2 = im2[:,::-1]
#        im2 = np.array(im2)
#        im2 = im2.reshape(-1,224,224,3)
#        ims = np.concatenate((im,im2))
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
            
## Avec le truc bizarre a faire sur les couches !!  [:,:,::-1,:]
#[[-3.47197604  0.19120267 -5.17651367 ..., -2.32900119  1.14006686
#   5.31669807]
# [-3.23206329 -1.92238402 -0.177055   ..., -2.62692785  4.10695839
#   4.26083565]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#154 : Pekinese, Pekingese, Peke
#152 : Japanese spaniel
#265 : toy poodle
#260 : chow, chow chow
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#281 : tabby, tabby cat
#478 : carton
#287 : lynx, catamount
#282 : tiger cat

# sans le np.transpose des kernels 
#[[-2.69815135 -1.34160721 -3.91112494 ..., -1.79113984  0.84197533
#   3.50710869]
# [-2.74866271 -1.84448302  1.16664696 ..., -3.25384331  2.18476605
#   2.35617161]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#154 : Pekinese, Pekingese, Peke
#151 : Chihuahua
#265 : toy poodle
#152 : Japanese spaniel
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#248 : Eskimo dog, husky
#281 : tabby, tabby cat
#434 : bath towel
#250 : Siberian husky


#VGG Matlab :
#    260
#Pomeranian
#13.9348
#155
#Pekinese, Pekingese, Peke
#11.3192
#266
#toy poodle
#9.8219
#153
#Japanese spaniel
#9.6906
#261
#chow, chow chow
#9.1936
#    
#    286
#Egyptian cat
#8.845
#282
#tabby, tabby cat
#7.9019
#479
#carton
#7.7792
#288
#lynx, catamount
#7.3395
#283
#tiger cat
#7.105
            
#[[-3.44510841  1.2140187  -7.61519289 ..., -4.4459734   2.71927929
#   7.72898293]
# [-5.36464787 -1.03326237 -1.86866057 ..., -6.12964725  8.30052757
#   8.96116066]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian : 27.126
#151 : Chihuahua : 18.7484
#154 : Pekinese, Pekingese, Peke : 17.9795
#152 : Japanese spaniel : 17.1226
#261 : keeshond : 16.9831
#
#Im  1
#5 first Predicted class : 
#478 : carton : 14.2492
#285 : Egyptian cat : 13.7479
#281 : tabby, tabby cat : 13.3356
#282 : tiger cat : 13.0345
#434 : bath towel : 12.8171            
