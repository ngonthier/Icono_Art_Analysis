#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:38:33 2019

Provide new way to load and crop image : 
    https://gist.github.com/rstml/bbd491287efc24133b90d4f7f3663905

@author: gonthier
"""

import random
from tensorflow.keras import preprocessing
from tensorflow.python.keras.preprocessing import image as kp_image
import numpy as np
from tensorflow.keras import applications
import keras_preprocessing

def load_and_crop_img(path,Net, grayscale=False, color_mode='rgb', target_size=224,
             crop_size=224,interpolation='lanczos:center'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_smallest_size: the smallest legnth of the image
        crop_size: size of the crop return
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "lanczos" is used for avoid aliasing
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    This function can be used to erase the load_img function 
    the @param target_size is the target_smallest_size if the crop parameter is not equal to none
    """

    if isinstance(target_size, int):
        target_smallest_size = target_size
    if isinstance(target_size, tuple):
        target_smallest_size = min(target_size[0],target_size[1])
    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")  

    if crop == "none":
        return preprocessing.image.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=(target_smallest_size,target_smallest_size),
                                            interpolation=interpolation)

    # Load original size image using Keras
    img = preprocessing.image.load_img(path, 
                                        grayscale=grayscale, 
                                        color_mode=color_mode, 
                                        target_size=None, 
                                        interpolation=interpolation)
    
    target_width = crop_size
    target_height = crop_size

    if crop not in ["center", "random"]:
        raise ValueError('Invalid crop method {} specified.', crop)
    if interpolation not in ["nearest", "bilinear", "bicubic", "lanczos","box", "hamming"]:
        raise ValueError('Invalid interpolation method {} specified.', interpolation)

    width, height = img.size # In PIl object size provide witdh,heigth
    # result should be no smaller than the targer size, include crop fraction overhead
    if width > height:
        ratio = target_smallest_size/height
        target_biggest_size = int(ratio*width)
        target_size_before_crop = (target_smallest_size, target_biggest_size) # (img_height, img_width)`
    else:
        ratio = target_smallest_size/width
        target_biggest_size = int(ratio*height)
        target_size_before_crop =  (target_biggest_size, target_smallest_size) # (img_height, img_width)`
        
    height,width = target_size_before_crop # (img_height, img_width)`
    
    # In preprocessing image according to the documentation 
    # target_size must be : (img_height, img_width)
    img = preprocessing.image.load_img(path, 
                                        grayscale=grayscale, 
                                        color_mode=color_mode, 
                                        target_size=target_size_before_crop, 
                                        interpolation=interpolation)

    if crop == "center":
        left_corner = int(round(width/2)) - int(round(target_width/2))
        top_corner = int(round(height/2)) - int(round(target_height/2))
        img = img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
        # Returns a rectangular region from this image. The box is a 4-tuple defining the left, 
        #    upper, right, and lower pixel coordinate
    elif crop == "random":
        print('This have never been tested')
        left_shift = random.randint(0, int((width - target_width)))
        down_shift = random.randint(0, int((height - target_height)))
        img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))
        # Returns a rectangular region from this image. 
        #­ The box is a 4-tuple defining the left, 
        #    upper, right, and lower pixel coordinate
    
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
    if not(Net is None):
        if 'VGG' in Net:
            preprocessing_function = applications.vgg19.preprocess_input
        elif 'ResNet' in Net:
            preprocessing_function = applications.resnet50.preprocess_input
        elif 'InceptionV1' in Net:
            preprocessing_function = applications.imagenet_utils.preprocess_input
        else:
            print(Net,'is unknwon, it can be None if you want no preprocessing.')
            raise(NotImplementedError)
        
        img =  preprocessing_function(img)

    return img

def load_and_crop_img_forImageGenerator(path,Net, grayscale=False, color_mode='rgb',\
                                        target_size=224,
                                        crop_size=224,interpolation='lanczos:center'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_smallest_size: the smallest legnth of the image
        crop_size: size of the crop return
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "lanczos" is used for avoid aliasing
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    #print(path)
    if isinstance(target_size, int):
        target_smallest_size = target_size
    if isinstance(target_size, tuple):
        target_smallest_size = min(target_size[0],target_size[1])
    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")  

    if crop == "none":
        return preprocessing.image.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=(target_smallest_size,target_smallest_size),
                                            interpolation=interpolation)

    # Load original size image using Keras, it will return a PIL image object 
    img = preprocessing.image.load_img(path, 
                                        grayscale=grayscale, 
                                        color_mode=color_mode, 
                                        target_size=None, 
                                        interpolation=interpolation)

    target_width = crop_size
    target_height = crop_size

    if crop not in ["center", "random"]:
        raise ValueError('Invalid crop method {} specified.', crop)
    if interpolation not in ["nearest", "bilinear", "bicubic", "lanczos","box", "hamming"]:
        raise ValueError('Invalid interpolation method {} specified.', interpolation)
    
    width, height = img.size # In a PIL image object : Image size, in pixels. 
    # The size is given as a 2-tuple (width, height).
    #print('w,h',width, height)
    # Resize keeping aspect ratio
    # result should be no smaller than the targer size, include crop fraction overhead
    if width > height:
        ratio = target_smallest_size/height
        target_biggest_size = int(ratio*width)
        new_width = target_biggest_size
        new_height = target_smallest_size
        #target_size_before_crop = (target_smallest_size, target_biggest_size) # (img_height, img_width)`
        #target_size_before_crop = (target_biggest_size, target_smallest_size) # (img_height, img_width)`
    elif  width < height:
        ratio = target_smallest_size/width
        target_biggest_size = int(ratio*height)
        new_width = target_smallest_size
        new_height = target_biggest_size
        #target_size_before_crop =  (target_biggest_size, target_smallest_size) # (img_height, img_width)`
        #target_size_before_crop =  (target_smallest_size, target_biggest_size) # (img_height, img_width)`
    else: # width == height
        new_width = target_smallest_size
        new_height = target_smallest_size
        
        
    #height,width = target_size_before_crop # (img_height, img_width)`
    #width,height = target_size_before_crop # (img_height, img_width)`

    #print(target_size_before_crop)

    # In preprocessing image according to the documentation 
    # target_size must be : (img_height, img_width)
    #print('new w,h',new_width,new_height)
#    target_size_before_crop = (new_height,new_width)
#    img = preprocessing.image.load_img(path, 
#                                    grayscale=grayscale, 
#                                    color_mode=color_mode, 
#                                    target_size=target_size_before_crop, 
#                                    interpolation=interpolation)
    target_size_before_crop_keep_ratio = new_width,new_height
    resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

    img = img.resize(target_size_before_crop_keep_ratio, resample=resample)
    #img2= kp_image.img_to_array(img)
    #print('before crop shape',img2.shape)

    if crop == "center":
        left_corner = int(round(new_width/2)) - int(round(target_width/2))
        top_corner = int(round(new_height/2)) - int(round(target_height/2))
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        #print('rect',left_corner, top_corner, left_corner + target_width, top_corner + target_height)
        img = img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
    elif crop == "random":
#        print('path',path)
#        print('img.size',img.size)
#        print('int((new_width - target_width))',int((new_width - target_width)))
#        print('int((new_height - target_height))',int((new_height - target_height)))
        left_shift = random.randint(0, int((new_width - target_width)))
        down_shift = random.randint(0, int((new_height - target_height)))
        img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))
    #img2= kp_image.img_to_array(img)
    
#    print('shape crop',img2.shape)
#    print(img2)
#    print('number of zero',len(np.where(img2.ravel() == 0)[0]))
#        
##    img = kp_image.img_to_array(img)
##    img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
#    if 'VGG' in Net:
#        preprocessing_function = applications.vgg19.preprocess_input
#    elif 'ResNet' in Net:
#        preprocessing_function = applications.resnet50.preprocess_input
#    else:
#        print(Net,'is unknwon')
#        raise(NotImplementedError)
#    
#    img =  preprocessing_function(img)

    return img

# Monkey patch
#import keras_preprocessing as kp
#kp.image.utils.load_img = load_and_crop_img
#tf.keras.preprocessing.image.load_img = load_and_crop_img
