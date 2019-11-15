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

def load_and_crop_img(path,Net, grayscale=False, color_mode='rgb', target_smallest_size=256,
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
#    print('original',img.size)

    target_width = crop_size
    target_height = crop_size

    if crop not in ["center", "random"]:
        raise ValueError('Invalid crop method {} specified.', crop)

    width, height = img.size

    # Resize keeping aspect ratio
    # result should be no smaller than the targer size, include crop fraction overhead
    if width > height:
        ratio = target_smallest_size/height
        target_biggest_size = int(ratio*width)
        target_size_before_crop = (target_smallest_size, target_biggest_size)
        width = target_smallest_size
        height = target_biggest_size
    else:
        ratio = target_smallest_size/width
        target_biggest_size = int(ratio*height)
        target_size_before_crop = (target_biggest_size, target_smallest_size)
        width = target_biggest_size
        height = target_smallest_size

    img = preprocessing.image.load_img(path, 
                                        grayscale=grayscale, 
                                        color_mode=color_mode, 
                                        target_size=target_size_before_crop, 
                                        interpolation=interpolation)

#    print('resize',img.size)
    if crop == "center":
        left_corner = int(round(width/2)) - int(round(target_width/2))
        top_corner = int(round(height/2)) - int(round(target_height/2))
        img = img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
    elif crop == "random":
        print('This have never been testes')
        left_shift = random.randint(0, int((width - target_width)))
        down_shift = random.randint(0, int((height - target_height)))
        img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))
        
    img = kp_image.img_to_array(img)
#    import matplotlib.pyplot as plt
#    plt.imshow(img.astype(int))
#    plt.title('Crop avec keras resizeing')
    img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
    if 'VGG' in Net:
        preprocessing_function = applications.vgg19.preprocess_input
    elif 'ResNet' in Net:
        preprocessing_function = applications.resnet50.preprocess_input
    else:
        print(Net,'is unknwon')
        raise(NotImplementedError)
    
    img =  preprocessing_function(img)
#    plt.figure()
#    plt.imshow(((img-np.min(img))/(np.max(img)-np.min(img)))[0,:,:,:])
#    plt.title('Crop avec keras resizeing after preprocessing fct')
#    import cv2
#    augmentation = False
#    im = cv2.imread(path)
#    if augmentation:
#        sizeIm = 256
#    else:
#        sizeIm = 224
#    if(im.shape[0] < im.shape[1]):
#        dim = (sizeIm, int(im.shape[1] * sizeIm / im.shape[0]),3)
#    else:
#        dim = (int(im.shape[0] * sizeIm / im.shape[1]),sizeIm,3)
#    tmp = (dim[1],dim[0])
#    dim = tmp
#    resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
#    resized = resized[:,:,[2,1,0]]
#
#    resizedf = resized.astype(np.float32)
#    crop = resizedf[int(resized.shape[0]/2 - 112):int(resized.shape[0]/2 +112),int(resized.shape[1]/2-112):int(resized.shape[1]/2+112),:] 
#    plt.figure()
#    plt.imshow(crop)
#    plt.title('Crop avec cv2 resizing')
#    
#    resized = kp_image.img_to_array(crop)
#    resized = np.expand_dims(resized, axis=0)
#    resized =  preprocessing_function(resized)
#    plt.figure()
#    plt.imshow(((resized-np.min(resized))/(np.max(resized)-np.min(resized)))[0,:,:,:])
#    plt.title('Crop avec keras resizeing apres preprocessing fct')
#    
    
    return img

def load_and_crop_img_forImageGenerator(path,Net, grayscale=False, color_mode='rgb',\
                                        target_smallest_size=224,
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
    
    width, height = img.size

    # Resize keeping aspect ratio
    # result should be no smaller than the targer size, include crop fraction overhead
    if width > height:
        ratio = target_smallest_size/height
        target_biggest_size = int(ratio*width)
        target_size_before_crop = (target_biggest_size, target_smallest_size)
    else:
        ratio = target_smallest_size/width
        target_biggest_size = int(ratio*height)
        target_size_before_crop = (target_smallest_size, target_biggest_size)

    img = preprocessing.image.load_img(path, 
                                    grayscale=grayscale, 
                                    color_mode=color_mode, 
                                    target_size=target_size_before_crop, 
                                    interpolation=interpolation)

    if crop == "center":
        left_corner = int(round(width/2)) - int(round(target_width/2))
        top_corner = int(round(height/2)) - int(round(target_height/2))
        img = img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
    elif crop == "random":
        left_shift = random.randint(0, int((width - target_width)))
        down_shift = random.randint(0, int((height - target_height)))
        img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))
        
#    img = kp_image.img_to_array(img)
#    img = np.expand_dims(img, axis=0) # Should be replace by expand_dims in tf
    if 'VGG' in Net:
        preprocessing_function = applications.vgg19.preprocess_input
    elif 'ResNet' in Net:
        preprocessing_function = applications.resnet50.preprocess_input
    else:
        print(Net,'is unknwon')
        raise(NotImplementedError)
    
    img =  preprocessing_function(img)

    return img
