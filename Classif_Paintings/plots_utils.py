# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:32:14 2020

This file contains useful plot function for the differents other scripts

@author: gonthier
"""


import scipy
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from tensorflow.python.framework import dtypes
import matplotlib.gridspec as gridspec
import math
from skimage import exposure
from PIL import Image
from preprocess_crop import load_and_crop_img

def plt_multiple_imgs(list_images,path_output,path_img='',name_fig='',\
                      cropCenter=False,Net='VGG'):
    number_imgs = len(list_images)
    hspace = 0.05
    wspace = 0.05
    if(number_imgs<6):
         fig, axes = plt.subplots(1,number_imgs)
    else:
         if(number_imgs%4==0):
             fig, axes = plt.subplots(number_imgs//4, 4, gridspec_kw = {'wspace':wspace, 'hspace':hspace})
         elif(number_imgs%3==0):
             fig, axes = plt.subplots(number_imgs//3, 3, gridspec_kw = {'wspace':wspace, 'hspace':hspace})
         elif(number_imgs%5==0):
             fig, axes = plt.subplots(number_imgs//5, 5, gridspec_kw = {'wspace':wspace, 'hspace':hspace})
         elif(number_imgs%2==0):
             fig, axes = plt.subplots(number_imgs//2, 2, gridspec_kw = {'wspace':wspace, 'hspace':hspace})
         else:
             j=6
             while(not(number_imgs%j==0)):
                 j += 1
             fig, axes = plt.subplots(number_imgs//j, j, gridspec_kw = {'wspace':wspace, 'hspace':hspace})
     
    i = 0
    axes = axes.flatten()
    for axis,img_name in zip(axes,list_images):
         img_name_path = os.path.join(path_img,img_name)
         if not('jpg' in img_name_path) or not('png' in img_name_path):
             img_name_path_ext = img_name_path + '.jpg'
             if not(os.path.exists(img_name_path_ext)):
                 img_name_path_ext  = img_name_path + '.png'
                 if not(os.path.exists(img_name_path_ext)):
                     print(img_name_path,'is not found')
                     raise(ValueError)
             img_name_path = img_name_path_ext
         if cropCenter:
             img = load_and_crop_img(path=img_name_path,Net=Net, grayscale=False, color_mode='rgb',\
                               target_size=224,crop_size=224,interpolation='lanczos:center')
             img = img[0,:,:,:] # Remove batch
             img = img.astype(np.uint8)
         else:
             img = Image.open(img_name_path)
         axis.imshow(img)
         axis.set_axis_off()
         axis.set_aspect('equal')
         i += 1
    pltname = os.path.join(path_output,name_fig+'.png')
    fig.savefig(pltname, dpi = 300)
    plt.close()