# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:32:14 2020

This file contains useful plot function for the differents other scripts

@author: gonthier
"""


#import scipy
import numpy as np
#import tensorflow as tf
#import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
#import pandas as pd
#import time
#import pickle
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
#import scipy.stats as stats
#from tensorflow.python.framework import dtypes
#import matplotlib.gridspec as gridspec
#import math
#from skimage import exposure
from PIL import Image
from preprocess_crop import load_and_crop_img

def plt_multiple_imgs(list_images,path_output,path_img='',name_fig='',\
                      cropCenter=False,Net='VGG',title_imgs=None,
                      roundColor=[],color='g'):
    """
    The image in the list roundColor will be rounded by a rectangle of color = color
    """
    
    matplotlib.use('Agg')
    number_imgs = len(list_images)
    assert(number_imgs>0)
    if title_imgs is None:
        hspace = 0.05
        wspace = 0.05
        gridspec_kw = {'wspace':wspace, 'hspace':hspace}
    else:
        gridspec_kw = {}
#    if(number_imgs<6):
#         fig, axes = plt.subplots(1,number_imgs)
#    else:
#         if(number_imgs%4==0):
#             fig, axes = plt.subplots(number_imgs//4, 4, gridspec_kw =gridspec_kw)
#         elif(number_imgs%3==0):
#             fig, axes = plt.subplots(number_imgs//3, 3, gridspec_kw =gridspec_kw)
#         elif(number_imgs%5==0):
#             fig, axes = plt.subplots(number_imgs//5, 5, gridspec_kw =gridspec_kw)
#         elif(number_imgs%2==0):
#             fig, axes = plt.subplots(number_imgs//2, 2, gridspec_kw =gridspec_kw)
#         else:
#             j=6
#             while(not(number_imgs%j==0)):
#                 fig, axes = plt.subplots(number_imgs//j, j, gridspec_kw =gridspec_kw)
#                 j += 1
    grid_size = int(np.ceil(np.sqrt(number_imgs)))
    dpi = 300
    if cropCenter:
        target_size = 224
        target_size_image_output = 1200*grid_size/3
        size_inch = target_size_image_output//dpi
        fig, axes = plt.subplots(grid_size, grid_size, gridspec_kw =gridspec_kw,figsize=(size_inch,size_inch))
    else:
        target_size = 500
        fig, axes = plt.subplots(grid_size, grid_size, gridspec_kw =gridspec_kw)
     
    i = 0
    if not(number_imgs==1):
        axes = axes.flatten()
    else:
        axes= [axes]
        
    linewidth_rect = 10
    
        
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
                               target_size=target_size,crop_size=224,interpolation='lanczos:center')
             img = img[0,:,:,:] # Remove batch
             img = img.astype(np.uint8)
         else:
             img = Image.open(img_name_path)
         axis.imshow(img)
         
         # If in list roundColor will be rounded by a color rectangle
         if img_name in roundColor:
             #print(img_name,len(roundColor),roundColor.index(img_name))
             xdim,ydim,c = img.shape
             axis.set_ylim(ydim,-linewidth_rect)
             axis.set_xlim(-linewidth_rect,xdim)
             rect = patches.Rectangle((-linewidth_rect,-linewidth_rect),xdim+linewidth_rect,ydim+linewidth_rect,
                                      linewidth=linewidth_rect,
                                      edgecolor=color,facecolor='none')
             axis.add_patch(rect)
             
         if not(title_imgs is None):
             axis.set_title(title_imgs[i],fontdict={'fontsize':5})
         axis.set_axis_off()
         axis.set_aspect('equal')
         i += 1
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.985, wspace=0.15, hspace=0.15)
    pltname = os.path.join(path_output,name_fig+'.png')
    fig.savefig(pltname, dpi = dpi)
    plt.close()