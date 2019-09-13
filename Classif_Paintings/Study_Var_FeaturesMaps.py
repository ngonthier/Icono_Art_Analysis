#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:14:24 2019

The goal of this scripts is to print the histogram of the variance of the 
featrues maps at certain layers of the VGG network for two datasets :
    - a subset of ImageNet 
    - an artistic dataset such as Paintings ArtUK

@author: gonthier
"""

import glob
import os.path
import os 
import numpy as np
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib

import tensorflow as tf

from Stats_Fcts import get_intermediate_layers_vgg,get_gram_mean_features



def Var_of_featuresMaps():
    """
    In this function we will compute the Gram Matrix for two subsets
    a small part of ImageNet validation set 
    Paintings datasets
    """

    dataset_tab = ['ImageNet','Paintings','watercolor','IconArt_v1']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 


    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

    num_style_layers = len(style_layers)
    # Load the VGG model
    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    
    # Get the covairances matrixes and the means
    number_im_considered = 1000
    # 6000 images pour IconArt
    # Un peu moins de 8700 images pour ArtUK
    # On devrait faire un test à 10000 
    
    if number_im_considered >= 10:
        if not(np.isinf(number_im_considered)):
            itera = number_im_considered//10
        else:
            itera =1000
    else:
        itera=1
    dict_of_dict = {}
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for dataset in dataset_tab:
        print('===',dataset,'===')
        filename = dataset + '_' + str(number_im_considered) + '_CovMean.pkl'
        filename_path= os.path.join(output_path,filename)
        if not os.path.isfile(filename_path):
            print('We will compute features')
            if dataset == 'ImageNet':
                ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD2','data','IMAGENET','val')
                list_imgs = glob.glob(os.path.join(ImageNet_val_path,'*.JPEG'))
            elif dataset == 'Paintings':
                images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
                list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
            elif dataset == 'watercolor':
                images_path = os.path.join(os.sep,'media','gonthier','HDD','data','cross-domain-detection','datasets','watercolor','JPEGImages')
                list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
            elif dataset == 'IconArt_v1':
                images_path = os.path.join(os.sep,'media','gonthier','HDD','data','Wikidata_Paintings','IconArt_v1','JPEGImages')
                list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
            print('Number of images :',len(list_imgs))
            dict_output = {}
            dict_var = {}
            for l,layer in enumerate(style_layers):
                dict_var[layer] = []
            for i,image_path in enumerate(list_imgs):
                if not(number_im_considered is None) and i < number_im_considered:
                    if i%itera==0: print(i,image_path)
                    head, tail = os.path.split(image_path)
                    short_name = '.'.join(tail.split('.')[0:-1])
                    try:
                        vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                    except IndexError as e:
                        print(e)
                        print(i,image_path)
                        raise(e)
                    dict_output[short_name] = vgg_cov_mean
                    for l,layer in enumerate(style_layers):
                        [cov,mean] = vgg_cov_mean[l]
                        # Here we only get a tensor we need to run the session !!! 
                        dict_var[layer] += [np.diag(cov)]
                else:
                    continue
            for l,layer in enumerate(style_layers):
                stacked = np.stack(dict_var[layer]) 
                dict_var[layer] =   stacked
            dict_of_dict[dataset] = dict_var
            
            # Save data
            with open(filename_path, 'wb') as handle:
                pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('We will load the features ')
            with open(filename_path, 'rb') as handle:
               dict_output = pickle.load(handle)
            dict_var = {}
            for l,layer in enumerate(style_layers):
                dict_var[layer] = []
            for elt in dict_output.keys():
               vgg_cov_mean =  dict_output[elt]
               for l,layer in enumerate(style_layers):
                    [cov,mean] = vgg_cov_mean[l]
                    # Here we only get a tensor we need to run the session !!! 
                    dict_var[layer] += [np.diag(cov)]
            for l,layer in enumerate(style_layers):
                stacked = np.stack(dict_var[layer]) 
                dict_var[layer] = stacked
            dict_of_dict[dataset] = dict_var
            
    sess.close()
    
    print('Start plotting')
    # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
    pltname = 'Hist_of_Var_fm_'
    labels = []
    for dataset in dataset_tab:
        pltname +=  dataset+'_'
        if dataset == 'ImageNet':
            labels += ['ImNet']
        elif dataset == 'Paintings':
            labels += ['ArtUK']
        elif dataset == 'watercolor':
            labels += ['w2k']
        elif dataset == 'IconArt_v1':
            labels += ['icon']
    pltname +=  str(number_im_considered)
        
    pltname +='.pdf'
    pltname= os.path.join(output_path,pltname)
    pp = PdfPages(pltname)
    
    alpha=0.7
    n_bins = 100
    colors_full = ['red','green','blue','purple','orange']
    colors = colors_full[0:len(dataset_tab)]
    
#    style_layers = [style_layers[0]]
    
    # Turn interactive plotting off
    plt.ioff()
    
    for l,layer in enumerate(style_layers):
        print("Layer",layer)
        tab_vars = []
        for dataset in dataset_tab: 
            vars_ = dict_of_dict[dataset][layer] 
            num_images,num_features = vars_.shape
            tab_vars +=[vars_]
 
        number_img_w = 4
        number_img_h= 4
        num_pages = num_features//(number_img_w*number_img_h)
        for p in range(num_pages):
            #f = plt.figure()  # Do I need this ?
            axes = []
            gs00 = gridspec.GridSpec(number_img_h, number_img_w)
            for j in range(number_img_w*number_img_h):
                ax = plt.subplot(gs00[j])
                axes += [ax]
            for k,ax in enumerate(axes):
                f_k = k + p*number_img_w*number_img_h
                xtab = []
                for l in range(len(dataset_tab)):
#                    x = np.vstack([tab_vars[0][:,f_k],tab_vars[1][:,f_k]])# Each line is a dataset 
#                    x = x.reshape((-1,2))
                    vars_values = tab_vars[l][:,f_k].reshape((-1,))
                    print(l,vars_values[0:10])
                    xtab += [vars_values]
                im = ax.hist(xtab,n_bins, density=True, histtype='step',color=colors,\
                             stacked=False,alpha=alpha,label=labels)
                ax.tick_params(axis='both', which='major', labelsize=3)
                ax.tick_params(axis='both', which='minor', labelsize=3)
                ax.legend(loc='upper right', prop={'size': 2})
            titre = layer +' ' +str(p)
            plt.suptitle(titre)
            
            #gs0.tight_layout(f)
            plt.savefig(pp, format='pdf')
            plt.close()
    pp.close()
    plt.clf()
    
if __name__ == '__main__':         
    Var_of_featuresMaps()
                    
        