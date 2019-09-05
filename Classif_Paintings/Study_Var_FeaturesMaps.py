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

from Stats_Fcts import get_intermediate_layers_vgg,get_gram_mean_features



def Var_of_featuresMaps():
    """
    In this function we will compute the Gram Matrix for two subsets
    a small part of ImageNet validation set 
    Paintings datasets
    """

    dataset_tab = ['ImageNet','Paintings']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')

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
    number_im_considered = 2
    dict_of_dict = {}
    for dataset in dataset_tab:
        
        if dataset == 'ImageNet':
            ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD2','data','IMAGENET','val')
            list_imgs = glob.glob(os.path.join(ImageNet_val_path,'*.JPEG'))
        elif dataset == 'Paintings':
            ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD','data','Painting_Dataset')
            list_imgs = glob.glob(os.path.join(ImageNet_val_path,'*.jpg'))
            #TODO
    
        dict_output = {}
        dict_var = {}
        for l,layer in enumerate(style_layers):
            dict_var[layer] = []
        for i,image_path in enumerate(list_imgs):
            if not(number_im_considered is None) and i < number_im_considered:
                head, tail = os.path.split(image_path)
                short_name = '.'.join(tail.split('.')[0:-1])
                vgg_cov_mean = get_gram_mean_features(vgg_inter,image_path)
                dict_output[short_name] = vgg_cov_mean
                for l,layer in enumerate(style_layers):
                    [cov,mean] = vgg_cov_mean[l]
                    # Here we only get a tensor we need to run the session !!! 
                    print(cov)
                    dict_var[layer] += [np.diag(cov)]
            else:
                continue
        for l,layer in enumerate(style_layers):
            dict_var[layer] = np.stack(dict_var[layer])   
        dict_of_dict[dataset] = dict_var
        
        # Save data
        filename = dataset + '_' + str(number_im_considered) + '_CovMean.pkl'
        filename_path= os.path.join(output_path,filename)
        with open(filename_path, 'wb') as handle:
            pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
    pltname = 'Hist_of_Var_fm_'
    for dataset in ['ImageNet','Paintings']:
        pltname +=  dataset+'_'
        
    pltname +='.pdf'
    pp = PdfPages(pltname)
    
    alpha=0.7
    n_bins = 10
    colors = ['red', 'green']
    tab_vars = []
    for l,layer in enumerate(style_layers):
        tab_vars += []
        for database in dataset_tab:        
            vars_ = dict_of_dict[dataset][layer] 
            tab_vars +=[vars_]
            num_features,_ = vars_.shape
        
        f = plt.figure()
        gs0 = gridspec.GridSpec(1,3, width_ratios=[0.05,4,4]) # 2 columns
        axcm = plt.subplot(gs0[0])     
        number_img_large = 4
        gs00 = gridspec.GridSpecFromSubplotSpec(num_features//number_img_large, number_img_large, subplot_spec=gs0[1])
        axes = []
        for j in range(num_features):
            ax = plt.subplot(gs00[j])
            axes += [ax]
        for l,ax in enumerate(axes):
            x = np.stack(tab_vars[0][:,l],tab_vars[1][:,l])# Each line is a dataset 
            im = ax.hist(x,n_bins, density=True, histtype='bar',color=colors, stacked=True,alpha=alpha)
#            ax.axis('off')

        titre = layer
        plt.suptitle(titre)
        #gs0.tight_layout(f)
        plt.savefig(pp, format='pdf')
        plt.close()
    pp.close()
    plt.clf()
    
if __name__ == '__main__':         
    Var_of_featuresMaps()
                    
        