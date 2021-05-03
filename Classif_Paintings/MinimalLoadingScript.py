#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:48:03 2019

@author: gonthier
"""

import pickle
import os
import h5py
import numpy as np

keras_vgg_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
                'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
                'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
                'block5_conv1','block5_conv2','block5_conv3','block5_conv4',
                'block5_pool','flatten','fc1','fc2','predictions']

def numeral_layers_index(style_layers):
    string = ''
    for elt in style_layers:
        string+= str(keras_vgg_layers.index(elt))
    return(string)

def load_precomputed_mean_cov(filename_path,style_layers,dataset,saveformat='h5',
                              whatToload='var'):
    """
    @param : whatToload mention what you want to load by default only return variances
    """
    
    if not(whatToload in ['var','cov','mean','covmean','varmean','all','']):
        print(whatToload,'is not known')
        raise(NotImplementedError)
    dict_var = {}
    for l,layer in enumerate(style_layers):
        dict_var[layer] = []
    if saveformat=='pkl':
        with open(filename_path, 'rb') as handle:
           dict_output = pickle.load(handle)
        for elt in dict_output.keys():
           vgg_cov_mean =  dict_output[elt]
           for l,layer in enumerate(style_layers):
                [cov,mean] = vgg_cov_mean[l]
                cov = cov[0,:,:]
                mean = mean[0,:]
                if whatToload=='var':
                    dict_var[layer] += [np.diag(cov)]
                elif whatToload=='cov':
                    dict_var[layer] += [cov]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload in ['covmean','','all']:
                    dict_var[layer] += [[cov,mean]]
                elif whatToload=='varmean':
                    dict_var[layer] += [[np.diag(cov),mean]]
        for l,layer in enumerate(style_layers):
            stacked = np.stack(dict_var[layer]) 
            dict_var[layer] = stacked
    if saveformat=='h5':
        store = h5py.File(filename_path, 'r')
        for elt in store.keys():
            vgg_cov_mean =  store[elt]
            for l,layer in enumerate(style_layers):
                if 'cov' in whatToload or 'var' in whatToload or whatToload=='' or whatToload=='all':
                    cov_str = layer + '_cov'
                    cov = vgg_cov_mean[cov_str] # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    cov = cov[0,:,:]
                if 'mean' in whatToload:
                    mean_str = layer + '_mean'
                    mean = vgg_cov_mean[mean_str] # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    mean = mean[0,:]
                if whatToload=='var':
                    dict_var[layer] += [np.diag(cov)]
                elif whatToload=='cov':
                    dict_var[layer] += [cov]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload in  ['covmean','','all']:
                    dict_var[layer] += [[cov,mean]]
                elif whatToload=='varmean':
                    dict_var[layer] += [[np.diag(cov),mean]]
        for l,layer in enumerate(style_layers):
            stacked = np.stack(dict_var[layer]) 
            dict_var[layer] = stacked
        store.close()
    return(dict_var)
    
    
def LoadAllDatasets(whatToload= 'var',data_path='data'):
    """
    @param whatToload= 'var' or 'mean'
    @param data_path : path to the data
    """
    
    style_layers = ['block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
           ]
    getBeforeReLU = True
    saveformat ='h5'
    dataset_tab = ['ImageNetTrain','ImageNetTest','ImageNet','Paintings','watercolor','IconArt_v1']
    number_im_considered_tab = [10000,10000,10000,None,None,None]
    dict_of_dict = {}
    for dataset,number_im_considered in zip(dataset_tab,number_im_considered_tab):
        print('===',dataset,'===')
        str_layers = numeral_layers_index(style_layers)
        filename = dataset + '_' + str(number_im_considered) + '_CovMean'+\
            '_'+str_layers
        if not(set=='' or set is None):
            filename += '_'+set
        if getBeforeReLU:
            filename += '_BeforeReLU'
        if saveformat=='pkl':
            filename += '.pkl'
        if saveformat=='h5':
            filename += '.h5'
        filename_path= os.path.join(data_path,filename)
        dict_stats =load_precomputed_mean_cov(filename_path,style_layers,dataset,
                                                    saveformat=saveformat,whatToload=whatToload)
        dict_of_dict[dataset] = dict_stats
    
    for l,layer in enumerate(style_layers):
        print("Layer",layer)
        tab_vars = []
        for dataset in dataset_tab: 
            stats_ = dict_of_dict[dataset][layer] 
            # The statistics per dataset and per layer 
            # each line is a different image and each column a different feature
            num_images,num_features = stats_.shape
            print(dataset,layer,'num_images,num_features ',num_images,num_features )
            tab_vars +=[stats_]
            
    return(dict_of_dict)
            
