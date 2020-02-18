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
from scipy.stats import kurtosis,skew
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import pathlib
#from pandas import HDFStore
#import pandas as pd

import h5py

import tensorflow as tf
from IMDB import get_database
from Stats_Fcts import get_intermediate_layers_vgg,get_gram_mean_features,\
    load_resize_and_process_img,get_VGGmodel_gram_mean_features,get_BaseNorm_gram_mean_features,\
    get_ResNet_ROWD_gram_mean_features,get_VGGmodel_4Param_features,get_VGGmodel_features,\
    get_cov_mean_of_InputImages,get_those_layers_output
from keras_resnet_utils import getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion
from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator
from OnlineHistogram import NumericHistogram
### To copy only the image from the dataset
#dataset='watercolor'
#item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
#path_data,Not_on_NicolasPC = get_database(dataset)
#images_in_set = df_label[item_name].values
#import shutil
#for name_img in images_in_set:
#    src = os.path.join(os.sep,'media','gonthier','HDD','data','cross-domain-detection','datasets','watercolor','JPEGImagesAll',name_img+'.jpg')
#    dst = os.path.join(images_path,name_img+'.jpg')
#    shutil.copyfile(src, dst)

keras_vgg_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
                'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
                'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
                'block5_conv1','block5_conv2','block5_conv3','block5_conv4',
                'block5_pool','flatten','fc1','fc2','predictions']

def numeral_layers_index(style_layers):
    """For VGG 19"""
    string = ''
    for elt in style_layers:
        string+= str(keras_vgg_layers.index(elt))+'_'
    return(string)
    
def numeral_layers_index_bitsVersion(style_layers):
    """For VGG 19"""
    
    list_bool = [False]*len(keras_vgg_layers)
    for elt in style_layers:
        try:
            list_bool[keras_vgg_layers.index(elt)] = True
        except ValueError as e:
            print(e)
    string = 'BV'+ str(int(''.join(['1' if i else '0' for i in list_bool]), 2)) # Convert the boolean version of index list to int
    return(string)

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    else:
        first = collection[0]
        for smaller in partition(collection[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
            # put `first` in its own subset 
            yield [ [ first ] ] + smaller

def get_partition(collection):
    l =  []
    for p in partition(collection):
        l += p
    return(l)



def get_list_im(dataset,set=''):
    """
    Returns the list of images and the number of images
    """    
    if dataset == 'ImageNet': # It is the validation set
        ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD2','data','IMAGENET','val')
        list_imgs = glob.glob(os.path.join(ImageNet_val_path,'*.JPEG'))
    if dataset == 'ImageNetTest':
        ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD2','data','IMAGENET','test')
        list_imgs = glob.glob(os.path.join(ImageNet_val_path,'*.JPEG'))
    if dataset == 'ImageNetTrain':
        ImageNet_val_path = os.path.join(os.sep,'media','gonthier','HDD2','data','IMAGENET','train')
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
    elif dataset == 'T':
        images_path = os.path.join(os.sep,'media','gonthier','HDD','data','T')
        list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
    elif dataset == 'OIV5':
        images_path = os.path.join(os.sep,'media','gonthier','HDD2','data','OIV5','Images')
        list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
    # Attention si jamais tu fais pour les autres bases il faut verifier que tu n'as que les images du datasets dans le dossier en question
    if not(set is None or set==''):
        if dataset in ['ImageNet','OIV5','ImageNetTest','ImageNetTrain']:
            print('Sorry we do not have the splitting information on ',dataset)
            raise(NotImplementedError)
        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset)
        if set=='trainval' or set=='trainvalidation':
            images_in_set = np.concatenate([df_label[df_label['set']=='train'][item_name].values,df_label[df_label['set']==str_val][item_name].values])
        else:
            images_in_set = df_label[df_label['set']==set][item_name].values
    else:
        images_in_set = None
    
    if dataset=='RASTA':
        if not(images_in_set is None):
            images_in_set_tmp = []
            for elt in images_in_set:
                elt_split = elt.split('/')[-1]
                images_in_set_tmp += [elt_split]
            images_in_set = images_in_set_tmp
        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset)
        list_imgs = df_label[item_name].values
        list_imgs =path_to_img +'/'+ list_imgs +'.jpg'
        list_imgs = list(list_imgs)       

    number_im_list = len(list_imgs)
    return(list_imgs,images_in_set,number_im_list)
 
def get_four_moments(vars_values):
    m = np.mean(vars_values)
    v = np.var(vars_values)
    s = skew(vars_values)
    k = kurtosis(vars_values)
    return(m,v,s,k)
    
def get_four_momentsAxis(vars_values):
    m = np.mean(vars_values,axis=(0,1))
    v = np.var(vars_values,axis=(0,1))
    s = skew(vars_values.reshape(-1,len(m)),axis=0)
    k = kurtosis(vars_values.reshape(-1,len(m)),axis=0)
    return(m,v,s,k)
    
def Precompute_Cumulated_Hist_4Moments(filename_path,model_toUse,Net,list_of_concern_layers,number_im_considered,\
                        dataset,set,saveformat='h5',cropCenter=True,histoFixe=True,bins_tab=None):
    """
    In this function we compute a cumulated histogram of the value of the features maps but also 
    the 4 first moments of them for the different layers involved for a given ne on a given dataset
    @param model_toUse : the model used for extracting data
    @param : Net : VGG or ResNet etc name in string
    @param : if histoFixe == True we will use a fixe bins histogram (bins argument)
            can be use only for the first layer of the ResNet
            otherwise we will use an adapatative bins algorithm
    @param : bins the bins used 
    """

    number_of_bins = 1000
    if saveformat=='h5':
        # Create a storage file where data is to be stored
        store = h5py.File(filename_path, 'a')
    else:
        raise(NotImplementedError)
    print('We will compute statistics and histogram for ',Net)
    list_imgs,images_in_set,number_im_list = get_list_im(dataset,set=set)
    
    # 6000 images pour IconArt
    # Un peu moins de 8700 images pour ArtUK
    # On devrait faire un test à 10000 
    if not(number_im_considered is None):
        if number_im_considered >= 10:
            if not(np.isinf(number_im_considered)):
                itera = number_im_considered//10
            else:
                itera =1000
        else:
            itera=1
    else:
        itera=1000
    print('Number of images :',number_im_list)
    if not(number_im_considered is None):
        if number_im_considered >= number_im_list:
            number_im_considered =None
    dict_output = {}
    dict_var = {}

    dict_histo = {}
    dict_num_f = {}
    
    for l,layer in enumerate(list_of_concern_layers):
        dict_var[layer] = []
        dict_histo[layer] = {}
        
    # Get only the concern layers : list_of_concern_layers
    model = get_those_layers_output(model_toUse,list_of_concern_layers)
        
    firstIm = True
    for i,image_path in enumerate(list_imgs):
        if number_im_considered is None or i < number_im_considered:
            if i%itera==0: print(i,image_path)
            head, tail = os.path.split(image_path)
            short_name = '.'.join(tail.split('.')[0:-1])
            if not(set is None or set==''):
                if not(short_name in images_in_set):
                    # The image is not in the set considered
                    continue
            # Get the covairances matrixes and the means
            try:
                #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                if cropCenter:
                    image_array= load_and_crop_img(path=image_path,Net=Net,target_size=224,
                                            crop_size=224,interpolation='lanczos:center')
                          # For VGG or ResNet size == 224
                else:
                    image_array = load_resize_and_process_img(image_path,Net=Net)
                features_tab = model.predict(image_array, batch_size=1) 
               
            except IndexError as e:
                print(e)
                print(i,image_path)
                raise(e)
            
            if saveformat=='h5':
                grp = store.create_group(short_name)
                for l,z in enumerate(zip(list_of_concern_layers,bins_tab)):
                    layer,bins = z
                    features_l = features_tab[l][0]
                    
                    
                    # Compute the statistics 
                    #print(layer,features_l.shape)
                    num_features = features_l.shape[-1]

                    mean,var,skew,kurt = get_four_momentsAxis(features_l)
                    
                    if histoFixe:
                        if firstIm: # Premiere image
                            dict_num_f[layer] = num_features
                            for k in range(num_features):
                                f_k = features_l[:,:,k].reshape((-1,))
                                hist, bin_edges = np.histogram(f_k, bins=bins, density=False) 
                                dict_histo[layer][k] = hist
                                #print(hist)
                            if len(list_of_concern_layers)==l+1:
                                firstIm = False
                        else:
                            for k in range(num_features):
                                f_k = features_l[:,:,k].reshape((-1,))
                                hist, bin_edges = np.histogram(f_k, bins=bins, density=False) 
                                dict_histo[layer][k] += hist
                    else:
                    
                        # Update the cumulated histogram : with an adaptative bins computation
                        if i==0: # Premiere image
                            dict_num_f[layer] = num_features
                            for k in range(num_features): # Number of canal
                                hist = NumericHistogram()
                                hist.allocate(number_of_bins)
                                dict_histo[layer][k] = hist
                        
                        for k in range(num_features):   
                            hist = dict_histo[layer][k]
                            for elt in list(features_l[:,:,k].reshape((-1,))):
                                hist.add(elt)
                            dict_histo[layer][k] = hist
    
                    dict_var[layer] += [[mean,var,skew,kurt]]
                    var_str = layer + '_var'
                    mean_str = layer + '_mean'
                    skew_str = layer + '_skew'
                    kurt_str = layer + '_kurt'
                    grp.create_dataset(mean_str,data=mean) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(var_str,data=var) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(skew_str,data=skew) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(kurt_str,data=kurt) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
#            elif saveformat=='pkl':
#                dict_output[short_name] = net_params
                
        else:
            continue
    
    # When we run on all the image
    grp = store.create_group('Histo')
    for l,layer in enumerate(list_of_concern_layers):
        num_features = dict_num_f[layer]
        grp.create_dataset(layer,data=[int(num_features)])
        for k in range(num_features):   
            hist = dict_histo[layer][k]
            l_str = layer+'_hist_'+str(k)
            grp.create_dataset(l_str,data=hist)
    
    # Save data
    if saveformat=='pkl':
        with open(filename_path, 'wb') as handle:
            pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if saveformat=='h5':
        store.close()
    return(dict_var,dict_histo,dict_num_f)
    
    
    
def Precompute_Mean_Cov(filename_path,style_layers,number_im_considered,\
                        dataset='ImageNet',set='',saveformat='h5',whatToload='var',\
                        getBeforeReLU=False,Net='VGG',style_layers_imposed=[],\
                        list_mean_and_std_source=[],list_mean_and_std_target=[],\
                        cropCenter=False,sizeIm=224):
    """
    In this function we precompute the mean and cov for certain dataset
    @param : whatToload mention what you want to load by default only return variances
    """
    if not(whatToload in ['var','cov','mean','covmean','varmean','all','']):
        print(whatToload,'is not known')
        raise(NotImplementedError)
    
    if saveformat=='h5':
        # Create a storage file where data is to be stored
        store = h5py.File(filename_path, 'a')
    print('We will compute features')
    list_imgs,images_in_set,number_im_list = get_list_im(dataset,set=set)
    
    # 6000 images pour IconArt
    # Un peu moins de 8700 images pour ArtUK
    # On devrait faire un test à 10000 
    if not(number_im_considered is None):
        if number_im_considered >= 10:
            if not(np.isinf(number_im_considered)):
                itera = number_im_considered//10
            else:
                itera =1000
        else:
            itera=1
    else:
        itera=1000
    print('Number of images :',number_im_list)
    if not(number_im_considered is None):
        if number_im_considered >= number_im_list:
            number_im_considered =None
    dict_output = {}
    dict_var = {}
    
    if not(sizeIm==224) and not(Net=='VGG'):
        print('/!\ I did not implement the gram matrices computation for a size different from 224*244 for the moment !')
        raise(NotImplementedError)
    
    if Net=='VGG':
        net_get_cov =  get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=getBeforeReLU)
        # Don t need sizeIm because we don t load the head of the model
    elif Net=='VGGBaseNorm' or Net=='VGGBaseNormCoherent':
        style_layers_exported = style_layers
        net_get_cov = get_BaseNorm_gram_mean_features(style_layers_exported,\
                        style_layers_imposed,list_mean_and_std_source,list_mean_and_std_target,\
                        getBeforeReLU=getBeforeReLU)
    elif Net=='ResNet50_ROWD': # Base coherent here also but only update the batch normalisation
        style_layers_exported = style_layers
        net_get_cov = get_ResNet_ROWD_gram_mean_features(style_layers_exported,style_layers_imposed,\
                                    list_mean_and_std_target,transformOnFinalLayer=None,
                                    res_num_layers=50,weights='imagenet')
        
    elif Net=='Input': # In this case we only compute mean and covariance of RGB channel
        net_get_cov = get_cov_mean_of_InputImages()
        style_layers = ['input']
    else:
        print(Net,'is inknown')
        raise(NotImplementedError)
    
    for l,layer in enumerate(style_layers):
        dict_var[layer] = []
    for i,image_path in enumerate(list_imgs):
        if number_im_considered is None or i < number_im_considered:
            if i%itera==0: print(i,image_path)
            head, tail = os.path.split(image_path)
            short_name = '.'.join(tail.split('.')[0:-1])
            if not(set is None or set==''):
                if not(short_name in images_in_set):
                    # The image is not in the set considered
                    continue
            # Get the covairances matrixes and the means
            try:
                #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                if cropCenter:
                    image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                            crop_size=sizeIm,interpolation='lanczos:center')
                    # For VGG or ResNet with classification head size == 224
                else:
                    image_array = load_resize_and_process_img(image_path,Net=Net,max_dim=sizeIm)
                net_cov_mean = net_get_cov.predict(image_array, batch_size=1)
            except IndexError as e:
                print(e)
                print(i,image_path)
                raise(e)
            
            if saveformat=='h5':
                grp = store.create_group(short_name)
                for l,layer in enumerate(style_layers):
                    cov = net_cov_mean[2*l]
                    mean = net_cov_mean[2*l+1]
                    cov_str = layer + '_cov'
                    mean_str = layer + '_mean'
                    grp.create_dataset(cov_str,data=cov) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(mean_str,data=mean) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
            elif saveformat=='pkl':
                dict_output[short_name] = net_cov_mean
                
            for l,layer in enumerate(style_layers):
#                        [cov,mean] = vgg_cov_mean[l]
                cov = net_cov_mean[2*l][0,:,:]
                mean = net_cov_mean[2*l+1][0,:] # car batch size == 1
                # Here we only get a tensor we need to run the session !!! 
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
        else:
            continue
    for l,layer in enumerate(style_layers):
        stacked = np.stack(dict_var[layer]) 
        dict_var[layer] = stacked
    
    # Save data
    if saveformat=='pkl':
        with open(filename_path, 'wb') as handle:
            pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if saveformat=='h5':
        store.close()
    return(dict_var)
    
def Precompute_4Param(filename_path,style_layers,number_im_considered,\
                        dataset='ImageNet',set='',saveformat='h5',whatToload='var',\
                        getBeforeReLU=False,Net='VGG',style_layers_imposed=[],\
                        list_mean_and_std_source=[],list_mean_and_std_target=[],cropCenter=False):
    """
    In this function we precompute the mean and cov for certain dataset
    @param : whatToload mention what you want to load by default only return variances
    """
    if not(whatToload in ['var','kurt','mean','skew','all','']):
        print(whatToload,'is not known')
        raise(NotImplementedError)
    
    if saveformat=='h5':
        # Create a storage file where data is to be stored
        store = h5py.File(filename_path, 'a')
    print('We will compute features')
    list_imgs,images_in_set,number_im_list = get_list_im(dataset,set=set)
    
    # 6000 images pour IconArt
    # Un peu moins de 8700 images pour ArtUK
    # On devrait faire un test à 10000 
    if not(number_im_considered is None):
        if number_im_considered >= 10:
            if not(np.isinf(number_im_considered)):
                itera = number_im_considered//10
            else:
                itera =1000
        else:
            itera=1
    else:
        itera=1000
    print('Number of images :',number_im_list)
    if not(number_im_considered is None):
        if number_im_considered >= number_im_list:
            number_im_considered =None
    dict_output = {}
    dict_var = {}
    
    if Net=='VGG':
        net_get_params =  get_VGGmodel_4Param_features(style_layers,getBeforeReLU=getBeforeReLU)
#    elif Net=='VGGBaseNorm' or Net=='VGGBaseNormCoherent':
#        style_layers_exported = style_layers
#        net_get_params = get_BaseNorm_gram_mean_features(style_layers_exported,\
#                        style_layers_imposed,list_mean_and_std_source,list_mean_and_std_target,\
#                        getBeforeReLU=getBeforeReLU)
#    elif Net=='ResNet50_ROWD': # Base coherent here also but only update the batch normalisation
#        style_layers_exported = style_layers
#        net_get_params = get_ResNet_ROWD_gram_mean_features(style_layers_exported,style_layers_imposed,\
#                                    list_mean_and_std_target,transformOnFinalLayer=None,
#                                    res_num_layers=50,weights='imagenet')
    else:
        print(Net,'is inknown')
        raise(NotImplementedError)
    
    for l,layer in enumerate(style_layers):
        dict_var[layer] = []
    for i,image_path in enumerate(list_imgs):
        if number_im_considered is None or i < number_im_considered:
            if i%itera==0: print(i,image_path)
            head, tail = os.path.split(image_path)
            short_name = '.'.join(tail.split('.')[0:-1])
            if not(set is None or set==''):
                if not(short_name in images_in_set):
                    # The image is not in the set considered
                    continue
            # Get the covairances matrixes and the means
            try:
                #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                if cropCenter:
                    image_array= load_and_crop_img(path=image_path,Net=Net,target_size=224,
                                            crop_size=224,interpolation='lanczos:center')
                          # For VGG or ResNet size == 224
                else:
                    image_array = load_resize_and_process_img(image_path,Net=Net)
                net_params = net_get_params.predict(image_array, batch_size=1)
            except IndexError as e:
                print(e)
                print(i,image_path)
                raise(e)
            
            if saveformat=='h5':
                grp = store.create_group(short_name)
                for l,layer in enumerate(style_layers):
                    mean = net_params[4*l]
                    var = net_params[4*l+1]
                    skew = net_params[4*l+2]
                    kurt = net_params[4*l+3]
                    var_str = layer + '_var'
                    mean_str = layer + '_mean'
                    skew_str = layer + '_skew'
                    kurt_str = layer + '_kurt'
                    grp.create_dataset(mean_str,data=mean) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(var_str,data=var) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(skew_str,data=skew) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(kurt_str,data=kurt) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
            elif saveformat=='pkl':
                dict_output[short_name] = net_params
                
            for l,layer in enumerate(style_layers):
                # car batch size == 1
                mean = net_params[4*l][0,:]
                var = net_params[4*l+1][0,:]
                skew = net_params[4*l+2][0,:]
                kurt = net_params[4*l+3][0,:]
                # Here we only get a tensor we need to run the session !!! 
                if whatToload=='var':
                    dict_var[layer] += [var]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload=='skew':
                    dict_var[layer] += [skew]
                elif whatToload=='kurt':
                    dict_var[layer] += [kurt]
                elif whatToload in ['','all']:
                    dict_var[layer] += [[mean,var,skew,kurt]]
        else:
            continue
    for l,layer in enumerate(style_layers):
        stacked = np.stack(dict_var[layer]) 
        dict_var[layer] = stacked
    
    # Save data
    if saveformat=='pkl':
        with open(filename_path, 'wb') as handle:
            pickle.dump(dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if saveformat=='h5':
        store.close()
    return(dict_var)

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
        try:
            store = h5py.File(filename_path, 'r')
        except OSError as e:
            print('OSError: Unable to open file (bad object header version number)')
            print('Trying to open :',filename_path)
            raise(e)
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
    
def load_precomputed_4Param(filename_path,style_layers,dataset,saveformat='h5',
                              whatToload='var'):
    """
    @param : whatToload mention what you want to load by default only return variances
    """
    
    if not(whatToload in ['var','kurt','mean','skew','all','']):
        print(whatToload,'is not known')
        raise(NotImplementedError)
    dict_var = {}
    for l,layer in enumerate(style_layers):
        dict_var[layer] = []
    if saveformat=='pkl':
        with open(filename_path, 'rb') as handle:
           dict_output = pickle.load(handle)
        for elt in dict_output.keys():
           net_params =  dict_output[elt]
           for l,layer in enumerate(style_layers):
                [mean,var,skew,kurt] = net_params[l]
                mean = mean[0,:]
                var = var[0,:]
                skew = skew[0,:]
                kurt = kurt[0,:]
                
                if whatToload=='var':
                    dict_var[layer] += [var]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload=='skew':
                    dict_var[layer] += [skew]
                elif whatToload=='kurt':
                    dict_var[layer] += [kurt]
                elif whatToload in ['','all']:
                    dict_var[layer] += [[mean,var,skew,kurt]]

        for l,layer in enumerate(style_layers):
            stacked = np.stack(dict_var[layer]) 
            dict_var[layer] = stacked
            
    if saveformat=='h5':
        store = h5py.File(filename_path, 'r')
        for elt in store.keys():
            net_params = store[elt]
            for l,layer in enumerate(style_layers):
                var_str = layer + '_var'
                mean_str = layer + '_mean'
                skew_str = layer + '_skew'
                kurt_str = layer + '_kurt'
                mean = net_params[mean_str]
                var = net_params[var_str]
                skew = net_params[skew_str]
                kurt = net_params[kurt_str]
                if whatToload=='var':
                    dict_var[layer] += [var]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload=='skew':
                    dict_var[layer] += [skew]
                elif whatToload=='kurt':
                    dict_var[layer] += [kurt]
                elif whatToload=='mean':
                    dict_var[layer] += [mean]
                elif whatToload in ['','all']:
                    dict_var[layer] += [[mean,var,skew,kurt]]
        for l,layer in enumerate(style_layers):
            stacked = np.vstack(dict_var[layer]) 
            dict_var[layer] = stacked
        store.close()
    return(dict_var)
    
def load_Cumulated_Hist_4Moments(filename_path,list_of_concern_layers,dataset,saveformat='h5'):
    """
    @param : whatToload mention what you want to load by default only return variances
    """
    
    dict_var = {}
    dict_histo = {}
    dict_num_f = {}
    for l,layer in enumerate(list_of_concern_layers):
        dict_var[layer] = []
#    if saveformat=='pkl':
#        with open(filename_path, 'rb') as handle:
#           dict_output = pickle.load(handle)
#        for elt in dict_output.keys():
#           net_params =  dict_output[elt]
#           for l,layer in enumerate(style_layers):
#                [mean,var,skew,kurt] = net_params[l]
#                mean = mean[0,:]
#                var = var[0,:]
#                skew = skew[0,:]
#                kurt = kurt[0,:]
#                
#                if whatToload=='var':
#                    dict_var[layer] += [var]
#                elif whatToload=='mean':
#                    dict_var[layer] += [mean]
#                elif whatToload=='skew':
#                    dict_var[layer] += [skew]
#                elif whatToload=='kurt':
#                    dict_var[layer] += [kurt]
#                elif whatToload in ['','all']:
#                    dict_var[layer] += [[mean,var,skew,kurt]]
#
#        for l,layer in enumerate(style_layers):
#            stacked = np.stack(dict_var[layer]) 
#            dict_var[layer] = stacked
            
    if saveformat=='h5':
        try:
            store = h5py.File(filename_path, 'r')
            for elt in store.keys():
                if not(elt=='Histo'):
                    net_params = store[elt]
                    for l,layer in enumerate(list_of_concern_layers):
                        var_str = layer + '_var'
                        mean_str = layer + '_mean'
                        skew_str = layer + '_skew'
                        kurt_str = layer + '_kurt'
                        mean = np.array(net_params[mean_str])
                        var = np.array(net_params[var_str])
                        skew = np.array(net_params[skew_str])
                        kurt = np.array(net_params[kurt_str])
                        dict_var[layer] += [[mean,var,skew,kurt]]
            for l,layer in enumerate(list_of_concern_layers):
                stacked = np.vstack(dict_var[layer]) 
                dict_var[layer] = stacked
                
            grp = store['Histo']
            for l,layer in enumerate(list_of_concern_layers):
                dict_histo[layer] = {}
                num_features = grp[layer][0]
                dict_num_f[layer] = num_features
                for k in range(num_features):   
                    l_str = layer+'_hist_'+str(k)
                    hist = grp[l_str]
                    dict_histo[layer][k] = np.array(hist)
                
            store.close()
        except OSError as e:
            print('Unable to open :',filename_path)
            raise(e)
    return(dict_var,dict_histo,dict_num_f)

def get_dict_stats(source_dataset,number_im_considered,style_layers,\
                   whatToload,saveformat='h5',set='',getBeforeReLU=False,\
                   Net='VGG',style_layers_imposed=[],\
                   list_mean_and_std_source=[],list_mean_and_std_target=[],\
                   cropCenter=False,BV=True,sizeIm=224):
    if 'VGG' in Net:
        if BV:
            str_layers = numeral_layers_index(style_layers)
        else:
            str_layers = numeral_layers_index_bitsVersion(style_layers)
    elif 'ResNet50' in Net:
        if BV:
            str_layers = getResNetLayersNumeral_bitsVersion(style_layers,num_layers=50)
        else:
            str_layers = getResNetLayersNumeral(style_layers,num_layers=50)
    else:
        raise(NotImplementedError)
    filename = source_dataset + '_' + str(number_im_considered) + '_CovMean'+'_'+str_layers
    
    if not(sizeIm==224):
        filename += '_ImSize'+str(sizeIm)
    
    if not(Net=='VGG'):
        if 'VGG' in Net:
            if BV:
                filename += '_'+Net + numeral_layers_index_bitsVersion(style_layers_imposed)
            else:
                filename += '_'+Net + numeral_layers_index(style_layers_imposed)
        elif 'ResNet50' in Net:
            if BV:
                filename += '_'+Net + getResNetLayersNumeral_bitsVersion(style_layers_imposed,num_layers=50)
            else:
                filename += '_'+Net + getResNetLayersNumeral(style_layers_imposed,num_layers=50)
        else:
            raise(NotImplementedError)
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp')
    
    if os.path.isdir(output_path):
        output_path_full = os.path.join(output_path,'Covdata')
    else:
        output_path_full = os.path.join('data','Covdata')
    pathlib.Path(output_path_full).mkdir(parents=True, exist_ok=True)  
    if not(set=='' or set is None):
        filename += '_'+set
    if getBeforeReLU:
        filename += '_BeforeReLU'
    if cropCenter:
        filename += '_cropCenter'
    if saveformat=='pkl':
        filename += '.pkl'
    if saveformat=='h5':
        filename += '.h5'
    filename_path= os.path.join(output_path_full,filename)
    if not os.path.isfile(filename_path):
        dict_stats = Precompute_Mean_Cov(filename_path,style_layers,number_im_considered,\
                                       dataset=source_dataset,set=set,saveformat=saveformat,\
                                       whatToload=whatToload,Net=Net,style_layers_imposed=style_layers_imposed,\
                                       list_mean_and_std_source=list_mean_and_std_source,\
                                       list_mean_and_std_target=list_mean_and_std_target,\
                                       cropCenter=cropCenter,sizeIm=sizeIm)
    else:
        dict_stats = load_precomputed_mean_cov(filename_path,style_layers,source_dataset,\
                                            saveformat=saveformat,whatToload=whatToload)
    return(dict_stats)
    

def ResNetComparison_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered = np.inf,dataset_tab=None
                        ,getBeforeReLU=False,printoutput='Var',cropCenter=False,BV=False,
                        Net = 'VGG'):
    """
    In this function we will compute the first or second moments of ResNet net
    for different subsets such as a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    
    Fonction non finie non testee
    
    """
    if not(printoutput in get_partition(['Mean','Var'])):
        print(printoutput,"is unknown. It must be 'Var' or 'Mean' or this of those two terms.")
        raise(NotImplementedError)
    if type(printoutput)==list:
        list_printoutput = printoutput
    else:
        list_printoutput = [printoutput]

    if dataset_tab is None:
        dataset_tab = ['ImageNet','Paintings','watercolor','IconArt_v1','OIV5']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    
    if Net=='VGG':
        style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
    elif Net=='ResNet50':
        style_layers = ['conv1',
                        'bn_conv1',
                        'activation']

#    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    
    set = None
    
    dict_of_dict = {}
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    for printoutput in list_printoutput:
        if printoutput=='Var':
            whatToload= 'var'
        elif printoutput=='Mean':
            whatToload= 'mean' 
        for dataset in dataset_tab:
            print('===',dataset,'===')
            list_imgs,images_in_set,number_im_list = get_list_im(dataset,set='')
            if not(number_im_considered is None):
                if number_im_considered >= number_im_list:
                    number_im_considered_tmp =None
                else:
                    number_im_considered_tmp=number_im_considered
            if BV:
                str_layers = numeral_layers_index_bitsVersion(style_layers)
            else:
                str_layers = numeral_layers_index(style_layers)
            filename = dataset + '_' + str(number_im_considered_tmp) + '_CovMean'+\
                '_'+str_layers
            if not(set=='' or set is None):
                filename += '_'+set
            if getBeforeReLU:
                filename += '_BeforeReLU'
            if cropCenter:
                filename += '_cropCenter'
            if saveformat=='pkl':
                filename += '.pkl'
            if saveformat=='h5':
                filename += '.h5'
            filename_path= os.path.join(output_path,filename)
            
            if not os.path.isfile(filename_path):
                dict_var = Precompute_Mean_Cov(filename_path,style_layers,number_im_considered_tmp,\
                                               dataset=dataset,set=set,saveformat=saveformat,
                                               whatToload=whatToload,getBeforeReLU=getBeforeReLU,cropCenter=cropCenter)
                dict_of_dict[dataset] = dict_var
            else:
                print('We will load the features ')
                dict_var =load_precomputed_mean_cov(filename_path,style_layers,dataset,
                                                    saveformat=saveformat,whatToload=whatToload)
                dict_of_dict[dataset] = dict_var
    
    
        print('Start plotting ',printoutput)
        # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
        pltname = 'Hist_of_'+printoutput+'_fm_'
        labels = []
        for dataset in dataset_tab:
            pltname +=  dataset+'_'
            if dataset == 'ImageNet':
                labels += ['ImNet']
            if dataset == 'ImageNetTest':
                labels += ['ImNetTest']
            if dataset == 'ImageNetTrain': # Warning in this case the images are ordered 
                # So we have a semantic bias
                labels += ['ImNetTrain']
            elif dataset == 'Paintings':
                labels += ['ArtUK']
            elif dataset == 'watercolor':
                labels += ['w2k']
            elif dataset == 'IconArt_v1':
                labels += ['icon']
            elif dataset == 'OIV5':
                labels += ['OIV5']
        pltname +=  str(number_im_considered)
        if getBeforeReLU:
            pltname+= '_BeforeReLU'
        if cropCenter:
            pltname += '_cropCenter'   
        pltname +='.pdf'
        pltname= os.path.join(output_path,pltname)
        pp = PdfPages(pltname)
        
        alpha=0.7
        n_bins = 100
        colors_full = ['red','green','blue','purple','orange','pink']
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
                print('num_images,num_features ',num_images,num_features )
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
        
def Net_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered = np.inf,dataset_tab=None
                        ,getBeforeReLU=False,printoutput='Var',cropCenter=False,BV=False,
                        Net = 'VGG'):
    """
    In this function we will compute the first or second moments of VGG or ResNet networks
    for different subsets such as a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    """
    if not(printoutput in get_partition(['Mean','Var'])):
        print(printoutput,"is unknown. It must be 'Var' or 'Mean' or this of those two terms.")
        raise(NotImplementedError)
    if type(printoutput)==list:
        list_printoutput = printoutput
    else:
        list_printoutput = [printoutput]

    if dataset_tab is None:
        dataset_tab = ['ImageNet','Paintings','watercolor','IconArt_v1','OIV5']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 

    
    if Net=='VGG':
        style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
    elif Net=='ResNet50':
        style_layers = ['conv1',
                        'bn_conv1',
                        'activation']

#    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    
    set = None
    
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    for printoutput in list_printoutput:
        dict_of_dict = return_dicts_of_Var_or_Mean_for_VGGfeatures(dataset_tab,printoutput,output_path,\
           style_layers,number_im_considered,set,getBeforeReLU,cropCenter,\
           saveformat,Net,BV)
    
        print('Start plotting ',printoutput)
        # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
        pltname = 'Hist_of_'+printoutput+'_fm_'
        if not(Net=='VGG'):
            pltname+=Net+'_'
        labels = []
        for dataset in dataset_tab:
            
            pltname +=  dataset+'_'
            if dataset == 'ImageNet':
                labels += ['ImNet']
            if dataset == 'ImageNetTest':
                labels += ['ImNetTest']
            if dataset == 'ImageNetTrain': # Warning in this case the images are ordered 
                # So we have a semantic bias
                labels += ['ImNetTrain']
            elif dataset == 'Paintings':
                labels += ['ArtUK']
            elif dataset == 'watercolor':
                labels += ['w2k']
            elif dataset == 'IconArt_v1':
                labels += ['icon']
            elif dataset == 'OIV5':
                labels += ['OIV5']
        pltname +=  str(number_im_considered)
        if getBeforeReLU:
            pltname+= '_BeforeReLU'
        if cropCenter:
            pltname += '_cropCenter'   
        pltname +='.pdf'
        pltname= os.path.join(output_path,pltname)
        pp = PdfPages(pltname)
        
        alpha=0.7
        n_bins = 100
        colors_full = ['red','green','blue','purple','orange','pink']
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
                print('num_images,num_features ',num_images,num_features )
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
 
def return_dicts_of_Var_or_Mean_for_VGGfeatures(dataset_tab,printoutput,output_path,\
           style_layers,number_im_considered,set,getBeforeReLU,cropCenter,\
           saveformat,Net,BV):
    dict_of_dict = {}
    if printoutput=='Var':
        whatToload= 'var'
    elif printoutput=='Mean':
        whatToload= 'mean' 
    for dataset in dataset_tab:
        print('===',dataset,'===')
        list_imgs,images_in_set,number_im_list = get_list_im(dataset,set=set)
        if not(number_im_considered is None):
            if number_im_considered >= number_im_list:
                number_im_considered_tmp =None
            else:
                number_im_considered_tmp=number_im_considered
        else:
             number_im_considered_tmp=number_im_considered
        if BV:
            str_layers = numeral_layers_index_bitsVersion(style_layers)
        else:
            str_layers = numeral_layers_index(style_layers)
        filename = dataset + '_' + str(number_im_considered_tmp) + '_CovMean'+\
            '_'+str_layers
        if not(set=='' or set is None):
            filename += '_'+set
        if getBeforeReLU:
            filename += '_BeforeReLU'
        if cropCenter:
            filename += '_cropCenter'
        if saveformat=='pkl':
            filename += '.pkl'
        if saveformat=='h5':
            filename += '.h5'
        filename_path= os.path.join(output_path,filename)
        
        #print('filename_path',filename_path)
        
        if not os.path.isfile(filename_path):
            dict_var = Precompute_Mean_Cov(filename_path,style_layers,number_im_considered_tmp,\
                                           dataset=dataset,set=set,saveformat=saveformat,
                                           whatToload=whatToload,getBeforeReLU=getBeforeReLU,cropCenter=cropCenter,\
                                           Net=Net)
            dict_of_dict[dataset] = dict_var
        else:
            print('We will load the features ')
            dict_var =load_precomputed_mean_cov(filename_path,style_layers,dataset,
                                                saveformat=saveformat,whatToload=whatToload)
            dict_of_dict[dataset] = dict_var
    return(dict_of_dict)
       
def classSlitted_Net_Boxplots_of_featuresMaps(saveformat='h5',number_im_considered = np.inf,\
                        dataset_tab=None
                        ,getBeforeReLU=False,printoutput='Var',cropCenter=False,BV=False,
                        Net = 'VGG'):
    """
    In this function we will compute the first or second moments of VGG or ResNet networks
    for different subsets such as a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    
    Output files in SemanticDiff folder 
    
    """
    matplotlib.use('Agg') 
    ## Fonction pas du tout tester a finir !!!
    # 
    
    if not(printoutput in get_partition(['Mean','Var'])):
        print(printoutput,"is unknown. It must be 'Var' or 'Mean' or this of those two terms.")
        raise(NotImplementedError)
    if type(printoutput)==list:
        list_printoutput = printoutput
    else:
        list_printoutput = [printoutput]

    for dataset in dataset_tab:
        if 'OIV5' in dataset or 'ImageNet' in dataset:
            print(dataset,'is not implemented yet for class split plotting')
            raise(NotImplementedError)

    if dataset_tab is None:
        dataset_tab = ['Paintings','watercolor','IconArt_v1']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')
    pdf_output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','SemanticDiff')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(pdf_output_path).mkdir(parents=True, exist_ok=True) 

    
    if Net=='VGG':
        style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
    elif Net=='ResNet50':
        style_layers = ['conv1',
                        'bn_conv1',
                        'activation']

#    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    
    set = 'test'
    

#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    for printoutput in list_printoutput:
        dict_of_dict = return_dicts_of_Var_or_Mean_for_VGGfeatures(dataset_tab,printoutput,output_path,\
           style_layers,number_im_considered,set,getBeforeReLU,cropCenter,\
           saveformat,Net,BV)
    
    
        print('Start plotting ',printoutput)
        # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
        pltname = 'Class_split_Bar_of_'+printoutput+'_fm_'
        if not(Net=='VGG'):
            pltname+=Net+'_'
        labels = []
        
        NUM_COLORS = 10*len(dataset_tab)
        cm = plt.get_cmap('gist_rainbow')
        cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm) # use : color=scalarMap.to_rgba(fig_i)
        
        for dataset in dataset_tab:
            
            pltname +=  dataset+'_'
            if dataset == 'ImageNet':
                labels += ['ImNet']
            if dataset == 'ImageNetTest':
                labels += ['ImNetTest']
            if dataset == 'ImageNetTrain': # Warning in this case the images are ordered 
                # So we have a semantic bias
                labels += ['ImNetTrain']
            elif dataset == 'Paintings':
                labels += ['ArtUK']
            elif dataset == 'watercolor':
                labels += ['w2k']
            elif dataset == 'IconArt_v1':
                labels += ['icon']
            elif dataset == 'OIV5':
                labels += ['OIV5']
            elif dataset == 'RASTA':
                labels += ['RASTA']
        pltname +=  str(number_im_considered)
        if getBeforeReLU:
            pltname+= '_BeforeReLU'
        if cropCenter:
            pltname += '_cropCenter'   
        pltname +='.pdf'
        pltname= os.path.join(pdf_output_path,pltname)
        pp = PdfPages(pltname)
        
        alpha=0.7
        n_bins = 100

        
    #    style_layers = [style_layers[0]]
        
# /!\ En fatit ca serait plutot la valeur moyenne + std de la statistics ce que tu voudrais  afficher
        plt.ioff()
        fig_i = 0
        
        new_labels = []
        colors_tab = []
        FirstTime = True
        for l,layer in enumerate(style_layers):
            print("Layer",layer)
            tab_vars = []

            for dataset,label_dataset in zip(dataset_tab,labels): 
                item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
                    path_data,Not_on_NicolasPC = get_database(dataset)
                    
                if not(set is None) and not(set==''):
                    if set == 'trainval' or set=='trainvalidation':
                        df_label=df_label[df_label['set']=='train'].append(df_label[df_label['set']==str_val])
                    else:
                        df_label = df_label[df_label['set']==set]
                    
                if FirstTime:
                    if dataset=='Paintings':
                        sLength = len(df_label[item_name])
                        classes_vectors = np.zeros((sLength,num_classes))
                        for i in range(sLength):
                            for j in range(num_classes):
                                if( classes[j] in df_label['classe'][i]):
                                    classes_vectors[i,j] = 1
                                
                #print(df_label.head(5))
                vars_ = dict_of_dict[dataset][layer]
                num_images,num_features = vars_.shape
                #print('num_images,num_features ',num_images,num_features)
                for c_i,c in enumerate(classes):
                     
                    if dataset=='Paintings':
                        vars_c = vars_[classes_vectors[:,c_i]==1.0]
                    else:
                        vars_c = vars_[df_label[c]==1.0]
                    tab_vars +=[vars_c]
                        
                    #print(c,'vars_c.shape',vars_c.shape)
                    if FirstTime:
                        colors_tab += [scalarMap.to_rgba(fig_i)]
                        fig_i += 1
                        new_labels +=  [label_dataset + ' ' + c]
     
            FirstTime =  False
        
            number_img_w = 4
            number_img_h= 4
            num_pages = num_features//(number_img_w*number_img_h)
            x_pos = np.arange(len(tab_vars))
            for p in range(num_pages):
                #f = plt.figure()  # Do I need this ?
                axes = []
                gs00 = gridspec.GridSpec(number_img_h, number_img_w)
                for j in range(number_img_w*number_img_h):
                    ax = plt.subplot(gs00[j])
                    axes += [ax]
                for k,ax in enumerate(axes):
                    f_k = k + p*number_img_w*number_img_h
                    mean_of_stats = []
                    mean_of_stats = []
                    X_stats = []
                    for l in range(len(tab_vars)):
                        vars_values = tab_vars[l][:,f_k].reshape((-1,))
                        X_stats += [vars_values]
                        #mean_of_stats += [np.mean(vars_values)]
                        #std_of_stats += [np.std(vars_values)]
                        #print('number points',len(vars_values),'mean',mean_of_stats,'std :',std_of_stats)
                        #xtab += [vars_values]
                    flierprops = dict(marker='+', markersize=3,
                                      linestyle='none')
                    im = ax.boxplot(X_stats,flierprops=flierprops,notch=True,patch_artist=True,labels=new_labels)
                    # im = ax.bar(x_pos, mean_of_stats, yerr=std_of_stats, align='center', 
                    #             alpha=alpha, ecolor='black',capsize=3,color=colors_tab,label=new_labels)
                    #ax.set_xticks(x_pos)
                    #ax.set_xticklabels(new_labels)

    #                 xtab = []
    #                 for l in range(len(dataset_tab)):
    # #                    x = np.vstack([tab_vars[0][:,f_k],tab_vars[1][:,f_k]])# Each line is a dataset 
    # #                    x = x.reshape((-1,2))
    #                     vars_values = tab_vars[l][:,f_k].reshape((-1,))
    #                     xtab += [vars_values]
    #                 im = ax.hist(xtab,n_bins, density=True, histtype='step',color=colors_tab,\
    #                              stacked=False,alpha=alpha,label=new_labels)
                    ax.tick_params(axis='both', which='major', labelsize=3)
                    ax.tick_params(axis='both', which='minor', labelsize=3)
                   # ax.legend(loc='upper right', prop={'size': 2})
                titre = layer +' ' +str(p)
                plt.suptitle(titre)
                
                #gs0.tight_layout(f)
                plt.savefig(pp, format='pdf')
                plt.close()
        pp.close()
        plt.clf()
        
def VGG_Hist_of_featuresMaps(number_im_considered = 10,
                             dataset_tab=None,getBeforeReLU=True,cropCenter=True,BV=False):
    """
    In this function we will compute features of VGG net and plot theirs histogram
    for different subsets such as a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    """
    matplotlib.use('Agg') 

    if number_im_considered>10:
        print('You should use a small value for number_im_considered because we will create a pdf file of more than 45 pages per image')

    if dataset_tab is None:
        dataset_tab = ['ImageNet','Paintings','watercolor','IconArt_v1','OIV5']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',\
                               'HistOfFM')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 


    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

    vgg_inter =  get_intermediate_layers_vgg(style_layers,getBeforeReLU=getBeforeReLU) 
 
    set = None
    itera = 1
    Net = 'VGG'
    n_bins= 100
    for dataset in dataset_tab:
        print('===',dataset,'===')
        list_imgs,images_in_set,number_im_list = get_list_im(dataset,set='')
        
        for i,image_path in enumerate(list_imgs):
            if number_im_considered is None or i < number_im_considered:
                if i%itera==0: print(i,image_path)
                
                
                
                head, tail = os.path.split(image_path)
                short_name = '.'.join(tail.split('.')[0:-1])
                if not(set is None or set==''):
                    if not(short_name in images_in_set):
                        # The image is not in the set considered
                        continue
                # Get the covairances matrixes and the means
                try:
                    #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                    if cropCenter:
                        image_array= load_and_crop_img(path=image_path,Net=Net,target_size=224,
                                                crop_size=224,interpolation='lanczos:center')
                              # For VGG or ResNet size == 224
                    else:
                        image_array = load_resize_and_process_img(image_path,Net=Net)
                    vgg_features = vgg_inter.predict(image_array, batch_size=1)
                except IndexError as e:
                    print(e)
                    print(i,image_path)
                    raise(e)
                # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
                pltname = 'Hist_of_'+dataset+'_'+short_name+'_fm'
                if getBeforeReLU:
                    pltname+= '_BeforeReLU'
                if cropCenter:
                    pltname += '_cropCenter'
                pltname +='.pdf'
                pltname= os.path.join(output_path,pltname)
                pp = PdfPages(pltname)
        
                # Turn interactive plotting off
                plt.ioff()
                
                # Plot the image on the first page
                image_array_ = image_array[0,:,:,::-1] # To get an RGB image
                image_array_between01 = (image_array_ - np.min(image_array_))/(np.max(image_array_)-np.min(image_array_))
                plt.figure()        
                plt.imshow(image_array_between01)
                plt.title(short_name)
                plt.savefig(pp, format='pdf')
                plt.close()
                
                for l,layer in enumerate(style_layers):
                    print("Layer",layer)
                    features_l = vgg_features[l]
                    num_images,h,w,num_features = features_l.shape
                    print('num_images,num_features ',num_images,num_features )
             
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
                            vars_values = features_l[:,:,:,f_k].reshape((-1,))
                            m,v,s,k = get_four_moments(vars_values)
                            im = ax.hist(vars_values,n_bins, density=False, histtype='step',\
                                         stacked=False)
                            ax.tick_params(axis='both', which='major', labelsize=3)
                            ax.tick_params(axis='both', which='minor', labelsize=3)
                            #ax.legend(loc='upper right', prop={'size': 2})
                            fig_title = "m: {:.2E}, v: {:.2E}, s: {:.2E}, k: {:.2E}".format(m,v,s,k)
                            ax.set_title(fig_title, fontsize=3)
                        titre = layer +' ' +str(p)
                        plt.suptitle(titre)
                        
                        #gs0.tight_layout(f)
                        plt.savefig(pp, format='pdf')
                        plt.close()
                pp.close()
                plt.clf()
        
        
def VGG_4Param_of_featuresMaps(saveformat='h5',number_im_considered = np.inf,dataset_tab=None
                        ,getBeforeReLU=True,printoutput='Var',cropCenter=False,BV=False):
    """
    In this function we will compute the 4 first moments of VGG net
    for different subsets such as a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    """
    matplotlib.use('Agg') 
    if type(printoutput)==list:
        list_printoutput = printoutput
    else:
        list_printoutput = [printoutput]
    if not(list_printoutput in get_partition(['Mean','Var','Skewness','Kurtosis'])):
        print(list_printoutput,'is not known')
        raise(NotImplementedError)

    if dataset_tab is None:
        dataset_tab = ['ImageNet','Paintings','watercolor','IconArt_v1','OIV5']
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','4Param')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 


    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

#    num_style_layers = len(style_layers)
    # Load the VGG model
#    vgg_inter =  get_intermediate_layers_vgg(style_layers) 
    
    set = None

    dict_of_dict = {}
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    vgg_get_cov = get_VGGmodel_gram_mean_features(style_layers)
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
    for printoutput in list_printoutput:
        if printoutput=='Var':
            whatToload= 'var'
        elif printoutput=='Mean':
            whatToload= 'mean' 
        elif printoutput=='Skewness':
            whatToload= 'skew' 
        elif printoutput=='Kurtosis':
            whatToload= 'kurt' 
        else:
            print(printoutput,'is unkwnon')
            raise(NotImplementedError)
        for dataset in dataset_tab:
            print('===',dataset,'===')
            list_imgs,images_in_set,number_im_list = get_list_im(dataset,set='')
            if not(number_im_considered is None):
                if number_im_considered >= number_im_list:
                    number_im_considered_tmp =None
                else:
                    number_im_considered_tmp=number_im_considered
            if BV:
                str_layers = numeral_layers_index_bitsVersion(style_layers)
            else:
                str_layers = numeral_layers_index(style_layers)
            filename = dataset + '_' + str(number_im_considered_tmp) + '_4Param'+\
                '_'+str_layers
            if not(set=='' or set is None):
                filename += '_'+set
            if getBeforeReLU:
                filename += '_BeforeReLU'
            if cropCenter:
                filename +=  '_cropCenter' 
            if saveformat=='pkl':
                filename += '.pkl'
            if saveformat=='h5':
                filename += '.h5'
            filename_path= os.path.join(output_path,filename)
            
            if not os.path.isfile(filename_path):
                dict_var = Precompute_4Param(filename_path,style_layers,number_im_considered_tmp,\
                                               dataset=dataset,set=set,saveformat=saveformat,
                                               whatToload=whatToload,getBeforeReLU=getBeforeReLU,cropCenter=cropCenter)
                dict_of_dict[dataset] = dict_var
            else:
                print('We will load the features ')
                dict_var =load_precomputed_4Param(filename_path,style_layers,dataset,
                                                    saveformat=saveformat,whatToload=whatToload)
                dict_of_dict[dataset] = dict_var
    
    
        print('Start plotting ',printoutput)
        # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
        pltname = 'Hist_of_'+printoutput+'_fm_'
        labels = []
        for dataset in dataset_tab:
            pltname +=  dataset+'_'
            if dataset == 'ImageNet':
                labels += ['ImNet']
            if dataset == 'ImageNetTest':
                labels += ['ImNetTest']
            if dataset == 'ImageNetTrain':
                labels += ['ImNetTrain']# Warning in this case the images are ordered 
                # So we have a semantic bias
            elif dataset == 'Paintings':
                labels += ['ArtUK']
            elif dataset == 'watercolor':
                labels += ['w2k']
            elif dataset == 'IconArt_v1':
                labels += ['icon']
            elif dataset == 'OIV5':
                labels += ['OIV5']
        pltname +=  str(number_im_considered)
        if getBeforeReLU:
            pltname+= '_BeforeReLU'
        if cropCenter:
            pltname += '_cropCenter'
            
        pltname +='.pdf'
        pltname= os.path.join(output_path,pltname)
        pp = PdfPages(pltname)
        
        alpha=0.7
        n_bins = 100
        colors_full = ['red','green','blue','purple','orange','pink']
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
                print('num_images,num_features ',num_images,num_features )
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
        
        # Plot Error bar
        pltname = 'ErrorBar_of_'+printoutput+'_fm_'
        labels = []
        for dataset in dataset_tab:
            pltname +=  dataset+'_'
            if dataset == 'ImageNet':
                labels += ['ImNet']
            if dataset == 'ImageNetTest':
                labels += ['ImNetTest']
            if dataset == 'ImageNetTrain':
                labels += ['ImNetTrain']# Warning in this case the images are ordered 
                # So we have a semantic bias
            elif dataset == 'Paintings':
                labels += ['ArtUK']
            elif dataset == 'watercolor':
                labels += ['w2k']
            elif dataset == 'IconArt_v1':
                labels += ['icon']
            elif dataset == 'OIV5':
                labels += ['OIV5']
        pltname +=  str(number_im_considered)
        if getBeforeReLU:
            pltname+= '_BeforeReLU'
        if cropCenter:
            pltname += '_cropCenter'
            
        pltname +='.pdf'
        pltname= os.path.join(output_path,pltname)
        pp = PdfPages(pltname)
        
        alpha=0.7
        n_bins = 100
        colors_full = ['red','green','blue','purple','orange','pink']
        colors = colors_full[0:len(dataset_tab)]
        
        for l,layer in enumerate(style_layers):
            print("Layer",layer,"pour bar plot")
            tab_vars = []
            for dataset in dataset_tab: 
                vars_ = dict_of_dict[dataset][layer]
                num_images,num_features = vars_.shape
                print('num_images,num_features ',num_images,num_features )
                tab_vars +=[vars_]
     
            x_pos = np.arange(len(dataset_tab))
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
                    mean_of_stats = []
                    std_of_stats = []
                    for l in range(len(dataset_tab)):
                        vars_values = tab_vars[l][:,f_k].reshape((-1,))
                        mean_of_stats += [np.mean(vars_values)]
                        std_of_stats += [np.std(vars_values)]
                        #xtab += [vars_values]
#                    print(mean_of_stats)
#                    print(std_of_stats)
                    im = ax.bar(x_pos, mean_of_stats, yerr=std_of_stats, align='center', 
                                alpha=alpha, ecolor='black',capsize=3,color=colors,label=labels)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels)
                    ax.tick_params(axis='both', which='major', labelsize=3)
                    ax.tick_params(axis='both', which='minor', labelsize=3)
                    ax.yaxis.grid(True)
                    #ax.legend(loc='upper right', prop={'size': 2})
                titre = layer +' ' +str(p)
                plt.suptitle(titre)
                
                #gs0.tight_layout(f)
                plt.savefig(pp, format='pdf')
                plt.close()
        pp.close()
        plt.clf()
    
## Potentiellement pour calculer une distance entre matrices de Gram : 
# https://github.com/pymanopt/pymanopt/blob/master/pymanopt/manifolds/psd.py
    
if __name__ == '__main__':         
    #VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab=None)
    #VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab= ['ImageNet','OIV5'])
    #VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab=  ['ImageNet','Paintings','watercolor','IconArt_v1'])
#    VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =10000,
#                        dataset_tab= ['ImageNet','Paintings','watercolor','IconArt_v1'],
#                        getBeforeReLU=True,printoutput=['Mean','Var'])
#    VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =10000,
#                        dataset_tab= ['ImageNetTrain','ImageNetTest','ImageNet'],
##                        getBeforeReLU=True,printoutput=['Mean','Var'])
#    VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =10000,
#                        dataset_tab= ['ImageNet'],
#                        getBeforeReLU=True,printoutput=['Mean','Var'])
    # VGG_4Param_of_featuresMaps(saveformat='h5',number_im_considered =10000,
    #                     dataset_tab= ['ImageNetTrain','ImageNetTest','ImageNet','Paintings','watercolor','IconArt_v1'],
    #                     getBeforeReLU=True,printoutput=['Mean','Var','Skewness','Kurtosis'],cropCenter =True)
#    VGG_4Param_of_featuresMaps(saveformat='h5',number_im_considered =10000,
#                        dataset_tab= ['ImageNetTrain'],
#                        getBeforeReLU=True,printoutput=['Var'],cropCenter =True)
    #VGG_MeanAndVar_of_featuresMaps(saveformat='h5',number_im_considered =np.inf,dataset_tab=  ['ImageNet','Paintings','watercolor','IconArt_v1'])              
    # classSlitted_Net_Boxplots_of_featuresMaps(saveformat='h5',number_im_considered =None,
    #                     dataset_tab= ['Paintings'],
    #                     getBeforeReLU=True,printoutput=['Var'],cropCenter =True)
    classSlitted_Net_Boxplots_of_featuresMaps(saveformat='h5',number_im_considered =None,
                        dataset_tab= ['RASTA'],
                        getBeforeReLU=True,printoutput=['Var'],cropCenter =True)
