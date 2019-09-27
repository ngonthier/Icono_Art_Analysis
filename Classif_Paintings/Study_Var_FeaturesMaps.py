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
#from pandas import HDFStore
#import pandas as pd

import h5py

import tensorflow as tf
from IMDB import get_database
from Stats_Fcts import get_intermediate_layers_vgg,get_gram_mean_features,\
    load_crop_and_process_img,get_VGGmodel_gram_mean_features


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
    string = ''
    for elt in style_layers:
        string+= str(keras_vgg_layers.index(elt))
    return(string)


def get_list_im(dataset,set=''):
    """
    Returns the list of images and the number of images
    """    
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
    elif dataset == 'OIV5':
        images_path = os.path.join(os.sep,'media','gonthier','HDD2','data','OIV5','Images')
        list_imgs = glob.glob(os.path.join(images_path,'*.jpg'))
    # Attention si jamais tu fais pour les autres bases il faut verifier que tu n'as que les images du datasets dans le dossier en question
    if not(set is None or set==''):
        if dataset in ['ImageNet','OIV5']:
            print('Sorry we do not have the splitting information on ',dataset)
            raise(NotImplementedError)
        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset)
        images_in_set = df_label[df_label['set']==set][item_name].values
    else:
        images_in_set = None
    number_im_list = len(list_imgs)
    return(list_imgs,images_in_set,number_im_list)
    
def Precompute_Mean_Cov(filename_path,style_layers,number_im_considered,\
                        dataset='ImageNet',set='',saveformat='h5',whatToload='var',
                        getBeforeReLU=False):
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
    list_imgs,images_in_set,number_im_list = get_list_im(dataset)
    
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
    
    vgg_get_cov =  get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=getBeforeReLU)
    
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
                image_array = load_crop_and_process_img(image_path)
                vgg_cov_mean = vgg_get_cov.predict(image_array, batch_size=1)
            except IndexError as e:
                print(e)
                print(i,image_path)
                raise(e)
            
            if saveformat=='h5':
                grp = store.create_group(short_name)
                for l,layer in enumerate(style_layers):
                    cov = vgg_cov_mean[2*l]
                    mean = vgg_cov_mean[2*l+1]
                    cov_str = layer + '_cov'
                    mean_str = layer + '_mean'
                    grp.create_dataset(cov_str,data=cov) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
                    grp.create_dataset(mean_str,data=mean) # , dtype=np.float32,shape=vgg_cov_mean[l].shape
            elif saveformat=='pkl':
                dict_output[short_name] = vgg_cov_mean
                
            for l,layer in enumerate(style_layers):
#                        [cov,mean] = vgg_cov_mean[l]
                cov = vgg_cov_mean[2*l][0,:,:]
                mean = vgg_cov_mean[2*l+1][0,:] # car batch size == 1
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

def get_dict_stats(source_dataset,number_im_considered,style_layers,\
                   whatToload,saveformat='h5',set='',getBeforeReLU=False):
    str_layers = numeral_layers_index(style_layers)
    filename = source_dataset + '_' + str(number_im_considered) + '_CovMean'+'_'+str_layers
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
    if saveformat=='pkl':
        filename += '.pkl'
    if saveformat=='h5':
        filename += '.h5'
    filename_path= os.path.join(output_path_full,filename)
    if not os.path.isfile(filename_path):
        dict_stats = Precompute_Mean_Cov(filename_path,style_layers,number_im_considered,\
                                       dataset=source_dataset,set=set,saveformat=saveformat,whatToload=whatToload)
    else:
        dict_stats = load_precomputed_mean_cov(filename_path,style_layers,source_dataset,
                                            saveformat=saveformat,whatToload=whatToload)
    return(dict_stats)

def Mom_of_featuresMaps(saveformat='h5',number_im_considered = np.inf,dataset_tab=None
                        ,getBeforeReLU=False,printoutput='Var'):
    """
    In this function we will compute the first or second moments for two subsets
    a small part of ImageNet validation set 
    Paintings datasets
    @param : saveformat use h5 if you use more than 1000 images
    @param :number_im_considered number of image considered in the computation 
        if == np.inf we will use all the image in the folder of the dataset
    @param : printoutput : print in a pdf the output Var or Mean
    """
    if not(printoutput in ['Var','Mean',['Mean','Var']]):
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
        for dataset in dataset_tab:
            print('===',dataset,'===')
            list_imgs,images_in_set,number_im_list = get_list_im(dataset,set='')
            if not(number_im_considered is None):
                if number_im_considered >= number_im_list:
                    number_im_considered_tmp =None
                else:
                    number_im_considered_tmp=number_im_considered
            str_layers = numeral_layers_index(style_layers)
            filename = dataset + '_' + str(number_im_considered_tmp) + '_CovMean'+\
                '_'+str_layers
            if not(set=='' or set is None):
                filename += '_'+set
            if getBeforeReLU:
                filename += '_BeforeReLU'
            if saveformat=='pkl':
                filename += '.pkl'
            if saveformat=='h5':
                filename += '.h5'
            filename_path= os.path.join(output_path,filename)
            
            if not os.path.isfile(filename_path):
                dict_var = Precompute_Mean_Cov(filename_path,style_layers,number_im_considered_tmp,\
                                               dataset=dataset,set=set,saveformat=saveformat,
                                               whatToload=whatToload,getBeforeReLU=getBeforeReLU)
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
    
if __name__ == '__main__':         
    #Mom_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab=None)
    #Mom_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab= ['ImageNet','OIV5'])
    #Mom_of_featuresMaps(saveformat='h5',number_im_considered =1000,dataset_tab=  ['ImageNet','Paintings','watercolor','IconArt_v1'])
    Mom_of_featuresMaps(saveformat='h5',number_im_considered =10000,
                        dataset_tab= ['Paintings','watercolor','IconArt_v1','ImageNet'],
                        getBeforeReLU=True,printoutput=['Mean','Var'])
    #Mom_of_featuresMaps(saveformat='h5',number_im_considered =np.inf,dataset_tab=  ['ImageNet','Paintings','watercolor','IconArt_v1'])
    
                    
        