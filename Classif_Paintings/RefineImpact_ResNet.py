#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:19:01 2019

The goal of this script is to study the impact of the refinement of the batch 
normalisation on the features of the ResNet model

@author: gonthier
"""

import numpy as np
import matplotlib
import os.path
import matplotlib.gridspec as gridspec
from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index,numeral_layers_index_bitsVersion,\
    Precompute_Cumulated_Hist_4Moments,load_Cumulated_Hist_4Moments,get_list_im
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
    load_resize_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model,\
    MLP_model,Perceptron_model,vgg_adaDBN,ResNet_AdaIn,ResNet_BNRefinements_Feat_extractor,\
    ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction,ResNet_cut,vgg_suffleInStats,\
    get_ResNet_ROWD_meanX_meanX2_features,get_BaseNorm_meanX_meanX2_features,\
    get_VGGmodel_meanX_meanX2_features,add_head_and_trainable,extract_Norm_stats_of_ResNet,\
    vgg_FRN,get_those_layers_output
from StatsConstr_ClassifwithTL import learn_and_eval


import pickle
import pathlib

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from keras_resnet_utils import getBNlayersResNet50,getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion,\
    fit_generator_ForRefineParameters


from sklearn.feature_selection import mutual_info_classif

def compare_Statistics_inFirstLayer_for_ResNet(target_dataset='Paintings'):
    """ The goal of this function is to compare the statistics of the features maps 
    between the base ResNet50 and the ResNet50_ROWD for Some set of ImageNet and ArtUK paintings 
    We will plot the histogram of all the possible values of each of the features maps for some model
    """
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','HistoOfAllValuesFM')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    number_im_considered = 2 #10000
    nets = ['ResNet50','ResNet50_ROWD_CUMUL']
    style_layers = getBNlayersResNet50()
    features = 'activation_48'
    normalisation = False
    getBeforeReLU = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    kind_method=  'TL'
    transformOnFinalLayer='GlobalAveragePooling2D'
    computeGlobalVariance_tab = [False,True]
    cropCenter = True
    saveformat = 'h5'
    # Load ResNet50 normalisation statistics
    
    list_bn_layers = getBNlayersResNet50()

    list_of_concern_layers = ['conv1','bn_conv1','activation']

    Model_dict = {}
    list_markers = ['o','s','X','*']
    alpha = 0.7
    
    dict_of_dict_hist = {}
    dict_of_dict = {}
    
    for constrNet,computeGlobalVariance in zip(nets,computeGlobalVariance_tab):          
        output = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                               constrNet,kind_method,style_layers=style_layers,
                               normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                               batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                               returnStatistics=True,cropCenter=cropCenter,\
                               computeGlobalVariance=computeGlobalVariance)
        if 'ROWD' in constrNet:
            dict_stats_target,list_mean_and_std_target = output
            # Need to create the model 
            model = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(
                                   style_layers,list_mean_and_std_target=list_mean_and_std_target,\
                                   final_layer=features,\
                                   transformOnFinalLayer=transformOnFinalLayer,res_num_layers=50,\
                                   weights='imagenet')
        else:
            model = output # In the case of ResNet50
            
        output_of_first_layer_net = get_those_layers_output(model,list_of_concern_layers)
            
        if constrNet == 'ResNet50':
            dataset_used = ['ImageNet','Paintings']
            set_used = [None,'trainval']
        elif  constrNet == 'ResNet50_ROWD_CUMUL':
            dataset_used = ['Paintings','Paintings']
            set_used = ['trainval','test']
        
        for dataset,set_ in zip(dataset_used,set_used):
            
            Netdatasetfull = constrNet + dataset + str(set_)
            list_imgs,images_in_set,number_im_list = get_list_im(dataset,set=set_)
            if not(number_im_considered is None):
                if number_im_considered >= number_im_list:
                    number_im_considered_tmp =None
                else:
                    number_im_considered_tmp=number_im_considered
            str_layers = getResNetLayersNumeral_bitsVersion(style_layers)
            filename = dataset + '_' + str(number_im_considered_tmp) + '_Hist4Params' +\
                '_'+str_layers
            if not(set_=='' or set_ is None):
                filename += '_'+set_
            if getBeforeReLU:
                filename += '_BeforeReLU'
            if cropCenter:
                filename += '_cropCenter'
            if computeGlobalVariance:
                filename += '_computeGlobalVariance'
            if saveformat=='pkl':
                filename += '.pkl'
            if saveformat=='h5':
                filename += '.h5'
            filename_path= os.path.join(output_path,filename) 
            if not os.path.isfile(filename_path):
                dict_var,dict_histo = Precompute_Cumulated_Hist_4Moments(filename_path,model_toUse=output_of_first_layer_net,\
                                                Net=constrNet,\
                                                list_of_concern_layers=list_of_concern_layers,number_im_considered=number_im_considered,\
                                                dataset=dataset,set=set_,saveformat=saveformat,cropCenter=cropCenter)

            else:
                print('We will load the features ')
                dict_var,dict_histo = load_Cumulated_Hist_4Moments(filename_path,list_of_concern_layers,dataset)
            dict_of_dict[Netdatasetfull] = dict_var
            dict_of_dict_hist[Netdatasetfull] = dict_histo
    
    print('Start plotting ')
    # Plot the histograms (one per kernel for the different layers and save all in a pdf file)
    pltname = 'Hist_of_AllFeaturesMaps'
    labels = []
    dataset_tab = []
    for constrNet,computeGlobalVariance in zip(nets,computeGlobalVariance_tab):
        if constrNet == 'ResNet50':
            dataset_used = ['ImageNet','Paintings']
            set_used = [None,'trainval']
        elif  constrNet == 'ResNet50_ROWD_CUMUL':
            dataset_used = ['Paintings','Paintings']
            set_used = ['trainval','test']
        
        for dataset,set_ in zip(dataset_used,set_used):
            Netdatasetfull = constrNet + dataset + str(set_)
            dataset_tab += [Netdatasetfull]
            labels += [Netdatasetfull]
    pltname +=  str(number_im_considered)
    if getBeforeReLU:
        pltname+= '_BeforeReLU'
    if cropCenter:
        pltname += '_cropCenter'
        
    pltname +='.pdf'
    pltname= os.path.join(output_path,pltname)
    pp = PdfPages(pltname)
    
    alpha=0.7
    colors_full = ['red','green','blue','purple','orange','pink']
    colors_full = ['red','green','blue','purple','orange','pink']
    colors = colors_full[0:len(dataset_tab)]
    
#    style_layers = [style_layers[0]]
    
    # Turn interactive plotting off
    plt.ioff()
        
    for l,layer in enumerate(list_of_concern_layers):
        print("Layer",layer)
        tab_vars = []
        for dataset in dataset_tab: 
            dict_k = dict_of_dict_hist[Netdatasetfull][layer]
            num_features = 64 # Need to generalize that
            
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
                
                n_bins = 1000
                for l,Netdatasetfull in enumerate(dataset_tab):
#                    x = np.vstack([tab_vars[0][:,f_k],tab_vars[1][:,f_k]])# Each line is a dataset 
#                    x = x.reshape((-1,2))
                    dict_k = dict_of_dict_hist[Netdatasetfull][layer]
                    xtab,ytab = dict_k[f_k]
                    xtab_density = xtab/np.sum(xtab)
                    vars_values = tab_vars[l][:,f_k].reshape((-1,))
                    xtab += [vars_values]
                    im = ax.bar(xtab_density,ytab, color=colors[l],\
                             alpha=alpha,label=labels[l])
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
        
    # Plot Error bar of means of the 4 paramaters statistics
      
def compare_Models_Plot_HistoDistrib_for_ResNet(target_dataset='Paintings'):
    """ The goal of this function is to compare the models fine tuned of with some statistics imposed
    
    We will compare , ROWD (mean of variance) and variance global in the case 
    of ResNet50 TL of FT """
    
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    dataset =  target_dataset
    nets = ['ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_BNRF']
    nets = ['ResNet50','ResNet50','ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL']
    nets = ['ResNet50','ResNet50','ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL_AdaIn']
    #nets = ['ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL']
    style_layers = getBNlayersResNet50()
    features = 'activation_48'
    normalisation = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    kind_method=  'TL'
    transformOnFinalLayer='GlobalAveragePooling2D'
    computeGlobalVariance_tab = [False,False,False,False,True]
    computeGlobalVariance_tab = [False,False,False,False,True,True,True]
    #computeGlobalVariance_tab = [False,False,True]
    pretrainingModif_tab = [False,106,106,False,False]
    pretrainingModif_tab = [False,106,106,False,False,False,False]
    #pretrainingModif_tab = [False,False,False]
    kindmethod_tab = ['TL','FT','FT','TL','TL']
    kindmethod_tab = ['TL','FT','FT','TL','TL','FT','FT']
    #kindmethod_tab = ['TL','TL','TL']
    
    final_clf = 'MLP2'
    epochs = 20
    optimizer = 'SGD'
    opt_option_tab = [[10**(-2)],[10**(-2)],[0.1,10**(-2)],[10**(-2)],[10**(-2)]]
    opt_option_tab = [[10**(-2)],[10**(-2)],[0.1,10**(-2)],[10**(-2)],[10**(-2)],[10**(-2)],[10**(-2)]]
    return_best_model = True
    batch_size= 16
    
    cropCenter = True
    # Load ResNet50 normalisation statistics
    
    list_bn_layers = getBNlayersResNet50()

    Model_dict = {}
    Model_dict_histo = {}
    dict_num_fS = {}
    list_markers = ['o','s','X','*','v','^','<','>','d','1','2','3','4','8','h','H','p','d','$f$','P']
    alpha = 0.7
    number_im_considered = None
    
    ArtUKlist_imgs,images_in_set,number_im_list = get_list_im(dataset,set='')
    if not(number_im_considered is None):
        if number_im_considered >= number_im_list:
            number_im_considered_tmp =None
        else:
            number_im_considered_tmp=number_im_considered
    else:
        number_im_considered_tmp = None
    
    idex = 0 
    
    list_of_concern_layers = ['conv1','bn_conv1','activation']
    set = 'trainval'
    saveformat='h5'
    cropCenter=True
    histoFixe=True
    bins=np.arange(-500,501)
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    for constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option in \
        zip(nets,computeGlobalVariance_tab,kindmethod_tab,pretrainingModif_tab,opt_option_tab): 
        print('loading :',constrNet)         
        output = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                               constrNet,kind_method,style_layers=style_layers,
                               normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                               batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                               returnStatistics=True,cropCenter=cropCenter,\
                               computeGlobalVariance=computeGlobalVariance,\
                               epochs=epochs,optimizer=optimizer,opt_option=opt_option,return_best_model=return_best_model,\
                               batch_size=batch_size,gridSearch=False)

        if 'ResNet50_ROWD_CUMUL' == constrNet and kind_method=='TL':
            dict_stats_target,list_mean_and_std_target = output
            list_mean_and_std_source = None
            target_number_im_considered = None
            whatToload = 'varmean'
            target_set = 'trainval'
                
            model_toUse = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(
                           style_layers,list_mean_and_std_target=list_mean_and_std_target,\
                           final_layer=features,\
                           transformOnFinalLayer='GlobalMaxPooling2D',res_num_layers=50,\
                           weights='imagenet')

        else:
            model_toUse = output
            
            
        Net = constrNet
        str_model = constrNet
        if computeGlobalVariance:
            str_model += 'GlobalVar' 
        if kind_method=='FT':
            str_model += ' FT '
            if len(opt_option)==1:
               str_model +='lr '+str(opt_option[0]) 
            if len(opt_option)==2:
               str_model +='lrp '+str(opt_option[0]) +' lr '+str(opt_option[1])
        #str_model = str_model.replace('ResNet50_','')
               
        filename_path = dataset + '_' + str(number_im_considered_tmp) + '_CovMean'+ str_model 
        for l in list_of_concern_layers:
            filename_path += '_'+l
        if cropCenter:
            filename_path += '_cropCenter'
        if histoFixe:
            filename_path += '_histoFixe'
        
        if saveformat=='pkl':
            filename_path += '.pkl'
        if saveformat=='h5':
            filename_path += '.h5'
        
        filename_path= os.path.join(output_path,filename_path)
               
        str_model += str(idex)
        idex += 1
        if not os.path.isfile(filename_path):
            dict_var,dict_histo,dict_num_f = Precompute_Cumulated_Hist_4Moments(filename_path,model_toUse,Net,list_of_concern_layers,number_im_considered_tmp,\
                        dataset,set=set,saveformat=saveformat,cropCenter=cropCenter,histoFixe=histoFixe,\
                        bins=bins)
        else:
            dict_var,dict_histo,dict_num_f  = load_Cumulated_Hist_4Moments(filename_path,list_of_concern_layers,dataset,saveformat='h5')
        Model_dict[str_model] = dict_var
        Model_dict_histo[str_model] = dict_histo
        dict_num_fS[str_model] = dict_num_f
        
    
    # Start plotting 
        
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',\
                               dataset,'CompModels')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pltname = 'ResNetS_Hist_of_featuresValues'+dataset+'_'
    if cropCenter:
        pltname += '_cropCenter'
    pltname +='.pdf'
    pltname= os.path.join(output_path,pltname)
    pp = PdfPages(pltname)

    # Turn interactive plotting off
    plt.ioff()
    
    num_features = 64
    
        
    number_img_w = len(nets)
    number_img_h= len(list_of_concern_layers)
    num_pages = num_features
    for p in range(num_pages):
        
        axes = []
        gs00 = gridspec.GridSpec(number_img_h, number_img_w)
        for j in range(number_img_w*number_img_h):
            ax = plt.subplot(gs00[j])
            axes += [ax]
        for l,layer in enumerate(list_of_concern_layers):
            n = 0
            for constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option in \
                zip(nets,computeGlobalVariance_tab,kindmethod_tab,pretrainingModif_tab,opt_option_tab): 
                Net = constrNet
                str_model = constrNet
                if computeGlobalVariance:
                    str_model += 'GlobalVar' 
                if kind_method=='FT':
                    str_model += ' FT '
                    if len(opt_option)==1:
                       str_model +='lr '+str(opt_option[0]) 
                    if len(opt_option)==2:
                       str_model +='lrp '+str(opt_option[0]) +' lr '+str(opt_option[1])
                label = str_model.replace('ResNet50_','')
                    
                hists = Model_dict_histo[str_model][layer]
                #num_features = dict_num_fS[str_model][layer]

                ax = axes[l*number_img_h+n]
                hist = hists[p]
                im = ax.bar(bins+0.5, hist, width=1, label=label)
                ax.tick_params(axis='both', which='major', labelsize=3)
                ax.tick_params(axis='both', which='minor', labelsize=3)
                    #ax.legend(loc='upper right', prop={'size': 2})
                    # fig_title = "m: {:.2E}, v: {:.2E}, s: {:.2E}, k: {:.2E}".format(m,v,s,k)
                    # ax.set_title(fig_title, fontsize=3)
                n += 1 
                titre = layer +' ' +str(p)
                plt.suptitle(titre)
                
                #gs0.tight_layout(f)
                plt.savefig(pp, format='pdf')
                plt.close()
    pp.close()
    plt.clf()
    
        
    
        
        
def compare_new_normStats_for_ResNet(target_dataset='Paintings'):
    """ The goal of this function is to compare the new normalisation statistics of BN
    computed in the case of the adaptation of them 
    We will compare  ROWD (mean of variance) and variance global in the case 
    of ResNet50 
    
    We not use BNRF for the moment because it diverge
    """
    
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    nets = ['ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_BNRF']
    nets = ['ResNet50','ResNet50','ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL']
    nets = ['ResNet50','ResNet50','ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL_AdaIn']
    #nets = ['ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL']
    style_layers = getBNlayersResNet50()
    features = 'activation_48'
    normalisation = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    kind_method=  'TL'
    transformOnFinalLayer='GlobalAveragePooling2D'
    computeGlobalVariance_tab = [False,False,False,False,True]
    computeGlobalVariance_tab = [False,False,False,False,True,True,True]
    #computeGlobalVariance_tab = [False,False,True]
    pretrainingModif_tab = [False,106,106,False,False]
    pretrainingModif_tab = [False,106,106,False,False,False,False]
    #pretrainingModif_tab = [False,False,False]
    kindmethod_tab = ['TL','FT','FT','TL','TL']
    kindmethod_tab = ['TL','FT','FT','TL','TL','FT','FT']
    #kindmethod_tab = ['TL','TL','TL']
    
    final_clf = 'MLP2'
    epochs = 20
    optimizer = 'SGD'
    opt_option_tab = [[10**(-2)],[10**(-2)],[0.1,10**(-2)],[10**(-2)],[10**(-2)]]
    opt_option_tab = [[10**(-2)],[10**(-2)],[0.1,10**(-2)],[10**(-2)],[10**(-2)],[10**(-2)],[10**(-2)]]
    return_best_model = True
    batch_size= 16
    
    cropCenter = True
    # Load ResNet50 normalisation statistics
    
    list_bn_layers = getBNlayersResNet50()

    Model_dict = {}
    list_markers = ['o','s','X','*','v','^','<','>','d','1','2','3','4','8','h','H','p','d','$f$','P']
    alpha = 0.7
    
    idex = 0 
    for constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option in \
        zip(nets,computeGlobalVariance_tab,kindmethod_tab,pretrainingModif_tab,opt_option_tab): 
        print('loading :',constrNet)         
        output = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                               constrNet,kind_method,style_layers=style_layers,
                               normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                               batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                               returnStatistics=True,cropCenter=cropCenter,\
                               computeGlobalVariance=computeGlobalVariance,\
                               epochs=epochs,optimizer=optimizer,opt_option=opt_option,return_best_model=return_best_model,\
                               batch_size=batch_size,gridSearch=False)

        if 'ResNet50_ROWD_CUMUL' == constrNet and kind_method=='TL':
            dict_stats_target,list_mean_and_std_target = output
            # print('dict_stats_target',dict_stats_target)
            # input('wait')
        else:
            dict_stats_target,list_mean_and_std_target = extract_Norm_stats_of_ResNet(output,\
                                                    res_num_layers=50,model_type=constrNet)
        str_model = constrNet
        if computeGlobalVariance:
            str_model += 'GlobalVar' 
        if kind_method=='FT':
            str_model += ' FT '
            if len(opt_option)==1:
               str_model +='lr '+str(opt_option[0]) 
            if len(opt_option)==2:
               str_model +='lrp '+str(opt_option[0]) +' lr '+str(opt_option[1])
        #str_model = str_model.replace('ResNet50_','')
        str_model += str(idex)
        idex += 1
        Model_dict[str_model] = dict_stats_target
      
    print('Plotting the statistics')
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',\
                               target_dataset,'CompBNstats') 
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    pltname = 'ResNet50_comparison_BN_statistics_ROWD'
    if cropCenter:
        pltname += '_cropCenter'   
    pltname +='.pdf'
    pltname= os.path.join(output_path,pltname)
    pp = PdfPages(pltname)    
    
    distances_means = {}
    distances_stds = {}
    ratios_means = {}
    ratios_stds = {}

    for layer_name in list_bn_layers:
        distances_means[layer_name] = []
        distances_stds[layer_name] = []
        ratios_means[layer_name] = []
        ratios_stds[layer_name] = []
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        str_title = 'Normalisation statistics ' + layer_name
        fig.suptitle(str_title)
        i = 0
        idex = 0
        for constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option in \
            zip(nets,computeGlobalVariance_tab,kindmethod_tab,pretrainingModif_tab,opt_option_tab):
                
            str_model = constrNet
            if computeGlobalVariance:
                str_model += 'GlobalVar' 
            if kind_method=='FT':
                str_model += ' FT '
                if len(opt_option)==1:
                   str_model +='lr '+str(opt_option[0]) 
                if len(opt_option)==2:
                   str_model +='lrp '+str(opt_option[0]) +' lr '+str(opt_option[1])
            label = str_model.replace('ResNet50_','')
            str_model += str(idex)
            idex += 1
            dict_stats_target = Model_dict[str_model]
            stats_target =  dict_stats_target[layer_name]
            means,stds = stats_target
            if constrNet=='ResNet50' and kind_method=='TL':
                ref_means = means
                ref_stds = stds
            else:
                diff_means = np.abs(ref_means-means)
                diff_stds = np.abs(ref_stds-stds)
                ratio_means = np.abs(means/ref_means)
                ratio_stds = np.abs(stds/ref_stds)
                distances_means[layer_name] += [diff_means]
                distances_stds[layer_name] += [diff_stds]
                ratios_means[layer_name] += [ratio_means]
                ratios_stds[layer_name] += [ratio_stds]
            x = np.arange(0,len(means))
            ax1.scatter(x, means,label=label,marker=list_markers[i],alpha=alpha)
            ax1.set_title('Normalisation Means')
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Mean')
            ax1.tick_params(axis='both', which='major', labelsize=3)
            ax1.tick_params(axis='both', which='minor', labelsize=3)
            ax1.legend(loc='best', prop={'size': 4})
            ax2.scatter(x, stds,label=label,marker=list_markers[i],alpha=alpha)
            ax2.set_title('Normalisation STDs')
            ax2.set_xlabel('Channel')
            ax2.set_ylabel('Std')
            ax2.tick_params(axis='both', which='major', labelsize=3)
            ax2.tick_params(axis='both', which='minor', labelsize=3)
            ax2.legend(loc='best', prop={'size': 4})
            i+=1
 
        #plt.show()
        plt.savefig(pp, format='pdf')
        plt.close()
     
    # Plot the boxplot of the distance between normalisation statistics
    # fig = plt.figure()
    # ax = plt.axes()
    # set_xticks= []
    # c = ['C1','C2','C3']
    # c = ['orange','green','red']
    # for i,layer_name in enumerate(list_bn_layers):     
    #     positions = [i*3,i*3+1,i*3+2]
    #     set_xticks += [i*3+1]
    #     bp = plt.boxplot(np.log(distances_means[layer_name]).tolist(), positions = positions, 
    #                      widths = 0.6,notch=True, patch_artist=True)
    #     for patch, color in zip(bp['boxes'], c):
    #         patch.set_facecolor(color)
    # ax.set_xticklabels(list_bn_layers)
    # ax.set_xticks(set_xticks)
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation='vertical')
    # hO, = plt.plot([1,1],'C1-')
    # hG, = plt.plot([1,1],'C2-')
    # hR, = plt.plot([1,1],'C3-')
    # plt.title('Log Abs distance between means of refined and orignal.', fontsize=10)
    # plt.legend((hO, hG,hR),('ROWD','ROWD_global', 'BNRF'))
    # hO.set_visible(False)
    # hG.set_visible(False)
    # hR.set_visible(False)
    # plt.savefig(pp, format='pdf')
    # plt.close()
    
    # fig = plt.figure()
    # ax = plt.axes()
    # set_xticks= []
    
    # for i,layer_name in enumerate(list_bn_layers):     
    #     positions = [i*3,i*3+1,i*3+2]
    #     set_xticks += [i*3+1]
    #     bp = plt.boxplot(np.log(distances_stds[layer_name]).tolist(), positions = positions, 
    #                      widths = 0.6,notch=True, patch_artist=True)
    #     for patch, color in zip(bp['boxes'], c):
    #         patch.set_facecolor(color) 
    # ax.set_xticklabels(list_bn_layers)
    # ax.set_xticks(set_xticks)
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation='vertical')
    # hO, = plt.plot([1,1],'C1-')
    # hG, = plt.plot([1,1],'C2-')
    # hR, = plt.plot([1,1],'C3-')
    # plt.title('Log Abs distance between  stds of refined and orignal.', fontsize=10)
    # plt.legend((hO, hG,hR),('ROWD','ROWD_global', 'BNRF'))
    # hO.set_visible(False)
    # hG.set_visible(False)
    # hR.set_visible(False)
    # plt.savefig(pp, format='pdf')
    # plt.close()
    
    # # Plot the boxplot of the ratio between normalisation statistics
    # fig = plt.figure()
    # ax = plt.axes()
    # set_xticks= []
    # c = ['C1','C2','C3']
    # c = ['orange','green','red']
    # for i,layer_name in enumerate(list_bn_layers):     
    #     positions = [i*3,i*3+1,i*3+2]
    #     set_xticks += [i*3+1]
    #     bp = plt.boxplot(np.log(1.+np.array(ratios_means[layer_name])).tolist(), positions = positions, 
    #                      widths = 0.6,notch=True, patch_artist=True)
    #     for patch, color in zip(bp['boxes'], c):
    #         patch.set_facecolor(color)
    # ax.set_xticklabels(list_bn_layers)
    # ax.set_xticks(set_xticks)
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation='vertical')
    # hO, = plt.plot([1,1],'C1-')
    # hG, = plt.plot([1,1],'C2-')
    # hR, = plt.plot([1,1],'C3-')
    # plt.title('Log 1+ Ratio between means of refined and orignal.', fontsize=10)
    # plt.legend((hO, hG,hR),('ROWD','ROWD_global', 'BNRF'))
    # hO.set_visible(False)
    # hG.set_visible(False)
    # hR.set_visible(False)
    # plt.savefig(pp, format='pdf')
    # plt.close()
    
    # fig = plt.figure()
    # ax = plt.axes()
    # set_xticks= []
    
    # for i,layer_name in enumerate(list_bn_layers):     
    #     positions = [i*3,i*3+1,i*3+2]
    #     set_xticks += [i*3+1]
    #     bp = plt.boxplot(np.log(1.+np.array(ratios_stds[layer_name])).tolist(), positions = positions, 
    #                      widths = 0.6,notch=True, patch_artist=True)
    #     for patch, color in zip(bp['boxes'], c):
    #         patch.set_facecolor(color) 
    # ax.set_xticklabels(list_bn_layers)
    # ax.set_xticks(set_xticks)
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation='vertical')
    # hO, = plt.plot([1,1],'C1-')
    # hG, = plt.plot([1,1],'C2-')
    # hR, = plt.plot([1,1],'C3-')
    # plt.title('Log 1+ ratio between stds of Refined model and original', fontsize=10)
    # plt.legend((hO, hG,hR),('ROWD','ROWD_global', 'BNRF'))
    # hO.set_visible(False)
    # hG.set_visible(False)
    # hR.set_visible(False)
    # plt.savefig(pp, format='pdf')
    # plt.close()
   
    pp.close()
    plt.clf()
    
    
def compute_MutualInfo(target_dataset='Paintings'):
    """ The goal of this function is to compute the entropy and the mutual information 
    of the features in the ResNet model and the refined versions 
    We will compare BNRF, ROWD (mean of variance) and variance global in the case 
    of ResNet50 """
    
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    nets = ['ResNet50','ResNet50_ROWD_CUMUL','ResNet50_ROWD_CUMUL','ResNet50_BNRF']
    style_layers = getBNlayersResNet50()
    features = 'activation_48'
    normalisation = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    kind_method=  'TL'
    transformOnFinalLayer='GlobalAveragePooling2D'
    computeGlobalVariance_tab = [False,False,True,False]
    cropCenter = True
    # Load ResNet50 normalisation statistics
    Model_dict = {}
    
    for constrNet,computeGlobalVariance in zip(nets,computeGlobalVariance_tab):    
        str_model = constrNet
        if computeGlobalVariance:
            str_model += 'GlobalVar'
        print(str_model)
        Model_dict[str_model] = {}
        
        output = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                               constrNet,kind_method,style_layers=style_layers,
                               normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                               batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                               returnFeatures=True,cropCenter=cropCenter,\
                               computeGlobalVariance=computeGlobalVariance)
        Xtrainval,ytrainval,X_test,y_test =  output
        _,num_classes = ytrainval.shape
        
        # Mutual Information
        for c in range(num_classes):
            print('For class',c)
            MI_trainval_c = mutual_info_classif(Xtrainval, ytrainval[:,c], discrete_features=False, n_neighbors=3, \
                                              copy=True, random_state=0)
            sum_MI_trainval_c = np.sum(MI_trainval_c)
            MI_test_c = mutual_info_classif(X_test, y_test[:,c], discrete_features=False, n_neighbors=3, \
                                              copy=True, random_state=0)
            sum_MI_test_c = np.sum(MI_test_c)
            Model_dict[str_model][c] = {}
            Model_dict[str_model][c]['trainval'] =  sum_MI_trainval_c
            Model_dict[str_model][c]['test'] =  sum_MI_test_c
     
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp')
    
    if os.path.isdir(output_path):
        output_path_full = os.path.join(output_path,'Covdata')
    else:
        output_path_full = os.path.join('data','Covdata')
    filename_path = os.path.join(output_path_full,'MutualInfo_'+target_dataset+'.pkl')
    # Warning ici tu ecrases le meme fichier
    with open(filename_path, 'wb') as handle:
        pickle.dump(Model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    for c in range(num_classes):
        string = 'Classs '+str(c) 
        for set_ in ['trainval','test']:
            strings = string + ' '+set_
            for constrNet,computeGlobalVariance in zip(nets,computeGlobalVariance_tab):
                   str_model = constrNet
                   if computeGlobalVariance:
                       str_model += 'GlobalVar' 
                   strings += ' '+str_model + ' : '
                   sum_MI =  Model_dict[str_model][c][set_] 
                   strings += "{:.2E}".format(sum_MI) 
            strings += '\n'
            print(strings)
