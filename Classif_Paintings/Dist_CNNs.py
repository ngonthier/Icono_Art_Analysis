# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:28:29 2020

The goal of this script is to compare models with differents methods
used in What is being transferred in transfer learning? Neyshabur 2020

To measure the distance between CNN models

@author: gonthier
"""

import numpy as np
import os
import platform
import pathlib
import pickle
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D
import gzip

from StatsConstr_ClassifwithTL import predictionFT_net
import Stats_Fcts
from googlenet import inception_v1_oldTF as Inception_V1
from IMDB import get_database
from plots_utils import plt_multiple_imgs
from CompNet_FT_lucidIm import get_fine_tuned_model

from CompNet_FT_lucidIm import get_imageNet_weights,get_fine_tuned_model,print_stats_on_diff,get_weights_and_name_layers

from Activation_for_model import get_Model_that_output_Activation,get_Network

from CKA import linear_CKA

def get_l2_norm_weights(list_name_layers,list_weights,net_finetuned,verbose=False):
    finetuned_layers = net_finetuned.layers
        
    j = 0
    dict_layers_diff = {}
    l2_norm_total = 0.
    for finetuned_layer in finetuned_layers:
        # check for convolutional layer
        layer_name = finetuned_layer.name
        if not(layer_name in list_name_layers):
            continue
        if isinstance(finetuned_layer, Conv2D) :
            
            list_weights_j = list_weights[j]
            if len(list_weights)==2:
                o_filters, o_biases = list_weights_j
                f_filters, f_biases = finetuned_layer.get_weights()
            else:
                o_filters = np.array(list_weights_j[0]) # We certainly are in the Inception_V1 case with no biases
                f_filters = np.array(finetuned_layer.get_weights()[0])
            j+=1

            # Norm 2 between the weights of the filters
                
            diff_filters = o_filters - f_filters
            frobenium_norm = np.linalg.norm(diff_filters)
            l2_norm_total += frobenium_norm
            if verbose:
                print('== For layer :',layer_name,' ==')
                print('= Frobenius norm =')
                print_stats_on_diff(frobenium_norm)
            
            dict_layers_diff[layer_name] = frobenium_norm
            
        
    return(dict_layers_diff,l2_norm_total)

def get_model_name_wo_oldModel(model_name):
    if 'XX' in model_name:
        splittedXX = model_name.split('XX')
        weights = splittedXX[1] # original model
        model_name_wo_oldModel = model_name.replace('_XX'+weights+'XX','')
    else:
        model_name_wo_oldModel = model_name
        if 'RandForUnfreezed' in model_name or 'RandInit' in model_name:
            weights = 'random'
        else:
            weights = 'pretrained'
            
    return(model_name_wo_oldModel,weights)
    
def l2norm_bw_nets(netA,netB,constrNet='InceptionV1',suffixA='',suffixB='',initA=False,initB=False):
    """ 
    Distance in parameters space between network. 
    if initA is True : we will use the initialization of A as model
    """
    
    
    net_P = None  
    if netA=='pretrained':
        net_P = 'pretrained'
        net_Autre = netB
        suffix_Autre = suffixB
        init_Autre = initB
    elif netB=='pretrained':
        net_P = 'pretrained'
        suffix_Autre = suffixA
        init_Autre = initA

    if not(net_P is None): # Dans le cas ou un des deux networks est pretrained weights
        list_weights,list_name_layers = get_imageNet_weights(Net=constrNet)
    
        net_finetuned, init_net = get_fine_tuned_model(net_Autre,constrNet=constrNet,suffix=suffix_Autre)
        if init_Autre:
            autre_model = init_net
        else:
            autre_model = net_finetuned
        
        dict_layers_diff,l2_norm_total = get_l2_norm_weights(list_name_layers,list_weights,autre_model,verbose=False)
        
        data_to_save = list_name_layers,dict_layers_diff,l2_norm_total
        
    else: # No pretrained model compared to an other one
        net_finetuned_A, init_net_A = get_fine_tuned_model(netA,constrNet=constrNet,suffix=suffixA)
        if initA:
            modelA = init_net_A
        else:
            modelA = net_finetuned_A
        list_weights_A,list_name_layers_A = get_weights_and_name_layers(modelA)
        net_finetuned_B, init_net_B = get_fine_tuned_model(netB,constrNet=constrNet,suffix=suffixB)
        if initB:
            modelB = init_net_B
        else:
            modelB = net_finetuned_B
        dict_layers_diff,l2_norm_total = get_l2_norm_weights(list_name_layers_A,list_weights_A,modelB,verbose=False)
    
        data_to_save = list_name_layers_A,dict_layers_diff,l2_norm_total
    
    print('===',netA,netB,l2_norm_total)
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,'Dists')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'Dists')
    # For output data
    output_path_full = os.path.join(output_path,'data')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_full).mkdir(parents=True, exist_ok=True)
    
    name_data = 'l2norm-'+netA+'-'+suffixA+'-'+netB+'-'+suffixB+'.pkl'
    with open(name_data, 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def compute_CKA_similarity():
    
    # According to 
    # https://github.com/google/svcca/blob/master/tutorials/002_CCA_for_Convolutional_Layers.ipynb
    # If the two conv layers being compared have the same spatial dimensions, 
    # we can flatten the spatial dimensions into the number of datapoints:
    # But we can do average pooling on the two spatial dimension or interpolation between points
    # Another thing one can do is to interpolate the spatial dimensions of the 
    # smaller sized conv layer so that they match the large layer. 
    # There are many ways to do this, and the interpolate library in 
    # scipy provides access to many different methods.

    return(0)

def get_list_full_activations(dataset,output_path,list_layers,
                         model_name,constrNet,suffix,cropCenter,FTmodel,
                         model_name_wo_oldModel,stats_on_layer):
    
    if dataset in model_name_wo_oldModel:
        dataset_str = ''
    else:
        dataset_str = dataset
    
    if not(FTmodel):
        extra_str = '_InitModel'
    else:
        extra_str = ''
    
    output_path  = os.path.join(output_path,'act_data')
    
#    if platform.system()=='Windows': 
#        output_path = os.path.join('CompModifModel',constrNet,folder_name,'act_data')
#    else:
#        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,folder_name,'act_data')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    if not(stats_on_layer is None):
        if stats_on_layer=='mean':
            save_file = os.path.join(output_path,'activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='meanAfterRelu':
            save_file = os.path.join(output_path,'meanAfterRelu_activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='max':
            save_file = os.path.join(output_path,'max_activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='min':
            save_file = os.path.join(output_path,'min_activations_per_test_img'+extra_str+dataset_str+'.p')
        else:
            raise(ValueError(stats_on_layer+' is unknown'))
    else:
        
        layers_str = ''
        for layer_name in list_layers:
            layers_str += layer_name
        save_file = os.path.join(output_path,'full_activations_per_test_img'+extra_str+dataset_str+layers_str+'.p')
        
    if os.path.exists(save_file):
        # The file exist
        # Load the activations for the main model :
        with gzip.open(save_file, "rb") as f:
            activations = pickle.load(f)
    else:
        # compute the activations for the main model :
        list_outputs_name,activations = compute_Full_Feature(dataset,
                                            model_name,constrNet,suffix=suffix,
                                            list_layers=list_layers,
                                            cropCenter=cropCenter,
                                            FTmodel=FTmodel,
                                            stats_on_layer=stats_on_layer)
    return(activations)
    
    
def compute_Full_Feature(dataset,model_name,constrNet,list_layers=None,
                                 suffix='',cropCenter = True,FTmodel=True,
                                 stats_on_layer=None):
    """
    This function will compute the full spatial activation of each features maps for all
    the convolutionnal layers 
    @param : FTmodel : in the case of finetuned from scratch if False use the initialisation
    networks
    """
    K.set_learning_phase(0) #IE no training
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_test = df_label[df_label['set']=='test']

    extra_str = ''
    
    if 'XX' in model_name:
        splittedXX = model_name.split('XX')
        weights = splittedXX[1]
        model_name_wo_oldModel = model_name.replace('_XX'+weights+'XX','')
    else:
        model_name_wo_oldModel = model_name
    if dataset in model_name_wo_oldModel:
        dataset_str = ''
    else:
        dataset_str = dataset
        
    if model_name=='pretrained':
        base_model = get_Network(constrNet)
    else:
        # Pour ton windows il va falloir copier les model .h5 finetuné dans ce dossier la 
        # C:\media\gonthier\HDD2\output_exp\Covdata\RASTA\model
        if 'RandInit' in model_name:
            FT_model,init_model = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if FTmodel:
                base_model = FT_model
            else:
                extra_str = '_InitModel'
                base_model = init_model 
        else:
            output = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
            if len(output)==2:
                base_model, init_model = output
            else:
                base_model = output
                
    model,_ = get_Model_that_output_Activation(base_model,list_layers=list_layers)
    print(model.summary())
    activations = predictionFT_net(model,df_test,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter,verbose_predict=1)
    print('activations len and shape of first element',len(activations),activations[0].shape)
    
    folder_name = model_name+suffix
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,folder_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,folder_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    

    
    if not(stats_on_layer is None):
        if stats_on_layer=='mean':
            save_file = os.path.join(output_path,'activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='meanAfterRelu':
            save_file = os.path.join(output_path,'meanAfterRelu_activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='max':
            save_file = os.path.join(output_path,'max_activations_per_test_img'+extra_str+dataset_str+'.p')
        elif stats_on_layer=='min':
            save_file = os.path.join(output_path,'min_activations_per_test_img'+extra_str+dataset_str+'.p')
        else:
            raise(ValueError(stats_on_layer+' is unknown'))
    else:
        
        layers_str = ''
        for layer_name in list_layers:
            layers_str += layer_name
        save_file = os.path.join(output_path,'full_activations_per_test_img'+extra_str+dataset_str+layers_str+'.p')
    
    with gzip.open(save_file, "wb") as f:
        pickle.dump(activations, f)
    
    
    return(activations)
    
def get_data_pts_for_analysis(model_name,dataset,list_layers,
                                 constrNet='InceptionV1',suffix='',
                                 cropCenter=True,FTmodel=True,
                                 stats_on_layer=None):
    """
    Get the activation for feature activation
    @return : a list of activations values for the test set of the dataset
    """
    
    K.set_learning_phase(0) #IE no training
    
    model_name_wo_oldModel,weights = get_model_name_wo_oldModel(model_name)
    
    folder_name = model_name+suffix
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,folder_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,folder_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)     
    
    if not(FTmodel): 
        # Load the activation on the initialisation model
        if weights == 'pretrained':
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,'pretrained')
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'pretrained')
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            activations= get_list_full_activations(dataset,
                                                    output_path_init,list_layers,
                                                    'pretrained',constrNet,
                                                    '',cropCenter,FTmodel,
                                                    'pretrained',
                                                    stats_on_layer=stats_on_layer)
        elif weights == 'random':
            activations = get_list_full_activations(dataset,
                                                    output_path,list_layers,
                                                    model_name,constrNet,
                                                    suffix,cropCenter,False,
                                                    model_name,
                                                    stats_on_layer=stats_on_layer) # FTmodel = False to get the initialisation model
        else:
            if platform.system()=='Windows': 
                output_path_init = os.path.join('CompModifModel',constrNet,weights)
            else:
                output_path_init = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,weights)
            pathlib.Path(output_path_init).mkdir(parents=True, exist_ok=True) 
            activations= get_list_full_activations(dataset,
                                                    output_path,list_layers,
                                                    weights,constrNet,
                                                    '',cropCenter,FTmodel,
                                                    weights,
                                                    stats_on_layer=stats_on_layer)
    else:
        activations = get_list_full_activations(dataset,
                                                 output_path,list_layers,
                                                 model_name,constrNet,
                                                 suffix,cropCenter,FTmodel,
                                                 model_name_wo_oldModel,
                                                 stats_on_layer=stats_on_layer)
        
    return(activations)
    
def feat_sim(model_nameA,model_nameB,dataset,list_layers=['conv2d0','mixed5b']
             ,constrNet='InceptionV1',suffixA=''
             ,suffixB='',initA=False,initB=False,kind_feat_sim='linearCKA',
             stats_on_layer=None):
    """
    Compute the feature similarity between two models
    
    list_layers=['conv2d0','conv2d1',
                                                          'conv2d2','mixed3a',
                                                          'mixed3b','mixed4a',
                                                          'mixed4b','mixed4c',
                                                          'mixed4d','mixed4e',
                                                          'mixed5a','mixed5b']
    """
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,'Dists')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'Dists')
    # For output data

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    name_data = kind_feat_sim+'-'+model_nameA+'-'+suffixA+'-'+model_nameB+'-'+suffixB
    if stats_on_layer is None:
        name_data += '.pkl'
    else:
        name_data +=stats_on_layer+ '.pkl'
    name_data_path= os.path.join(output_path,name_data)
    
    if os.path.exists(name_data_path):
            
        with open(name_data, 'rb') as handle:
            dico = pickle.load(handle)
    else:
        dico = {}
        if stats_on_layer is None:
        
            for layer_name in list_layers:
                dict_acts_A = get_data_pts_for_analysis(model_nameA,dataset,[layer_name],
                                             constrNet=constrNet,suffix=suffixA,
                                             cropCenter=True,FTmodel=not(initA),
                                             stats_on_layer=stats_on_layer)
                dict_acts_B = get_data_pts_for_analysis(model_nameB,dataset,[layer_name],
                                             constrNet=constrNet,suffix=suffixB,
                                             cropCenter=True,FTmodel=not(initB),
                                             stats_on_layer=stats_on_layer)
            
            
                acts_A = dict_acts_A[layer_name]
                num_datapoints, h, w, channels = acts_A.shape
                f_actsA = acts_A.reshape((num_datapoints*h*w, channels))
                acts_B = dict_acts_B[layer_name]
                num_datapoints, h, w, channels = acts_B.shape
                f_actsB = acts_B.reshape((num_datapoints*h*w, channels))
                cka_layer = linear_CKA(f_actsA, f_actsB)
                dico[layer_name] = cka_layer
        else:
            
            dict_acts_A = get_data_pts_for_analysis(model_nameA,dataset,list_layers,
                                         constrNet=constrNet,suffix=suffixA,
                                         cropCenter=True,FTmodel=not(initA),stats_on_layer=stats_on_layer)
            dict_acts_B = get_data_pts_for_analysis(model_nameB,dataset,list_layers,
                                         constrNet=constrNet,suffix=suffixB,
                                         cropCenter=True,FTmodel=not(initB),
                                         stats_on_layer=stats_on_layer)
            for layer_name in list_layers:
            
            
                acts_A = dict_acts_A[layer_name]
                acts_B = dict_acts_B[layer_name]
                cka_layer = linear_CKA(acts_A, acts_B)
                dico[layer_name] = cka_layer
            
        with open(name_data, 'wb') as handle:
            pickle.dump(dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(model_nameA,model_nameB,dico)
        
    return(dico)

if __name__ == '__main__': 
    # Petit test 
    
    #l2norm_bw_nets(netA='pretrained',netB='RASTA_small01_modif')
    #l2norm_bw_nets(netA='RASTA_small01_modif',netB='RASTA_small01_modif',suffixA='',suffixB='1')
    #l2norm_bw_nets(netA='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',netB='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',initA=False,initB=True)
    feat_sim(model_nameA='pretrained',model_nameB='RASTA_small01_modif',dataset='RASTA',stats_on_layer='mean')

