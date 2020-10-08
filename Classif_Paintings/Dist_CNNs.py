# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:28:29 2020

The goal of this script is to compare models with differents methods
used in What is being transferred in transfer learning? Neyshabur 2020

To measure the distance between CNN models

Remarques tu peux parfois avoir l'erreur suivante : 
UnknownError: 2 root error(s) found.(0) Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
    
Il te faudra alors peut etre vider le cache qui se trouve a l'endroit suivant : 
    Users gonthier AppData Roaming NVIDIA ComputeCache

@author: gonthier
"""

import numpy as np
import os
import platform
import pathlib
import pickle
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D,Activation,Concatenate,Input,Lambda
from tensorflow.python.keras import Model
import tensorflow as tf
import gzip
import gc
import itertools
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from StatsConstr_ClassifwithTL import predictionFT_net
import Stats_Fcts
from googlenet import inception_v1_oldTF as Inception_V1
from IMDB import get_database
from plots_utils import plt_multiple_imgs
from CompNet_FT_lucidIm import get_fine_tuned_model

from lucid_utils import get_pretrained_model
from CompNet_FT_lucidIm import get_imageNet_weights,get_fine_tuned_model,print_stats_on_diff,get_weights_and_name_layers

from Activation_for_model import get_Model_that_output_Activation,get_Network,get_model_name_wo_oldModel

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


    
def get_l2norm_bw_nets(netA,netB,constrNet='InceptionV1',suffixA='',suffixB='',
                   initA=False,initB=False,
                   ReDo=False):
    """ 
    Distance in parameters space between network. 
    if initA is True : we will use the initialization of A as model
    """
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,'Dists')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'Dists')
    # For output data
    output_path_full = os.path.join(output_path,'data')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_full).mkdir(parents=True, exist_ok=True)
    
    name_data = 'l2norm-'+netA+'-'+suffixA+'-'+netB+'-'+suffixB+'.pkl'
    
    name_data = os.path.join(output_path_full,name_data)
    
    if os.path.exists(name_data) and not(ReDo):
        # The file exist
        with open(name_data, 'rb') as handle:
            data_to_save = pickle.load(handle)
    else:
    
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

        with open(name_data, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    list_name_layers_A,dict_layers_diff,l2_norm_total = data_to_save
    
    return(list_name_layers_A,dict_layers_diff,l2_norm_total)
    
#def compute_CKA_similarity():
#    
#    # According to 
#    # https://github.com/google/svcca/blob/master/tutorials/002_CCA_for_Convolutional_Layers.ipynb
#    # If the two conv layers being compared have the same spatial dimensions, 
#    # we can flatten the spatial dimensions into the number of datapoints:
#    # But we can do average pooling on the two spatial dimension or interpolation between points
#    # Another thing one can do is to interpolate the spatial dimensions of the 
#    # smaller sized conv layer so that they match the large layer. 
#    # There are many ways to do this, and the interpolate library in 
#    # scipy provides access to many different methods.
#
#    return(0)

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
            FT_model,init_model = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix,
                                          verbose=False)
            if FTmodel:
                base_model = FT_model
            else:
                extra_str = '_InitModel'
                base_model = init_model 
        else:
            output = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix,
                                          verbose=False)
            if len(output)==2:
                base_model, init_model = output
            else:
                base_model = output
                
    model,_ = get_Model_that_output_Activation(base_model,list_layers=list_layers)
    #print(model.summary())
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

def layer_3dots(inputA,inputB,sampling_FM=None,k=1,name=''):
    
    ReduceSumA = Lambda(lambda z: K.sum(z, axis=0),name=name+'_sumA')
    ReduceSumB = Lambda(lambda z: K.sum(z, axis=0),name=name+'_sumB')
    ReduceSumSquaredA = Lambda(lambda z: K.sum(K.square(z), axis=0),name=name+'_sumSA')
    ReduceSumSquaredB = Lambda(lambda z: K.sum(K.square(z), axis=0),name=name+'_sumSB')
    
    #ReduceSumSquaredA = Lambda(lambda z: K.sum(K.square(z), axis=0))
    #ReduceSumB = Lambda(lambda z: K.sum(z, axis=1))
    #ReduceSumSquaredB = Lambda(lambda z: K.sum(K.square(z), axis=0))
    
    dotlayer = Lambda(lambda z: tf.linalg.matmul(z[0], z[1], transpose_a=True),name=name+'_dotAB')
    dotlayerAA = Lambda(lambda z: tf.linalg.matmul(z[0], z[1], transpose_a=True),name=name+'_dotAA')
    dotlayerBB = Lambda(lambda z: tf.linalg.matmul(z[0], z[1], transpose_a=True),name=name+'_dotBB')
    
    
    if sampling_FM is None or sampling_FM=='':
        layer_outputA = layers.Flatten(name=name+'_flatten_A')(inputA) # batch size and on the other one channel * h * w
        layer_outputB = layers.Flatten(name=name+'flatten_B')(inputB)
    elif sampling_FM=='selectk2points':
        Ashape = K.int_shape(inputA)
        #print(Ashape)
        b,h,w,c = Ashape
        step = h//(k+1)
        select_k2_A = Lambda(lambda z: z[:,step:-1:step,step:-1:step,:],name=name+'_selectkA')
        # start:end:step
        Bshape = K.int_shape(inputB)
        #print(Bshape)
        b,h,w,c = Bshape
        step = h//(k+1)
        select_k2_B = Lambda(lambda z: z[:,step:-1:step,step:-1:step,:],name=name+'_selectkB') #start:end:step
        layer_outputA = select_k2_A(inputA) # batch size and on the other one channel * h * w
        layer_outputB = select_k2_B(inputB)
        layer_outputA = layers.Flatten(name=name+'_flatten_A')(layer_outputA) # batch size and on the other one channel * h * w
        layer_outputB = layers.Flatten(name=name+'_flatten_B')(layer_outputB)       
        #layer_outputA = layers.GlobalAveragePooling2D()(inputA) # batch size and on the other one channel * h * w
        #layer_outputB = layers.GlobalAveragePooling2D()(inputB)
#    elif sampling_FM=='centralpoints':
#        Ashape = K.int_shape(inputA)
#        #print(Ashape)
#        b,h,w,c = Ashape
#        step = h//(k+1)
#        select_k2_A = Lambda(lambda z: z[:,step:-1:step,step:-1:step,:],name=name+'_selectkA')
#        # start:end:step
#        Bshape = K.int_shape(inputB)
#        #print(Bshape)
#        b,h,w,c = Bshape
#        step = h//(k+1)
#        select_k2_B = Lambda(lambda z: z[:,step:-1:step,step:-1:step,:],name=name+'_selectkB') #start:end:step
#        layer_outputA = select_k2_A(inputA) # batch size and on the other one channel * h * w
#        layer_outputB = select_k2_B(inputB)
#        layer_outputA = layers.Flatten(name=name+'_flatten_A')(layer_outputA) # batch size and on the other one channel * h * w
#        layer_outputB = layers.Flatten(name=name+'_flatten_B')(layer_outputB)       
#        #layer_outputA = layers.GlobalAveragePooling2D()(inputA) # batch size and on the other one channel * h * w
#        #layer_outputB = layers.GlobalAveragePooling2D()(inputB)
    elif sampling_FM=='GlobalAveragePooling2D':
        layer_outputA = layers.GlobalAveragePooling2D(name=name+'_GAP_A')(inputA) # batch size and on the other one channel * h * w
        layer_outputB = layers.GlobalAveragePooling2D(name=name+'_GAP_B')(inputB)
    elif sampling_FM=='AveragePooling2D':
        raise(NotImplementedError)
        pool_size = (7,7) # Il faudrait une maniere automatique de calculer cela en 
        # fct de la taille de la feature maps et du nombre de points voulu
        layer_outputA = layers.AveragePooling2D(pool_size=pool_size)(inputA) # batch size and on the other one channel * h * w
        layer_outputB = layers.AveragePooling2D(pool_size=pool_size)(inputB)
        layer_outputA = layers.Flatten()(layer_outputA) # batch size and on the other one channel * h * w
        layer_outputB = layers.Flatten()(layer_outputB)
    else:
       raise(ValueError(sampling_FM+' is unknown')) 
        
    sumA = ReduceSumA(layer_outputA)
    sumB = ReduceSumB(layer_outputB)
    sum_squaredA =ReduceSumSquaredA(layer_outputA)#,name=name+'_sumSA')
    sum_squaredB = ReduceSumSquaredB(layer_outputB)#,name=name+'_sumSB')
    #layer_outputA_T = TransposeA(layer_outputA)
    #layer_outputB_T = TransposeA(layer_outputB)
    dotAB = dotlayer([layer_outputA,layer_outputB])#
    dotAA = dotlayerAA([layer_outputA,layer_outputA])#,name=name+'_dotAA')
    dotBB = dotlayerBB([layer_outputB,layer_outputB])#,name=name+'_dotBB')
    
    return(sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB)


def get_cumulated_output_model(modelA,modelB,list_layers=None,sampling_FM=None,k=1):
    """
    Provide a keras model which outputs the stats_on_layer == mean or max of each 
    features maps
    """
    
    list_outputs = []
    list_outputs_name = []
    
#    commonInput = Input(input_shape)
#
#    outA = modelA(commonInput)    
#    outB = modelB(commonInput) 
#    
#    print('outA',outA)
#    
#    modelA = Model(commonInput,outA)
#    modelB = Model(commonInput,outB)
#        
#    print(modelA,modelA.summary())
    

    for layerA in modelA.layers:
        
        name_of_this_layer = layerA.name
        layerA._name = name_of_this_layer + '_A'
        
        if list_layers is None:
            if  isinstance(layerA, Conv2D) or isinstance(layerA,Concatenate) or isinstance(layerA,Activation):
                layerA_output = layerA.output
                layerB_output = modelB.get_layer(layerA.name).output
#                layer_outputA = K.flatten(layerA.output) # batch size and on the other one channel * h * w
#                layer_outputB = K.flatten(layerB.output)
#                sumA = K.sum(layer_outputA,axis=0)
#                sumB = K.sum(layer_outputB,axis=0)
#                sum_squaredA =K.sum(K.square(layer_outputA),axis=0)
#                sum_squaredB = K.sum(K.square(layer_outputB),axis=0)
#                dotAB = K.dot(K.transpose(layer_outputA),layer_outputB)
#                dotAA = K.dot(K.transpose(layer_outputA),layer_outputA)
#                dotBB = K.dot(K.transpose(layer_outputB),layer_outputB)
                sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB= layer_3dots(inputA=layerA_output,inputB=layerB_output,
                                                                                   sampling_FM=sampling_FM,k=k,
                                                                                   name=name_of_this_layer)
                list_outputs += [[sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB]]
                list_outputs_name += [layerA.name]
        else:
            if name_of_this_layer in list_layers:
                
                layerA_output = layerA.output
                layerB_output = modelB.get_layer(name_of_this_layer).output
#                layer_outputA = K.flatten(layerA.output) #
#                sumA = K.sum(layer_outputA,axis=0) batch size and on the other one channel * h * w
#                layer_outputB = K.flatten(layerB.output)
#                sumB = K.sum(layer_outputB,axis=0)
#                sum_squaredA =K.sum(K.square(layer_outputA),axis=0)
#                sum_squaredB = K.sum(K.square(layer_outputB),axis=0)
#                dotAB = K.dot(K.transpose(layer_outputA),layer_outputB)
#                dotAA = K.dot(K.transpose(layer_outputA),layer_outputA)
#                dotBB = K.dot(K.transpose(layer_outputB),layer_outputB)
                #print(layerA.output)
                sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB= layer_3dots(inputA=layerA_output,inputB=layerB_output,
                                                                                   sampling_FM=sampling_FM,k=k,
                                                                                   name=name_of_this_layer)
                list_outputs += [[sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB]]
                #print(dotAB)
                list_outputs_name += [name_of_this_layer]
        # Need to rename layers
        
            
    #print(list_outputs)
                
    new_model = Model([modelA.input,modelB.input],list_outputs)
    #new_model = Model(commonInput,list_outputs)
    
    return(new_model,list_outputs_name)
    
def get_linearCKA_bw_nets(dataset,netA,netB,constrNet='InceptionV1',
                                                     suffixA='',suffixB='',
                                                     initA=False,initB=False,
                                                     list_layers=['conv2d0'],
                                                     suffix='',cropCenter = True,
                                                     sampling_FM='GlobalAveragePooling2D',
                                                     k = 1,ReDo=False):
    """
    This function will compute the cumulated sum of the fatures value and the cumulated of the 
    squared of the fatures value and the cumulated dot product between the features of 
    the two models in order to compute the linear CKA    
    @param : sampling_FM='GlobalAveragePooling2D' or AveragePooling2D or None or ''
        selectk2points
    """
    owncloud_mode = True
    if platform.system()=='Windows': 
        if owncloud_mode:
            output_path = os.path.join('C:\\','Users','gonthier','ownCloud','tmp3','Lucid_outputs',constrNet,'Dists')
        else:
            output_path = os.path.join('CompModifModel',constrNet,'Dists')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'Dists')
    # For output data
    output_path_full = os.path.join(output_path,'data')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_full).mkdir(parents=True, exist_ok=True)
    
    name_data = 'linearCKA'+sampling_FM
    if sampling_FM=='selectk2points':
        name_data += str(k)
    name_data+='-'+netA+'-'+suffixA+'-'+netB+'-'+suffixB+'.pkl'
    
    name_data = os.path.join(output_path_full,name_data)
    #print(name_data)
    #return(0)
    if os.path.exists(name_data) and not(ReDo):
        # The file exist
        with open(name_data, 'rb') as handle:
            data_to_save = pickle.load(handle)
    else:
    
        
        
        K.set_learning_phase(0) #IE no training
        tf.compat.v1.disable_eager_execution()
    
        curr_session = tf.get_default_session()
        # close current session
        if curr_session is not None:
            curr_session.close()
        # reset graph
        K.clear_session()
        # create new session
        s = tf.Session()
        K.set_session(s)
        
        # Load info about dataset
        item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset)
        df_test = df_label[df_label['set']=='test']
        #df_test = df_test.sample(128)
        num_samples= len(df_test)
    
        extra_str = ''
        
        net_P = None  
        if netA=='pretrained':
            net_P = 'pretrained'
            net_Autre = netB
            suffix_Autre = suffixB
            init_Autre = initB
        elif netB=='pretrained':
            net_P = 'pretrained'
            net_Autre = netA
            suffix_Autre = suffixA
            init_Autre = initA
    
        if not(net_P is None): # Dans le cas ou un des deux networks est pretrained weights


            keras_netA = get_pretrained_model(Net=constrNet,include_top=False)
            net_finetuned, init_net = get_fine_tuned_model(net_Autre,constrNet=constrNet,suffix=suffix_Autre,
                                                           clearSessionTf=False,verbose=False)
            
            if init_Autre:
                keras_netB = init_net
            else:
                keras_netB = net_finetuned
            
        else: # No pretrained model compared to an other one
            net_finetuned_A, init_net_A = get_fine_tuned_model(netA,constrNet=constrNet,suffix=suffixA,
                                                           clearSessionTf=False,verbose=False)
            if initA:
                keras_netA = init_net_A
            else:
                keras_netA = net_finetuned_A
            net_finetuned_B, init_net_B = get_fine_tuned_model(netB,constrNet=constrNet,suffix=suffixB,
                                                           clearSessionTf=False,verbose=False)
            if initB:
                keras_netB = init_net_B
            else:
                keras_netB = net_finetuned_B
            
        # We get the two networks here
        # Now we will put them together to get the different cumulated information 
        # we want
        
        #with g.as_default():
        
        model,_ = get_cumulated_output_model(keras_netA,keras_netB,list_layers,sampling_FM=sampling_FM,k=k) # the merge model
        
        #print(model.summary())
        if k >1:
            batch_size = 32//(k**2)
        else:
            batch_size = 32
        activations = predictionFT_net(model,df_test,x_col=item_name,y_col=classes,path_im=path_to_img,
                         Net=constrNet,cropCenter=cropCenter,verbose_predict=1,
                         two_images_as_input=True,batch_size=batch_size)
        
        num_steps = num_samples // batch_size
        
        if len(list_layers) == 1:
            activations = [activations]
            
    #    print(len(activations))
    #    print(len(activations[0]))
    #    print(activations[0][0].shape)
    #    print(activations[0][4].shape)
        data_to_save = {}
        l = 0
        for layer_name in list_layers:
            #i = 0
            
            activations_l = activations[l*7:(l+1)*7]
            [sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB] = activations_l
            #print(sumA.shape,sumB.shape,sum_squaredA.shape,sum_squaredB.shape,dotAB.shape,dotAA.shape,dotBB.shape)
            sumA = sumA.reshape(num_steps,-1)
            _,num_pA = sumA.shape
            sumB = sumB.reshape(num_steps,-1)
            _,num_pB = sumB.shape
            #sum_squaredA = sum_squaredA.reshape(-1,num_steps)
            #sum_squaredB = sum_squaredB.reshape(-1,num_steps)
            dotAB = dotAB.reshape(num_steps,num_pA,num_pB)
            dotAA = dotAA.reshape(num_steps,num_pA,num_pA)
            dotBB = dotBB.reshape(num_steps,num_pB,num_pB)
            meanAt = np.sum(sumA/num_samples,axis=0)
            meanBt = np.sum(sumB/num_samples,axis=0)
            dotABt = np.sum(dotAB,axis=0)
            dotAAt = np.sum(dotAA,axis=0)
            dotBBt = np.sum(dotBB,axis=0)
            
            l += 1
    #        for elt in activations_l:
    #            if i ==0:
    #                [sumA,sumB,sum_squaredA,sum_squaredB,dotAB,dotAA,dotBB] = elt
    #                meanAt = sumA/num_samples
    #                meanBt = sumB/num_samples
    #                sum_squaredAt = sum_squaredA/num_samples
    #                sum_squaredBt = sum_squaredB/num_samples
    #                dotABt = dotAB
    #                dotAAt = dotAA
    #                dotBBt = dotBB
    #            else:
    #                [sumA,sumB,sum_squaredA,sum_squaredB,dot] = elt
    #                meanAt += sumA/num_samples
    #                meanBt += sumB/num_samples
    #                sum_squaredAt += sum_squaredA/num_samples
    #                sum_squaredBt += sum_squaredB/num_samples
    #                dotABt += dotAB
    #                dotAAt += dotAA
    #                dotBBt += dotBB
    #            i += 1
            multi_means_AB = np.reshape(meanAt,(-1,1))*np.reshape(meanBt,(1,-1))
            multi_means_AA = np.reshape(meanAt,(-1,1))*np.reshape(meanAt,(1,-1))
            multi_means_BB = np.reshape(meanBt,(-1,1))*np.reshape(meanBt,(1,-1))
            centered_dotABt = dotABt - num_samples*multi_means_AB
            centered_dotAAt = dotAAt - num_samples*multi_means_AA
            centered_dotBBt = dotBBt - num_samples*multi_means_BB
            
            frobenium_norm_AB_squared = np.square(np.linalg.norm(centered_dotABt))
            frobenium_norm_AA = np.linalg.norm(centered_dotAAt)
            frobenium_norm_BB = np.linalg.norm(centered_dotBBt)
            
            linearCKA = frobenium_norm_AB_squared / (frobenium_norm_AA*frobenium_norm_BB)
            
            print(layer_name,linearCKA)
            data_to_save[layer_name] = linearCKA
      
        
        with open(name_data, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        # To clean GPU memory
        s.close()
        K.clear_session()
        gc.collect()

    return(data_to_save)
    
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
        
    #print(model_nameA,model_nameB,dico)
        
    return(dico)

def comp_l2_for_paper(dataset='RASTA',verbose=False):
    """
    compute all the l2 difference for the different pair of models
    """
    
    if dataset=='RASTA':
        
        l_rasta_dico = []
        l_rasta_pairs = []
        
        list_models = ['pretrained',
                       'RASTA_small01_modif',
                       'RASTA_small001_modif',
                       'RASTA_big001_modif',
                       'RASTA_small001_modif_deepSupervision',
                       'RASTA_big001_modif_deepSupervision',
                       'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                       'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
        
        list_with_suffix = ['RASTA_small01_modif',
                       'RASTA_small001_modif',
                       'RASTA_big001_modif',
                       'RASTA_small001_modif_deepSupervision',
                       'RASTA_big001_modif_deepSupervision']
        
        all_pairs = itertools.combinations(list_models, r=2)
         
        for pair in all_pairs:
            netA,netB = pair
            dico = get_l2norm_bw_nets(netA=netA,netB=netB)
            if verbose:print(netA,netB,dico)
            l_rasta_dico += [dico]
            l_rasta_pairs += [(netA,netB)]
            
            if netA in list_with_suffix:
                dico = get_l2norm_bw_nets(netA=netA,netB=netB,
                                                suffixA='1')
                if verbose:print(netA,'1',netB,dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA+'1',netB)]
            if netB in list_with_suffix:
                dico = get_l2norm_bw_nets(netA=netA,netB=netB,
                                                suffixB='1')
                if verbose: print(netA,netB,'1',dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA,netB+'1')]
            if (netB in list_with_suffix) and (netA in list_with_suffix):
                dico = get_l2norm_bw_nets(netA=netA,netB=netB,
                                                suffixA='1',suffixB='1')
                if verbose:print(netA,'1',netB,'1',dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA+'1',netB+'1')]
                
        list_net_init = ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                       'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
                
        for net_init in list_net_init: # Net with a random initialisation at some moment
            dico = get_l2norm_bw_nets(netA=net_init,netB=net_init,
                                         suffixB='1')
            l_rasta_dico += [dico]
            l_rasta_pairs += [(netA,netB+'_init')]
        for net_ in list_with_suffix: # Net with a random initialisation at some moment
            dico = get_l2norm_bw_nets(netA=net_,netB=net_,
                                         initB=True)
            l_rasta_dico += [dico]
            l_rasta_pairs += [(netA,netB+'_init')]
            
        return(l_rasta_dico,l_rasta_pairs)
        
    elif dataset=='Paintings':
        # Paintings dataset 
        l_paintings_dico = []
        l_paintings_pairs = []
        
        # Paintings dataset 
        list_models_name_P = ['Paintings_small01_modif',
                            'Paintings_big01_modif',
                            'Paintings_big001_modif',
                            'Paintings_big001_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_small01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big001_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_small01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_small01_modif_XXRASTA_small01_modifXX',
                            'Paintings_big01_modif_XXRASTA_small01_modifXX',
                            'Paintings_big001_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        # Version plus courte
        list_models_name_P = ['Paintings_small01_modif',
                            'Paintings_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_big01_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        all_pairs = itertools.combinations(list_models_name_P, r=2)
        for pair in all_pairs:
            netA,netB = pair
            dico = get_l2norm_bw_nets(netA=netA,netB=netB)   
            if verbose:print(netA,netB,dico)
            l_paintings_dico += [dico]
            l_paintings_pairs += [(netA,netB)]
        
        return(l_paintings_pairs,l_paintings_dico)
        
    elif dataset=='IconArt_v1':
        l_iconart_dico = []
        l_iconart_pairs = []
        # IconArt v1 dataset 
        # IconArt_v1_small01_modif diverge 
        list_models_name_I = ['IconArt_v1_small01_modif',
                              'IconArt_v1_big01_modif',
                            'IconArt_v1_big001_modif',
                            'IconArt_v1_big001_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_small01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big001_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_small01_modifXX',
                            'IconArt_v1_big01_modif_XXRASTA_small01_modifXX',
                            'IconArt_v1_big001_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        list_models_name_I = ['IconArt_v1_small01_modif',
                            'IconArt_v1_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        all_pairs = itertools.combinations(list_models_name_I, r=2)
        for pair in all_pairs:
            netA,netB = pair
            dico = get_l2norm_bw_nets(netA=netA,netB=netB)      
            if verbose:print(netA,netB,dico)
            l_iconart_dico += [dico]
            l_iconart_pairs += [(netA,netB)]
        
        return(l_iconart_pairs,l_iconart_dico)
        
    else:
        raise(ValueError(dataset+' is unknown'))
    
def test_pb_nan_in_cka():
    netA = 'pretrained'
    netB = 'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200'
    
    dico = get_linearCKA_bw_nets(dataset='RASTA',netA=netA,netB=netB,
                                 list_layers=['conv2d0','conv2d1',
                                      'conv2d2','mixed3a',
                                      'mixed3b','mixed4a',
                                      'mixed4b','mixed4c',
                                      'mixed4d','mixed4e',
                                      'mixed5a','mixed5b'])
    
def comp_cka_for_paper(dataset='RASTA',verbose=False):
    """
    Compute the linear CKA for the different pair of models
    """
    
    if dataset=='RASTA':
    
        l_rasta_dico = []
        l_rasta_pairs = []
        
        # On RASTA first 
        list_models = ['pretrained',
                       'RASTA_small01_modif',
                       'RASTA_small001_modif',
                       'RASTA_big001_modif',
                       'RASTA_small001_modif_deepSupervision',
                       'RASTA_big001_modif_deepSupervision',
                       'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                       'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
        
        list_with_suffix = ['RASTA_small01_modif',
                       'RASTA_small001_modif',
                       'RASTA_big001_modif',
                       'RASTA_small001_modif_deepSupervision',
                       'RASTA_big001_modif_deepSupervision']
        
        all_pairs = itertools.combinations(list_models, r=2)
         
        for pair in all_pairs:
            netA,netB = pair
            dico = get_linearCKA_bw_nets(dataset='RASTA',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'])
            if verbose: print(netA,netB,dico)
            l_rasta_dico += [dico]
            l_rasta_pairs += [pair]
            if netA in list_with_suffix:
                dico = get_linearCKA_bw_nets(dataset='RASTA',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'],
                                                suffixA='1')
                if verbose: print(netA,'1',netB,dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA+'1',netB)]
            if netB in list_with_suffix:
                dico = get_linearCKA_bw_nets(dataset='RASTA',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'],
                                                suffixB='1')
                if verbose: print(netA,netB,'1',dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA,netB+'1')]
            if (netB in list_with_suffix) and (netA in list_with_suffix):
                dico = get_linearCKA_bw_nets(dataset='RASTA',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'],
                                                suffixA='1',suffixB='1')
                if verbose: print(netA,'1',netB,'1',dico)
                l_rasta_dico += [dico]
                l_rasta_pairs += [(netA+'1',netB+'1')]
                
        list_net_init = ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                       'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                       'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
                
        for net_init in list_net_init: # Net with a random initialisation at some moment
            dico = get_linearCKA_bw_nets(dataset='RASTA',netA=net_init,netB=net_init,
                                         initB=True,
                                         list_layers=['conv2d0','conv2d1',
                                              'conv2d2','mixed3a',
                                              'mixed3b','mixed4a',
                                              'mixed4b','mixed4c',
                                              'mixed4d','mixed4e',
                                              'mixed5a','mixed5b'])
            l_rasta_dico += [dico]
            l_rasta_pairs += [(net_init,net_init+'_init')]
            
        # A decommenter et faire tourner sur Morisot
        for net_ in list_with_suffix: # Net with a random initialisation at some moment
            dico = get_linearCKA_bw_nets(dataset='RASTA',netA=net_,netB=net_,
                                         suffixB='1',
                                         list_layers=['conv2d0','conv2d1',
                                              'conv2d2','mixed3a',
                                              'mixed3b','mixed4a',
                                              'mixed4b','mixed4c',
                                              'mixed4d','mixed4e',
                                              'mixed5a','mixed5b'])
            l_rasta_dico += [dico]
            l_rasta_pairs += [(net_init,net_init+'1')]
        
        return(l_rasta_pairs,l_rasta_dico)
            
    elif dataset=='Paintings':
        # Paintings dataset 
        l_paintings_dico = []
        l_paintings_pairs = []
        list_models_name_P = ['Paintings_small01_modif',
                            'Paintings_big01_modif',
                            'Paintings_big001_modif',
                            'Paintings_big001_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_small01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big001_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_small01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_small01_modif_XXRASTA_small01_modifXX',
                            'Paintings_big01_modif_XXRASTA_small01_modifXX',
                            'Paintings_big001_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        # Version courte
        list_models_name_P = ['Paintings_small01_modif',
                            'Paintings_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'Paintings_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'Paintings_big01_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        all_pairs = itertools.combinations(list_models_name_P, r=2)
        for pair in all_pairs:
            netA,netB = pair
            dico = get_linearCKA_bw_nets(dataset='Paintings',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'])   
            if verbose: print(netA,netB,dico)
            l_paintings_dico += [dico]
            l_paintings_pairs += [(netA,netB)]
        
        return(l_paintings_pairs,l_paintings_dico)
        
    elif dataset=='IconArt_v1':
        
        # IconArt dataset 
        l_iconart_dico = []
        l_iconart_pairs = []
        # 'IconArt_v1_small01_modif', diverge
        list_models_name_I = ['IconArt_v1_small01_modif',
                              'IconArt_v1_big01_modif',
                            'IconArt_v1_big001_modif',
                            'IconArt_v1_big001_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_small01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big001_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_small01_modifXX',
                            'IconArt_v1_big01_modif_XXRASTA_small01_modifXX',
                            'IconArt_v1_big001_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        list_models_name_I = ['IconArt_v1_small01_modif',
                            'IconArt_v1_big01_modif_XXRASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                            'IconArt_v1_big01_modif_XXRASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                            'IconArt_v1_small01_modif_XXRASTA_small01_modifXX',
                            'RASTA_small01_modif',
                            'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            'pretrained'
                            ]
        all_pairs = itertools.combinations(list_models_name_I, r=2)
        for pair in all_pairs:
            netA,netB = pair
            dico = get_linearCKA_bw_nets(dataset='IconArt_v1',netA=netA,netB=netB,
                                                         list_layers=['conv2d0','conv2d1',
                                                              'conv2d2','mixed3a',
                                                              'mixed3b','mixed4a',
                                                              'mixed4b','mixed4c',
                                                              'mixed4d','mixed4e',
                                                              'mixed5a','mixed5b'])      
            if verbose: print(netA,netB,dico)
            l_iconart_dico += [dico]
            l_iconart_pairs += [(netA,netB)]
        
        return(l_iconart_pairs,l_iconart_dico)
    
    else:
        raise(ValueError(dataset+' is unknown'))
    
def produce_latex_tab_result_cka(dataset = 'RASTA'):
    
    
    # Realisation d un tableau mais aussi d une matrice geante avec les cka moyen
    
    list_layers=['conv2d0','conv2d1',
                  'conv2d2','mixed3a',
                  'mixed3b','mixed4a',
                  'mixed4b','mixed4c',
                  'mixed4d','mixed4e',
                  'mixed5a','mixed5b']
    list_modified_in_unfreeze50 = ['mixed4a',
                  'mixed4b','mixed4c',
                  'mixed4d','mixed4e',
                  'mixed5a','mixed5b']
    
    
    constrNet = 'InceptionV1'
    
    l_pairs,l_dico = comp_cka_for_paper(dataset=dataset)
    
    main = '\\begin{tabular}{|c|c|'
    for _ in list_layers:
        main += 'c'
    main +='|c|} \\\\ \\hline  \n '
    print(main)
        
    second_line = 'NetA & NetB '
    for layer in list_layers:
        second_line += ' & ' +layer.replace('_','\_')
    second_line += ' & mean'    
    second_line += "\\\\ \\hline "
    print(second_line)

    list_net = []
    list_mean_cka = []
    for pair, dico in zip(l_pairs,l_dico):
        netA = pair[0]
        netB = pair[1]

        if not(netA in list_net):
            list_net += [netA]
        if not(netB in list_net):
            list_net += [netB]
        latex_str = netA.replace('_','\_') + ' & ' + netB.replace('_','\_')
        list_cka = []
        for layer in list_layers:
            cka_l = dico[layer]
            
            if dataset == 'RASTA' and ('RandForUnfreezed' in  netA or 'RandForUnfreezed' in  netB):# cas du randinit
               if not('unfreeze50' in  netA or 'unfreeze50' in  netB):
                   raise(NotImplementedError)
               if layer in list_modified_in_unfreeze50:
                   latex_str += ' & ' + '{0:.4f}'.format(cka_l)
                   list_cka += [cka_l]
                   print(list_cka)
               else:
                   latex_str += ' & '
            else:
               latex_str += ' & ' + '{0:.4f}'.format(cka_l)
               list_cka += [cka_l]
               

        mean_cka = np.mean(list_cka)
        list_mean_cka += [mean_cka]
        latex_str += ' & ' + '{0:.4f}'.format(mean_cka)
        latex_str += "\\\\"
        print(latex_str)

    last_line = '\\hline  \n \\end{tabular} '
    print(last_line)
    
    #print(len(list_net))
    cka_matrice = np.ones((len(list_net),len(list_net)))
    cka_matrice = cka_matrice*np.nan
    
    for pair, mean_cka in zip(l_pairs,list_mean_cka):
        netA = pair[0]
        netB = pair[1]
        cka_matrice[list_net.index(netA),list_net.index(netB)] = mean_cka
        
        
    case_str = dataset
    ext_name = 'linearCKA'
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,'Dists')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,'Dists')
    # For output data
    print(cka_matrice)
    create_matrices_plot_values(matrice=cka_matrice,labels=list_net,
                                min_val=0., max_val=1.,
                                output_path=output_path,
                                ext_name=ext_name,case_str=case_str)
        

def create_matrices_plot_values(matrice,labels,min_val=0., max_val=1.,
                                output_path='',save_or_show=True,
                                ext_name='',case_str='',
                                output_img='png'):

    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['axes.titlesize'] =8
    sns.set()
    #sns.set_style("whitegrid")
    fontsize_text = 16
    h,w = matrice.shape
    print(matrice.shape)

    fig, ax = plt.subplots(figsize=(15,15))
    ax.grid(False)

#    min_val, max_val, diff = 0., 10., 1.
    
    #imshow portion
#    N_points = (max_val - min_val) / diff
#    imshow_data = np.random.rand(N_points, N_points)
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')
    cm = 'viridis'
    pcm  = ax.matshow(matrice,cmap=cm, vmin=min_val, vmax=max_val,alpha=0.5)
    fig.colorbar(pcm, ax=ax)   
    
    
    
    #text portion
    ind_array = np.arange(0, h, 1)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = matrice[y_val,x_val]
        if not(np.isnan(c)):
            ax.text(x_val, y_val, '{0:.2f}'.format(c), va='center', ha='center',color='black',
                    fontsize=fontsize_text)
    
    #set tick marks for grid
    labels_ = []
    for l in labels:
        labels_+= [l.replace('_','\_')]
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels_, rotation=90)#%,fontdict={'fontsize':8}) #plt.xticks(list(range(0,len(list_methods)))
    ax.set_yticklabels(labels_)#,fontdict={'fontsize':8})
    #ax.set_xlim(min_val-diff/2, max_val-diff/2)
    #ax.set_ylim(min_val-diff/2, max_val-diff/2)
    #ax.grid()
    #plt.show()
    if save_or_show:
        if output_img=='png':
            plt.tight_layout()
            path_fig = os.path.join(output_path,ext_name+case_str+'_values.png')
            plt.savefig(path_fig,bbox_inches='tight')
            plt.close()
        if output_img=='tikz':
            path_fig = os.path.join(output_path,ext_name+case_str+'_values.tex')
            tikzplotlib.save(path_fig)
            # From from DataForPerceptual_Evaluation import modify_underscore,modify_labels,modify_fontsizeByInput
            # si besoin
#            modify_underscore(path_fig)
#            modify_labels(path_fig)
#            modify_fontsizeByInput(path_fig)
    else:
        plt.show()
        input('Enter to close.')
        plt.close()


if __name__ == '__main__': 
    # Petit test 
    
    #get_l2norm_bw_nets(netA='pretrained',netB='RASTA_small01_modif')
    #get_l2norm_bw_nets(netA='RASTA_small01_modif',netB='RASTA_small01_modif',suffixA='',suffixB='1')
    #get_l2norm_bw_nets(netA='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',netB='RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',initA=False,initB=True)
    #feat_sim(model_nameA='pretrained',model_nameB='RASTA_small01_modif',dataset='RASTA',stats_on_layer='mean')

#    get_linearCKA_bw_nets(dataset='RASTA',netA='pretrained',netB='RASTA_small01_modif',
#                                                     list_layers=['conv2d0','conv2d1',
#                                                          'conv2d2','mixed3a',
#                                                          'mixed3b','mixed4a',
#                                                          'mixed4b','mixed4c',
#                                                          'mixed4d','mixed4e',
#                                                          'mixed5a','mixed5b'],
#                                                    sampling_FM='selectk2points',
#                                                    k=1)
#    conv2d0 0.9993819
#    conv2d1 0.97327524
#    conv2d2 0.9140044
#    mixed3a 0.91450137
#    mixed3b 0.80959475
#    mixed4a 0.7767332
#    mixed4b 0.70974946
#    mixed4c 0.5975521
#    mixed4d 0.514714
#    mixed4e 0.5433922
#    mixed5a 0.52442396
#    mixed5b 0.2709231
    
#    get_linearCKA_bw_nets(dataset='RASTA',netA='pretrained',netB='RASTA_small01_modif',
#                                                     list_layers=['conv2d0','conv2d1',
#                                                          'conv2d2','mixed3a',
#                                                          'mixed3b','mixed4a',
#                                                          'mixed4b','mixed4c',
#                                                          'mixed4d','mixed4e',
#                                                          'mixed5a','mixed5b'])
#    conv2d0 0.99666345
#    conv2d1 0.90760374
#    conv2d2 0.90344805
#    mixed3a 0.9493459
#    mixed3b 0.89645696
#    mixed4a 0.85551107
#    mixed4b 0.8370101
#    mixed4c 0.74714226
#    mixed4d 0.62428975
#    mixed4e 0.5859339
#    mixed5a 0.57563436
#    mixed5b 0.37219697
#    get_linearCKA_bw_nets(dataset='RASTA',
#                                                     netA='RASTA_small01_modif',
#                                                     netB='RASTA_small01_modif',
#                                                     list_layers=['conv2d0','conv2d1',
#                                                          'conv2d2','mixed3a',
#                                                          'mixed3b','mixed4a',
#                                                          'mixed4b','mixed4c',
#                                                          'mixed4d','mixed4e',
#                                                          'mixed5a','mixed5b'])
    
    comp_cka_for_paper('RASTA')
    comp_cka_for_paper('Paintings')
    comp_cka_for_paper('IconArt_v1')
    comp_l2_for_paper(dataset='RASTA')
    comp_l2_for_paper(dataset='Paintings')
    comp_l2_for_paper(dataset='IconArt_v1')
