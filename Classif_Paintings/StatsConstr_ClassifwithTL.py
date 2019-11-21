#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:14:19 2019

In this script we are looking at the transfer learning of a VGG with some 
statistics imposed on the features maps of the layers

@author: gonthier
"""

from trouver_classes_parmi_K import TrainClassif
import numpy as np
import os.path
from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index,numeral_layers_index_bitsVersion
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
    load_resize_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model,\
    MLP_model,Perceptron_model,vgg_adaDBN,ResNet_AdaIn,ResNet_BNRefinements_Feat_extractor,\
    ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction,ResNet_cut,vgg_suffleInStats,\
    get_ResNet_ROWD_meanX_meanX2_features,get_BaseNorm_meanX_meanX2_features,get_VGGmodel_meanX_meanX2_features
from IMDB import get_database
import pickle
import pathlib
from Classifier_On_Features import TrainClassifierOnAllClass,PredictOnTestSet
from sklearn.metrics import average_precision_score,recall_score,make_scorer,\
    precision_score,label_ranking_average_precision_score,classification_report
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.preprocessing import StandardScaler
from Custom_Metrics import ranking_precision_score
from LatexOuput import arrayToLatex
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
from numba import cuda
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import gc 
import tempfile
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from keras_resnet_utils import getBNlayersResNet50,getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion,\
    fit_generator_ForRefineParameters

from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator

def compute_ref_stats(dico,style_layers,type_ref='mean',imageUsed='all',whatToload = 'varmean',applySqrtOnVar=False):
    """
    This function compute a reference statistics on the statistics of the whole dataset
    """
    vgg_stats_values = []
    for l,layer in enumerate(style_layers):
        stats = dico[layer]
        if whatToload == 'varmean':
            if imageUsed=='all':
                if type_ref=='mean':
                    mean_stats = np.mean(stats,axis=0) # First colunm = variance, second = mean
                    mean_of_means = mean_stats[1,:]
                    mean_of_vars = mean_stats[0,:]
                    if applySqrtOnVar:
                        mean_of_vars = np.sqrt(mean_of_vars) # STD
                    vgg_stats_values += [[mean_of_means,mean_of_vars]]
                    # To return vgg_mean_vars_values
    return(vgg_stats_values)

def get_dict_stats_BaseNormCoherent(target_dataset,source_dataset,target_number_im_considered,\
                                    style_layers,\
                                    list_mean_and_std_source,whatToload,saveformat='h5',\
                                    getBeforeReLU=False,target_set='trainval',applySqrtOnVar=True,\
                                    Net='VGG',cropCenter=False,BV=True,cumulativeWay=False,verbose=False,\
                                    useFloat32=False):
    """
    The goal of this function is to compute a version of the statistics of the 
    features of the VGG or ResNet50
    """
    
    if not(cumulativeWay):
        # We will compute and save the mean and covariance of all the images 
        # En then compute the variances on it
        dict_stats_coherent = {} 
        for i_layer,layer_name in enumerate(style_layers):
            if verbose: print(i_layer,layer_name)
            if i_layer==0:
                style_layers_firstLayer = [layer_name]
                dict_stats_target0 = get_dict_stats(target_dataset,target_number_im_considered,\
                                                    style_layers_firstLayer,\
                                                    whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                    set=target_set,Net=Net,cropCenter=cropCenter,BV=BV)
                dict_stats_coherent[layer_name] = dict_stats_target0[layer_name]
                list_mean_and_std_target_i_m1 = compute_ref_stats(dict_stats_target0,\
                                                style_layers_firstLayer,type_ref='mean',\
                                                imageUsed='all',whatToload=whatToload,\
                                                applySqrtOnVar=applySqrtOnVar)
                current_list_mean_and_std_target = list_mean_and_std_target_i_m1
            else:
                style_layers_imposed = style_layers[0:i_layer]
                style_layers_exported = [style_layers[i_layer]]
                
                if Net=='ResNet50_ROWD' or list_mean_and_std_source is None:
                    list_mean_and_std_source_i = None
                else:
                    list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
                
                dict_stats_target_i = get_dict_stats(target_dataset,target_number_im_considered,\
                                                     style_layers=style_layers_exported,whatToload=whatToload,\
                                                     saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                     set=target_set,Net=Net,\
                                                     style_layers_imposed=style_layers_imposed,\
                                                     list_mean_and_std_source=list_mean_and_std_source_i,\
                                                     list_mean_and_std_target=current_list_mean_and_std_target,\
                                                     cropCenter=cropCenter,BV=BV)
                dict_stats_coherent[layer_name] = dict_stats_target_i[layer_name]
                # Compute the next statistics 
                list_mean_and_std_target_i = compute_ref_stats(dict_stats_target_i,\
                                                style_layers_exported,type_ref='mean',\
                                                imageUsed='all',whatToload=whatToload,\
                                                applySqrtOnVar=applySqrtOnVar)
                current_list_mean_and_std_target += [list_mean_and_std_target_i[-1]]
    else:
        # In this case we will only return the list of the mean and std on the target set
        current_list_mean_and_std_target = []
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
        filename = 'OnlyCoherentStats_'+source_dataset + '_' + str(target_number_im_considered) + '_MeanStd'+'_'+str_layers
        
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp')
        if os.path.isdir(output_path):
            output_path_full = os.path.join(output_path,'Covdata','data')
        else:
            output_path_full = os.path.join('data','Covdata')
        pathlib.Path(output_path_full).mkdir(parents=True, exist_ok=True)  
        
        if not(target_set=='' or target_set is None):
            filename += '_'+ target_set
        if 'VGG' in Net and getBeforeReLU:
            filename += '_BeforeReLU'
        filename += '.pkl'
        filename_path= os.path.join(output_path_full,filename)
        if not os.path.isfile(filename_path):
            dict_stats_coherent = {} 
            for i_layer,layer_name in enumerate(style_layers):
                if verbose: print(i_layer,layer_name)
                if i_layer==0:
                    style_layers_firstLayer = [layer_name]
                    mean_and_std_layer = compute_mean_std_onDataset(target_dataset,target_number_im_considered,style_layers_firstLayer,\
                                                                    set=target_set,getBeforeReLU=getBeforeReLU,\
                                                                    Net=Net,style_layers_imposed=[],\
                                                                    list_mean_and_std_source=None,list_mean_and_std_target=None,\
                                                                    cropCenter=cropCenter,useFloat32=useFloat32)
                    dict_stats_coherent[layer_name] = mean_and_std_layer
                    current_list_mean_and_std_target = [mean_and_std_layer]
                else:
                    style_layers_imposed = style_layers[0:i_layer]
                    style_layers_exported = [style_layers[i_layer]]
                    
                    if Net=='ResNet50_ROWD' or list_mean_and_std_source is None:
                        list_mean_and_std_source_i = None
                    else:
                        list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
                    style_layers_firstLayer = [layer_name]
                    mean_and_std_layer = compute_mean_std_onDataset(target_dataset,target_number_im_considered,style_layers_firstLayer,\
                                                                    set=target_set,getBeforeReLU=getBeforeReLU,\
                                                                    Net=Net,style_layers_imposed=[],\
                                                                    list_mean_and_std_source=list_mean_and_std_source_i,\
                                                                    list_mean_and_std_target=current_list_mean_and_std_target,\
                                                                    cropCenter=cropCenter,useFloat32=useFloat32)
                    dict_stats_coherent[layer_name] = mean_and_std_layer
                    current_list_mean_and_std_target += [mean_and_std_layer]
            # Need to save the dict_stats_coherent 
            with open(filename_path, 'wb') as handle:
                pickle.dump(dict_stats_coherent, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filename_path, 'rb') as handle:
                dict_stats_coherent = pickle.load(handle)
            for i_layer,layer_name in enumerate(style_layers):
                mean_and_std_layer =  dict_stats_coherent[layer_name]
                current_list_mean_and_std_target += [mean_and_std_layer]

    return(dict_stats_coherent,current_list_mean_and_std_target)

def compute_mean_std_onDataset(dataset,number_im_considered,style_layers,\
                   set='',getBeforeReLU=False,\
                   Net='VGG',style_layers_imposed=[],\
                   list_mean_and_std_source=[],list_mean_and_std_target=[],\
                   cropCenter=False,useFloat32=True):
    """
    this function will directly compute mean and std of the features maps on the source_dataset
    in an efficient way without saving covariance matrices for the style_layers
    @param : dtype is the 
    """
    # Les differents reseaux retournr la moyenne spatiale des features et la 
    # moyenne spatiale des carrées des features
    if Net=='VGG':
        net_get_SpatialMean_SpatialMeanOfSquare =  get_VGGmodel_meanX_meanX2_features(style_layers,getBeforeReLU=getBeforeReLU)
    elif Net=='VGGBaseNorm' or Net=='VGGBaseNormCoherent':
        style_layers_exported = style_layers
        net_get_SpatialMean_SpatialMeanOfSquare = get_BaseNorm_meanX_meanX2_features(style_layers_exported,\
                        style_layers_imposed,list_mean_and_std_source,list_mean_and_std_target,\
                        getBeforeReLU=getBeforeReLU)
    elif Net=='ResNet50_ROWD_CUMUL': # Base coherent here also but only update the batch normalisation
        style_layers_exported = style_layers
        net_get_SpatialMean_SpatialMeanOfSquare = get_ResNet_ROWD_meanX_meanX2_features(style_layers_exported,style_layers_imposed,\
                                    list_mean_and_std_target,transformOnFinalLayer=None,
                                    res_num_layers=50,weights='imagenet')
    else:
        print(Net,'is inknown')
        raise(NotImplementedError)
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    
    if set=='train':
        df = df_label[df_label['set']=='train']
    elif set=='test':
        df = df_label[df_label['set']=='test']
    elif set==str_val or set=='val' or set=='validation':
        df = df_label[df_label['set']==str_val]
    elif set=='trainval':
        df1 = df_label[df_label['set']=='train']
        df2 = df_label[df_label['set']==str_val]
        df =df1.append(df2)
    x_col = item_name
    df[x_col] = df[x_col].apply(lambda x : x + '.jpg')
    
    datagen= tf.keras.preprocessing.image.ImageDataGenerator()
    
    if cropCenter:
        from functools import partial
        preprocessing_function = partial(load_and_crop_img_forImageGenerator,Net)
    else:
        if 'VGG' in Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet50' in Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        else:
            print(Net,'is unknwon')
            raise(NotImplementedError)
            
    test_generator=datagen.flow_from_dataframe(dataframe=df, directory=path_to_img,\
                                                x_col=x_col,\
                                                class_mode=None,shuffle=False,\
                                                target_size=(224,224), batch_size=32,
                                                preprocessing_function=preprocessing_function,
                                                use_multiprocessing=True,workers=3)
    predictions = net_get_SpatialMean_SpatialMeanOfSquare.predict_generator(test_generator)
    meanX,meanX2 = predictions
#    if dtype=='float64':
#        meanX.astype('float64')
#        meanX2.astype('float64')
#    print(meanX.shape,meanX2.shape)
#    print(meanX)
#    print(meanX2)
    expectation_meanX = np.mean(meanX,axis=0)
    varX = meanX2 - np.power(meanX,2)
#    varX_beforeClip = varX
#    varX = np.where(varX<0.0 and varX>=-10**(-5), 0.0, varX)
    varX = varX.clip(min=0.0)
#    print(varX)
    try:
        assert(varX>=0).all()
    except AssertionError as e:
        print('varX negative values :',varX[np.where(varX<0.0)])
        print('varX negative index :',np.where(varX<0.0))
        raise(e)
    expectation_stdX = np.mean(np.sqrt(varX),axis=0)
#    if dtype=='float64':
#        expectation_meanX.astype('float32')
#        expectation_stdX.astype('float32')
    return(expectation_meanX,expectation_stdX)

def learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
                   constrNet='VGG',kind_method='FT',\
                   style_layers = ['block1_conv1',
                                    'block2_conv1',
                                    'block3_conv1', 
                                    'block4_conv1', 
                                    'block5_conv1'
                                   ],normalisation=False,gridSearch=True,ReDo=False,\
                                   transformOnFinalLayer='',number_im_considered = 1000,\
                                   set='',getBeforeReLU=False,forLatex=False,epochs=20,\
                                   pretrainingModif=True,weights='imagenet',opt_option=[0.01],\
                                   optimizer='adam',freezingType='FromTop',verbose=False,\
                                   plotConv=False,batch_size=32,regulOnNewLayer=None,\
                                   regulOnNewLayerParam=[],return_best_model=False,\
                                   onlyReturnResult=False,dbn_affine=True,m_per_group=16,
                                   momentum=0.9,batch_size_RF=32,epochs_RF=20,cropCenter=True,\
                                   BV=True,dropout=None,nesterov=False,SGDmomentum=0.0,decay=0.0,\
                                   kind_of_shuffling='shuffle',useFloat32=True):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
        - linear SVM 'LinearSVC' or two layers NN 'MLP2' or MLP1 for perceptron
    @param : features : which features we will use
        - fc2, fc1, flatten block5_pool (need a transformation)
    @param : constrNet the constrained net used
    TODO : VGGInNorm, VGGInNormAdapt seulement sur les features qui répondent trop fort, VGGGram
    @param : kind_method the type of methods we will use : TL or FT
    @param : if we use a set to compute the statistics
    @param : getBeforeReLU=False if True we will impose the statistics before the activation ReLU fct
    @param : forLatex : only plot performance score to print them in latex
    @param : epochs number of epochs for the finetuning (FT case)
    @param : pretrainingModif : we modify the pretrained net for the case FT + VGG 
        it can be a boolean True of False or a 
    @param : opt_option : learning rate different for the SGD
    @param : freezingType : the way we unfreeze the pretained network : 'FromBottom','FromTop','Alter'
    @param : ReDo : we erase the output performance file
    @param : plotConv : plot the loss function in train and val
    @param : regulOnNewLayer : None l1 l2 or l1_l2
    @param : regulOnNewLayerParam the weight on the regularizer
    @param : return_best_model if True we will load and return the best model
    @param : onlyReturnResult : only return the results if True and nothing computed will return None
    @param : dbn_affine : use of Affine decorrelated BN in  VGGAdaDBN model
    @param : m_per_group : number of group for the VGGAdaDBN model (with decorrelated BN)
    @param : momentum : momentum for the refinement of the batch statistics
    @param : batch_size_RF : batch size for the refinement of the batch statistics
    @param : epochs_RF : number of epochs for the refinement of the batch statistics
    @param : cropCenter if True we only consider the central crop of the image as in Crowley 2016
    @param : BV : if true use the compress value for the layer index
    @param : dropout : if None no dropout otherwise on the new layer
    @param : nesterov : nesterov approximation for MLP
    @param : SGDmomentum : SGD momentum in the gradient descent
    @param : decay : learning rate decay for MLP model
    @param : kind_of_shuffling=='shuffle' or 'roll'  for VGGshuffleInStats
    @param : useFloat32 is the use of float32 for cumulated spatial mean of features and squared features
    """
#    tf.enable_eager_execution()
    # for ResNet you need to use different layer name such as  ['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1']
    assert(freezingType in ['FromBottom','FromTop','Alter'])
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if kind_method=='FT':
        model_output_path = os.path.join(output_path,'model')
        pathlib.Path(model_output_path).mkdir(parents=True, exist_ok=True) 
    
    if kind_method=='TL':
        if not (transformOnFinalLayer is None or transformOnFinalLayer=='') and features in ['fc2','fc1','flatten','avg_pool']:
            print('Incompatible feature layer and transformation applied to this layer')
            raise(NotImplementedError)
    
    if 'VGG' in  constrNet: 
        if BV:
            num_layers = numeral_layers_index_bitsVersion(style_layers)
        else:
            num_layers = numeral_layers_index(style_layers)
    elif 'ResNet' in constrNet:
        if BV: 
            num_layers = getResNetLayersNumeral_bitsVersion(style_layers)
        else: 
            num_layers = getResNetLayersNumeral(style_layers)
    # Compute statistics on the source_dataset
    if source_dataset is None:
        constrNet='VGG'
        
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(target_dataset)
        
    name_base = constrNet + '_'  +target_dataset +'_'
    if not(constrNet=='VGG') and not(constrNet=='ResNet50'):
        if kind_method=='TL' and constrNet in ['VGGInNorm','VGGInNormAdapt','VGGBaseNorm','VGGBaseNormCoherent']:
            name_base += source_dataset +str(number_im_considered)
        name_base +=  '_' + num_layers
    if kind_method=='FT' and (weights is None):
        name_base += '_RandInit' # Random initialisation 
    if kind_method=='FT':
        if not(optimizer=='adam'):
            name_base += '_'+optimizer
        if len(opt_option)==2:
            multiply_lrp, lr = opt_option
            name_base += '_lrp'+str(multiply_lrp)+'_lr'+str(lr)
        if len(opt_option)==1:
            lr = opt_option[0]
            name_base += '_lr'+str(lr)
        # Default value used
        if constrNet=='VGGAdaDBN':
            name_base += '_MPG'+str(m_per_group)
            if dbn_affine:
                name_base += '_Affine'
                
    if kind_method=='TL':
        if constrNet=='ResNet50_BNRF': # BN Refinement
            name_base += '_m'+str(momentum)+'_bsRF'+str(batch_size_RF)+'_ep'+str(epochs_RF)
            
    if not(set=='' or set is None):
        name_base += '_'+set
    if constrNet=='VGG':
        getBeforeReLU  = False
    if constrNet in ['VGG','ResNet50'] and kind_method=='FT':
        if type(pretrainingModif)==bool:
            if pretrainingModif==False:
                name_base +=  '_wholePretrainedNetFreeze'
        else:
            if not(freezingType=='FromTop'):
                name_base += '_'+freezingType
            name_base += '_unfreeze'+str(pretrainingModif)
            
    if getBeforeReLU:
        name_base += '_BeforeReLU'
#    if not(kind_method=='FT' and constrNet=='VGGAdaIn'):
#        name_base +=  features 
    if 'VGG' in constrNet: # VGG kind family
        if features in  ['fc2','fc1','flatten']:
            name_base += '_' + features
        else:
            if not(features=='block5_pool'):
                name_base += '_' + features
            if not((transformOnFinalLayer is None) or (transformOnFinalLayer=='')):
               name_base += '_'+ transformOnFinalLayer
    elif 'ResNet' in constrNet: # ResNet kind family
        if features in ['avg_pool']: # A remplir
            name_base += '_' + features
        else:
            if not(features=='activation_48'): # TODO ici
                name_base += '_' + features
            if not((transformOnFinalLayer is None) or (transformOnFinalLayer=='')):
               name_base += '_'+ transformOnFinalLayer
   
    if constrNet=='VGGsuffleInStats':
        if not(kind_of_shuffling=='shuffle'):
            name_base += '_'+ kind_of_shuffling
   
    if constrNet=='ResNet50_ROWD_CUMUL' and useFloat32:
        name_base += '_useFloat32'
    
    if cropCenter:   
        name_base += '_CropCenter'  
    name_base += '_' + kind_method   
    
    # features can be 'flatten' with will output a 25088 dimension vectors = 7*7*512 features
    
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    s = tf.InteractiveSession()
    K.set_session(s)
    
    if kind_method=='TL': # Transfert Learning
        final_layer = features
        name_pkl_im = target_dataset +'.pkl'
        name_pkl_values  = name_base+ '_Features.pkl'
        name_pkl_im = os.path.join(output_path,name_pkl_im)
        name_pkl_values = os.path.join(output_path,name_pkl_values)

        if  not(onlyReturnResult):
            if not os.path.isfile(name_pkl_values):
                if not(forLatex):
                    print('== We will compute the reference statistics and / or the extraction features network ==')
                    print('They will be saved in ',name_pkl_values)
                features_net = None
                im_net = []
                # Load Network 
                if constrNet=='VGG':
                    network_features_extraction = vgg_cut(final_layer,\
                                                          transformOnFinalLayer=transformOnFinalLayer,\
                                                          weights='imagenet')
                elif constrNet=='VGGInNorm' or constrNet=='VGGInNormAdapt':
                    whatToload = 'varmean'
                    dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                           whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set,Net='VGG',\
                           cropCenter=cropCenter,BV=BV)
                    # Compute the reference statistics
                    vgg_mean_stds_values = compute_ref_stats(dict_stats,style_layers,type_ref='mean',\
                                                         imageUsed='all',whatToload =whatToload,
                                                         applySqrtOnVar=True)
                    if constrNet=='VGGInNormAdapt':
                        network_features_extraction = vgg_InNorm_adaptative(style_layers,
                                                                           vgg_mean_stds_values,
                                                                           final_layer=final_layer,
                                                                           transformOnFinalLayer=transformOnFinalLayer,
                                                                           HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
                    elif constrNet=='VGGInNorm':
                        network_features_extraction = vgg_InNorm(style_layers,
                                                                vgg_mean_stds_values,
                                                                final_layer=final_layer,
                                                                transformOnFinalLayer=transformOnFinalLayer,
                                                                HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
                elif constrNet=='VGGBaseNorm':
                    whatToload = 'varmean'
                    dict_stats_source = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                           whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set,Net='VGG',\
                           cropCenter=cropCenter,BV=BV)
                    # Compute the reference statistics
                    list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                         imageUsed='all',whatToload =whatToload,
                                                         applySqrtOnVar=True)
                    target_number_im_considered = None
                    target_set = 'trainval' # Todo ici
                    dict_stats_target = get_dict_stats(target_dataset,target_number_im_considered,style_layers,\
                           whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=target_set,\
                           cropCenter=cropCenter,BV=BV)
                    # Compute the reference statistics
                    list_mean_and_std_target = compute_ref_stats(dict_stats_target,style_layers,type_ref='mean',\
                                                         imageUsed='all',whatToload =whatToload,
                                                         applySqrtOnVar=True)
    #                
    #                
                    network_features_extraction = vgg_BaseNorm(style_layers,list_mean_and_std_source,
                        list_mean_and_std_target,final_layer=final_layer,transformOnFinalLayer=transformOnFinalLayer,
                        getBeforeReLU=getBeforeReLU)
                    
                elif constrNet=='VGGBaseNormCoherent':
                    # A more coherent way to compute the VGGBaseNormalisation
                    # We will pass the dataset several time through the net modify bit after bit to 
                    # get the coherent mean and std of the target domain
                    whatToload = 'varmean'
                    dict_stats_source = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                           whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set,Net='VGG',\
                           cropCenter=cropCenter,BV=BV)
                    # Compute the reference statistics
                    list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                         imageUsed='all',whatToload =whatToload,
                                                         applySqrtOnVar=True)
                    target_number_im_considered = None
                    target_set = 'trainval'
                    dict_stats_target,list_mean_and_std_target = get_dict_stats_BaseNormCoherent(target_dataset,source_dataset,target_number_im_considered,\
                           style_layers,list_mean_and_std_source,whatToload,saveformat='h5',\
                           getBeforeReLU=getBeforeReLU,target_set=target_set,\
                           applySqrtOnVar=True,cropCenter=cropCenter,BV=BV,verbose=verbose) # It also computes the reference statistics (mean,var)
                    
                    network_features_extraction = vgg_BaseNorm(style_layers,list_mean_and_std_source,
                        list_mean_and_std_target,final_layer=final_layer,transformOnFinalLayer=transformOnFinalLayer,
                        getBeforeReLU=getBeforeReLU)
                
                elif constrNet=='ResNet50':
                    # in the case of ResNet50 : final_alyer = features = 'activation_48'
                    network_features_extraction = ResNet_cut(final_layer=features,\
                                     transformOnFinalLayer ='GlobalMaxPooling2D',\
                             verbose=verbose,weights='imagenet',res_num_layers=50)
                
                elif constrNet=='ResNet50_ROWD':
                    # Refinement the batch normalisation : normalisation statistics
                    # Once on the Whole train val Dataset on new dataset
                    list_mean_and_std_source = None
                    target_number_im_considered = None
                    whatToload = 'varmean'
                    target_set = 'trainval'
                    dict_stats_target,list_mean_and_std_target = get_dict_stats_BaseNormCoherent(
                            target_dataset,source_dataset,target_number_im_considered,\
                            style_layers,list_mean_and_std_source,whatToload,saveformat='h5',\
                            getBeforeReLU=getBeforeReLU,target_set=target_set,\
                            applySqrtOnVar=True,Net=constrNet,cropCenter=cropCenter,\
                            BV=BV,verbose=verbose) # It also computes the reference statistics (mean,var)
                    
                    network_features_extraction = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(
                                   style_layers,list_mean_and_std_target=list_mean_and_std_target,\
                                   final_layer=features,\
                                   transformOnFinalLayer=transformOnFinalLayer,res_num_layers=50,\
                                   weights='imagenet')
                    
                elif constrNet=='ResNet50_ROWD_CUMUL':
                    # Refinement the batch normalisation : normalisation statistics
                    # Once on the Whole train val Dataset on new dataset
                    # In a cumulative way to be more efficient
                    # Il faudrait peut etre mieux gerer les cas ou la variances est négatives
                    list_mean_and_std_source = None
                    target_number_im_considered = None
                    whatToload = 'varmean'
                    target_set = 'trainval'
                    dict_stats_target,list_mean_and_std_target = get_dict_stats_BaseNormCoherent(
                            target_dataset,source_dataset,target_number_im_considered,\
                            style_layers,list_mean_and_std_source,whatToload,saveformat='h5',\
                            getBeforeReLU=getBeforeReLU,target_set=target_set,\
                            applySqrtOnVar=True,Net=constrNet,cropCenter=cropCenter,\
                            BV=BV,cumulativeWay=True,verbose=verbose,useFloat32=useFloat32) # It also computes the reference statistics (mean,var)
                    
                    network_features_extraction = ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction(
                                   style_layers,list_mean_and_std_target=list_mean_and_std_target,\
                                   final_layer=features,\
                                   transformOnFinalLayer=transformOnFinalLayer,res_num_layers=50,\
                                   weights='imagenet')
                    
                elif constrNet=='ResNet50_BNRF':
                    res_num_layers = 50
                    network_features_extraction= get_ResNet_BNRefin(df=df_label,\
                                    x_col=item_name,path_im=path_to_img,\
                                    str_val=str_val,num_of_classes=len(classes),Net=constrNet,\
                                    weights=weights,res_num_layers=res_num_layers,\
                                    transformOnFinalLayer=transformOnFinalLayer,\
                                    kind_method=kind_method,\
                                    batch_size=batch_size_RF,momentum=momentum,
                                    num_epochs_BN=epochs_RF,output_path=output_path)

                else:
                    print(constrNet,'is unknown')
                    raise(NotImplementedError)
                    
                if not(forLatex):
                    print('== We will compute the bottleneck features ==')
                # Compute bottleneck features on the target dataset
                for i,name_img in  enumerate(df_label[item_name]):
                    im_path =  os.path.join(path_to_img,name_img+'.jpg')
                    if cropCenter:
                        image = load_and_crop_img(path=im_path,Net=constrNet,target_smallest_size=224,
                                            crop_size=224,interpolation='lanczos:center')
                          # For VGG or ResNet size == 224
                    else:
                        image = load_resize_and_process_img(im_path,Net=constrNet)
                    features_im = network_features_extraction.predict(image)
                    #print(i,name_img,features_im.shape,features_im[0,0:10])
                    if features_net is not None:
                        features_net = np.vstack((features_net,np.array(features_im)))
                        im_net += [name_img]
                    else:
                        features_net =np.array(features_im)
                        im_net = [name_img]
                with open(name_pkl_values, 'wb') as pkl:
                    pickle.dump(features_net,pkl)
                
                with open(name_pkl_im, 'wb') as pkl:
                    pickle.dump(im_net,pkl)
            else: # Load the precomputed data
                with open(name_pkl_values, 'rb') as pkl:
                    features_net = pickle.load(pkl)
                
                with open(name_pkl_im, 'rb') as pkl:
                    im_net = pickle.load(pkl)
                
    if not(batch_size==32):
        batch_size_str =''
    else:
        batch_size_str ='_bs'+str(batch_size)
    AP_file  = name_base
    if kind_method=='TL':
        AP_file += '_'+final_clf
        if final_clf in ['MLP2','MLP1','MLP3']:
            AP_file += '_'+str(epochs)+batch_size_str+'_'+optimizer
            lr = opt_option[-1]
            AP_file += '_lr' +str(lr) 
        if return_best_model:
            AP_file += '_BestOnVal'
        if normalisation:
            AP_file +=  '_Norm' 
            if gridSearch:
                AP_file += '_GS'
        if not(regulOnNewLayer is None):
           AP_file += '_'+regulOnNewLayer
           if len(regulOnNewLayerParam)>0:
               if regulOnNewLayer=='l1' or  regulOnNewLayer=='l1':
                   AP_file += '_'+  regulOnNewLayerParam[0]
               elif regulOnNewLayer=='l1_l2':
                   AP_file += '_'+  regulOnNewLayerParam[0]+'_'+ regulOnNewLayerParam[1]
        if not(dropout is None):
             AP_file += '_dropout'+str(dropout)
        if optimizer=='SGD':
            if nesterov:
                AP_file += '_nes'
            if not(SGDmomentum==0.0):
                AP_file += '_sgdm'+str(SGDmomentum)
        if not(decay==0.0):
            AP_file += '_dec'+str(decay)
    elif kind_method=='FT':
       AP_file += '_'+str(epochs)+batch_size_str
       if not(optimizer=='adam'):
           AP_file += '_'+optimizer
       if not(regulOnNewLayer is None):
           AP_file += '_'+regulOnNewLayer
           if len(regulOnNewLayerParam)>0:
               if regulOnNewLayer=='l1' or  regulOnNewLayer=='l1':
                   AP_file += '_'+  regulOnNewLayerParam[0]
               elif regulOnNewLayer=='l1_l2':
                   AP_file += '_'+  regulOnNewLayerParam[0]+'_'+ regulOnNewLayerParam[1]
       if not(dropout is None):
            AP_file += '_dropout'+str(dropout) 
       if optimizer=='SGD':
           if nesterov:
                AP_file += '_nes'
           if not(SGDmomentum==0.0):
                AP_file += '_sgdm'+str(SGDmomentum)
       if not(decay==0.0):
           AP_file += '_dec'+str(decay)
       if return_best_model:
            AP_file += '_BestOnVal'
    
    AP_file_base =  AP_file
    AP_file_pkl =AP_file_base+'_AP.pkl'
    APfilePath =  os.path.join(output_path,AP_file_pkl)
    if verbose: print(APfilePath)
    
    # TL or FT method
    if kind_method=='TL':
        Latex_str = constrNet 
        if style_layers==['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1']:
            Latex_str += ' Block1-5\_conv1'
        elif style_layers==['block1_conv1','block2_conv1']:
            Latex_str += ' Block1-2\_conv1' 
        elif style_layers==['block1_conv1']:
            Latex_str += ' Block1\_conv1' 
        else:
            for layer in style_layers:
                Latex_str += layer
        Latex_str += ' ' +features.replace('_','\_')
        Latex_str += ' ' + transformOnFinalLayer
        Latex_str += ' '+final_clf
        if final_clf=='LinearSVC': 
            if gridSearch:
                Latex_str += 'GS'
            else:
                Latex_str += ' no GS'
        if getBeforeReLU:
            Latex_str += ' BFReLU'
    elif kind_method=='FT':
        Latex_str = constrNet +' '+transformOnFinalLayer  + ' ep :' +str(epochs)
        if type(pretrainingModif)==bool:
            if not(pretrainingModif):
                Latex_str += ' All Freeze'
        else:
            Latex_str += ' Unfreeze '+str(pretrainingModif) + ' ' +freezingType
        if getBeforeReLU:
            Latex_str += ' BFReLU'
        if weights is None:
            Latex_str += ' RandInit'
    
    if (not(os.path.isfile(APfilePath)) or ReDo) and not(onlyReturnResult):
        
        if target_dataset=='Paintings':
            sLength = len(df_label[item_name])
            classes_vectors = np.zeros((sLength,num_classes))
            for i in range(sLength):
                for j in range(num_classes):
                    if( classes[j] in df_label['classe'][i]):
                        classes_vectors[i,j] = 1
            if kind_method=='FT':
                df_copy = df_label.copy()
                for j,c in enumerate(classes):
                    df_copy[c] = classes_vectors[:,j].astype(int)
                    #df_copy[c] = df_copy[c].apply(lambda x : bool(x))
                df_label = df_copy
                df_label_test = df_label[df_label['set']=='test']
            y_test = classes_vectors[df_label['set']=='test',:]
        elif target_dataset=='IconArt_v1':
            sLength = len(df_label[item_name])
            classes_vectors =  df_label[classes].values
            df_label_test = df_label[df_label['set']=='test']
            y_test = classes_vectors[df_label['set']=='test',:]
        else:
            raise(NotImplementedError)
    
        if kind_method=='TL':
            # Get Train set in form of numpy array
            index_train = df_label['set']=='train'
            if not(forLatex):
                print('classes_vectors.shape',classes_vectors.shape)
                print('features_net.shape',features_net.shape)
            X_train = features_net[index_train,:]
            y_train = classes_vectors[df_label['set']=='train',:]
            
            X_test= features_net[df_label['set']=='test',:]
            
            X_val = features_net[df_label['set']==str_val,:]
            y_val = classes_vectors[df_label['set']==str_val,:]
            
            Xtrainval = np.vstack([X_train,X_val])
            ytrainval = np.vstack([y_train,y_val])
            
            if normalisation:
                scaler = StandardScaler()
                Xtrainval = scaler.fit_transform(Xtrainval)
                X_test = scaler.transform(X_test)
            
            if final_clf=='LinearSVC':
                dico_clf=TrainClassifierOnAllClass(Xtrainval,ytrainval,clf=final_clf,gridSearch=gridSearch)
                # Prediction
                dico_pred = PredictOnTestSet(X_test,dico_clf,clf=final_clf)
                metrics = evaluationScoreDict(y_test,dico_pred)
            elif final_clf in ['MLP2','MLP1','MLP3']:
                if final_clf=='MLP2':
                    model = MLP_model(num_of_classes=num_classes,optimizer=optimizer,lr=lr,\
                                      regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam,dropout=dropout,\
                                      nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                if final_clf=='MLP3':
                    model = MLP_model(num_of_classes=num_classes,optimizer=optimizer,lr=lr,num_layers=3,\
                                      regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam,dropout=dropout,\
                                      nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                elif final_clf=='MLP1':
                    model = Perceptron_model(num_of_classes=num_classes,optimizer=optimizer,lr=lr,\
                                            regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam,dropout=dropout,\
                                            nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                
                model = TrainMLP(model,X_train,y_train,X_val,y_val,batch_size,epochs,\
                                 verbose=verbose,plotConv=plotConv,return_best_model=return_best_model)
                predictions = model.predict(X_test, batch_size=1)
                metrics = evaluationScore(y_test,predictions)  
            else:
                print(final_clf,'doesn t exist')
                raise(NotImplementedError)
                
        elif kind_method=='FT':
            # We fineTune a VGG
            if constrNet=='VGG':
                getBeforeReLU = False
                model = VGG_baseline_model(num_of_classes=num_classes,pretrainingModif=pretrainingModif,
                                           transformOnFinalLayer=transformOnFinalLayer,weights=weights,
                                           optimizer=optimizer,opt_option=opt_option,freezingType=freezingType,
                                           final_clf=final_clf,final_layer=features,verbose=verbose,
                                           regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam
                                           ,dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                
            elif constrNet=='VGGAdaIn':
                model = vgg_AdaIn(style_layers,num_of_classes=num_classes,weights=weights,\
                          transformOnFinalLayer=transformOnFinalLayer,getBeforeReLU=getBeforeReLU,\
                          final_clf=final_clf,final_layer=features,verbose=verbose,\
                          optimizer=optimizer,opt_option=opt_option,regulOnNewLayer=regulOnNewLayer,\
                          regulOnNewLayerParam=regulOnNewLayerParam,dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                
            elif constrNet=='VGGAdaDBN':
                model = vgg_adaDBN(style_layers,num_of_classes=num_classes,\
                          transformOnFinalLayer=transformOnFinalLayer,getBeforeReLU=getBeforeReLU,verbose=verbose,\
                          weights=weights,final_layer=features,final_clf=final_clf,\
                          optimizer=optimizer,opt_option=opt_option,regulOnNewLayer=regulOnNewLayer,\
                          regulOnNewLayerParam=regulOnNewLayerParam,\
                          dbn_affine=dbn_affine,m_per_group=m_per_group,dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
            
            elif constrNet=='VGGsuffleInStats':
                model = vgg_suffleInStats(style_layers,num_of_classes=num_classes,\
                          transformOnFinalLayer=transformOnFinalLayer,getBeforeReLU=getBeforeReLU,verbose=verbose,\
                          weights=weights,final_layer=features,final_clf=final_clf,\
                          optimizer=optimizer,opt_option=opt_option,regulOnNewLayer=regulOnNewLayer,\
                          regulOnNewLayerParam=regulOnNewLayerParam,\
                          dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay,\
                          kind_of_shuffling=kind_of_shuffling)
            
            elif constrNet=='ResNet50':
                getBeforeReLU = False
                model = ResNet_baseline_model(num_of_classes=num_classes,pretrainingModif=pretrainingModif,
                                           transformOnFinalLayer=transformOnFinalLayer,weights=weights,\
                                           res_num_layers=50,final_clf=final_clf,verbose=verbose,\
                                           freezingType=freezingType,dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
            elif constrNet=='ResNet50AdaIn':
                getBeforeReLU = False
                model = ResNet_AdaIn(style_layers,num_of_classes=num_classes,\
                                           transformOnFinalLayer=transformOnFinalLayer,weights=weights,\
                                           res_num_layers=50,final_clf=final_clf,verbose=verbose,dropout=dropout,nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
                
            else:
                print(constrNet,'is unkwon in the context of TL')
                raise(NotImplementedError)
            
            model = FineTuneModel(model,dataset=target_dataset,df=df_label,\
                                    x_col=item_name,y_col=classes,path_im=path_to_img,\
                                    str_val=str_val,num_classes=len(classes),epochs=epochs,\
                                    Net=constrNet,plotConv=plotConv,batch_size=batch_size,cropCenter=cropCenter)
            model_path = os.path.join(model_output_path,AP_file_base+'.h5')
            include_optimizer=False
            model.save(model_path,include_optimizer=include_optimizer)
            # Prediction
            predictions = predictionFT_net(model,df_test=df_label_test,x_col=item_name,\
                                           y_col=classes,path_im=path_to_img,Net=constrNet,\
                                           cropCenter=cropCenter)

            metrics = evaluationScore(y_test,predictions)    
            del model
            
        with open(APfilePath, 'wb') as pkl:
            pickle.dump(metrics,pkl)
                
            
    else:
        try:
            with open(APfilePath, 'rb') as pkl:
                metrics = pickle.load(pkl)
        except FileNotFoundError as e:
            if onlyReturnResult:
                metrics = None
                return(metrics)
            else:
                raise(e)
                
    if len(metrics)==5:
        AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
    if len(metrics)==4:
        AP_per_class,P_per_class,R_per_class,P20_per_class = metrics
        F1_per_class = None
    
    if not(forLatex):
        print(target_dataset,source_dataset,number_im_considered,final_clf,features,transformOnFinalLayer,\
              constrNet,kind_method,'GS',gridSearch,'norm',normalisation,'getBeforeReLU',getBeforeReLU)
        print(style_layers)
    
    
#    print(Latex_str)
    #VGGInNorm Block1-5\_conv1 fc2 LinearSVC no GS BFReLU
    str_part2 = arrayToLatex(AP_per_class,per=True)
    Latex_str += str_part2
    Latex_str = Latex_str.replace('\hline','')
    print(Latex_str)
    
    # To clean GPU memory
    K.clear_session()
    gc.collect()
    #cuda.select_device(0)
    #cuda.close()
    
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)

def get_ResNet_BNRefin(df,x_col,path_im,str_val,num_of_classes,Net,\
                       weights,res_num_layers,transformOnFinalLayer,kind_method,\
                       batch_size=32,momentum=0.9,num_epochs_BN=5,output_path=''):
    """
    This function refine the normalisation statistics of the batch normalisation 
    with an exponential moving average
    """
    
    model_file_name = 'ResNet'+str(res_num_layers)+'_ROWD_'+str(weights)+'_'+transformOnFinalLayer+\
        '_bs' +str(batch_size)+'_m'+str(momentum)+'_ep'+str(num_epochs_BN) 
    model_file_name_path = model_file_name + '.tf'
    model_file_name_path = os.path.join(output_path,'model',model_file_name_path) 
    
    print('model_file_name_path',model_file_name_path,os.path.isfile(model_file_name_path))
    verbose = True
    model = ResNet_BNRefinements_Feat_extractor(num_of_classes=num_of_classes,\
                                            transformOnFinalLayer =transformOnFinalLayer,\
                                            verbose=verbose,weights=weights,\
                                            res_num_layers=res_num_layers,momentum=momentum,\
                                            kind_method=kind_method)
    
    if os.path.isfile(model_file_name_path):
        print('We will load the weights of the model')
        #model = load_model(model_file_name_path)
        model.load_weights(model_file_name_path)
    else:
        print('We will refine the normalisation parameters')
        if 'ResNet50' in Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        else:
            print(Net,'is unknwon')
            raise(NotImplementedError)
    
        df_train = df[df['set']=='train']
        df_val = df[df['set']==str_val]
        df_train[x_col] = df_train[x_col].apply(lambda x : x + '.jpg')
        df_val[x_col] = df_val[x_col].apply(lambda x : x + '.jpg')
        if not(len(df_val)==0):
            df_train = df_train.append(df_val)
            
        datagen= tf.keras.preprocessing.image.ImageDataGenerator()
        # Todo should add the possibility to crop the center of the image here
        trainval_generator=datagen.flow_from_dataframe(dataframe=df_train, directory=path_im,\
                                                    x_col=x_col,y_col=None,\
                                                    class_mode=None, \
                                                    target_size=(224,224), batch_size=batch_size,\
                                                    shuffle=True,\
                                                    preprocessing_function=preprocessing_function)
        STEP_SIZE_TRAIN=trainval_generator.n//trainval_generator.batch_size

        model =  fit_generator_ForRefineParameters(model,
                      trainval_generator,
                      steps_per_epoch=STEP_SIZE_TRAIN,
                      epochs=num_epochs_BN,
                      verbose=1,
    #                  callbacks=None,
    #                  validation_data=None,
    #                  validation_steps=None,
    #                  validation_freq=1,
    #                  class_weight=None,
                      max_queue_size=10,
                      workers=3,
                      use_multiprocessing=True,
                      shuffle=True)

        model.save_weights(model_file_name_path)

    return(model)

def FineTuneModel(model,dataset,df,x_col,y_col,path_im,str_val,num_classes,epochs=20,\
                  Net='VGG',batch_size = 32,plotConv=False,test_size=0.15,\
                  return_best_model=False,cropCenter=False):
    """
    @param x_col : name of images
    @param y_col : classes
    @param path_im : path to images
    @param : return_best_model : return the best model on the val_loss
    """
    df_train = df[df['set']=='train']
    df_val = df[df['set']==str_val]
    df_train[x_col] = df_train[x_col].apply(lambda x : x + '.jpg')
    df_val[x_col] = df_val[x_col].apply(lambda x : x + '.jpg')
    if len(df_val)==0:
        df_train, df_val = train_test_split(df_train, test_size=test_size)
        
    if cropCenter:
        from functools import partial
        preprocessing_function = partial(load_and_crop_img_forImageGenerator,Net)
    else:
        if Net=='VGG' or Net=='VGGAdaIn' or Net=='VGGAdaDBN':
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet50' in Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        else:
            print(Net,'is unknwon')
            raise(NotImplementedError)

    datagen= tf.keras.preprocessing.image.ImageDataGenerator()
    
    train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory=path_im,\
                                                x_col=x_col,y_col=y_col,\
                                                class_mode="other", \
                                                target_size=(224,224), batch_size=batch_size,\
                                                shuffle=True,\
                                                preprocessing_function=preprocessing_function)
    valid_generator=datagen.flow_from_dataframe(dataframe=df_val, directory=path_im,\
                                                x_col=x_col,y_col=y_col,\
                                                class_mode="other", \
                                                target_size=(224,224), batch_size=batch_size,\
                                                preprocessing_function=preprocessing_function)
#    train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory=path_im,\
#                                                x_col=x_col,y_col=y_col,\
#                                                class_mode="other", \
#                                                target_size=(224,224), batch_size=32,\
#                                                shuffle=True,\
#                                                preprocessing_function=preprocessing_function)
#    valid_generator=datagen.flow_from_dataframe(dataframe=df_val, directory=path_im,\
#                                                x_col=x_col,y_col=y_col,\
#                                                class_mode="other", \
#                                                target_size=(224,224), batch_size=32,\
#                                                preprocessing_function=preprocessing_function)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    # TODO you should add an early stoppping 
#    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
#    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
#    
    callbacks = []
    if return_best_model:
        tmp_model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        mcp_save = ModelCheckpoint(tmp_model_path, save_best_only=True, monitor='val_loss', mode='min')
        callbacks += [mcp_save]
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,use_multiprocessing=True,
                    workers=3,callbacks=callbacks)
    
    if return_best_model: # We need to load back the best model !
        # https://github.com/keras-team/keras/issues/2768
        model = load_model(tmp_model_path) 
    
    if plotConv:
       plotKerasHistory(history) 
    
    return(model)

def plotKerasHistory(history):
    plt.ion()
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.figure()
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.title('Acc')
    plt.legend()
    plt.draw()
    plt.pause(0.001)
   
    
def TrainMLP(model,X_train,y_train,X_val,y_val,batch_size,epochs,verbose=False,\
             plotConv=False,return_best_model=False):
    
    callbacks = []
    if return_best_model:
        tmp_model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        mcp_save = ModelCheckpoint(tmp_model_path, save_best_only=True, monitor='val_loss',\
                                   mode='min')
        callbacks += [mcp_save]
    
    STEP_SIZE_TRAIN=len(X_train)//batch_size
    if not(len(X_val)==0):
        STEP_SIZE_VALID=len(X_val)//batch_size
        history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,\
                            validation_data=(X_val, y_val),\
                            steps_per_epoch=STEP_SIZE_TRAIN,\
                            validation_steps=STEP_SIZE_VALID,\
                            use_multiprocessing=True,workers=3,\
                            shuffle=True,callbacks=callbacks)
    else: # No validation set provided
        history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,\
                            validation_split=0.15,\
                            steps_per_epoch=STEP_SIZE_TRAIN,\
                            use_multiprocessing=True,workers=3,\
                            shuffle=True,callbacks=callbacks)
        plotKerasHistory(history)
        
    return(model)
    
def predictionFT_net(model,df_test,x_col,y_col,path_im,Net='VGG',cropCenter=False):
    df_test[x_col] = df_test[x_col].apply(lambda x : x + '.jpg')
    datagen= tf.keras.preprocessing.image.ImageDataGenerator()
    
    if cropCenter:
        from functools import partial
        preprocessing_function = partial(load_and_crop_img_forImageGenerator,Net)
    else:
        if 'VGG' in Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet50' in Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        else:
            print(Net,'is unknwon')
            raise(NotImplementedError)
            
    test_generator=datagen.flow_from_dataframe(dataframe=df_test, directory=path_im,\
                                                x_col=x_col,\
                                                class_mode=None,shuffle=False,\
                                                target_size=(224,224), batch_size=1,
                                                preprocessing_function=preprocessing_function,
                                                use_multiprocessing=True,workers=3)
    predictions = model.predict_generator(test_generator)
    return(predictions)
    
def evaluationScoreDict(y_gt,dico_pred,verbose=False,k = 20,seuil=0.5):
    """
    @param k for precision at rank k
    @param the seuil can change a lot of things on the F1 score
    """
    num_samples,num_classes = y_gt.shape
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    F1_per_class = []
    P20_per_class = []
    for c in range(num_classes):
        y_gt_c = y_gt[:,c]
        [y_predict_confidence_score,y_predict_test] = dico_pred[c]
        AP = average_precision_score(y_gt_c,y_predict_confidence_score,average=None)
        if verbose: print("Average Precision on all the data for classe",c," = ",AP)  
        AP_per_class += [AP] 
        test_precision = precision_score(y_gt_c,y_predict_test)
        test_recall = recall_score(y_gt_c,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = f1_score(y_gt_c,y_predict_test)
        F1_per_class +=[F1]
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        if verbose: print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)
    
def evaluationScore(y_gt,y_pred,verbose=False,k = 20,seuil=0.5):
    """
    y_gt must be between 0 or 1
    y_predmust be between 0 and 1
    @param k for precision at rank k
    @param the seuil can change a lot of things
    """
    num_samples,num_classes = y_gt.shape
    AP_per_class = []
    P_per_class = []
    F1_per_class = []
    R_per_class = []
    P20_per_class = []
    for c in range(num_classes):
        y_gt_c = y_gt[:,c]
        y_predict_confidence_score = y_pred[:,c] # The prediction by the model
        y_predict_test = (y_predict_confidence_score>seuil).astype(int)
        if verbose:
            print('classe num',c)
            print('GT',y_gt_c)
            print('Pred',y_predict_confidence_score)
        AP = average_precision_score(y_gt_c,y_predict_confidence_score,average=None)
        if verbose: print("Average Precision ofpickn all the data for classe",c," = ",AP)  
        AP_per_class += [AP] 
        test_precision = precision_score(y_gt_c,y_predict_test)
        test_recall = recall_score(y_gt_c,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = f1_score(y_gt_c,y_predict_test)
        F1_per_class +=[F1]
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        if verbose: print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)

def RunUnfreezeLayerPerformanceVGG(plot=False):
    """
    The goal is to unfreeze only some part of the network
    """
    list_freezingType = ['FromTop','FromBottom','Alter']
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D','']
    optimizer_tab = ['adam','SGD']
    opt_option_tab = [[0.01],[0.1,0.01]]
    range_l = range(0,17)
    target_dataset = 'Paintings'
    for transformOnFinalLayer in transformOnFinalLayer_tab:
        for optimizer,opt_option in zip(optimizer_tab,opt_option_tab):
            if plot: plt.figure()
            list_perf = [] 
            j = 0
            for freezingType in list_freezingType:
                list_perf += [[]]  
                for pretrainingModif in range_l:
                    metrics = learn_and_eval(target_dataset=target_dataset,constrNet='VGG',\
                                             kind_method='FT',epochs=20,transformOnFinalLayer=transformOnFinalLayer,\
                                             pretrainingModif=pretrainingModif,freezingType=freezingType,
                                             optimizer=optimizer,opt_option=opt_option)

        
                    AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
                    list_perf[j] += [np.mean(AP_per_class)]
                if plot: plt.plot(list(range_l),list_perf[j],label=freezingType)
                j += 1
            if plot:
                title = optimizer + ' ' + transformOnFinalLayer
                plt.xlabel('Number of layers retrained')
                plt.ylabel('mAP ArtUK')
                plt.title(title)
                plt.legend(loc='best')
                output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset,'fig')
                pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
                fname =  os.path.join(output_path,target_dataset+'_Unfreezed_'+ optimizer+'_'+transformOnFinalLayer+'.png')
                plt.show() 
                plt.savefig(fname)
                plt.close()
                
def PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'Paintings',short=False,
                           scenario=0,onlyPlot=False,BV=True):
    """
    Plot some mAP  on ArtUK Paintings dataset with different model just to see
    if we can say someting
    """
    # Normally metric = AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class but sometimes F1 is missing
    if metricploted=='mAP':
        metricploted_index = 0
    elif metricploted=='Precision':
        metricploted_index = 1
    elif metricploted=='Recall':
        metricploted_index = 2
    else:
        print(metricploted,' is unknown')
        raise(NotImplementedError)
        
    list_markers = ['o','s','X','*','v','^','<','>','d','1','2','3','4','8','h','H','p','d','$f$','P']
    # Les 3 frozen : 'o','s','X'
    # VGG : '*'
    
    NUM_COLORS = 20
    color_number_for_frozen = [0,NUM_COLORS//2,NUM_COLORS-1]
    cm = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    list_freezingType = ['FromTop','FromBottom','Alter']
    list_freezingType = ['FromTop']
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D'] # Not the flatten case for the moment
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D'] # Not the flatten case for the moment
    
    if scenario==0:
        final_clf = 'MLP2'
        epochs = 20
        optimizer_tab = ['SGD','adam']
        opt_option_tab = [[0.1,0.001],[0.01]] # Soit 0=10**-3
        return_best_model = False
    elif scenario==1:
        final_clf = 'MLP1'
        epochs = 20
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-4)]]
        return_best_model = True
    elif scenario==2:
        final_clf = 'MLP1'
        epochs = 100
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-5)]]
        return_best_model = True
        # Really small learning rate 10**(-5)
    elif scenario==3:
        final_clf = 'MLP1'
        epochs = 5
        optimizer_tab = ['adam']
        opt_option_tab = [[0.1,10**(-3)]]
        return_best_model = True
        # Really small learning rate 10**(-5)
    elif scenario==4:
        final_clf = 'MLP2'
        epochs = 20
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-4)]]
        return_best_model = True
    
    # Not the flatten case for the moment
#    optimizer_tab = ['SGD','SGD','adam']
#    opt_option_tab = [[0.1,0.01],[0.1,0.001],[0.01]]
#    optimizer_tab = ['SGD','adam']
    
    range_l = range(0,17)
    
    if short:
        transformOnFinalLayer_tab = ['GlobalMaxPooling2D']
        range_l = range(0,6)
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,0.001]]
        list_freezingType = ['FromTop']
#    opt_option_tab = [[0.1,0.01]]
#    optimizer_tab = ['SGD']
    for optimizer,opt_option in zip(optimizer_tab,opt_option_tab): 
        for transformOnFinalLayer in transformOnFinalLayer_tab:
            fig_i = 0
            plt.figure()
            list_perf = [] 
            j = 0
            # Plot the value with a certain number of freeze or unfreeze layer
            for freezingType in list_freezingType:
                list_perf += [[]]  
                for pretrainingModif in range_l:
                    metrics = learn_and_eval(target_dataset=target_dataset,constrNet='VGG',\
                                             kind_method='FT',epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                             pretrainingModif=pretrainingModif,freezingType=freezingType,
                                             optimizer=optimizer,opt_option=opt_option
                                             ,final_clf=final_clf,features='block5_pool',\
                                             return_best_model=return_best_model,onlyReturnResult=onlyPlot)
    
                    if metrics is None:
                        continue
                    metricI_per_class = metrics[metricploted_index]
                    list_perf[j] += [np.mean(metricI_per_class)]
                if not(len(list(range_l))==len(list_perf[j])):
                    layers_j = list(range_l)[0:len(list_perf[j])]
                else:
                    layers_j = list(range_l)

                plt.plot(layers_j,list_perf[j],label=freezingType,color=scalarMap.to_rgba(color_number_for_frozen[fig_i]),\
                         marker=list_markers[fig_i],linestyle=':')
                fig_i += 1
                j += 1
            
            if short:
                continue
#            # Plot other proposed solution
#            features = 'block5_pool'
#            net_tab = ['VGG','VGGInNorm','VGGInNormAdapt','VGGBaseNorm','VGGBaseNormCoherent']
#            style_layers_tab_forOther = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
#                             ['block1_conv1','block2_conv1'],['block1_conv1']]
#            style_layers_tab_foVGGBaseNormCoherentr = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
#                             ['block1_conv1','block2_conv1']]
#            number_im_considered = 1000
#            source_dataset = 'ImageNet'
#            kind_method = 'TL'
#            normalisation = False
            getBeforeReLU = True
#            forLatex = True
#            fig_i = 0
#            for constrNet in net_tab:
#                print(constrNet)
#                if constrNet=='VGGBaseNormCoherent':
#                    style_layers_tab = style_layers_tab_foVGGBaseNormCoherentr
#                elif constrNet=='VGG':
#                    style_layers_tab = [[]]
#                else:
#                    style_layers_tab = style_layers_tab_forOther
#                for style_layers in style_layers_tab:
#                    labelstr = constrNet 
#                    if not(constrNet=='VGG'):
#                        labelstr += '_'+ numeral_layers_index(style_layers)
#                    metrics = learn_and_eval(target_dataset,source_dataset,final_clf=final_clf,features='block5_pool',\
#                                       constrNet=constrNet,kind_method=kind_method,style_layers=style_layers,gridSearch=False,
#                                       number_im_considered=number_im_considered,
#                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
#                                       transformOnFinalLayer=transformOnFinalLayer,\
#                                       optimizer=optimizer,opt_option=[opt_option[-1]],\
#                                       forLatex=forLatex,epochs=epochs,return_best_model=return_best_model,\
#                                       onlyReturnResult=onlyPlot)
#                    
#                    if metrics is None:
#                        continue
#                    
#                    metricI_per_class = metrics[metricploted_index]
#                    mMetric = np.mean(metricI_per_class)
#                    if fig_i in color_number_for_frozen:
#                        fig_i_c = fig_i +1 
#                        fig_i_m = fig_i
#                        fig_i += 1
#                    else:
#                        fig_i_c = fig_i
#                        fig_i_m = fig_i
#                    plt.plot([0],[mMetric],label=labelstr,color=scalarMap.to_rgba(fig_i_c),\
#                             marker=list_markers[fig_i_m],linestyle='')
#                    fig_i += 1
            
            # Case of the fine tuning with batch normalization 
            constrNet = 'VGGAdaIn'
            style_layers_tab_VGGAdaIn = [['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
                                                        'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
                                                        'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
                                                        'block5_conv1','block5_conv2','block5_conv3','block5_conv4'],
                                                        ['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                                                        ['block1_conv1','block2_conv1'],['block2_conv1'],['block1_conv1']]
            for style_layers in style_layers_tab_VGGAdaIn:
    #            print(constrNet,style_layers)
                metrics = learn_and_eval(target_dataset,constrNet=constrNet,kind_method='FT',\
                                          epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                          forLatex=True,optimizer=optimizer,\
                                          opt_option=[opt_option[-1]],\
                                          style_layers=style_layers,getBeforeReLU=getBeforeReLU,\
                                          final_clf=final_clf,features='block5_pool',return_best_model=return_best_model,\
                                          onlyReturnResult=onlyPlot)
                
                if metrics is None:
                    continue
                
                metricI_per_class = metrics[metricploted_index]
                mMetric = np.mean(metricI_per_class)
                if fig_i in color_number_for_frozen:
                    fig_i_c = fig_i +1 
                    fig_i_m = fig_i
                    fig_i += 1
                else:
                    fig_i_c = fig_i
                    fig_i_m = fig_i
                labelstr = constrNet 
                if not(constrNet=='VGG'):
                    if BV:
                        labelstr += '_'+ numeral_layers_index_bitsVersion(style_layers)
                    else:
                        labelstr += '_'+ numeral_layers_index(style_layers)
                plt.plot([0],[mMetric],label=labelstr,color=scalarMap.to_rgba(fig_i_c),\
                         marker=list_markers[fig_i_m],linestyle='')
                fig_i += 1
                
            # Case of the fine tuning with decorellated batch normalization 
            constrNet = 'VGGAdaDBN'
            style_layers_tab_VGGAdaDBN = [['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
                                                        'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
                                                        'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
                                                        'block5_conv1','block5_conv2','block5_conv3','block5_conv4'],
                                                        ['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                                                        ['block1_conv1','block2_conv1'],['block2_conv1'],['block1_conv1']]
            for style_layers in style_layers_tab_VGGAdaDBN:
    #            print(constrNet,style_layers)
                metrics = learn_and_eval(target_dataset,constrNet=constrNet,kind_method='FT',\
                                          epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                          forLatex=True,optimizer=optimizer,\
                                          opt_option=[opt_option[-1]],\
                                          style_layers=style_layers,getBeforeReLU=getBeforeReLU,\
                                          final_clf=final_clf,features='block5_pool',return_best_model=return_best_model,\
                                          onlyReturnResult=onlyPlot,dbn_affine=True,m_per_group=16)
                
                if metrics is None:
                    continue
                
                metricI_per_class = metrics[metricploted_index]
                mMetric = np.mean(metricI_per_class)
                if fig_i in color_number_for_frozen:
                    fig_i_c = fig_i +1 
                    fig_i_m = fig_i
                    fig_i += 1
                else:
                    fig_i_c = fig_i
                    fig_i_m = fig_i
                labelstr = constrNet 
                if not(constrNet=='VGG'):
                    if BV:
                        labelstr += '_'+ numeral_layers_index_bitsVersion(style_layers)
                    else:
                        labelstr += '_'+ numeral_layers_index(style_layers)
                plt.plot([0],[mMetric],label=labelstr,color=scalarMap.to_rgba(fig_i_c),\
                         marker=list_markers[fig_i_m],linestyle='')
                fig_i += 1
                
            title = optimizer + ' ' + transformOnFinalLayer + ' ' + metricploted + ' ' + final_clf
            plt.ion()
            plt.xlabel('Number of layers retrained')
            plt.ylabel(metricploted+' ArtUK')
            plt.title(title)
            plt.legend(loc='best')
            output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset,'fig')
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            name_of_the_figure = 'Summary_'+ target_dataset+'_'+final_clf+'_Unfreezed_'+ optimizer+'_'+transformOnFinalLayer+'.png'
            if short:
                name_of_the_figure = 'Short_'+name_of_the_figure
            fname = os.path.join(output_path,name_of_the_figure)
    plt.show() 
    plt.pause(0.001)
#        input('Press to close')
    plt.savefig(fname)
#        plt.close()
        
def PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'Paintings',scenario=0,
                              onlyPlot=False,cropCenter=True,BV=True):
    """
    Plot some mAP  on ArtUK Paintings dataset with different model just to see
    if we can say someting
    """
    
    network = 'ResNet50'
    # Normally metric = AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class but sometimes F1 is missing
    if metricploted=='mAP':
        metricploted_index = 0
    elif metricploted=='Precision':
        metricploted_index = 1
    elif metricploted=='Recall':
        metricploted_index = 2
    else:
        print(metricploted,' is unknown')
        raise(NotImplementedError)
        
    list_markers = ['o','s','X','*','v','^','<','>','d','1','2','3','4','8','h','H','p','d','$f$','P']
    # Les 3 frozen : 'o','s','X'
    # VGG : '*'
    
    NUM_COLORS = 20
    color_number_for_frozen = [0,NUM_COLORS//2,NUM_COLORS-1]
    cm = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    list_freezingType = ['FromTop','FromBottom','Alter']
    #list_freezingType = ['FromTop']
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D'] # Not the flatten case for the moment
    #transformOnFinalLayer_tab = ['GlobalMaxPooling2D'] # Not the flatten case for the moment
    style_layers = ['bn_conv1']
    if scenario==0:
        final_clf = 'MLP2'
        epochs = 20
        optimizer_tab = ['SGD','adam']
        opt_option_tab = [[0.1,0.001],[0.01]] # Soit 0=10**-3
        return_best_model = False
    elif scenario==1:
        final_clf = 'MLP1'
        epochs = 20
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-4)]]
        return_best_model = True
    elif scenario==2:
        final_clf = 'MLP1'
        epochs = 100
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-5)]]
        return_best_model = True
        # Really small learning rate 10**(-5)
    elif scenario==3:
        final_clf = 'MLP1'
        epochs = 5
        optimizer_tab = ['adam']
        opt_option_tab = [[0.1,10**(-3)]]
        return_best_model = True
    elif scenario==4:
        final_clf = 'MLP2'
        epochs = 20
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,10**(-4)]]
        return_best_model = True
    elif scenario==5:
        final_clf = 'MLP1'
        epochs = 20
        optimizer_tab = ['SGD']
        opt_option_tab = [[0.1,0.01]] # WILDACT
        return_best_model = True
        # In WILDCAT we have lrp = 0.1, lp = 0.01 loss = MultiLabelSoftMarginLoss, Net = ResNet101
        # MultiLabelSoftMarginLoss seems to be the use of sigmoid on the outpur of the model and then the sum over classes of the binary cross entropy loss
    elif scenario==6:
        # In Recognizing Characters in Art History Using Deep Learning  -  Madhu 2019 ?
        raise(NotImplementedError)


    range_l = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102,106] # For Resnet50
    batch_size = 16 
    features = 'activation_48'
    for optimizer,opt_option in zip(optimizer_tab,opt_option_tab): 
        for transformOnFinalLayer in transformOnFinalLayer_tab:
            fig_i = 0
            plt.figure()
            list_perf = [] 
            j = 0
            # Plot the value with a certain number of freeze or unfreeze layer
            for freezingType in list_freezingType:
                list_perf += [[]]  
                for pretrainingModif in range_l:
                    metrics = learn_and_eval(target_dataset=target_dataset,constrNet=network,\
                                             kind_method='FT',epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                             pretrainingModif=pretrainingModif,freezingType=freezingType,\
                                             optimizer=optimizer,opt_option=opt_option,batch_size=batch_size\
                                             ,final_clf=final_clf,features=features,return_best_model=return_best_model,\
                                             onlyReturnResult=onlyPlot,style_layers=style_layers,
                                             cropCenter=cropCenter)
                                            # il faudra checker cela avec le ResNet 
        
                    if metrics is None:
                        continue
                    metricI_per_class = metrics[metricploted_index]
                    list_perf[j] += [np.mean(metricI_per_class)]
                if not(len(list(range_l))==len(list_perf[j])):
                    layers_j = list(range_l)[0:len(list_perf[j])]
                else:
                    layers_j = list(range_l)
                plt.plot(layers_j,list_perf[j],label=freezingType,color=scalarMap.to_rgba(color_number_for_frozen[fig_i]),\
                         marker=list_markers[fig_i],linestyle=':')
                fig_i += 1
                j += 1
    
            
        ## TODO il va falloir gerer les optimizers la !
    #        # Plot other proposed solution
            
    #        net_tab = ['VGG','VGGInNorm','VGGInNormAdapt','VGGBaseNorm','VGGBaseNormCoherent']
            net_tab = ['ResNet50','ResNet50_BNRF']
            style_layers_tab_forOther = [[]] # TODO il faudra changer cela a terme
            style_layers_tab_forResNet50_ROWD = [['bn_conv1'],['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1'],
                                         getBNlayersResNet50()]
            style_layers_tab_forResNet50_ROWD = [['bn_conv1'],['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1']]
            number_im_considered = 1000
            final_clf = 'MLP2'
            source_dataset = 'ImageNet'
            kind_method = 'TL'
            normalisation = False
            getBeforeReLU = True
            forLatex = True
            fig_i = 0
            for constrNet in net_tab:
                print('~~~ ',constrNet,' ~~~')
                if constrNet=='ResNet50_ROWD_CUMUL':
                    style_layers_tab = style_layers_tab_forResNet50_ROWD
                elif constrNet=='ResNet50' or constrNet=='ResNet50_BNRF':
                    style_layers_tab = [[]]
                else:
                    style_layers_tab = style_layers_tab_forOther
                for style_layers in style_layers_tab:
                    labelstr = constrNet 
                    if not(constrNet=='ResNet50' or constrNet=='ResNet50_BNRF'):
                        if BV:
                            labelstr += '_'+ getResNetLayersNumeral_bitsVersion(style_layers)
                        else:
                            labelstr += '_'+ getResNetLayersNumeral(style_layers)
                    metrics = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       transformOnFinalLayer=transformOnFinalLayer,\
                                       optimizer=optimizer,opt_option=opt_option,batch_size=batch_size\
                                       ,return_best_model=return_best_model,\
                                       forLatex=forLatex,cropCenter=cropCenter,\
                                       momentum=0.9,batch_size_RF=16,epochs_RF=20,\
                                       onlyReturnResult=onlyPlot,verbose=True)
                    if metrics is None:
                        continue
                    metricI_per_class = metrics[metricploted_index]
                    mMetric = np.mean(metricI_per_class)
                    if fig_i in color_number_for_frozen:
                        fig_i_c = fig_i +1 
                        fig_i_m = fig_i
                        fig_i += 1
                    else:
                        fig_i_c = fig_i
                        fig_i_m = fig_i
                    plt.plot([0],[mMetric],label=labelstr,color=scalarMap.to_rgba(fig_i_c),\
                             marker=list_markers[fig_i_m],linestyle='')
                    fig_i += 1
    #        
    #        # Case of the fine tuning with batch normalization 
            constrNet = 'ResNet50AdaIn'
            getBeforeReLU = True
            style_layers_tab_ResNet50AdaIn = [['bn_conv1'],['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1'],
                                         getBNlayersResNet50()]
            for style_layers in style_layers_tab_ResNet50AdaIn:
    #            print(constrNet,style_layers)
                metrics = learn_and_eval(target_dataset,constrNet=constrNet,kind_method='FT',\
                                          epochs=20,transformOnFinalLayer=transformOnFinalLayer,\
                                          final_clf='MLP2',forLatex=True,optimizer=optimizer,\
                                          style_layers=style_layers,getBeforeReLU=getBeforeReLU,\
                                          opt_option=opt_option,cropCenter=cropCenter,\
                                          onlyReturnResult=onlyPlot)
                if metrics is None:
                    continue
                metricI_per_class = metrics[metricploted_index]
                mMetric = np.mean(metricI_per_class)
                if fig_i in color_number_for_frozen:
                    fig_i_c = fig_i +1 
                    fig_i_m = fig_i
                    fig_i += 1
                else:
                    fig_i_c = fig_i
                    fig_i_m = fig_i
                labelstr = constrNet 
                if not(constrNet=='VGG' or constrNet=='ResNet50'):
                    if BV:
                        if getResNetLayersNumeral_bitsVersion(style_layers) == getResNetLayersNumeral_bitsVersion(getBNlayersResNet50()):
                            labelstr += '_all'
                        else:
                            labelstr += '_'+  getResNetLayersNumeral_bitsVersion(style_layers)
                    else:
                        if getResNetLayersNumeral(style_layers) == getResNetLayersNumeral(getBNlayersResNet50()):
                            labelstr += '_all'
                        else:
                            labelstr += '_'+  getResNetLayersNumeral(style_layers)
                plt.plot([0],[mMetric],label=labelstr,color=scalarMap.to_rgba(fig_i_c),\
                         marker=list_markers[fig_i_m],linestyle='')
                fig_i += 1
    #            
            optstr  = ''        
            for o in opt_option:     
                optstr += ' ' +str(o)
            optstr += ' ' +str(epochs)
            optstr_ = optstr.replace(' ','_')
            title = optimizer+ optstr + ' ' + transformOnFinalLayer + ' ' +final_clf + ' ' + metricploted
            plt.ion()
            plt.xlabel('Number of layers retrained')
            if target_dataset=='Paintings':
                target_dataset_str = 'ArtUK'
            else:
                target_dataset_str  = target_dataset
            plt.ylabel(metricploted+' '+target_dataset_str)
            plt.title(title)
            plt.legend(loc='best')
            output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset,'fig')
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            name_of_the_figure = 'Summary_'+ target_dataset+network+'_Unfreezed_'+\
                optimizer+optstr_+'_'+transformOnFinalLayer+final_clf+target_dataset_str
            if cropCenter:
                name_of_the_figure += '_CropCenter'   
            name_of_the_figure+='.png'
            fname = os.path.join(output_path,name_of_the_figure)
            plt.show() 
            plt.pause(0.001)
    #        input('Press to close')
            plt.savefig(fname)
    #        plt.close()
                    
def RunEval_MLP_onConvBlock():
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D','']
    for optimizer,opt_option in zip(['adam','SGD'],[[0.01],[0.01]]):
        for transformOnFinalLayer in transformOnFinalLayer_tab:
            if transformOnFinalLayer=='':
                feature='flatten'
            else:
                feature='block5_pool'
            metrics = learn_and_eval(target_dataset='Paintings',constrNet='VGG',features=feature,\
                                     kind_method='TL',epochs=20,transformOnFinalLayer=transformOnFinalLayer,\
                                     optimizer=optimizer,opt_option=opt_option,final_clf='MLP2')
    
def RunAllEvaluation(dataset='Paintings',forLatex=False):
    target_dataset='Paintings'
    source_dataset = 'ImageNet'
    ## Run the baseline
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D']
    for normalisation in [False]:
        final_clf_list = ['MLP2','LinearSVC'] # LinearSVC but also MLP
        features_list = ['fc2','fc1','flatten'] # We want to do fc2, fc1, max spatial and concat max and min spatial
        features_list = [] # We want to do fc2, fc1, max spatial and concat max and min spatial
         # We want to do fc2, fc1, max spatial and concat max and min spatial
        # Normalisation and not normalise
        kind_method = 'TL'
        style_layers = []
        
        # Baseline with just VGG
        constrNet = 'VGG'
        for final_clf in final_clf_list:
            for features in features_list:
                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False,
                           normalisation=normalisation,transformOnFinalLayer='')
            
            # Pooling on last conv block
            for transformOnFinalLayer in transformOnFinalLayer_tab:
                features = 'block5_pool'
                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False,
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,\
                           forLatex=forLatex)
         
        # With VGGInNorm
        style_layers_tab_forOther = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1','block2_conv1'],['block1_conv1']]
        style_layers_tab_foVGGBaseNormCoherentr = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1','block2_conv1']]
        
        features_list = ['fc2','fc1','flatten']
        features_list = ['fc2','fc1']
        features_list = []
        net_tab = ['VGGInNorm','VGGInNormAdapt','VGGBaseNorm','VGGBaseNormCoherent']
        number_im_considered_tab = [1000]
        for getBeforeReLU in [True,False]:
            for constrNet in net_tab:
                if constrNet=='VGGBaseNormCoherent':
                    style_layers_tab = style_layers_tab_foVGGBaseNormCoherentr
                else:
                    style_layers_tab = style_layers_tab_forOther
                for final_clf in final_clf_list:
                    for style_layers in style_layers_tab:
                        for features in features_list:
                            for number_im_considered in number_im_considered_tab:
                                if not(forLatex): print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers)
                                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,\
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       forLatex=forLatex)
                       
                        number_im_considered = 1000
                        # Pooling on last conv block
                        for transformOnFinalLayer in transformOnFinalLayer_tab:
                            if not(forLatex):  print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers,'transformOnFinalLayer',transformOnFinalLayer)
                            features = 'block5_pool'
                            learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       transformOnFinalLayer=transformOnFinalLayer,\
                                       forLatex=forLatex)
 
def TrucBizarre(target_dataset='Paintings'):
    # Ces deux manieres de faire devrait retourner les memes performances et ce n'est pas le cas....
    
    # Ici cas du transfert learning avec extraction de features puis entrainement d'un MLP2
    metrics = learn_and_eval(target_dataset=target_dataset,constrNet='VGG',pretrainingModif=False,\
                   kind_method='TL',epochs=1,transformOnFinalLayer='GlobalAveragePooling2D',\
                   final_clf='MLP2',forLatex=True,features='block5_pool',ReDo=True,plotConv=True,\
                   optimizer='adam')
    AP_TL = metrics[0]
    print('TL',AP_TL)
    
    # Ici fine-tuning du réseau mais avec l'ensemble du réseau pretained qui est freeze /
    # fixe / non trainable : on rajoute juste un MLP2 a la fin
    metrics2 = learn_and_eval(target_dataset=target_dataset,constrNet='VGG',pretrainingModif=False,\
                   kind_method='FT',epochs=1,transformOnFinalLayer='GlobalAveragePooling2D',\
                   final_clf='MLP2',forLatex=True,ReDo=True,plotConv=True,optimizer='adam')
    AP_FT = metrics2[0]
    print('FT',AP_FT)

def Test_Unfrozen_ResNet():
    print('=== From Top ===')
    metrics = learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',\
                         kind_method='FT',epochs=0,transformOnFinalLayer='GlobalMaxPooling2D',\
                         pretrainingModif=3,freezingType='FromTop',\
                         optimizer='adam',opt_option=[0.001],batch_size=16\
                         ,final_clf='MLP2',features='avg_pool',return_best_model=True,\
                         onlyReturnResult=False,style_layers=['bn_conv1'],verbose=True)
    
    
    print('=== From Bottom ===')
    metrics = learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',\
                         kind_method='FT',epochs=0,transformOnFinalLayer='GlobalMaxPooling2D',\
                         pretrainingModif=3,freezingType='FromBottom',\
                         optimizer='adam',opt_option=[0.001],batch_size=16\
                         ,final_clf='MLP2',features='avg_pool',return_best_model=True,\
                         onlyReturnResult=False,style_layers=['bn_conv1'],verbose=True)

def Test_Apropos_DuRebond():
    
    # il semblerait que dans certains cas on arrive a faire du 57% dans certains cas
    tab_AP =[]
    for i in range(3):
        AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class\
        =learn_and_eval(target_dataset='IconArt_v1',final_clf='MLP2',\
                        kind_method='FT',epochs=20,ReDo=True,optimizer='SGD',\
                        opt_option=[0.1,0.001],features='block5_pool',\
                        batch_size=32,constrNet='VGG',freezingType='FromTop',\
                        pretrainingModif=3,plotConv=False)
        print(i,np.mean(AP_per_class))
        tab_AP += [np.mean(AP_per_class)]
 
def Crowley_reproduction_results():
    
    target_dataset = 'Paintings'
    ReDo = False
    
    print('The following experiments will normally reproduce the performance of Crowley 2016 with VGG central crop, grid search on C parameter of SVM but no augmentation of the image (multi crop).')
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='LinearSVC',features='fc2',\
                   constrNet='VGG',kind_method='TL',gridSearch=True,ReDo=ReDo,cropCenter=True)
    
    print('Same experiment with ResNet50 ')
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='LinearSVC',features='activation_48',\
                   transformOnFinalLayer='GlobalAveragePooling2D',
                   constrNet='ResNet50',kind_method='TL',gridSearch=True,ReDo=ReDo,cropCenter=True)
    # ResNet50 Block1-5\_conv1 activation\_48 GlobalAveragePooling2D LinearSVCGS 
    #& 64.4 & 45.9 & 91.2 & 71.6 & 64.6 & 67.9 & 53.6 & 78.0 & 65.5 & 83.6 & 68.6 \\ 
    
    print('Same experiment with ResNet50 but a MLP2')
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='activation_48',\
                   constrNet='ResNet50',kind_method='TL',gridSearch=True,ReDo=ReDo,\
                   transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True)
    
    print('Same experiment with ResNet50 but a MLP3 with dropout etc')
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
                   constrNet='ResNet50',kind_method='TL',gridSearch=True,ReDo=ReDo,\
                   transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
                   dropout=0.5,regulOnNewLayer='l2',optimizer='SGD',opt_option=[10**(-4)],\
                   epochs=50)
    
    print('Same experiment with ResNet50 but a MLP3 with dropout decay etc')
    # 68.8 AP sur Paintings
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
                   constrNet='ResNet50',kind_method='TL',gridSearch=True,ReDo=ReDo,\
                   transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
                   dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[5*10**(-4)],\
                   epochs=50,nesterov=True,SGDmomentum=0.99,decay=0.0005)
    
    print('Same experiment with ResNet50 with a fine tuning of the whole model')
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='activation_48',\
                   constrNet='ResNet50',kind_method='FT',gridSearch=True,ReDo=ReDo,\
                   transformOnFinalLayer='GlobalAveragePooling2D',pretrainingModif=True,\
                   optimizer='SGD',opt_option=[0.1,0.0001],return_best_model=True,
                   epochs=20,cropCenter=True)   
    
def testVGGShuffle():
    target_dataset = 'Paintings'
    ReDo = False
#    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
#               constrNet='VGG',kind_method='FT',gridSearch=True,ReDo=ReDo,\
#               transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
#               dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[5*10**(-4)],\
#               epochs=50,nesterov=True,SGDmomentum=0.99,decay=0.0005)
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
               constrNet='VGG',kind_method='FT',gridSearch=True,ReDo=ReDo,\
               transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,pretrainingModif=False,\
               dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[10**(-2)],\
               epochs=20,nesterov=True,SGDmomentum=0.9,decay=0.0005)
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
               constrNet='VGGAdaIn',kind_method='FT',gridSearch=True,ReDo=ReDo,\
               transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
               dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[10**(-2)],\
               epochs=20,nesterov=True,SGDmomentum=0.9,decay=0.0005)
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
               constrNet='VGGsuffleInStats',kind_method='FT',gridSearch=True,ReDo=ReDo,\
               transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,kind_of_shuffling='roll',\
               dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[10**(-2)],\
               epochs=20,nesterov=True,SGDmomentum=0.9,decay=0.0005)
    learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP3',features='activation_48',\
               constrNet='VGGsuffleInStats',kind_method='FT',gridSearch=True,ReDo=ReDo,\
               transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,kind_of_shuffling='shuffle',\
               dropout=0.2,regulOnNewLayer='l2',optimizer='SGD',opt_option=[10**(-2)],\
               epochs=20,nesterov=True,SGDmomentum=0.9,decay=0.0005)

def testROWD_CUMUL():
    learn_and_eval(target_dataset='Paintings',final_clf='LinearSVC',\
                    kind_method='TL',ReDo=False,
                    constrNet='ResNet50_ROWD_CUMUL',transformOnFinalLayer='GlobalAveragePooling2D',
                    style_layers=['bn_conv1'],verbose=True,features='activation_48',useFloat32=True)
#    learn_and_eval(target_dataset='Paintings',final_clf='LinearSVC',\
#                    kind_method='TL',ReDo=False,
#                    constrNet='ResNet50_ROWD_CUMUL',transformOnFinalLayer='GlobalAveragePooling2D',
#                    style_layers=['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1'],\
#                    verbose=True,features='activation_48',useFloat32=True)
#    learn_and_eval(target_dataset='Paintings',final_clf='LinearSVC',\
#                    kind_method='TL',ReDo=False,
#                    constrNet='ResNet50_ROWD_CUMUL',transformOnFinalLayer='GlobalAveragePooling2D',
#                    style_layers=getBNlayersResNet50(),verbose=True,features='activation_48,useFloat32=True')
       
# TODO :
# Train the layer i and use it as initialization for training layer i+1 
# Test RASTA
        
### What we could add to improve the model performance : 
# change the learning rate
# data augmentation
# dropout
# use of L1 and L2 regularization (also known as "weight decay")
                  
if __name__ == '__main__': 
    # Ce que l'on 
    #RunAllEvaluation()
    ### TODO !!!!! Need to add a unbalanced way to deal with the dataset
    #PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'Paintings',scenario=0,onlyPlot=True)
    #PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=0,onlyPlot=True)

#    PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'Paintings',scenario=4)
#    PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=4)
#    PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'Paintings',scenario=5)
#    PlotSomePerformanceVGG(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=5)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'Paintings',scenario=4)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=4)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'Paintings',scenario=5)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=5)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'Paintings',scenario=3)
#    PlotSomePerformanceResNet(metricploted='mAP',target_dataset = 'IconArt_v1',scenario=3)
#    PlotSomePerformanceVGG()
#    RunUnfreezeLayerPerformanceVGG()
#    RunEval_MLP_onConvBlock()
    
#    learn_and_eval(target_dataset='Paintings',constrNet='VGG',kind_method='FT',weights=None,epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,pretrainingModif=False,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,pretrainingModif=False,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50AdaIn',kind_method='FT',epochs=2,style_layers=['bn_conv1','bn2a_branch1','bn3a_branch1','bn4a_branch1','bn5a_branch1'],transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
###    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',kind_method='FT',\
##                   epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
##    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',kind_method='FT',\
#                   epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',weights=None,\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,\
#                   transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True,\
#                   style_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
#                                  'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
#                                  'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
#                                  'block5_conv1','block5_conv2','block5_conv3','block5_conv4'])
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',weights='imagenet',\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,\
#                   transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True,\
#                   style_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
#                                  'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
#                                  'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
#                                  'block5_conv1','block5_conv2','block5_conv3','block5_conv4'])
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='VGG',pretrainingModif=False,\
#                   kind_method='FT',epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)

## Pour tester le MLP1 sur IconArt v1
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='VGG',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='fc2')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='VGG',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='block5_pool')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='ResNet50',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='ResNet50',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='avg_pool')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='VGGAdaIn',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='fc2')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='VGGAdaIn',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='block5_pool')
#    learn_and_eval(target_dataset='IconArt_v1',constrNet='VGG',kind_method='FT',weights='imagenet',epochs=5,final_clf='MLP1',features='fc2',pretrainingModif=6)

## To test return_best_model in FT mode and FT mode
#    learn_and_eval(target_dataset='Paintings',final_clf='MLP2',\
#                        kind_method='FT',epochs=3,ReDo=True,optimizer='adam',\
#                        opt_option=[0.01],features='block5_pool',\
#                        batch_size=32,constrNet='VGG',freezingType='FromTop',\
#                        pretrainingModif=6,plotConv=True,transformOnFinalLayer='GlobalAveragePooling2D',return_best_model=True)
#    learn_and_eval(target_dataset='Paintings',final_clf='MLP2',\
#                        kind_method='TL',epochs=3,ReDo=True,optimizer='adam',\
#                        opt_option=[0.01],features='block5_pool',\
#                        batch_size=32,constrNet='VGG',plotConv=True,\
#                        transformOnFinalLayer='GlobalAveragePooling2D',return_best_model=True)

## Test BN Refinement of ResNet50
#    learn_and_eval(target_dataset='Paintings',final_clf='LinearSVC',\
#                        kind_method='TL',
#                        constrNet='ResNet50_BNRF',batch_size_RF=16,\
#                        style_layers=['bn_conv1'],verbose=True,epochs_RF=20,\
#                        transformOnFinalLayer='GlobalAveragePooling2D',\
#                        features='activation_48',cropCenter=True)
#    Crowley_reproduction_results()
## Test BN Refinement Once on the Whole Dataset of ResNet50
#    learn_and_eval(target_dataset='Paintings',final_clf='LinearSVC',\
#                        kind_method='TL',ReDo=True,
#                        constrNet='ResNet50_ROWD_CUMUL',transformOnFinalLayer='GlobalAveragePooling2D',
#                        style_layers=['bn_conv1'],verbose=True,features='activation_48') # A finir
    testROWD_CUMUL()
