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
from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,load_crop_and_process_img
from IMDB import get_database
import pickle
import pathlib
from Classifier_On_Features import TrainClassifierOnAllClass,PredictOnTestSet
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score,label_ranking_average_precision_score,classification_report
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.preprocessing import StandardScaler
from Custom_Metrics import ranking_precision_score
from LatexOuput import arrayToLatex

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
                                    getBeforeReLU=False,target_set='trainval',applySqrtOnVar=True):
    """
    The goal of this function is to compute a version of the statistics of the 
    features of the VGG 
    """
    dict_stats_coherent = {} 
    for i_layer,layer_name in enumerate(style_layers):
        #print(i_layer,layer_name)
        if i_layer==0:
            style_layers_firstLayer = [layer_name]
            dict_stats_target0 = get_dict_stats(target_dataset,target_number_im_considered,\
                                                style_layers_firstLayer,\
                                                whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                set=target_set)
            dict_stats_coherent[layer_name] = dict_stats_target0[layer_name]
            list_mean_and_std_target_i_m1 = compute_ref_stats(dict_stats_target0,\
                                            style_layers_firstLayer,type_ref='mean',\
                                            imageUsed='all',whatToload=whatToload,\
                                            applySqrtOnVar=applySqrtOnVar)
            current_list_mean_and_std_target = list_mean_and_std_target_i_m1
        else:
            list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
            style_layers_imposed = style_layers[0:i_layer]
            style_layers_exported = [style_layers[i_layer]]
            list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
            
            dict_stats_target_i = get_dict_stats(target_dataset,target_number_im_considered,\
                                                 style_layers=style_layers_exported,whatToload=whatToload,\
                                                 saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                 set=target_set,Net='VGGBaseNormCoherent',\
                                                 style_layers_imposed=style_layers_imposed,\
                                                 list_mean_and_std_source=list_mean_and_std_source_i,\
                                                 list_mean_and_std_target=current_list_mean_and_std_target)
            dict_stats_coherent[layer_name] = dict_stats_target_i[layer_name]
            # Compute the next statistics 
            list_mean_and_std_target_i = compute_ref_stats(dict_stats_target_i,\
                                            style_layers_exported,type_ref='mean',\
                                            imageUsed='all',whatToload=whatToload,\
                                            applySqrtOnVar=applySqrtOnVar)
            current_list_mean_and_std_target += [list_mean_and_std_target_i[-1]]
            
    return(dict_stats_coherent,current_list_mean_and_std_target)

def learn_and_eval(target_dataset,source_dataset,final_clf,features,constrNet,kind_method,
                   style_layers = ['block1_conv1',
                                    'block2_conv1',
                                    'block3_conv1', 
                                    'block4_conv1', 
                                    'block5_conv1'
                                   ],normalisation=False,gridSearch=True,ReDo=False,\
                                   transformOnFinalLayer='',number_im_considered = 1000,\
                                   set='',getBeforeReLU=False):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
    TODO : linear SVM - MLP - perceptron - MLP one per class - MLP with finetuning of the net
    @param : features : which features we will use
    TODO : fc2, fc1, max spatial, max et min spatial
    @param : constrNet the constrained net used
    TODO : VGGInNorm, VGGInNormAdapt seulement sur les features qui répondent trop fort, VGGGram
    @param : kind_method the type of methods we will use : TL or FT
    @param : if we use a set to compute the statistics
    @param : getBeforeReLU=False if True we will impose the statistics before the activation ReLU fct
    """
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    num_layers = numeral_layers_index(style_layers)
    # Compute statistics on the source_dataset
    if source_dataset is None:
        constrNet='VGG'
        
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(target_dataset)
        
    name_base = constrNet + '_'  +target_dataset +'_'
    if not(constrNet=='VGG'):
        name_base += source_dataset +str(number_im_considered)+ '_' + num_layers
    if not(set=='' or set is None):
        name_base += '_'+set
    if getBeforeReLU:
        name_base += '_BeforeReLU'
    name_base +=  features 
    if not((transformOnFinalLayer is None) or (transformOnFinalLayer=='')):
       name_base += '_'+     transformOnFinalLayer
    name_base += '_' + kind_method   
    
    # features can be 'flatten' with will output a 25088 dimension vectors = 7*7*512 features
    
    if kind_method=='TL': # Transfert Learning
        final_layer = features
        name_pkl_im = target_dataset +'.pkl'
        name_pkl_values  = name_base+ '_Features.pkl'
        name_pkl_im = os.path.join(output_path,name_pkl_im)
        name_pkl_values = os.path.join(output_path,name_pkl_values)
        
        if not os.path.isfile(name_pkl_values):
            print('== We will compute the reference statistics ==')
            print('Saved in ',name_pkl_values)
            features_net = None
            im_net = []
            # Load Network 
            if constrNet=='VGG':
                network_features_extraction = vgg_cut(final_layer,\
                                                      transformOnFinalLayer=transformOnFinalLayer)
            elif constrNet=='VGGInNorm' or constrNet=='VGGInNormAdapt':
                whatToload = 'varmean'
                dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
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
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
                # Compute the reference statistics
                list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
                target_number_im_considered = None
                target_set = 'trainval' # Todo ici
                dict_stats_target = get_dict_stats(target_dataset,target_number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=target_set)
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
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
                # Compute the reference statistics
                list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
                target_number_im_considered = None
                target_set = 'trainval'
                dict_stats_target,list_mean_and_std_target = get_dict_stats_BaseNormCoherent(target_dataset,source_dataset,target_number_im_considered,\
                       style_layers,list_mean_and_std_source,whatToload,saveformat='h5',\
                       getBeforeReLU=getBeforeReLU,target_set=target_set,\
                       applySqrtOnVar=True) # It also computes the reference statistics (mean,var)
                
                network_features_extraction = vgg_BaseNorm(style_layers,list_mean_and_std_source,
                    list_mean_and_std_target,final_layer=final_layer,transformOnFinalLayer=transformOnFinalLayer,
                    getBeforeReLU=getBeforeReLU)
            
            else:
                raise(NotImplementedError)
            
            print('== We will compute the bottleneck features ==')
            # Compute bottleneck features on the target dataset
            for i,name_img in  enumerate(df_label[item_name]):
                im_path =  os.path.join(path_to_img,name_img+'.jpg')
                image = load_crop_and_process_img(im_path)
                features_im = network_features_extraction.predict(image)
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
    
    AP_file  = name_base+'_'+final_clf
    if normalisation:
       AP_file +=  '_Norm' 
    if gridSearch:
       AP_file += '_GS'
    AP_file +='_AP.pkl'
    APfilePath =  os.path.join(output_path,AP_file)
    
    if not(os.path.isfile(APfilePath)) or ReDo:
        # Get Train set in form of numpy array
        if target_dataset=='Paintings':
            sLength = len(df_label[item_name])
            classes_vectors = np.zeros((sLength,num_classes))
            for i in range(sLength):
                for j in range(num_classes):
                    if( classes[j] in df_label['classe'][i]):
                        classes_vectors[i,j] = 1
        else:
            raise(NotImplementedError)
        
        print('classes_vectors.shape',classes_vectors.shape)
        index_train = df_label['set']=='train'
        print('features_net.shape',features_net.shape)
        X_train = features_net[index_train,:]
        y_train = classes_vectors[df_label['set']=='train',:]
        
        X_test= features_net[df_label['set']=='test',:]
        y_test = classes_vectors[df_label['set']=='test',:]
        
        X_val = features_net[df_label['set']==str_val,:]
        y_val = classes_vectors[df_label['set']==str_val,:]
        
        Xtrainval = np.vstack([X_train,X_val])
        ytrainval = np.vstack([y_train,y_val])
        
        if normalisation:
            scaler = StandardScaler()
            Xtrainval = scaler.fit_transform(Xtrainval)
            X_test = scaler.transform(X_test)
        
        dico_clf=TrainClassifierOnAllClass(Xtrainval,ytrainval,clf=final_clf,gridSearch=gridSearch)
        dico_pred = PredictOnTestSet(X_test,dico_clf,clf=final_clf)
        
        metrics = evaluationScore(y_test,dico_pred)
        with open(APfilePath, 'wb') as pkl:
            pickle.dump(metrics,pkl)
    else:
        with open(APfilePath, 'rb') as pkl:
            metrics = pickle.load(pkl)
    AP_per_class,P_per_class,R_per_class,P20_per_class = metrics
    
    print(target_dataset,source_dataset,number_im_considered,final_clf,features,transformOnFinalLayer,\
          constrNet,kind_method,'GS',gridSearch,'norm',normalisation,'getBeforeReLU',getBeforeReLU)
    print(style_layers)
    print(arrayToLatex(AP_per_class,per=True))
    
    return(AP_per_class,P_per_class,R_per_class,P20_per_class)

def evaluationScore(y_gt,dico_pred,verbose=False,k = 20):
    """
    @param k for precision at rank k
    """
    num_samples,num_classes = y_gt.shape
    AP_per_class = []
    P_per_class = []
    R_per_class = []
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
        
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        if verbose: print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class)
    
def RunAllEvaluation(dataset='Paintings'):
    target_dataset='Paintings'
    source_dataset = 'ImageNet'
    ## Run the baseline
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D']
    for normalisation in [False]:
        final_clf_list = ['LinearSVC'] # LinearSVC but also MLP
        features_list = ['fc2','fc1','flatten'] # We want to do fc2, fc1, max spatial and concat max and min spatial
         # We want to do fc2, fc1, max spatial and concat max and min spatial
        # Normalisation and not normalise
        kind_method = 'TL'
        style_layers = []
        
        # Baseline with just VGG
#        constrNet = 'VGG'
#        for final_clf in final_clf_list:
#            for features in features_list:
#                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
#                           constrNet,kind_method,style_layers,gridSearch=False,
#                           normalisation=normalisation,transformOnFinalLayer='')
#            
#            # Pooling on last conv block
#            for transformOnFinalLayer in transformOnFinalLayer_tab:
#                features = 'block5_pool'
#                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
#                           constrNet,kind_method,style_layers,gridSearch=False,
#                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer)
            
    #    constrNet = 'VGG'
    #    features = 'block5_pool'
    #    transformOnFinalLayer_list = ['GlobalMaxPool2D']
    #    for final_clf in final_clf_list:
    #        for transformOnFinalLayer in transformOnFinalLayer_list:
    #            learn_and_eval(target_dataset,source_dataset,final_clf,features,\
    #                           constrNet,kind_method,style_layers,gridSearch=False,\
    #                           transformOnFinalLayer=transformOnFinalLayer) 
         
        # With VGGInNorm
        style_layers_tab = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1'],['block1_conv1','block2_conv1']]
        style_layers_tab = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1','block2_conv1']]
        
        features_list = ['fc2','fc1','flatten']
        features_list = ['fc2','fc1']
        net_tab = ['VGGBaseNormCoherent','VGGBaseNorm','VGGInNorm','VGGInNormAdapt']
        number_im_considered_tab = [1000]
        for getBeforeReLU in [True,False]:
            for constrNet in net_tab:
                for final_clf in final_clf_list:
                    for style_layers in style_layers_tab:
                        for features in features_list:
                            for number_im_considered in number_im_considered_tab:
                                print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers)
                                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,\
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU)
                       
                        number_im_considered = 1000
                        # Pooling on last conv block
                        for transformOnFinalLayer in transformOnFinalLayer_tab:
                            print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers,'transformOnFinalLayer',transformOnFinalLayer)
                            features = 'block5_pool'
                            learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       transformOnFinalLayer=transformOnFinalLayer)
                    
                
                       
                   
    
if __name__ == '__main__': 
    # Ce que l'on 
    RunAllEvaluation()