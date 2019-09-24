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
from Stats_Fcts import vgg_cut,vgg_AdaIn_adaptative,vgg_AdaIn,load_crop_and_process_img
from IMDB import get_database
import pickle
import pathlib
from Classifier_On_Features import TrainClassifierOnAllClass,PredictOnTestSet
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score,label_ranking_average_precision_score,classification_report
from sklearn.metrics import matthews_corrcoef,f1_score
from Custom_Metrics import ranking_precision_score
from LatexOuput import arrayToLatex

def compute_ref_stats(dico,style_layers,type_ref='mean',imageUsed='all',whatToload = 'varmean'):
    """
    This function compute a reference statistics on the statistics of the whole dataset
    """
    vgg_stats_values = []
    for l,layer in enumerate(style_layers):
        stats = dico[layer]
        if whatToload == 'varmean':
            if imageUsed=='all':
                if type_ref=='mean':
                    mean_stats = np.mean(stats,axis=0)
                    vgg_stats_values += [[mean_stats[1,:],mean_stats[0,:]]]
                    # To return vgg_mean_vars_values
    return(vgg_stats_values)

def learn_and_eval(target_dataset,source_dataset,final_clf,features,constrNet,kind_method,
                   style_layers = ['block1_conv1',
                                    'block2_conv1',
                                    'block3_conv1', 
                                    'block4_conv1', 
                                    'block5_conv1'
                                   ],normalisation=False,gridSearch=True,ReDo=False,\
                                   transformOnFinalLayer=''):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
    TODO : linear SVM - MLP - perceptron - MLP one per class - MLP with finetuning of the net
    @param : features : which features we will use
    TODO : fc2, fc1, max spatial, max et min spatial
    @param : constrNet the constrained net used
    TODO : VGGAdaIn, VGGAdaInAdapt seulement sur les features qui répondent trop fort, VGGGram
    @param : kind_method the type of methods we will use : TL or FT
    """
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    num_layers = numeral_layers_index(style_layers)
    # Compute statistics on the source_dataset
    if source_dataset is None:
        constrNet='VGG'
    elif constrNet=='VGG':
        pass
    else:
        if constrNet=='VGGAdaIn' or constrNet=='VGGAdaInAdapt':
            whatToload = 'varmean'
        else:
            raise(NotImplementedError)
        number_im_considered = 10000
        source_dataset = 'ImageNet'
        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                   whatToload,saveformat='h5')
        # Compute the reference statistics
        vgg_mean_vars_values = compute_ref_stats(dict_stats,style_layers,type_ref='mean',\
                                                 imageUsed='all',whatToload =whatToload)
        
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(target_dataset)
        
    name_base = constrNet + '_'  +target_dataset +'_'
    if not(constrNet=='VGG'):
        name_base += source_dataset +str(number_im_considered)+ '_' + num_layers
    name_base +=  features 
    if not(transformOnFinalLayer is None or transformOnFinalLayer==''):
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
            features_net = None
            im_net = []
            # Load Network 
            if constrNet=='VGG':
                network_features_extraction = vgg_cut(final_layer)
            elif constrNet=='VGGAdaInAdapt':
                network_features_extraction = vgg_AdaIn_adaptative(style_layers,vgg_mean_vars_values,final_layer=final_layer,
                             HomeMadeBatchNorm=True)
            elif constrNet=='VGGAdaIn':
                network_features_extraction = vgg_AdaIn(style_layers,vgg_mean_vars_values,final_layer=final_layer,
                             HomeMadeBatchNorm=True)
            
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
        
        print(classes_vectors.shape)
        print(df_label)
        index_train = df_label['set']=='train'
        print(index_train.shape)
        print(features_net.shape)
        X_train = features_net[index_train,:]
        y_train = classes_vectors[df_label['set']=='train',:]
        
        X_test= features_net[df_label['set']=='test',:]
        y_test = classes_vectors[df_label['set']=='test',:]
        
        X_val = features_net[df_label['set']==str_val,:]
        y_val = classes_vectors[df_label['set']==str_val,:]
        
        Xtrainval = np.vstack([X_train,X_val])
        ytrainval = np.vstack([y_train,y_val])
        
        if normalisation:
            raise(NotImplementedError)
        
        dico_clf=TrainClassifierOnAllClass(Xtrainval,ytrainval,clf=final_clf,gridSearch=gridSearch)
        dico_pred = PredictOnTestSet(X_test,dico_clf,clf=final_clf)
        
        metrics = evaluationScore(y_test,dico_pred)
        with open(APfilePath, 'wb') as pkl:
            pickle.dump(metrics,pkl)
    else:
        with open(APfilePath, 'rb') as pkl:
            metrics = pickle.load(pkl)
    AP_per_class,P_per_class,R_per_class,P20_per_class = metrics
    
    print(target_dataset,source_dataset,final_clf,features,transformOnFinalLayer,\
          constrNet,kind_method,'GS',gridSearch)
    print(style_layers)
    print(arrayToLatex(AP_per_class,per=True))
    
    return(AP_per_class,P_per_class,R_per_class,P20_per_class)

def evaluationScore(y_gt,dico_pred,k = 20):
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
        print("Average Precision on all the data for classe",c," = ",AP)  
        AP_per_class += [AP] 
        test_precision = precision_score(y_gt_c,y_predict_test)
        test_recall = recall_score(y_gt_c,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = f1_score(y_gt_c,y_predict_test)
        
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class)
    
def RunAllEvaluation(dataset='Paintings'):
    target_dataset='Paintings'
    source_dataset = 'ImageNet'
    ## Run the baseline
    
    final_clf_list = ['LinearSVC'] # LinearSVC but also MLP
    features_list = ['fc2','fc1','flatten'] # We want to do fc2, fc1, max spatial and concat max and min spatial
    # Normalisation and not normalise
    net_tab = ['VGG','VGGAdaIn']
    kind_method = 'TL'
    style_layers = []
    
    for final_clf in final_clf_list:
        for features in features_list:
            for constrNet  in net_tab:
                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False)
        
    constrNet = 'VGG'
    features = 'block5_pool'
    transformOnFinalLayer_list = ['GlobalMaxPool2D']
    for final_clf in final_clf_list:
        for transformOnFinalLayer in transformOnFinalLayer_list:
            learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False,\
                           transformOnFinalLayer=transformOnFinalLayer) 
                   
                   
    
if __name__ == '__main__': 
    # Ce que l'on 
    learn_and_eval()