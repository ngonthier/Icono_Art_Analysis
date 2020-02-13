#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:57:20 2020

The goal of this file is to regroup the RASTA performance evaluation, the main
idea is to try to use the GRAM matrices for classifying the artistic style

@author: gonthier
"""

from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator

from trouver_classes_parmi_K import TrainClassif
import numpy as np
import math
import matplotlib
import os.path
from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index,numeral_layers_index_bitsVersion
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
    load_resize_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model,\
    MLP_model,Perceptron_model,vgg_adaDBN,ResNet_AdaIn,ResNet_BNRefinements_Feat_extractor,\
    ResNet_BaseNormOnlyOnBatchNorm_ForFeaturesExtraction,ResNet_cut,vgg_suffleInStats,\
    get_ResNet_ROWD_meanX_meanX2_features,get_BaseNorm_meanX_meanX2_features,\
    get_VGGmodel_meanX_meanX2_features,add_head_and_trainable,extract_Norm_stats_of_ResNet,\
    vgg_FRN,set_momentum_BN,get_VGGmodel_gram_mean_features
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
from matplotlib.backends.backend_pdf import PdfPages
import gc 
import tempfile
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from keras_resnet_utils import getBNlayersResNet50,getResNetLayersNumeral,getResNetLayersNumeral_bitsVersion,\
    fit_generator_ForRefineParameters,fit_generator_ForRefineParameters_v2
import keras_preprocessing as kp

from Classifier_On_Features import TrainClassifierOnAllClass,PredictOnTestSet

from functools import partial
from sklearn.metrics import average_precision_score,make_scorer
from sklearn.model_selection import GridSearchCV

# Bayesian optimization of the hyper parameters of the networks
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt import JSONLogger
from bayes_opt.event import Events

from scipy.spatial.distance import pdist

from sklearn.metrics import pairwise_distances
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from StatsConstr_ClassifwithTL import TrainMLP

from sklearn.preprocessing import StandardScaler

#both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
def top_k_accuracy(truths,preds, k):
    best_k = np.argsort(preds, axis=1)[:,-k:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_k[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]

def get_top_scores(y,y_pred,top_k=[1,3,5]):
    scores = []
    for k in top_k:
        scores.append(top_k_accuracy(y,y_pred, k))
    return scores


def minimal_sizeOfRASTAImages():
    source_dataset = 'RASTA'
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
            path_data,Not_on_NicolasPC = get_database(source_dataset)
            
    list_h = []
    list_w = []
    for i,row in enumerate(df_label.iterrows()):
        row_value = row[1]
        image_path = path_to_img+'/'+ row_value[item_name] + '.jpg'
        im = Image.open(image_path)
        width, height = im.size
        list_h += [height]
        list_w += [width]
    print('Size of the images from RASTA dataset')
    print('Minimal width',np.min(list_w))
    print('Minimal height',np.min(list_h))
    print('Mean width',np.mean(list_w))
    print('Mean height',np.mean(list_h))
    print('Median width',np.median(list_w))
    print('Median height',np.median(list_h))
    
    # Minimal width 63
    # Minimal height 50
    # Mean width 955.2433613772783
    # Mean height 974.7609243544008
    # Median width 800.0
    # Median height 800.0

# def get_per_class_accuracy(labels,preds):
#     names = []
#     accs = []
#     dico = get_dico()
#     inv_dico = invert_dico(dico)
#     for value in set(labels):
#         s = 0
#         n = 0
#         for i in range(len(labels)):
#             if (labels[i] == value):
#                 n = n + 1
#                 if (preds[i] == value):
#                     s = s + 1
#         names.append(inv_dico.get(value))
#         accs.append(s / n * 100,)
#     return accs,names

def simpleRASTAclassification_withGramMatrices():
    """
    In this function we just try to see if one distance of the Gram matrices at 
    one of the layer can be a good proxy to the classification task
    """
    Net = 'VGG'
    source_dataset = 'RASTA'
    cropCenter = True
    # Get dataset information
    set_ = 'trainval'
    getBeforeReLU = True
    whatToload = 'covmean'
    
    number_im_considered = None
    
    style_layers_all = ['block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
           ]
    style_layers_all = ['block3_conv1'
           ]
    layer_used = style_layers_all[0]
    
    sizeIm_tab = [224,800]
    sizeIm = sizeIm_tab[1]
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(source_dataset)    
    
    df_train = df_label[df_label['set']=='train']
    df_test = df_label[df_label['set']=='test']
    df_val = df_label[df_label['set']==str_val]
    df_trainval = df_train.append(df_val)
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',\
                               'RASTA')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    for layer_used in style_layers_all:
        style_layers = [layer_used]

        for sizeIm in sizeIm_tab: # Il va falloir modifier cela a un moment : calculer la taille minimale du dataset RASTA
            
            name = 'Triu_cov_of_VGG_'+ layer_used 
            name_feat = name+ '_ImSize'+str(sizeIm)
            outfile_test = os.path.join(output_path, name_feat + '_test.npy')
            outfile_trainval = os.path.join(output_path,name_feat + '_trainval.npy')
            outfile_test_y_get = os.path.join(output_path, name + '_test_labels.npy')
            outfile_trainval_y_gt = os.path.join(output_path,name + '_trainval_labels.npy')
            
            if os.path.isfile(outfile_test) and  os.path.isfile(outfile_trainval):
                Xtest=np.load(outfile_test)
                Xtrain=np.load(outfile_trainval)
                Ytest_gt=np.load(outfile_test_y_get)
                Ytrain_gt=np.load(outfile_trainval_y_gt)
                Ytest_pred_minOfDist = np.empty(Ytest_gt.shape, dtype=np.float32)
                Ytest_pred_meanOfDist = np.empty(Ytest_gt.shape, dtype=np.float32)       
                Ytest_pred_meanOf_kNN_Dist = np.empty(Ytest_gt.shape, dtype=np.float32)  
            else:
            
                net_get_cov =  get_VGGmodel_gram_mean_features(style_layers,getBeforeReLU=getBeforeReLU)
                itera = 1000
                
                Xtrain = None
                Xtest = None
                Xtrain_dict = {}
                Xtest_dict = {}
                for l,layer in enumerate(style_layers):
                    Xtrain_dict[layer] = None
                    Xtest_dict[layer] = None
                Ytrain_gt = np.empty((len(df_trainval),num_classes), dtype=np.float32)
                Ytrain_pred = np.empty((len(df_trainval),num_classes), dtype=np.float32)
                for i,row in enumerate(df_trainval.iterrows()):
                    row_value = row[1]
                    image_path = path_to_img+'/'+ row_value[item_name] + '.jpg'
                    labels = row_value[classes].values
                    Ytrain_gt[i,:] = labels
                    X_i = None
                    X_i = None
                    if number_im_considered is None or i < number_im_considered:
                        if i%itera==0: print(i,image_path)
                        head, tail = os.path.split(image_path)
                        short_name = '.'.join(tail.split('.')[0:-1])
                        # Get the covairances matrixes and the means
                        try:
                            #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                            if cropCenter:
                                image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                                        crop_size=sizeIm,interpolation='lanczos:center')
                                # For VGG or ResNet with classification head size == 224
                            else:
                                image_array = load_resize_and_process_img(image_path,Net=Net,sizeIm=sizeIm)
                            net_cov_mean = net_get_cov.predict(image_array, batch_size=1)
                        except IndexError as e:
                            print(e)
                            print(i,image_path)
                            raise(e)
            
                        for l,layer in enumerate(style_layers):
            #                        [cov,mean] = vgg_cov_mean[l]
                            cov = net_cov_mean[2*l][0,:,:]
                            #cov = cov.reshape(int(np.sqrt(cov.size)),int(np.sqrt(cov.size)))
                            #print('cov.size',cov.shape)
                            iu1 = np.triu_indices(cov.shape[0])
                            cov_vectorized = cov[iu1] # Only the superior triangular part because the Gram matrices is symetric
                            mean = net_cov_mean[2*l+1][0,:] # car batch size == 1
                            if X_i is None:
                                X_i = cov_vectorized
                            else:
                                X_i = np.concatenate((X_i,cov_vectorized))
                            if Xtrain_dict[layer] is None:
                                Xtrain_dict[layer]={}
                            #Xtrain_dict[layer][short_name]= cov
                        if Xtrain is None:
                            Xtrain = np.empty((len(df_trainval),len(X_i)), dtype=np.float32)
                        Xtrain[i,:] = X_i
        
                
                Ytest_gt = np.empty((len(df_test),num_classes), dtype=np.float32)
                Ytest_pred_minOfDist = np.empty((len(df_test),num_classes), dtype=np.float32)
                Ytest_pred_meanOfDist = np.empty((len(df_test),num_classes), dtype=np.float32)       
                Ytest_pred_meanOf_kNN_Dist = np.empty((len(df_test),num_classes), dtype=np.float32)       
                for i,row in enumerate(df_test.iterrows()):
                    row_value = row[1]
                    image_path = path_to_img+'/'+ row_value[item_name] + '.jpg'
                    labels = row_value[classes].values
                    Ytest_gt[i,:] = labels
                    X_i = None
                    X_i = None
                    if number_im_considered is None or i < number_im_considered:
                        if i%itera==0: print(i,image_path)
                        head, tail = os.path.split(image_path)
                        short_name = '.'.join(tail.split('.')[0:-1])
                        # Get the covairances matrixes and the means
                        try:
                            #vgg_cov_mean = sess.run(get_gram_mean_features(vgg_inter,image_path))
                            if cropCenter:
                                image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                                        crop_size=sizeIm,interpolation='lanczos:center')
                                # For VGG or ResNet with classification head size == 224
                            else:
                                image_array = load_resize_and_process_img(image_path,Net=Net,sizeIm=sizeIm)
                            net_cov_mean = net_get_cov.predict(image_array, batch_size=1)
                        except IndexError as e:
                            print(e)
                            print(i,image_path)
                            raise(e)
            
                        for l,layer in enumerate(style_layers):
            #                        [cov,mean] = vgg_cov_mean[l]
                            cov = net_cov_mean[2*l][0,:,:]
                            iu1 = np.triu_indices(cov.shape[0])
                            cov_vectorized = cov[iu1] # Only the superior triangular part because the Gram matrices is symetric
                            mean = net_cov_mean[2*l+1][0,:] # car batch size == 1
                            if X_i is None:
                                X_i = cov_vectorized
                            else:
                                X_i = np.concatenate((X_i,cov_vectorized))
                            if Xtest_dict[layer] is None:
                                Xtest_dict[layer]={}
                            #Xtest_dict[layer][short_name]= cov
                        if Xtest is None:
                            Xtest = np.empty((len(df_test),len(X_i)), dtype=np.float32)
                        Xtest[i,:] = X_i
                
                np.save(outfile_test, Xtest)
                np.save(outfile_trainval, Xtrain)
                np.save(outfile_test_y_get, Ytest_gt)
                np.save(outfile_trainval_y_gt, Ytrain_gt)
                
                del net_get_cov
            
            #argmin_test = pairwise_distances_argmin() # Y[argmin[i], :] is the row in Y that is closest to X[i, :].
            
            # metrics = ['euclidean','l2','manhattan']
            # allMethod = False
            # dist_metric = 'l2'
            # #dist_metric = 'manhattan'
            
            # pairwise_dist = pairwise_distances(Xtest,Xtrain,metric=dist_metric)
            # # D_{i, j} is the distance between the ith array from X and the jth array from Y
            
            # # Problem ici avec block3_conv1
            # # MemoryError: Unable to allocate array with shape (4268, 131328) and data type float64
            
            # max_distance = np.max(pairwise_dist)
            # min_distance = np.min(pairwise_dist)
            # mean_distance = np.mean(pairwise_dist)
            
            # k_n = 5
            # for c,classe in enumerate(classes):
            #     index_c_trainval_samples = np.where(Ytrain_gt[:,c]==1.0)[0]
            #     for j in range(len(df_test)):
            #         image_j_pairwise_dist_images_classe_c = pairwise_dist[j,index_c_trainval_samples]
            #         min_image_j_images_classe_c = np.min(image_j_pairwise_dist_images_classe_c)
                    
            #         # if j < 2:
            #         #     argmin_image_j_images_classe_c = np.argmin(image_j_pairwise_dist_images_classe_c)
                        
                    
            #         Ytest_pred_minOfDist[j,c] = np.exp(-min_image_j_images_classe_c)
            #         if allMethod:
            #             best_k = np.argsort(image_j_pairwise_dist_images_classe_c, axis=0)[:k_n]
            #             Ytest_pred_meanOfDist[j,c] = np.exp(-np.mean(image_j_pairwise_dist_images_classe_c))
            #             Ytest_pred_meanOf_kNN_Dist[j,c] = np.exp(-np.mean(image_j_pairwise_dist_images_classe_c[best_k]))
        
                
            top_k = [1,3,5]
            
            # print('\nFor layer :',layer_used,'and size :',sizeIm)
            # if allMethod: print('\nMinimal distance')
            # scores = get_top_scores(Ytest_gt,Ytest_pred_minOfDist,top_k=top_k)
            # for val,pred in zip(top_k,scores):
            #     print('Top-{} accuracy : {}%'.format(val,pred*100))
              
            # if allMethod:
            #     print('\nMean distance')
            #     scores = get_top_scores(Ytest_gt,Ytest_pred_meanOfDist,top_k=top_k)
            #     for val,pred in zip(top_k,scores):
            #         print('Top-{} accuracy : {}%'.format(val,pred*100))
                    
            #     print('\nMean distance of',k_n,'NN')
            #     scores = get_top_scores(Ytest_gt,Ytest_pred_meanOf_kNN_Dist,top_k=top_k)
            #     for val,pred in zip(top_k,scores):
            #         print('Top-{} accuracy : {}%'.format(val,pred*100))
                    
                
                    
            # SVM classification
            # class_weight = 'balanced'
            # C_finalSVM = 1.0
            # classifier = LinearSVC(penalty='l2',class_weight=class_weight,
            # loss='squared_hinge',max_iter=1000,dual=False,C=C_finalSVM) # dual=False if number of samples > num of features
            
            print(Xtrain.shape)
            
            num_classes = 25
            optimizer = 'SGD'
            regulOnNewLayer = None
            regulOnNewLayerParam = []
            dropout = None
            verbose = True
            decay = 10**(-4)
            SGDmomentum= 0.9
            lr = 0.001
            nesterov = False
            model = MLP_model(num_of_classes=num_classes,optimizer=optimizer,lr=lr,\
                                          regulOnNewLayer=regulOnNewLayer,regulOnNewLayerParam=regulOnNewLayerParam,dropout=dropout,\
                                          nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay,verbose=verbose,\
                                          final_activation='softmax',metrics='top_k_categorical_accuracy',loss='categorical_crossentropy')
            
            # X_train_big, X_train_small, y_train_big, y_train_small = train_test_split(Xtrain, Ytrain_gt, test_size=0.1, random_state=42)
            
            # Yi = np.where(y_train_small==1.0)
            # Yi1 = Yi[1]
            # classifier.fit(X_train_small,Yi1)

            scaler = StandardScaler(copy=False)
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            if layer_used=='block1_conv1':
                batch_size =2048
            elif layer_used=='block3_conv1': # Dans ce cas là on a un vector de features de
                # taille 32896
                batch_size = 64
            else:
                batch_size = 1024*2048//Xtrain.shape[1] 
            print('batch_size :';batch_size)
                
            model = TrainMLP(model,Xtrain,Ytrain_gt,None,None,batch_size=batch_size,epochs=20,\
             verbose=True,plotConv=False,return_best_model=True,\
             NoValidationSetUsed=False,RandomValdiationSet=True)

            Y_clf_prediction = model.predict(Xtest)
            print('\nFor layer :',layer_used,'and size :',sizeIm)
            scores = get_top_scores(Ytest_gt,Y_clf_prediction,top_k=top_k)
            for val,pred in zip(top_k,scores):
                print('Top-{} accuracy : {}%'.format(val,pred*100))
                
           # Avec dropout 
            #Top-1 accuracy : 12.303377636873552%
            #Top-3 accuracy : 14.961590049993903%
            #Top-5 accuracy : 18.68064870137788%
                
            # Sans dropout 
            # Top-1 accuracy : 9.55980977929521%
            # Top-3 accuracy : 12.242409462260701%
            # Top-5 accuracy : 15.998049018412388%
                
            # Sans dropout et lr at 0.1
            #     For layer : block1_conv1 and size : 224
            # Top-1 accuracy : 12.303377636873552%
            # Top-3 accuracy : 14.961590049993903%
            # Top-5 accuracy : 18.68064870137788%
                
             # Avec sizeIm 800 et lr = 0.1 et standardisation of the input image
            # Top-1 accuracy : 25.57005243263017%
            # Top-3 accuracy : 48.8477014998171%
            # Top-5 accuracy : 62.077795390806%
                
            # # For lr = 0.01
            # For layer : block1_conv1 and size : 800
            # Top-1 accuracy : 26.155346908913547%
            # Top-3 accuracy : 50.49384221436411%
            # Top-5 accuracy : 63.723936105353005%
                
            #dico_clf=TrainClassifierOnAllClass(Xtrainval,ytrainval,clf=final_clf,gridSearch=gridSearch)
            # Prediction
            #dico_pred = PredictOnTestSet(X_test,dico_clf,clf=final_clf)
          
    # For the first layer !         
    # For euclidian distance 
    # Minimal distance

    # Top-1 accuracy : 1.6095598097792954%
    
    # Top-3 accuracy : 5.377393000853554%
    
    # Top-5 accuracy : 11.620534081209609%
    # Mean distance
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%
    # Mean distance of 5 NN
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%        
    
    # For l2 distance
    # Minimal distance

    # Top-1 accuracy : 1.6095598097792954%
    
    # Top-3 accuracy : 5.377393000853554%
    
    # Top-5 accuracy : 11.620534081209609%
    # Mean distance
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%
    # Mean distance of 5 NN
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%
    
    
    # Mannathan distance 
    # Minimal distance

    # Top-1 accuracy : 1.5729789050115839%
    
    # Top-3 accuracy : 5.340812096085843%
    
    # Top-5 accuracy : 11.583953176441897%
    # Mean distance
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%
    # Mean distance of 5 NN
    
    # Top-1 accuracy : 1.4266552859407389%
    
    # Top-3 accuracy : 5.206682111937569%
    
    # Top-5 accuracy : 11.449823192293623%
    
    # # Load the covariance matrices of the trainval set
    # dict_stats_trainval = get_dict_stats(source_dataset,number_im_considered,style_layers,\
    #                whatToload,saveformat='h5',set=set_,getBeforeReLU=getBeforeReLU,\
    #                Net='VGG',style_layers_imposed=[],\
    #                list_mean_and_std_source=[],list_mean_and_std_target=[],\
    #                cropCenter=False,BV=True,sizeIm=sizeIm)
        
    # set_test = 'test'
    # # Load the covariance matrices of the trainval set
    # dict_stats_test = get_dict_stats(source_dataset,number_im_considered,style_layers,\
    #                whatToload,saveformat='h5',set=set_test,getBeforeReLU=getBeforeReLU,\
    #                Net='VGG',style_layers_imposed=[],\
    #                list_mean_and_std_source=[],list_mean_and_std_target=[],\
    #                cropCenter=False,BV=True,sizeIm=sizeIm)
        
    
        
if __name__ == '__main__': 
    simpleRASTAclassification_withGramMatrices()
        
    

