#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:04:28 2018

Scripts pour la baseline pour faire les calculs pour le papier VISART2018 
et pour faire tourner sur le cluster

@author: gonthier
"""

import time
import pickle
import gc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tf_faster_rcnn.lib.model.nms_wrapper import nms
#import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
#from sklearn.model_selection import PredefinedSplit,train_test_split
#from nltk.classify.scikitlearn import SklearnClassifier
#from tf_faster_rcnn.tools.demo import vis_detections
import numpy as np
import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score,recall_score,precision_score,make_scorer,f1_score
from Custom_Metrics import ranking_precision_score
from tf_faster_rcnn.lib.model.test import get_blobs
#from Classifier_Evaluation import Classification_evaluation
import os.path
from LatexOuput import arrayToLatex
from sklearn.linear_model import SGDClassifier
#import pathlib
#from sklearn.externals import joblib # To save the classifier
from tool_on_Regions import reduce_to_k_regions
from tf_faster_rcnn.lib.datasets.factory import get_imdb
#from Estimation_Param import kde_sklearn,findIntersection
#from utils.save_param import create_param_id_file_and_dir,write_results,tabs_to_str
#from hpsklearn import HyperoptEstimator,sgd
#from hyperopt import tpe
from random import uniform
from sklearn.metrics import hinge_loss
from IMDB import get_database

# For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64

def rand_convex(n):
    rand = np.matrix([uniform(0.0, 1.0) for i in range(n)])
    return(rand / np.sum(rand))

def TrainClassif(X,y,clf='LinearSVC',class_weight=None,gridSearch=True,n_jobs=-1,C_finalSVM=1,cskind=None,
                 testMode=False):
    """
    @param clf : LinearSVC, defaultSGD or SGDsquared_hinge 
    """
    if cskind =='' or cskind is None:
        # default case
        cs = np.logspace(-5, -2, 20)
        cs = np.hstack((cs,[0.01,0.2,1.,2.,10.,100.]))
    elif cskind=='small':
        cs = np.logspace(-5, 3, 9)
    param_grid = dict(C=cs)
    # TODO  class_weight='balanced' TODO add this parameter ! 
    #Prefer dual=False when n_samples > n_features.
    n_samples,n_features = X.shape
    dual = True
#    if n_samples > n_features:
#        dual = False

    if testMode:
        max_iter = 2
    else:
        max_iter=1000

    if gridSearch:
        if clf == 'LinearSVC':
            estimator = LinearSVC(penalty='l2',class_weight=class_weight, 
                            loss='squared_hinge',max_iter=max_iter,dual=dual)
            param_grid = dict(C=cs)
        elif clf == 'defaultSGD':
            estimator = SGDClassifier(max_iter=max_iter, tol=0.0001)
            param_grid = dict(alpha=cs)
        elif clf == 'SGDsquared_hinge':
            estimator = SGDClassifier(max_iter=max_iter, tol=0.0001,loss='squared_hinge')
            param_grid = dict(alpha=cs)
    
        classifier = GridSearchCV(estimator, refit=True,
                                  scoring =make_scorer(average_precision_score,
                                                       needs_threshold=True),
                                  param_grid=param_grid,n_jobs=n_jobs)
    else:
        # class_weight='balanced'
        if clf == 'LinearSVC':
            classifier = LinearSVC(penalty='l2',class_weight=class_weight,
                                   loss='squared_hinge',max_iter=max_iter,dual=dual,C=C_finalSVM)
        elif clf == 'defaultSGD':
            classifier = SGDClassifier(max_iter=max_iter)
        elif clf == 'SGDsquared_hinge':
            classifier = SGDClassifier(max_iter=max_iter, tol=0.0001,loss='squared_hinge')
    
    classifier.fit(X,y)
    
    return(classifier)


def Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'Paintings',Test_on_k_bag = False,
                             normalisation= False,baseline_kind = 'MAX1',
                             verbose = True,gridSearch=False,k_per_bag=300,jtest=0,testMode=False,
                             n_jobs=-1,clf='LinearSVC',PCAuse=False,variance_thres= 0.9,
                             restarts = 0,max_iter = 10,reDo=True):
    """ 
    27 juin 2018 ==> Il faut que les dossiers soit calculés avant sur mon ordi 
    puis passer sur le cluster
    ---------------------------------------------------------------------------
    
    Detection based on CNN features with Transfer Learning on Faster RCNN output
    This is used to compute the baseline MAX1 (only the best score object as training exemple)
    and MAXA_lowMem (all the regions for the negatives)
    This one use pickle precomputed saved data
    
    
    @param : demonet : the kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : database : the database used for the classification task
    @param : verbose : Verbose option classical
    @param : testMode : boolean True we only run on one class
    @param : jtest : the class on which we run the test
    @param : PlotRegions : plot the regions used for learn and the regions in the positive output response
    @param : saved_clf : [default : True] Too sva ethe classifier 
    @param : RPN=False trace la boite autour de l'element ayant le plus haut score du RPN object
    @param : CompBest : Comparaison with the CompBest classifier trained
    @param : Stocha : Use of a SGD for the MIL SVM SAID [default : False]
    @param : k_per_bag : number of element per batch in the slection phase [defaut : 30]
    @param : PCAuse : boolean to know if we do a PCA or not before learning
    @param : variance_thres variance threshold keep features for PCA
    @param : restarts number of restarts in the MISVM case
    @param : max_iter : maximum number of iteration in the MISVM et miSVM case
    The idea of thi algo is : 
        1/ Compute CNN features
        2/ Do NMS on the regions 
    
    option to train on background part also
    option on  scaling : sklearn.preprocessing.StandardScaler
    option : add a wieghted balanced of the SVM because they are really unbalanced classes
    
    Cette fonction permet de calculer les performances AP pour les differents dataset 
    Wikidata et Your Paintings avec l'algo de selection de Said et l'entrainement du SVM final 
    en dehors du code de Said trouver_classes_parmi_k
    
    FasterRCNN_TL_MISVM est sense etre la meme chose avec en utilisant les algos MISVM et miSVm de Andrews
    
    
    """
    # TODO be able to train on background 
    list_methods =['MAXA_lowMem','MAX1','MEAN','MISVM','miSVM','SISVM','MAXA']
    print('==========')
    print('Baseline for ',database,demonet,baseline_kind,'gridSearch',gridSearch,'clf',clf)
    try:
        if demonet == 'vgg16_COCO':
            num_features = 4096
        elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
            num_features = 2048
        item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC \
            = get_database(database)

        if(jtest>len(classes)) and testMode:
           print("We are in test mode but jtest>len(classes), we will use jtest =0" )
           jtest =0
        N = 1
        extL2 = ''
        nms_thresh = 0.7
        savedstr = '_all'
        name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+ \
            '_TLforMIL_nms_'+str(nms_thresh)+savedstr+'.pkl'
           
        features_resnet_dict = {}
        sLength_all = len(df_label[item_name])


        if not(os.path.isfile(name_pkl)):
            # Compute the features
            if Not_on_NicolasPC: 
                print("You need to compute the CNN features before and put them in the folder :",path_data)
                return(0)
            if verbose: print("We will computer the CNN features")
            filesave = 'pkl'
            from FasterRCNN import Compute_Faster_RCNN_features
            Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                         database=database,augmentation=False,L2 =False,
                                         saved='all',verbose=verbose,filesave=filesave)
        
#        features_resnet = np.empty((sLength_all,k_per_bag,size_output),dtype=np.float32)  
        classes_vectors = np.zeros((sLength_all,num_classes),dtype=np.float32)
        if database in ['Wikidata_Paintings_miniset_verif','VOC2007','watercolor','IconArt_v1','PeopleArt']:
            classes_vectors = df_label.as_matrix(columns=classes)
        f_test = {}

        # Parameters important
        new_nms_thresh = 0.0
        score_threshold = 0.1
        minimal_surface = 36*36
        # In the case of Wikidata
        if database=='Wikidata_Paintings_miniset_verif':
            raise(NotImplementedError)
    
        if database in['VOC2007','watercolor','clipart','IconArt_v1','PeopleArt']:
            if database=='VOC2007' : imdb = get_imdb('voc_2007_test',data_path=path_data)
            if database=='watercolor' : imdb = get_imdb('watercolor_test',data_path=path_data)
            if database=='clipart' : imdb = get_imdb('clipart_test',data_path=path_data)
            if database=='IconArt_v1' : imdb = get_imdb('IconArt_v1_test',data_path=path_data)
            if database=='PeopleArt' : imdb = get_imdb('PeopleArt_test',data_path=path_data)
            imdb.set_force_dont_use_07_metric(True)
            if database in ['IconArt_v1']:
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            else:
                num_images = len(imdb.image_index)
        else:
            num_images = len(df_label[df_label['set']=='test'])
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

        # First we save the test images
        roi_test = {}
        name_test = {}
        key_test = 0
        if database=='VOC12' or database=='Paintings':
            for i,name_img in  enumerate(df_label[item_name]):
                    for j in range(num_classes):
                        if(classes[j] in df_label['classe'][i]):
                            classes_vectors[i,j] = 1
        
        # Separation training, validation, test set
        if database in['VOC12','Paintings','VOC2007','watercolor','IconArt_v1','WikiTenLabels','PeopleArt']:
            if database in ['VOC2007','watercolor','IconArt_v1','WikiTenLabels','PeopleArt']:
                str_val ='val' 
            else: 
                str_val='validation'
#            X_train = features_resnet[df_label['set']=='train',:,:]
            y_train = classes_vectors[df_label['set']=='train',:].astype(np.float32)
#            X_test= features_resnet[df_label['set']=='test',:,:]
            y_test = classes_vectors[df_label['set']=='test',:].astype(np.float32)
#            X_val = features_resnet[df_label['set']==str_val,:,:]
            y_val = classes_vectors[df_label['set']==str_val,:].astype(np.float32)
#            X_trainval = np.append(X_train,X_val,axis=0)
            y_trainval =np.append(y_train,y_val,axis=0).astype(np.float32)
            names = df_label.as_matrix(columns=[item_name])
            name_train = names[df_label['set']=='train']
            name_val = names[df_label['set']==str_val]
            name_all_test =  names[df_label['set']=='test']
            name_trainval = np.append(name_train,name_val,axis=0)
#        elif database=='Wikidata_Paintings_miniset_verif' :
#            name = df_label.as_matrix(columns=[item_name])
#            name_trainval = name[index_trainval]
#            #name_test = name[index_test]
##            X_test= features_resnet[index_test,:,:]
#            y_test = classes_vectors[index_test,:]
##            X_trainval =features_resnet[index_trainval,:,:]
#            y_trainval =  classes_vectors[index_trainval,:]
        
        name_trainval = name_trainval.ravel()
        
        if baseline_kind in list_methods:
            if verbose: print("Start loading precomputed data",name_pkl)
            with open(name_pkl, 'rb') as pkl:
                for i,name_img in  enumerate(df_label[item_name]):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                        features_resnet_dict_tmp = pickle.load(pkl)
                        if i==1000:
                            features_resnet_dict = features_resnet_dict_tmp
                        else:
                            features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
                features_resnet_dict_tmp = pickle.load(pkl)
                features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
            if verbose: print("Data loaded",len(features_resnet_dict))
        else:
            print(baseline_kind,' unknown')
            raise(NotImplementedError)
        
        
        # Then we run on the different classes
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        if baseline_kind == 'MAX1':
            number_neg = 1
        elif baseline_kind in ['MAXA_lowMem','MISVM','miSVM','SISVM','MAXA']:
            number_neg = 300
            
        # Training time
        dict_clf = {}
        for j,classe in enumerate(classes):
            gc.collect()
            if testMode and not(j==jtest):
                print('Skip because test mode')
                continue
            
            if baseline_kind == 'MAX1':
                if j==0: X_trainval_all  = []
                number_pos_ex = int(np.sum(y_trainval[:,j]))
                number_neg_ex = len(y_trainval) - number_pos_ex
                number_ex = number_pos_ex + number_neg*number_neg_ex
                y_trainval_select = np.zeros((number_ex,),dtype=np.float32)
                X_trainval_select = np.empty((number_ex,num_features),dtype=np.float32)
                index_nav = 0
                for i,name_img in  enumerate(name_trainval):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                    rois,roi_scores,fc7 = features_resnet_dict[name_img]
                    if y_trainval[i,j] == 1: # Positive exemple
                        y_trainval_select[index_nav] = 1
                        X_trainval_select[index_nav,:] = fc7[0,:] # The roi_scores vector is sorted
                        index_nav += 1 
                    else:
                        X_trainval_select[index_nav,:] = fc7[0,:]
                        index_nav += 1
                    if j==0:
                        if not(len(X_trainval_all)==0):
                            X_trainval_all += [fc7.astype(np.float32)]
                        else:
                            X_trainval_all += [fc7.astype(np.float32)]
                if j==0: X_trainval_all = np.concatenate(X_trainval_all,axis=0).astype(np.float32)
            elif baseline_kind == 'MAXA':
                if j==0: X_trainval_all  = []
                number_pos_ex = int(np.sum(y_trainval[:,j]))
                number_neg_ex = (len(y_trainval) - number_pos_ex)
                number_ex = number_pos_ex + number_neg*number_neg_ex
                y_trainval_select = []
                X_trainval_select = []  
                index_nav = 0
                for i,name_img in  enumerate(name_trainval):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                    rois,roi_scores,fc7 = features_resnet_dict[name_img]
                    if y_trainval[i,j] == 1: # Positive exemple
                        X_to_add = fc7[0,:].astype(np.float32).reshape((1,-1))
                        y_to_add = [1]
                    else:
                        X_to_add = fc7.astype(np.float32)
                        y_to_add = [0]*len(fc7)
                    y_trainval_select += y_to_add
                    X_trainval_select += [X_to_add]
                    if j==0:
                        if not(len(X_trainval_all)==0):
                            X_trainval_all += [fc7.astype(np.float32)]
                        else:
                            X_trainval_all = [fc7.astype(np.float32)]
                if j==0: X_trainval_all = np.concatenate(X_trainval_all,axis=0).astype(np.float32)
                X_trainval_select = np.concatenate(X_trainval_select,axis=0).astype(np.float32)
                y_trainval_select = np.hstack(y_trainval_select).astype(np.float32)
            elif baseline_kind in ['MISVM','miSVM']:
                if j==0: X_trainval_all  = []
                number_pos_ex = int(np.sum(y_trainval[:,j]))
                number_neg_ex = len(y_trainval) - number_pos_ex
                number_ex = number_pos_ex + number_neg*number_neg_ex
#                y_trainval_select_neg = np.zeros((300*number_neg_ex,),dtype=np.float32)
                y_trainval_select_neg = []
                y_trainval_select_pos = np.ones((number_pos_ex,),dtype=np.float32)
                X_trainval_select_neg = []
                X_trainval_select_pos = [] # TODO change that number
#                X_trainval_select_neg = np.empty((300*number_neg_ex,num_features),dtype=np.float32)
#                X_trainval_select_pos = np.empty((number_pos_ex,300,num_features),dtype=np.float32) # TODO change that number
                for i,name_img in  enumerate(name_trainval):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                    rois,roi_scores,fc7 = features_resnet_dict[name_img]
                    if y_trainval[i,j] == 1: # Positive exemple
                        if not(len(X_trainval_select_pos)==0):
                            X_trainval_select_pos += [fc7.astype(np.float32)] 
                        else:
                            X_trainval_select_pos = [fc7.astype(np.float32)] 
                    else:
                        if not(len(X_trainval_select_neg)==0):
                            X_trainval_select_neg += [fc7.astype(np.float32)] 
                            y_trainval_select_neg +=[0]*len(fc7)
                        else:
                            X_trainval_select_neg = [fc7.astype(np.float32)] 
                            y_trainval_select_neg =[0]*len(fc7)
                    if j==0:
                        if not(len(X_trainval_all)==0):
                            X_trainval_all += [fc7.astype(np.float32)]
                        else:
                            X_trainval_all += [fc7.astype(np.float32)]
                X_trainval_select_neg = np.concatenate(X_trainval_select_neg,axis=0).astype(np.float32)
                y_trainval_select_neg = np.array(y_trainval_select_neg,dtype=np.float32)
                y_trainval_select = np.hstack((y_trainval_select_neg,y_trainval_select_pos))
                if j==0: X_trainval_all = np.concatenate(X_trainval_all,axis=0).astype(np.float32)
            elif baseline_kind == 'SISVM':
                number_pos_ex = int(np.sum(y_trainval[:,j]))
                number_neg_ex = len(y_trainval) - number_pos_ex
                number_ex = number_pos_ex + number_neg*number_neg_ex
#                y_trainval_select_neg = np.zeros((300*number_neg_ex,),dtype=np.float32)
                y_trainval_select = []
                
                if j==0: X_trainval_select = []
                for i,name_img in  enumerate(name_trainval):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                    rois,roi_scores,fc7 = features_resnet_dict[name_img]
                    if y_trainval[i,j] == 1: # Positive exemple
                        if not(len(y_trainval_select)==0):
                            if j==0: X_trainval_select += [fc7.astype(np.float32)] 
                            y_trainval_select +=[1]*len(fc7)
                        else:
                            if j==0: X_trainval_select = [fc7.astype(np.float32)] 
                            y_trainval_select =[1]*len(fc7)
                    else:
                        if not(len(y_trainval_select)==0):
                            if j==0: X_trainval_select += [fc7.astype(np.float32)] 
                            y_trainval_select +=[0]*len(fc7)
                        else:
                            if j==0: X_trainval_select = [fc7.astype(np.float32)] 
                            y_trainval_select =[0]*len(fc7)
                if j==0: 
                    X_trainval_select = np.concatenate(X_trainval_select,axis=0).astype(np.float32)
                    X_trainval_all = X_trainval_select
                y_trainval_select = np.array(y_trainval_select,dtype=np.float32)
               
            elif baseline_kind=='MAXA_lowMem':
                # Version low memory of MAXA but it doesn't work
                if j==0: X_trainval_all  = []
                y_trainval_select = []
                X_trainval_select = []
                index_nav = 0
                with open(name_pkl, 'rb') as pkl:
                    for i,name_img in enumerate(name_trainval):
                        if i%1000==0:
                            if verbose: print(i,name_img)
                            features_resnet_dict = pickle.load(pkl)
                        InTestSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
                        if not(InTestSet):
                            rois,roi_scores,fc7 = features_resnet_dict[name_img]
                            if y_trainval[i,j] == 1: # Positive exemple
                                if not(len(X_trainval_select)==0):
                                    y_trainval_select+= [1]
                                    X_trainval_select += [np.expand_dims(fc7[0,:],axis=0).astype(np.float32)] # The roi_scores vector is sorted
                                else:
                                    y_trainval_select = [1]
                                    X_trainval_select = [np.expand_dims(fc7[0,:],axis=0).astype(np.float32)] # The roi_scores vector is sorted
                            else:
                                if not(len(X_trainval_select)==0):
                                    X_trainval_select += [fc7.astype(np.float32)]
                                    y_trainval_select += [0]*len(fc7)
                                else:
                                    X_trainval_select = [fc7.astype(np.float32)]
                                    y_trainval_select = [0]*len(fc7)
                            if j==0:
                                if not(len(X_trainval_all)==0):
                                    X_trainval_all += [fc7.astype(np.float32)]
                                else:
                                    X_trainval_all += [fc7.astype(np.float32)]
                    del features_resnet_dict
                if j==0: X_trainval_all = np.concatenate(X_trainval_all,axis=0).astype(np.float32)
                X_trainval_select = np.concatenate(X_trainval_select,axis=0).astype(np.float32)
                X_trainval_all = X_trainval_select
                y_trainval_select = np.array(y_trainval_select,dtype=np.float32)
            else:
                print(baseline_kind,' unknown method')
                raise(NotImplementedError)
            if verbose: 
                try:
                    print("Shape X and y",X_trainval_select.shape,y_trainval_select.shape)
                except UnboundLocalError:
                    if not( baseline_kind in ['MISVM','miSVM']):
                        print('UnboundLocalError')
                        raise(UnboundLocalError)
                
            if (normalisation == True) or PCAuse:
                if (baseline_kind in ['SISVM','MISVM','miSVM','MAXA_lowMem','MAX1','MAXA'] and j==0):
                    if verbose: print('Normalisation of the data (X-mean)/std')
                    scaler = StandardScaler()
                    scaler.fit(X_trainval_all)
                    X_trainval_all = scaler.transform(X_trainval_all)
                    if not(baseline_kind in ['miSVM','MISVM']):
                        X_trainval_select = scaler.transform(X_trainval_select)
                elif  (baseline_kind in ['MAXA_lowMem','MAX1','MAXA']):
                    X_trainval_select = scaler.transform(X_trainval_select)
                    
                if baseline_kind in ['miSVM','MISVM']:
                    X_trainval_select_neg = scaler.transform(X_trainval_select_neg)
                    for k in range(len(X_trainval_select_pos)):
                        X_trainval_select_pos[k] = scaler.transform(X_trainval_select_pos[k])

                 
            # PCA computation
            if PCAuse :
                if (baseline_kind in ['SISVM','MISVM','miSVM','MAXA_lowMem','MAX1','MAXA'] and j==0):
                    if verbose: print("Use of a PCA for dimensionality reduction")
                    pca = PCA()
                    pca.fit(X_trainval_all)
                    del X_trainval_all
                    
                    cumsum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                    number_composant = 1+np.where(cumsum_explained_variance_ratio>variance_thres)[0][0]
                    print('We will reduce the number of features to : ',number_composant,' for variance_thres',variance_thres)
                    if not(baseline_kind in ['miSVM','MISVM']):
                        X_trainval_select = pca.transform(X_trainval_select)
                        X_trainval_select = X_trainval_select[:,0:number_composant]
                    else:
                        X_trainval_select_neg = pca.transform(X_trainval_select_neg)
                        X_trainval_select_neg = np.ascontiguousarray(X_trainval_select_neg[:,0:number_composant])
                        for k in range(len(X_trainval_select_pos)):
                            X_trainval_select_pos[k] = np.ascontiguousarray(pca.transform(X_trainval_select_pos[k])[:,0:number_composant])
                elif (baseline_kind in ['MISVM','miSVM'] and not(j==0)):
                    X_trainval_select_neg = pca.transform(X_trainval_select_neg)
                    X_trainval_select_neg = np.ascontiguousarray(X_trainval_select_neg[:,0:number_composant])
                    for k in range(len(X_trainval_select_pos)):
                        X_trainval_select_pos[k] = np.ascontiguousarray(pca.transform(X_trainval_select_pos[k])[:,0:number_composant])
                elif (baseline_kind in ['MAXA_lowMem','MAX1','MAXA']):
                    X_trainval_select = pca.transform(X_trainval_select)
                    X_trainval_select = X_trainval_select[:,0:number_composant]
            else:
                number_composant = num_features


            det_name_file = 'Detect_Baseline_'+database+'_'+baseline_kind
            if testMode:
                det_name_file += '_TestMode'
            if normalisation:
                det_name_file += '_Normed'
            if baseline_kind in ['MISVM','miSVM']:
                det_name_file += '_r'+str(restarts) +'_m'+str(max_iter)
            if PCAuse:
                det_name_file += '_PCA'+str(number_composant)
            if gridSearch:
                det_name_file += '_GS'
            det_name_fileAP = os.path.join(path_data,det_name_file + '_AP.pkl')
            if os.path.isfile(det_name_fileAP) and not(reDo):
                print('The results file already exists')
                return(0)

            # Training time
            if verbose: print("Start learning for class",j)
            if not(baseline_kind in ['miSVM','MISVM']):
                X_trainval_select = np.ascontiguousarray(X_trainval_select)
                classifier_trained = TrainClassif(X_trainval_select,y_trainval_select,
                    clf=clf,class_weight='balanced',gridSearch=gridSearch,
                    n_jobs=n_jobs,C_finalSVM=1,cskind='small',testMode=testMode)  # TODO need to put it in parameters 
                dict_clf[j] = classifier_trained
            elif baseline_kind=='miSVM': ## miSVM
                ## Implementation of the miSVM of Andrews 2006
                #Initialisation  
                hinge_loss_value_best = np.inf
                for rr in range(restarts+1):
                    X_pos = np.empty((number_pos_ex*number_neg_ex,number_composant),dtype=np.float32)
                    if rr==0:
                        if verbose: print('Non-random start...')
                        # The initialization is that all the instances of positive bags are considered as positive
                        X_pos= np.concatenate(X_trainval_select_pos,axis=0).astype(np.float32)
                        y_trainval_select_pos = np.ones(shape=(len(X_pos),))
                    else:
                        if verbose: print('Random restart %d of %d...' % (rr, restarts))
                        # In the other cases, we do draw random label for the elements
                        X_pos= np.concatenate(X_trainval_select_pos,axis=0).astype(np.float32) 
                        # The label are 0 or 1 !
                        y_trainval_select_pos = np.random.randint(low=0,high=2,size=(len(X_pos),))

                    X_trainval_select = np.ascontiguousarray(np.vstack((X_trainval_select_neg,X_pos)))

                    y_trainval_select_pos_old = y_trainval_select_pos
                    iteration = 0
                    SelectirVar_haveChanged = True
                    while((iteration < max_iter) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
                        y_trainval_select = np.hstack((y_trainval_select_neg,y_trainval_select_pos))
                        svm_trained = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1)
                        
                        labels_k_tab = []
                        for k in range(len(X_trainval_select_pos)):
                            labels_k = svm_trained.predict(X_trainval_select_pos[k])
                            if len(np.nonzero(labels_k)[0])==0:
                                decision_fct = svm_trained.decision_function(X_trainval_select_pos[k])
                                argmax_k = np.argmax(decision_fct)
                                labels_k[argmax_k] = 1 # We assign the highest case to the value 1 in each of the positive bag
                            labels_k_tab +=[labels_k]
                        y_trainval_select_pos_old = y_trainval_select_pos
                        y_trainval_select_pos = np.array(np.hstack(labels_k_tab))
#                        assert(len(y_trainval_select_pos)==len(y_trainval_select_pos_old))
#                        print(y_trainval_select_pos==y_trainval_select_pos_old)
                        yy_equal = y_trainval_select_pos==y_trainval_select_pos_old
                        if all(yy_equal):
                            SelectirVar_haveChanged=False
                        
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                        
                    # Need to evaluate the objective value 
                    pred_decision = svm_trained.decision_function(X_trainval_select)
                    hinge_loss_value = hinge_loss(y_trainval_select, pred_decision)
                    
                    if hinge_loss_value <= hinge_loss_value_best:
                        hinge_loss_value_best =hinge_loss_value
                        best_svm = svm_trained
                        if verbose: print('New best SVM; ,new value of the loss :',hinge_loss_value_best)
                    

                    if verbose: print("End after ",iteration,"iterations on",max_iter)
                # Sur Watercolor avec LinearSVC et sans GridSearch on a 7 iterations max
                # Training ended
                dict_clf[j] = best_svm   
                del X_trainval_select
                
            elif baseline_kind=='MISVM': ## MISVM 
                ## Implementation of the MISVM of Andrews 2006
                #Initialisation
                number_of_pos_bag = len(X_trainval_select_pos)
                hinge_loss_value_best = np.inf
                for rr in range(restarts+1):
                    X_pos = np.empty((number_pos_ex,number_composant),dtype=np.float32)
                    if rr==0:
                        if verbose: print('Non-random start...')
                        for k in range(number_of_pos_bag):
                            # The initialization is the mean of all the element of the bag in the positive case
                            X_pos[k,:] = np.mean(X_trainval_select_pos[k],axis=0).astype(np.float32)
                    else:
                        if verbose: print('Random restart %d of %d...' % (rr, restarts))
                        # In the other cases, we do a random weighted sum of the element of the bags
                        for k in range(number_of_pos_bag):
                            weighted_random = rand_convex(len(X_trainval_select_pos[k]))
                            X_pos[k,:] = np.sum(weighted_random*X_trainval_select_pos[k],axis=0).astype(np.float32)
                    
                    S_I = [-1]*number_of_pos_bag
                    iteration = 0
                    SelectirVar_haveChanged = True
                    while((iteration < max_iter) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
                        X_trainval_select = np.ascontiguousarray(np.vstack((X_trainval_select_neg,X_pos)))
                        svm_trained = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1)
                        
                        S_I_old = S_I
                        S_I = []
                        for k in range(number_of_pos_bag):
                            argmax_k = np.argmax(svm_trained.decision_function(X_trainval_select_pos[k]))
                            S_I += [argmax_k]
                            X_S_I = X_trainval_select_pos[k][argmax_k,:]
                            X_pos[k,:] = X_S_I
                        if S_I==S_I_old:
                            SelectirVar_haveChanged=False
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                    # Need to evaluate the objective value 
                    pred_decision = svm_trained.decision_function(X_trainval_select)
                    hinge_loss_value = hinge_loss(y_trainval_select, pred_decision)
                    
                    if hinge_loss_value <= hinge_loss_value_best:
                        hinge_loss_value_best =hinge_loss_value
                        best_svm = svm_trained
                        if verbose: print('New best SVM; ,new value of the loss :',hinge_loss_value_best)
                    
                   
                    if verbose: print("End after ",iteration,"iterations on",max_iter)
                # Sur Watercolor avec LinearSVC et sans GridSearch on a 7 iterations max
                # Training ended
                dict_clf[j] = best_svm   
                del X_trainval_select
                
            # End of the training 
            if verbose: print("End learning for class",j)
            
        gc.collect()
        
        
        # In the case of the test mode
        if testMode:
            for j,classe in enumerate(classes):
                if not(j==jtest): 
                    dict_clf[j] = dict_clf[jtest]
        
        #Load test set 
        if baseline_kind in ['MAXA_lowMem']:
            #del features_resnet_dict
            with open(name_pkl, 'rb') as pkl:
                for i,name_img in  enumerate(df_label[item_name]):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                        features_resnet_dict_tmp = pickle.load(pkl)
                        if i==1000:
                            features_resnet_dict = features_resnet_dict_tmp
                        else:
                            features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
                features_resnet_dict_tmp = pickle.load(pkl)
                features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
        
        roi_test = {}
        name_test = {}
        key_test = 0
        for i,name_img in  enumerate(df_label[item_name]):
            if i%1000==0 and not(i==0):
                if verbose: print(i,name_img)
            if database in ['VOC2007','VOC12','Paintings','watercolor','IconArt_v1','PeopleArt']:          
                InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
#            elif database=='Wikidata_Paintings_miniset_verif':
#                InSet = (i in index_test)
            if InSet:
                rois,roi_scores,fc7 = features_resnet_dict[name_img]
                #print(rois.shape,roi_scores.shape)
                if Test_on_k_bag:
                    rois_reduce,roi_scores,fc7_reduce =  reduce_to_k_regions(k_per_bag,rois, \
                                                               roi_scores, fc7,new_nms_thresh, \
                                                               score_threshold,minimal_surface)
                    if(len(fc7_reduce) >= k_per_bag):
                        bag = np.expand_dims(fc7_reduce[0:k_per_bag,:],axis=0)
                    else:
                        number_repeat = k_per_bag // len(fc7_reduce)  +1
                        f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
                        bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0) 
                    fc7 = np.array(bag)
                 
                if PCAuse:
                    fc7 = scaler.transform(fc7)
                    fc7 = pca.transform(fc7)
                    fc7 = fc7[:,0:number_composant]
                if normalisation:
                    fc7 = scaler.transform(fc7)
                    
                f_test[key_test] = fc7
                roi_test[key_test] = rois
                name_test[key_test] = name_img
                key_test += 1
            del features_resnet_dict[name_img]
        del features_resnet_dict
        if verbose: print("End load test image")
        
        for j,classe in enumerate(classes):
            classifier_trained = dict_clf[j]
            y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j],dtype=np.float32)
            labels_test_predited = np.zeros_like(y_test[:,j],dtype=np.float32)
            
            # Test Time
            for k in range(len(f_test)): 
                if Test_on_k_bag: 
                    raise(NotImplementedError)
#                    decision_function_output = classifier_trained.decision_function(X_test[k,:,:])
                else:
                    elt_k = f_test[k]
                    decision_function_output = classifier_trained.decision_function(elt_k)
                
                y_predict_confidence_score_classifier[k]  = np.max(decision_function_output)
#                roi_with_object_of_the_class = np.argmax(decision_function_output)
                
                # For detection 
                if database in ['VOC2007','watercolor','IconArt_v1']:
                    thresh = 0.05 # Threshold score or distance MILSVM
                    TEST_NMS = 0.3 # Recouvrement entre les classes
                    complet_name = path_to_img + str(name_test[k]) + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    inds = np.where(decision_function_output > thresh)[0]
                    cls_scores = decision_function_output[inds]
                    roi = roi_test[k]
                    roi_boxes =  roi[:,1:5] / im_scales[0] 
                    cls_boxes = roi_boxes[inds,:]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(cls_dets, TEST_NMS)
                    cls_dets = cls_dets[keep, :]
                    all_boxes[j][k] = cls_dets
                   
                if np.max(decision_function_output) > 0:
                    labels_test_predited[k] = 1 
                else: 
                    labels_test_predited[k] =  0 # Label of the class 0 or 1
            AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                raise(NotImplementedError)
#                print("Baseline SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("Baseline ",baseline_kind," version Average Precision classification for",classes[j]," = ",AP)
            test_precision = precision_score(y_test[:,j],labels_test_predited)
            test_recall = recall_score(y_test[:,j],labels_test_predited)
            F1 = f1_score(y_test[:,j],labels_test_predited)
            print("Test on all the data precision = {0:.2f}, recall = {1:.2f},F1 = {2:.2f}".format(test_precision,test_recall,F1))
            precision_at_k = ranking_precision_score(np.array(y_test), y_predict_confidence_score_classifier,20)
            P20_per_class += [precision_at_k]
            AP_per_class += [AP]
            R_per_class += [test_recall]
            P_per_class += [test_precision]
            
        print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
        print("mean Precision for all the data = {0:.3f}".format(np.mean(P_per_class)))  
        print("mean Recall for all the data = {0:.3f}".format(np.mean(R_per_class)))  
        print("mean Precision @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    
        print(AP_per_class)
        print(arrayToLatex(AP_per_class))
    
        if database in ['VOC2007','watercolor','IconArt_v1','PeopleArt']:
            if testMode:
                for j in range(0, imdb.num_classes-1):
                    if not(j==jtest):
                        #print(all_boxes[jtest])
                        all_boxes[j] = all_boxes[jtest]
#            det_file = os.path.join(path_data, 'detections_aux.pkl')
#            with open(det_file, 'wb') as f:
#                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            max_per_image = 100
            num_images_detect = len(imdb.image_index)
            all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
            for i in range(num_images_detect):
                name_img = imdb.image_path_at(i)
                if database=='PeopleArt':
                    name_img_wt_ext = name_img.split('/')[-2] +'/' +name_img.split('/')[-1]
                    #name_img_wt_ext_tab =name_img_wt_ext.split('.')
                    #name_img_wt_ext = '.'.join(name_img_wt_ext_tab[0:-1])
                else:
                    name_img_wt_ext = name_img.split('/')[-1]
                    name_img_wt_ext =name_img_wt_ext.split('.')[0]
                #print(name_img_wt_ext)
                name_img_ind = np.where(np.array(name_all_test)==name_img_wt_ext)[0]
                #print(name_img_ind)
                if len(name_img_ind)==0:
                    print('len(name_img_ind)==0, that s strange')
                    print('name_all_test',name_all_test)
                    print('name_img_wt_ext',name_img_wt_ext)
                    print('name_img_ind',name_img_ind)
                    raise(Exception)
                #print(name_img_ind[0])
                for j in range(1, imdb.num_classes):
                    j_minus_1 = j-1
                    all_boxes_order[j][i]  = all_boxes[j_minus_1][name_img_ind[0]]
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes_order[j][i][:, -1]
                                for j in range(1, imdb.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, imdb.num_classes):
                            keep = np.where(all_boxes_order[j][i][:, -1] >= image_thresh)[0]
                            all_boxes_order[j][i] = all_boxes_order[j][i][keep, :]
            results_pkl = {}
            
#            det_name_file = 'Detect_Baseline_'+database+'_'+baseline_kind
#            if normalisation:
#                det_name_file += '_Normed'
#            if baseline_kind in ['MISVM','miSVM']:
#                det_name_file += '_r'+str(restarts) +'_m'+str(max_iter)
#            if PCAuse:
#                det_name_file += '_PCAc'+str(number_composant)
#            det_name_fileAP = os.path.join(path_data,det_name_file + '_AP.pkl')
            det_name_filef = det_name_file + '.pkl'
            det_file = os.path.join(path_data, det_name_filef)
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
            output_dir =  os.path.join(path_data,'tmp',database) # path_data +'tmp/' + database + '/'

            aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
            results_pkl['AP_IOU05'] = aps
            print("Detection scores at 0.5 for Baseline :",baseline_kind,'gridSearch',gridSearch,'on ',database)
            if PCAuse: print("With PCA and ",number_composant," componants")
            if baseline_kind in ['MISVM','miSVM']:
                print('restarts',restarts,'max_iter',max_iter)
            print(arrayToLatex(aps,per=True))
            ovthresh_tab = [0.1]
            for ovthresh in ovthresh_tab:
                aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
                if ovthresh == 0.1:
                    apsAt01 = aps
                print("Detection score with thres at ",ovthresh)
                print(arrayToLatex(aps,per=True))
            results_pkl['AP_IOU01'] = apsAt01
            
            pkl = open(det_name_fileAP, 'wb')
            pickle.dump(results_pkl,pkl)

    except KeyboardInterrupt:
        gc.collect()  
  
def BaselineRunAll():
    """ Run severals baseline model on two datasets
    """
    
    datasets = ['IconArt_v1','watercolor']
    list_methods =['MISVM','miSVM','SISVM','MAX1','MAXA']
    normalisation = False
    restarts = 0
    max_iter = 50
    variance_thres = 0.9
    for database in datasets:
        for method in list_methods: 
            if method in ['MAXA','MAX1','SISVM']:
                GS_tab = [False,True]
                PCA_tab = [True,False]
            else:
                GS_tab = [False]
                PCA_tab = [True]
            for GS in GS_tab:
                for PCAuse in PCA_tab:
                    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                            normalisation= normalisation,baseline_kind=method,verbose=False,
                            gridSearch=GS,k_per_bag=300,n_jobs=4,PCAuse=PCAuse,variance_thres= variance_thres,
                            restarts=restarts,max_iter=max_iter,reDo=False)
    restarts = 10
    for database in datasets:
        for method in ['MISVM','miSVM']:
            Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                    normalisation= normalisation,baseline_kind=method,verbose=False,
                    gridSearch=False,k_per_bag=300,n_jobs=4,PCAuse=PCAuse,variance_thres= variance_thres,
                    restarts=restarts,max_iter=max_iter,reDo=False)
            
def RunTest():
    """ Run severals baseline model on two datasets
    """
    
    datasets = ['IconArt_v1','watercolor']
    list_methods =['SISVM','MISVM','miSVM','MAX1','MAXA','MAXA_lowMem']
    for database in datasets:
        for method in list_methods: 
            if method in ['MAXA','MAX1','SISVM']:
                GS_tab = [False]
                PCA_tab = [True,False]
            else:
                GS_tab = [False]
                PCA_tab = [True]
            for GS in GS_tab:
                for PCAuse in PCA_tab:
                    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                            normalisation= False,baseline_kind=method,verbose=False,
                            gridSearch=GS,k_per_bag=300,n_jobs=3,PCAuse=PCAuse,variance_thres= 0.9,
                            restarts=0,max_iter=2,testMode=True)

  
if __name__ == '__main__':
    #BaselineRunAll()
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=1)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=True,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=100)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=100)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MAX1',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MAXA_lowMem',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=0,max_iter=50,testMode=True)
    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'PeopleArt',Test_on_k_bag=False,
                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
                        gridSearch=False,k_per_bag=300,n_jobs=4,PCAuse=False,variance_thres= 0.9,
                        restarts=0,max_iter=50,testMode=False)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'SISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=0,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MAX1',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=0,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MAXA_lowMem',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=0,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'MISVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=50)
#
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=100)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'IconArt_v1',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.9,
#                        restarts=10,max_iter=100)
#   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                        normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                        gridSearch=False,k_per_bag=300,n_jobs=3,PCAuse=True,variance_thres= 0.75,
#                        restarts=10,max_iter=100)
