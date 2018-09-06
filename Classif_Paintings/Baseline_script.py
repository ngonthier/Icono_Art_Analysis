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
from sklearn.preprocessing import StandardScaler
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

def rand_convex(n):
    rand = np.matrix([uniform(0.0, 1.0) for i in range(n)])
    return(rand / np.sum(rand))

def TrainClassif(X,y,clf='LinearSVC',class_weight=None,gridSearch=True,n_jobs=-1,C_finalSVM=1,cskind=None):
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
    if gridSearch:
        if clf == 'LinearSVC':
            clf = LinearSVC(penalty='l2',class_weight=class_weight, 
                            loss='squared_hinge',max_iter=1000,dual=True)
            param_grid = dict(C=cs)
        elif clf == 'defaultSGD':
            clf = SGDClassifier(max_iter=1000, tol=0.0001)
            param_grid = dict(alpha=cs)
        elif clf == 'SGDsquared_hinge':
            clf = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
            param_grid = dict(alpha=cs)
    
        classifier = GridSearchCV(clf, refit=True,
                                  scoring =make_scorer(average_precision_score,
                                                       needs_threshold=True),
                                  param_grid=param_grid,n_jobs=n_jobs)
    else:
        # ,class_weight='balanced'
        if clf == 'LinearSVC':
            classifier = LinearSVC(penalty='l2',class_weight=class_weight,
                                   loss='squared_hinge',max_iter=1000,dual=True,C=C_finalSVM)
        elif clf == 'defaultSGD':
            classifier = SGDClassifier(max_iter=1000)
        elif clf == 'SGDsquared_hinge':
            classifier = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
    
    classifier.fit(X,y)
    
    return(classifier)

def Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'Paintings',Test_on_k_bag = False,
                             normalisation= False,baseline_kind = 'MAX1',
                             verbose = True,gridSearch=False,k_per_bag=300,jtest=0,testMode=False,
                             n_jobs=-1,clf='LinearSVC'):
    """ 
    27 juin 2018 ==> Il faut que les dossiers soit calculés avant sur mon ordi 
    puis passer sur le cluster
    ---------------------------------------------------------------------------
    
    Detection based on CNN features with Transfer Learning on Faster RCNN output
    This is used to compute the baseline MAX1 (only the best score object as training exemple)
    and MAXA (all the regions for the negatives)
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
    print('==========')
    print('Baseline for ',database,demonet,baseline_kind,'gridSearch',gridSearch,'clf',clf)
    try:
        if demonet == 'vgg16_COCO':
            num_features = 4096
        elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
            num_features = 2048
        ext = '.txt'
        dtypes = str
        if database=='Paintings':
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/Painting_Dataset/'
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        elif database=='VOC12':
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        elif database=='VOC2007':
            ext = '.csv'
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
            classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
        elif(database=='WikiTenLabels'):
            ext='.csv'
            item_name='item'
            classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien','turban']
        elif database=='watercolor':
            ext = '.csv'
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
            classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
        elif(database=='Wikidata_Paintings'):
            item_name = 'image'
            path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
            raise NotImplemented # TODO implementer cela !!! 
        elif(database=='Wikidata_Paintings_miniset_verif'):
            item_name = 'image'
            path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
            classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
        
        if(jtest>len(classes)) and testMode:
           print("We are in test mode but jtest>len(classes), we will use jtest =0" )
           jtest =0
        
        path_data = '/media/HDD/output_exp/ClassifPaintings/'
        Not_on_NicolasPC = False
        if not(os.path.exists(path_data)):
            Not_on_NicolasPC = True
            path_data_tab = path_data.split('/')
            path_to_img_tab = path_to_img.split('/')
            path_tmp = 'data' 
            path_to_img = path_tmp + '/'+path_to_img_tab[-2] +'/'
            path_data = path_tmp + '/'+path_data_tab[-2] +'/'
        
        databasetxt =path_data + database + ext    
        if database=='VOC2007' or database=='watercolor':
            dtypes = {0:str,'name_img':str,'aeroplane':int,'bicycle':int,'bird':int, \
                      'boat':int,'bottle':int,'bus':int,'car':int,'cat':int,'cow':int,\
                      'dinningtable':int,'dog':int,'horse':int,'motorbike':int,'person':int,\
                      'pottedplant':int,'sheep':int,'sofa':int,'train':int,'tvmonitor':int,'set':str}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
            df_label[classes] = df_label[classes].apply(lambda x: np.floor((x + 1.0) /2.0))
        else:
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
            if database=='Wikidata_Paintings_miniset_verif':
                df_label = df_label[df_label['BadPhoto'] <= 0.0]
    
        num_classes = len(classes)
        N = 1
        extL2 = ''
        nms_thresh = 0.7
        savedstr = '_all'
        # TODO improve that 
        name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+ \
            '_TLforMIL_nms_'+str(nms_thresh)+savedstr+'.pkl'
           
        features_resnet_dict = {}
        sLength_all = len(df_label[item_name])
        if demonet == 'vgg16_COCO':
            size_output = 4096
        elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
            size_output = 2048
#        filesave = 'pkl'
        if not(os.path.isfile(name_pkl)):
            # Compute the features
            if Not_on_NicolasPC: print("You need to compute the CNN features before")
            return(0)
#            if verbose: print("We will computer the CNN features")
#            Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
#                                         database=database,augmentation=False,L2 =False,
#                                         saved='all',verbose=verbose,filesave=filesave)
        
        if baseline_kind == 'MAX1' or baseline_kind == 'MEAN' or baseline_kind =='MISVM'or baseline_kind =='miSVM':
            if verbose: print("Start loading data",name_pkl)
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
        
        
#        features_resnet = np.empty((sLength_all,k_per_bag,size_output),dtype=np.float32)  
        classes_vectors = np.zeros((sLength_all,num_classes),dtype=np.float32)
        if database=='Wikidata_Paintings_miniset_verif' or database=='VOC2007' or database=='watercolor':
            classes_vectors = df_label.as_matrix(columns=classes)
        f_test = {}

        # Parameters important
        new_nms_thresh = 0.0
        score_threshold = 0.1
        minimal_surface = 36*36
        # In the case of Wikidata
        if database=='Wikidata_Paintings_miniset_verif':
            raise(NotImplemented)
#            random_state = 0
#            index = np.arange(0,len(features_resnet_dict))
#            index_trainval, index_test = train_test_split(index, test_size=0.6, random_state=random_state)
#            index_trainval = np.sort(index_trainval)
#            index_test = np.sort(index_test)
    
        if database=='VOC2007'  or database=='watercolor' or database=='clipart':
            if database=='VOC2007' : imdb = get_imdb('voc_2007_test')
            if database=='watercolor' : imdb = get_imdb('watercolor_test')
            if database=='clipart' : imdb = get_imdb('clipart_test')
            imdb.set_force_dont_use_07_metric(True)
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
        if database=='VOC12' or database=='Paintings' or database=='VOC2007'  or database=='watercolor':
            if database=='VOC2007'  or database=='watercolor' or database=='WikiTenLabels':
                str_val ='val' 
            else: 
                str_val='validation'
#            X_train = features_resnet[df_label['set']=='train',:,:]
            y_train = classes_vectors[df_label['set']=='train',:]
#            X_test= features_resnet[df_label['set']=='test',:,:]
            y_test = classes_vectors[df_label['set']=='test',:]
#            X_val = features_resnet[df_label['set']==str_val,:,:]
            y_val = classes_vectors[df_label['set']==str_val,:]
#            X_trainval = np.append(X_train,X_val,axis=0)
            y_trainval =np.append(y_train,y_val,axis=0).astype(np.float32)
            names = df_label.as_matrix(columns=['name_img'])
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
        
        # Then we run on the different classes
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        if baseline_kind == 'MAX1':
            number_neg = 1
        elif baseline_kind == 'MAXA' or baseline_kind =='MISVM' or baseline_kind == 'miSVM':
            number_neg = 300
            
        # Training time
        dict_clf = {}
        for j,classe in enumerate(classes):
            gc.collect()
            if testMode and not(j==jtest):
                continue
            
            if baseline_kind == 'MAX1':
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
            elif baseline_kind == 'MEAN':
                number_ex = len(y_trainval)
                y_trainval_select = y_trainval[:,j]
                X_trainval_select = np.empty((number_ex,num_features),dtype=np.float32)
                index_nav = 0
                for i,name_img in  enumerate(name_trainval):
                    if i%1000==0 and not(i==0):
                        if verbose: print(i,name_img)
                    rois,roi_scores,fc7 = features_resnet_dict[name_img]
                    X_trainval_select[index_nav,:] = np.mean(fc7[0,:]) # The roi_scores vector is sorted
                    index_nav += 1 
            elif baseline_kind == 'MISVM' or baseline_kind == 'miSVM':
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
                X_trainval_select_neg = np.concatenate(X_trainval_select_neg,axis=0).astype(np.float32)
                y_trainval_select_neg = np.array(y_trainval_select_neg,dtype=np.float32)
                y_trainval_select = np.hstack((y_trainval_select_neg,y_trainval_select_pos))
            elif baseline_kind=='MAXA':
                y_trainval_select = []
                X_trainval_select = []
                index_nav = 0
                with open(name_pkl, 'rb') as pkl:
                    for i,name_img in  enumerate(df_label[item_name]):
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
                    del features_resnet_dict
                X_trainval_select = np.concatenate(X_trainval_select,axis=0).astype(np.float32)
                y_trainval_select = np.array(y_trainval_select,dtype=np.float32)
            if verbose: 
                try:
                    print("Shape X and y",X_trainval_select.shape,y_trainval_select.shape)
                except UnboundLocalError:
                    if not( baseline_kind == 'MISVM'):
                        print('UnboundLocalError')
                        raise(UnboundLocalError)
                
            if normalisation == True:
                if verbose: print('Normalisation, never tested')
                scaler = StandardScaler()
                scaler.fit(X_trainval_select.reshape(-1,size_output))
                X_trainval_select = scaler.transform(X_trainval_select.reshape(-1,size_output))
                X_trainval_select = X_trainval_select.reshape(-1,k_per_bag,size_output)
#                X_test_norm = scaler.transform(X_test.reshape(-1,size_output))
#                X_test_norm = X_test_norm.reshape(-1,k_per_bag,size_output)
                        
            # Training time
            if verbose: print("Start learning for class",j)
            if not(baseline_kind in ['miSVM','MISVM']):
                classifier_trained = TrainClassif(X_trainval_select,y_trainval_select,
                    clf=clf,class_weight='balanced',gridSearch=gridSearch,
                    n_jobs=n_jobs,C_finalSVM=1,cskind='small')  # TODO need to put it in parameters 
                dict_clf[j] = classifier_trained
            elif baseline_kind=='MISVM':
                ## Implementation of the MISVM of Andrews 2006
                #Initialisation  
                restarts = 1
                for rr in range(restarts+1):
                    X_pos = np.empty((number_pos_ex,num_features),dtype=np.float32)
                    if rr==0:
                        for k in range(len(X_trainval_select_pos)):
                            X_pos[k,:] = np.mean(X_trainval_select_pos[k],axis=0).astype(np.float32)
                    else:
                        weighted_random = rand_convex(len(X_trainval_select_pos[k]))
                        for k in range(len(X_trainval_select_pos)):
                            X_pos[k,:] = np.sum(weighted_random*X_trainval_select_pos[k],axis=0).astype(np.float32)
                    S_I = [-1]*len(X_trainval_select_pos)
                    max_iter = 10
                    iteration = 0
                    SelectirVar_haveChanged = True
                    while((iteration < max_iter) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
                        X_trainval_select = np.vstack((X_trainval_select_neg,X_pos))
                        clf = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1)
                        
                        S_I_old = S_I
                        S_I = []
                        for k in range(len(X_trainval_select_pos)):
                            argmax_k = np.argmax(clf.decision_function(X_trainval_select_pos[k]))
                            S_I += [argmax_k]
                            X_S_I = X_trainval_select_pos[k][argmax_k,:]
                            X_pos[k,:] = X_S_I
                        if S_I==S_I_old:
                            SelectirVar_haveChanged=False
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                    if verbose: print("End after ",iteration,"iterations on",max_iter)
                # Sur Watercolor avec LinearSVC et sans GridSearch on a 7 iterations max
                # Training ended
                dict_clf[j] = clf   
                del X_trainval_select
                
            elif baseline_kind=='miSVM':
                ## Implementation of the MISVM of Andrews 2006
                #Initialisation
                raise(NotImplemented) # TODO a faire
                restarts = 0
                for rr in range(restarts+1):
                    X_pos = np.empty((number_pos_ex,num_features),dtype=np.float32)
                    if rr==0:
                        for k in range(len(X_trainval_select_pos)):
                            X_pos[k,:] = np.mean(X_trainval_select_pos[k],axis=0).astype(np.float32)
                    else:
                        weighted_random = rand_convex(len(X_trainval_select_pos[k]))
                        for k in range(len(X_trainval_select_pos)):
                            X_pos[k,:] = np.sum(weighted_random*X_trainval_select_pos[k],axis=0).astype(np.float32)
#                        pos_bag_avgs = np.vstack([ * bag for bag in bs.pos_bags])
            
    #                X_pos = np.mean(np.vstack(X_trainval_select_pos,axis=0),axis=1)
                    S_I = [-1]*len(X_trainval_select_pos)
                    max_iter = 10
                    iteration = 0
                    SelectirVar_haveChanged = True
    #                clf = LinearSVC(penalty='l2',class_weight='balanced', 
    #                            loss='squared_hinge',max_iter=1000,dual=True)
                    
                    while((iteration < max_iter) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
                        X_trainval_select = np.vstack((X_trainval_select_neg,X_pos))
    #                    clf.fit(X_trainval_select,y_trainval_select)
                        
                        clf = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1)
                        
                        S_I_old = S_I
                        S_I = []
                        for k in range(len(X_trainval_select_pos)):
                            argmax_k = np.argmax(clf.decision_function(X_trainval_select_pos[k]))
                            S_I += [argmax_k]
                            X_S_I = X_trainval_select_pos[k][argmax_k,:]
                            X_pos[k,:] = X_S_I
                        if S_I==S_I_old:
                            SelectirVar_haveChanged=False
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                    if verbose: print("End after ",iteration,"iterations on",max_iter)
                # Sur Watercolor avec LinearSVC et sans GridSearch on a 7 iterations max
                
                # Training ended
                dict_clf[j] = clf   
                del X_trainval_select
            if verbose: print("End learning for class",j)
            
        gc.collect()
        
        #Load test set 
        if baseline_kind == 'MAXA':
            del features_resnet_dict
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
            if database=='VOC2007' or database=='VOC12' or database=='Paintings'  or database=='watercolor':          
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
                    raise(NotImplemented)
#                    decision_function_output = classifier_trained.decision_function(X_test[k,:,:])
                else:
                    if normalisation:
                        elt_k =  scaler.transform(f_test[k])
                    else:
                        elt_k = f_test[k]
                    decision_function_output = classifier_trained.decision_function(elt_k)
                
                y_predict_confidence_score_classifier[k]  = np.max(decision_function_output)
#                roi_with_object_of_the_class = np.argmax(decision_function_output)
                
                # For detection 
                if database=='VOC2007'  or database=='watercolor':
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
                raise(NotImplemented)
#                print("Baseline SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("Baseline SVM version Average Precision for",classes[j]," = ",AP)
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
    
        if database=='VOC2007'  or database=='watercolor':
            if testMode:
                for j in range(0, imdb.num_classes-1):
                    if not(j==jtest):
                        #print(all_boxes[jtest])
                        all_boxes[j] = all_boxes[jtest]
            det_file = os.path.join(path_data, 'detections_aux.pkl')
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            max_per_image = 100
            all_boxes_order = [[[] for _ in range(num_images)] for _ in range(imdb.num_classes)]
            for i in range(num_images):
                name_img = imdb.image_path_at(i)
                name_img_wt_ext = name_img.split('/')[-1]
                name_img_wt_ext =name_img_wt_ext.split('.')[0]
                #print(name_img_wt_ext)
                name_img_ind = np.where(np.array(name_all_test)==name_img_wt_ext)[0]
                #print(name_img_ind)
                if len(name_img_ind)==0:
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
            det_file = os.path.join(path_data, 'detections.pkl')
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
            output_dir = path_data +'tmp/' + database + '/'
            aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
            print("Detection scores for Baseline algo")
            print(arrayToLatex(aps,per=True))

    except KeyboardInterrupt:
        gc.collect()  
            
if __name__ == '__main__':
   Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'VOC2007',Test_on_k_bag=False,
                        normalisation= False,baseline_kind = 'MAXA',verbose = True,
                        gridSearch=False,k_per_bag=300,n_jobs=1)