#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:12:26 2018

@author: gonthier
"""

import time
import pickle
import gc
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import SGDClassifier
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
from tf_faster_rcnn.lib.model.test import im_detect,TL_im_detect,TL_im_detect_end,get_blobs
from tf_faster_rcnn.lib.model.nms_wrapper import nms
from tf_faster_rcnn.lib.nms.py_cpu_nms import py_cpu_nms
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit,train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
#from tf_faster_rcnn.tools.demo import vis_detections
import numpy as np
import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score,recall_score,precision_score,make_scorer,f1_score
from Custom_Metrics import ranking_precision_score
from Classifier_Evaluation import Classification_evaluation
import os.path
import misvm # Library to do Multi Instance Learning with SVM
from sklearn.preprocessing import StandardScaler
from trouver_classes_parmi_K import MILSVM,TrainClassif,tf_MILSVM
from LatexOuput import arrayToLatex
from FasterRCNN import vis_detections_list,Compute_Faster_RCNN_features
import pathlib
from milsvm import mi_linearsvm # Version de nicolas avec LinearSVC et TODO SGD 
from sklearn.externals import joblib # To save the classifier
from tool_on_Regions import reduce_to_k_regions
from sklearn import linear_model
from tf_faster_rcnn.lib.datasets.factory import get_imdb
from Estimation_Param import kde_sklearn,findIntersection
from utils.save_param import create_param_id_file_and_dir,write_results,tabs_to_str
from Transform_Box import py_cpu_modif
from TL_MILSVM import parser_w_rois_all_class
#from hpsklearn import HyperoptEstimator,sgd
#from hyperopt import tpe
from random import uniform


def LearnOn30andTestOn300():
    """ This function is just used to do the test with a model learn on 30 regions"""
    demonet = 'res152_COCO'
    database = 'WikiTenLabels' 
    verbose = True
    testMode = False
    jtest = 'cow'

    PlotRegions = False
    saved_clf=False
    RPN=False
    CompBest=False
    Stocha=True
    k_per_bag=30
    parallel_op=True
    CV_Mode=''
    num_split=2
    WR=True
    init_by_mean =None
    seuil_estimation=''
    restarts=11
    max_iters_all_base=300
    LR=0.01
    with_tanh=True
    C=1.0
    Optimizer='GradientDescent'
    norm=''
      
    transform_output='tanh'
    with_rois_scores_atEnd=False
    optim_wt_Reg = False  
    with_scores=False
    epsilon=0.01
    restarts_paral='paral'
    Max_version=''
    w_exp=10.0
    seuillage_by_score=False
    seuil=0.9
    k_intopk=1
    C_Searching=False
    predict_with='MILSVM'
    gridSearch=False
    thres_FinalClassifier=0.5
    n_jobs=1
    thresh_evaluation=0.05
    TEST_NMS=0.3
    ext = '.txt'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif database=='VOC2007':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif database=='watercolor':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
        classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
    elif database=='PeopleArt':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/PeopleArt/JPEGImages/'
        classes =  ["person"]
    elif database=='WikiTenLabels':
        ext = '.csv'
        item_name = 'item'
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
    elif database=='clipart':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/cross-domain-detection/datasets/clipart/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        raise NotImplemented # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    
    if testMode and not(type(jtest)==int):
        assert(type(jtest)==str)
        jtest = int(np.where(np.array(classes)==jtest)[0][0])# Conversion of the jtest string to the value number
        assert(type(jtest)==int)
        
    if testMode and (jtest>len(classes)) :
       print("We are in test mode but jtest>len(classes), we will use jtest =0" )
       jtest =0
    
        
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    databasetxt =path_data + database + ext
    if database=='WikiTenLabels':
        dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                      'turban':int,'set':str,'Anno':int}
        df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)    
    else:
        df_label = pd.read_csv(databasetxt,sep=",")
    str_val = 'val'
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
        str_val = 'validation'
    elif database=='Paintings':
        str_val = 'validation'
    elif database in ['VOC2007','watercolor','clipart','PeopleArt']:
        str_val = 'val'
        df_label[classes] = df_label[classes].apply(lambda x:(x + 1.0)/2.0)
    num_trainval_im = len(df_label[df_label['set']=='train'][item_name]) + len(df_label[df_label['set']==str_val][item_name])
    num_classes = len(classes)
    print(database,'with ',num_trainval_im,' images in the trainval set')
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    thresh_evaluation = 0.05
    TEST_NMS =0.3
    thresh = 0.05
    
    sets = ['train','val','trainval','test']
    dict_name_file = {}
    data_precomputeed= True
    if k_per_bag==300:
        k_per_bag_str = ''
    else:
        k_per_bag_str = '_k'+str(k_per_bag)
    eval_onk300 = True
    for set_str in sets:
        name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
        if not(k_per_bag==300) and eval_onk300 and set_str=='test': # We will evaluate on all the 300 regions and not only the k_per_bag ones
            name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
        dict_name_file[set_str] = name_pkl_all_features
        if not(os.path.isfile(name_pkl_all_features)):
            data_precomputeed = False

#    sLength_all = len(df_label[item_name])
    if demonet == 'vgg16_COCO':
        num_features = 4096
    elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
        num_features = 2048
    
    if not(data_precomputeed):
        # Compute the features
        if verbose: print("We will computer the CNN features")
        Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                     database=database,augmentation=False,L2 =False,
                                     saved='all',verbose=verbose,filesave='tfrecords',k_regions=k_per_bag)
    dont_use_07_metric = True
    if database=='VOC2007':
        imdb = get_imdb('voc_2007_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index)
    elif database=='watercolor':
        imdb = get_imdb('watercolor_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index)
    elif database=='PeopleArt':
        imdb = get_imdb('PeopleArt_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index)
    elif database=='clipart':
        imdb = get_imdb('clipart_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index) 
    elif database=='WikiTenLabels':
        imdb = get_imdb('WikiTenLabels_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        #num_images = len(imdb.image_index) 
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    else:
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
   
    data_path_train= dict_name_file['trainval']
        
    # Config param for TF session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
            
    # Data for the MILSVM Latent SVM
    # All those parameter are design for my GPU 1080 Ti memory size 
    performance = False
    if parallel_op:
        sizeMax = 30*10000 // (k_per_bag*num_classes) 
    else:
        sizeMax = 30*10000 // k_per_bag
    if not(init_by_mean is None) and not(init_by_mean==''):
        if not(CV_Mode=='CV' and num_split==2):
            sizeMax //= 2
     # boolean paralleliation du W
    if not(num_features==2048):
        sizeMax //= (num_features//2048)
    if restarts_paral=='Dim': # It will create a new dimension
        restarts_paral_str = '_RP'
        sizeMax //= max(int((restarts+1)//2),1) # To avoid division by zero
        # it seems that using a different size batch drasticly change the results
    elif restarts_paral=='paral': # Version 2 of the parallelisation
        restarts_paral_str = '_RPV2'
        sizeMax = 30*200000 // (k_per_bag*20)
    else:
        restarts_paral_str=''
    # InternalError: Dst tensor is not initialized. can mean that you are running out of GPU memory
    mini_batch_size = min(sizeMax,num_trainval_im)
    buffer_size = 10000
    if testMode:
        ext_test = '_Test_Mode'
    else:
        ext_test= ''
    max_iters = ((num_trainval_im // mini_batch_size)+ \
                 np.sign(num_trainval_im % mini_batch_size))*max_iters_all_base

    
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    AP_per_classbS = []
    final_clf = None
    class_weight = None
    ReDo = False
    if C_Searching:C_Searching_str ='_Csearch'
    else: C_Searching_str = ''
    if with_scores:
        with_scores_str = '_WRC'+str(epsilon)
    else:
        with_scores_str=''
    if seuillage_by_score:
        seuillage_by_score_str = '_SBS'+str(seuil)
    else: seuillage_by_score_str = ''
    if norm=='L2':
        extNorm = '_L2'
    elif norm=='STDall':
        extNorm = '_STDall'
    elif norm=='STDSaid':
        extNorm = '_STDSaid'
    elif norm=='STD':
        extNorm = '_STD'
        raise(NotImplemented)
    elif norm=='' or norm is None:
        extNorm = ''
    if parallel_op:
        extPar = '_p'
    else:
        extPar =''
    if CV_Mode=='CV':
        max_iters = (max_iters*(num_split-1)//num_split) # Modification d iteration max par rapport au nombre de split
        extCV = '_cv'+str(num_split)
    elif CV_Mode=='LA':
        max_iters = (max_iters*(num_split-1)//num_split) # Modification d iteration max par rapport au nombre de split
        extCV = '_la'+str(num_split)
    elif CV_Mode=='CV' and WR==True:
        extCV = '_cv'+str(num_split)+'_wr'
    elif CV_Mode is None or CV_Mode=='':
        extCV =''
    if WR: extCV += '_wr'

    if Optimizer=='Adam':
        opti_str=''
    elif Optimizer=='GradientDescent':
        opti_str='_gd'
    else:
        raise(NotImplemented)
    if init_by_mean is None or init_by_mean=='':
        init_by_mean_str = ''
    elif init_by_mean=='First':
        init_by_mean_str= '_ibnF'
    elif init_by_mean=='All':
        init_by_mean_str= '_ibnA'
    if LR==0.01:
        LR_str = ''
    else:
        LR_str='_LR'+str(LR)
    if C == 1.0:
        C_str=''
    else:
        C_str = '_C'+str(C) # regularisation term 
    if Max_version=='max' or Max_version=='' or Max_version is None:
        Max_version_str =''
    elif Max_version=='softmax':
        Max_version_str ='_MVSF'
        if not(w_exp==1.0): Max_version_str+=str(w_exp)
    elif Max_version=='sparsemax':
        Max_version_str ='_MVSM'
    elif Max_version=='mintopk':
        Max_version_str ='_MVMT'+str(k_intopk)
    optimArg= None
    verboseMILSVM = True
    shuffle = True
    if num_trainval_im==mini_batch_size:
        shuffle = False
    if shuffle:
        shuffle_str = ''
    else:
        shuffle_str = '_allBase'
    if optim_wt_Reg:
        optim_wt_Reg_str = '_OptimWTReg'
    else:
        optim_wt_Reg_str =''
    Number_of_positif_elt = 1 
    number_zone = k_per_bag
    
#    thresh_evaluation,TEST_NMS = 0.05,0.3
    dont_use_07_metric = True
    symway = True
    
    arrayParam = [demonet,database,N,extL2,nms_thresh,savedstr,mini_batch_size,
                  performance,buffer_size,predict_with,shuffle,C,testMode,restarts,max_iters_all_base,
                  max_iters,CV_Mode,num_split,parallel_op,WR,norm,Optimizer,LR,optimArg,
                  Number_of_positif_elt,number_zone,seuil_estimation,thresh_evaluation,
                  TEST_NMS,init_by_mean,transform_output,with_rois_scores_atEnd,
                  with_scores,epsilon,restarts_paral,Max_version,w_exp,seuillage_by_score,seuil,
                  k_intopk,C_Searching,gridSearch,thres_FinalClassifier,optim_wt_Reg]
    arrayParamStr = ['demonet','database','N','extL2','nms_thresh','savedstr',
                     'mini_batch_size','performance','buffer_size','predict_with',
                     'shuffle','C','testMode','restarts','max_iters_all_base','max_iters','CV_Mode',
                     'num_split','parallel_op','WR','norm','Optimizer','LR',
                     'optimArg','Number_of_positif_elt','number_zone','seuil_estimation'
                     ,'thresh_evaluation','TEST_NMS','init_by_mean','transform_output','with_rois_scores_atEnd',
                     'with_scores','epsilon','restarts_paral','Max_version','w_exp','seuillage_by_score',
                     'seuil','k_intopk','C_Searching','gridSearch','thres_FinalClassifier','optim_wt_Reg']
    assert(len(arrayParam)==len(arrayParamStr))
    print(tabs_to_str(arrayParam,arrayParamStr))
#    print('database',database,'mini_batch_size',mini_batch_size,'max_iters',max_iters,'norm',norm,\
#          'parallel_op',parallel_op,'CV_Mode',CV_Mode,'WR',WR,'restarts',restarts,'demonet',demonet,
#          'Optimizer',Optimizer,'init_by_mean',init_by_mean,'with_tanh',with_tanh)
    

    cachefile_model_base= database +'_'+demonet+'_r'+str(restarts)+'_s' \
        +str(mini_batch_size)+'_k'+str(k_per_bag)+'_m'+str(max_iters)+extNorm+extPar+\
        extCV+ext_test+opti_str+LR_str+C_str+init_by_mean_str+with_scores_str+restarts_paral_str\
        +Max_version_str+seuillage_by_score_str+shuffle_str+C_Searching_str+optim_wt_Reg_str
    cachefile_model = path_data +  cachefile_model_base+'_MILSVM.pkl'
    with open(cachefile_model, 'rb') as f:
        name_milsvm = pickle.load(f)
    export_dir,np_pos_value,np_neg_value= name_milsvm
    export_dir_path = ('/').join(export_dir.split('/')[:-1])
    name_model_meta = export_dir + '.meta'
    get_roisScore = False
    train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
    if not(k_per_bag==300) and eval_onk300:
         train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,num_rois=300))
    else:
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,num_rois=k_per_bag))
    dataset_batch = train_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()
    true_label_all_test =  []
    predict_label_all_test =  []
    name_all_test =  []
    FirstTime= True
    i = 0
    ii = 0
    load_model = False
    scoreInMILSVM =False
    seuil_estimation_bool = False
    with_rois_scores_atEnd=False
    seuillage_by_score=False
    gridSearch=False
    with_softmax_a_intraining=  False
    with_softmax = False
    sess = tf.Session(config=config)
    if load_model==False:
        new_saver = tf.train.import_meta_graph(name_model_meta)
        new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
        graph= tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        if scoreInMILSVM: 
            scores_tf = graph.get_tensor_by_name("scores:0")
            Prod_best = graph.get_tensor_by_name("ProdScore:0")
        else:
            Prod_best = graph.get_tensor_by_name("Prod:0")
        if with_tanh:
            print('use of tanh')
            Tanh = tf.tanh(Prod_best)
            mei = tf.argmax(Tanh,axis=2)
            score_mei = tf.reduce_max(Tanh,axis=2)
        elif with_softmax:
            Softmax = tf.nn.softmax(Prod_best,axis=-1)
            mei = tf.argmax(Softmax,axis=2)
            score_mei = tf.reduce_max(Softmax,axis=2)
        elif with_softmax_a_intraining:
            Softmax=tf.multiply(tf.nn.softmax(Prod_best,axis=-1),Prod_best)
            mei = tf.argmax(Softmax,axis=2)
            score_mei = tf.reduce_max(Softmax,axis=2)
        else:
            mei = tf.argmax(Prod_best,axis=2)
            score_mei = tf.reduce_max(Prod_best,axis=2)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        while True:
            try:
                if not(with_rois_scores_atEnd) and not(scoreInMILSVM):
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                else:
                    fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                fc7s_full,roiss_full, labels_full,name_imgs_full = fc7s,roiss, labels,name_imgs
                PositiveExScoreAll_tab = []
                for iii in range(10):
                    fc7s = fc7s_full[:,iii*30:iii*30+30,:]
                    roiss = roiss_full[:,iii*30:iii*30+30]
                    labels = labels_full
                    name_imgs = name_imgs_full
                
                    if predict_with=='MILSVM':
                        if scoreInMILSVM:
                            feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                        else:
                            feed_dict_value = {X: fc7s, y: labels}
                        if with_tanh:
                            PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                            sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                        elif with_softmax or with_softmax_a_intraining:
                            PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                            sess.run([mei,score_mei,Softmax], feed_dict=feed_dict_value)
                        else:
                            PositiveRegions,get_RegionsScore,PositiveExScoreAll = \
                            sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)
                        if with_rois_scores_atEnd:
                            PositiveExScoreAll = PositiveExScoreAll*rois_scores
                            get_RegionsScore = np.max(PositiveExScoreAll,axis=2)
                            PositiveRegions = np.amax(PositiveExScoreAll,axis=2)
                        if with_tanh: assert(np.max(PositiveExScoreAll) <= 1.)
                    PositiveExScoreAll_tab += [PositiveExScoreAll]
                roiss = roiss_full

                PositiveExScoreAll= np.concatenate(PositiveExScoreAll_tab,axis=2)
                PositiveRegions  = np.amax(PositiveExScoreAll,axis=2)
                get_RegionsScore= np.max(PositiveExScoreAll,axis=2)
                print('PositiveExScoreAll',PositiveExScoreAll.shape)
                print('PositiveRegions',PositiveRegions.shape)
                print('get_RegionsScore',get_RegionsScore.shape)
                print('name_imgs',len(name_imgs))
                true_label_all_test += [labels]
                
                if predict_with=='MILSVM':
                    predict_label_all_test +=  [get_RegionsScore]
                elif 'LinearSVC' in predict_with:
                    predict_label_all_test_tmp = []
                    for j in range(num_classes):
                        predict_label_all_test_tmp += [np.reshape(classifier_trained_dict[j].decision_function( 
                                np.reshape(fc7s,(-1,fc7s.shape[-1]))),(fc7s.shape[0],fc7s.shape[1]))]
                    predict_label_all_test_batch = np.stack(predict_label_all_test_tmp,axis=0)
    #                    print(predict_label_all_test_batch.shape)
                    predict_label_all_test += [np.max(predict_label_all_test_batch,axis=2)]
    #                    print('predict_label_all_test',predict_label_all_test[-1].shape)
                    # predict_label_all_test is used only for the classification score !
    #                if predict_with=='LinearSVC':
                    
                for k in range(len(labels)):
                    if database in ['VOC2007','watercolor','Paintings','clipart','WikiTenLabels','PeopleArt'] :
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    else:
                         complet_name = path_to_img + name_imgs[k] + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    if predict_with=='MILSVM':
                        scores_all = PositiveExScoreAll[:,k,:]
                    elif 'LinearSVC' in predict_with:
                        scores_all = predict_label_all_test_batch[:,k,:]

                    roi = roiss[k,:]
                    roi_boxes =  roi[:,1:5] / im_scales[0] 
                    
                    for j in range(num_classes):
                        scores = scores_all[j,:]
                        if seuil_estimation_bool:
                            inds = np.where(scores > list_thresh[j])[0]
                        else:
                            inds = np.where(scores > thresh)[0]
                        cls_scores = scores[inds]
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                        
                        modif_box = '' # Possibility : SumPond, Inter
                        if(not(modif_box=='') and not(modif_box is None)):
                            # Modification of the bounding box 
                            cls_dets = py_cpu_modif(cls_dets,kind=modif_box)
                        
                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        
                        all_boxes[j][i] = cls_dets
                    i+=1
    
                for l in range(len(name_imgs)): 
                    if database in ['VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt']:
                        name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                    else:
                        name_all_test += [[name_imgs[l]]]
                
                if PlotRegions and predict_with=='MILSVM':
                    if verbose and (ii%1000==0):
                        print("Plot the images :",ii)
                    if verbose and FirstTime: 
                        FirstTime = False
                        print("Start ploting Regions on test set")
                    for k in range(len(labels)):   
                        if ii > number_im:
                            continue
                        if  database in ['VOC2007','Paintings','watercolor','clipart','WikiTenLabels','PeopleArt']:
                            name_img = str(name_imgs[k].decode("utf-8") )
                        else:
                            name_img = name_imgs[k]
                        rois = roiss[k,:]
                        if database in ['VOC12','Paintings','VOC2007','clipart','watercolor','WikiTenLabels','PeopleArt']:
                            complet_name = path_to_img + name_img + '.jpg'
                            name_sans_ext = name_img
                        elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                            name_sans_ext = os.path.splitext(name_img)[0]
                            complet_name = path_to_img +name_sans_ext + '.jpg'
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        roi_boxes_and_score = []
                        local_cls = []
                        for j in range(num_classes):
                            cls_dets = all_boxes[j][ii] # Here we have #classe x box dim + score
                            # Last element is the score
#                            print(cls_dets.shape)
                            if len(cls_dets) > 0:
                                local_cls += [classes[j]]
#                                roi_boxes_score = np.expand_dims(cls_dets,axis=0)
                                roi_boxes_score = cls_dets
#                                print(roi_boxes_score.shape)
                                if roi_boxes_and_score is None:
                                    roi_boxes_and_score = [roi_boxes_score]
                                else:
                                    roi_boxes_and_score += [roi_boxes_score] 
                                    #np.vstack((roi_boxes_and_score,roi_boxes_score))

                        if roi_boxes_and_score is None: roi_boxes_and_score = [[]]
                        ii += 1    
                        if RPN:
                            best_RPN_roi = rois[0,:]
                            best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                            best_RPN_roi_scores = [PositiveExScoreAll[j,k,0]]
                            cls = local_cls + ['RPN']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                            #best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                            best_RPN_roi_boxes_score =  np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0)
#                            print(best_RPN_roi_boxes_score.shape)
                            #roi_boxes_and_score = np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                            roi_boxes_and_score += [best_RPN_roi_boxes_score] #np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                        else:
                            cls = local_cls
                        #print(len(cls),len(roi_boxes_and_score))
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                        name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions.jpg'
                        if database=='PeopleArt':
                            path_tmp = '/'.join(name_output.split('/')[0:-1])
                            pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                        plt.savefig(name_output)
                        plt.close()
            except tf.errors.OutOfRangeError:
                break
    print('End compute')
    tf.reset_default_graph()
    true_label_all_test = np.concatenate(true_label_all_test)
    predict_label_all_test = np.transpose(np.concatenate(predict_label_all_test,axis=1))
    name_all_test = np.concatenate(name_all_test)
    labels_test_predited = (np.sign(predict_label_all_test) +1.)/2
    labels_test_predited[np.where(labels_test_predited==0.5)] = 0 # To deal with the case where predict_label_all_test == 0 
     
    for j,classe in enumerate(classes):
        AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
        print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
        test_precision = precision_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        test_recall = recall_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        F1 = f1_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        print("Test on all the data precision = {0:.2f}, recall = {1:.2f},F1 = {2:.2f}".format(test_precision,test_recall,F1))
        precision_at_k = ranking_precision_score(np.array(true_label_all_test[:,j]), predict_label_all_test[:,j],20)
        P20_per_class += [precision_at_k]
        AP_per_class += [AP]
        R_per_class += [test_recall]
        P_per_class += [test_precision]
    
    det_file = os.path.join(path_data, 'detections_aux.pkl') 
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    max_per_image = 100
    num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels case
    all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
    number_im = 0
    for i in range(num_images_detect):
        name_img = imdb.image_path_at(i)
        if database=='PeopleArt':
            name_img_wt_ext = name_img.split('/')[-2] +'/' +name_img.split('/')[-1]
            name_img_wt_ext_tab =name_img_wt_ext.split('.')
            name_img_wt_ext = '.'.join(name_img_wt_ext_tab[0:-1])
        else:
            name_img_wt_ext = name_img.split('/')[-1]
            name_img_wt_ext =name_img_wt_ext.split('.')[0]
        name_img_ind = np.where(np.array(name_all_test)==name_img_wt_ext)[0]
        #print(name_img_ind)
        if len(name_img_ind)==0:
            print('len(name_img_ind), images not found in the all_boxes')
            print(name_img_wt_ext)
            raise(Exception)
        else:
            number_im += 1 
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
    assert (number_im==num_images_detect) # To check that we have the all the images in the detection prediction
    det_file = os.path.join(path_data, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
    output_dir = path_data +'tmp/' + database+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
    print("Detection score (thres = 0.5): ",database)
    print(arrayToLatex(aps,per=True))
    ovthresh_tab = [0.3,0.1,0.]
    for ovthresh in ovthresh_tab:
        aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
        print("Detection score with thres at ",ovthresh)
        print(arrayToLatex(aps,per=True))
    imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb 
    aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
    print("Detection score with the difficult element")
    print(arrayToLatex(aps,per=True))
    imdb.set_use_diff(False)

           
    print('~~~~~~~~')        
    print("mean Average Precision Classification for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
    if CompBest: print("mean Average Precision for BEst Score = {0:.3f}".format(np.mean(AP_per_classbS))) 
    print("mean Precision Classification for all the data = {0:.3f}".format(np.mean(P_per_class)))  
    print("mean Recall Classification for all the data = {0:.3f}".format(np.mean(R_per_class)))  
    print("mean Precision Classification @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    print('Mean Average Precision Classification :')
    print(AP_per_class)
    print(arrayToLatex(AP_per_class,per=True))
      
    return(0)
    
        
if __name__ == '__main__':
    LearnOn30andTestOn300()