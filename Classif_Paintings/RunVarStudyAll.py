#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:12:57 2019

This script will run our code 100 times only for the MIMAX model for the moment

@author: gonthier
"""

from TL_MIL import tfR_FRCNN,tfR_evaluation_parall

import time
import pickle
#import gc
import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import IsolationForest
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.covariance import EllipticEnvelope
#from sklearn.linear_model import SGDClassifier
#from tf_faster_rcnn.lib.nets.vgg16 import vgg16
#from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
#from tf_faster_rcnn.lib.model.test import im_detect,TL_im_detect,TL_im_detect_end,get_blobs
#from tf_faster_rcnn.lib.model.nms_wrapper import nms
#from tf_faster_rcnn.lib.nms.py_cpu_nms import py_cpu_nms
#import matplotlib.pyplot as plt
#from sklearn.svm import LinearSVC, SVC
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import PredefinedSplit,train_test_split
#from nltk.classify.scikitlearn import SklearnClassifier
#from tf_faster_rcnn.tools.demo import vis_detections
import numpy as np
#import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score
#,recall_score,precision_score,make_scorer,f1_score
#from Custom_Metrics import ranking_precision_score
#from Classifier_Evaluation import Classification_evaluation
import os.path
#import misvm # Library to do Multi Instance Learning with SVM
#from sklearn.preprocessing import StandardScaler
#from trouver_classes_parmi_K import MI_max,TrainClassif,tf_MI_max,ModelHyperplan
from trouver_classes_parmi_K import ModelHyperplan
#from trouver_classes_parmi_K_mi import tf_mi_model
#from LatexOuput import arrayToLatex
#from FasterRCNN import vis_detections_list,vis_detections,Compute_Faster_RCNN_features,vis_GT_list
#import pathlib
#from milsvm import mi_linearsvm # Version de nicolas avec LinearSVC et TODO SGD 
#from sklearn.externals import joblib # To save the classifier
#from tool_on_Regions import reduce_to_k_regions
#from sklearn import linear_model
from tf_faster_rcnn.lib.datasets.factory import get_imdb
#from Estimation_Param import kde_sklearn,findIntersection
#from utils.save_param import create_param_id_file_and_dir,write_results,tabs_to_str
#from Transform_Box import py_cpu_modif
##from hpsklearn import HyperoptEstimator,sgd
##from hyperopt import tpe
#from random import uniform
from shutil import copyfile

from IMDB import get_database
import numpy as np
import pickle

CLASSESVOC = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSESCOCO = ('__background__','person', 'bicycle','car','motorcycle', 'aeroplane','bus','train','truck','boat',
 'traffic light','fire hydrant', 'stop sign', 'parking meter','bench','bird',
 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
 'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball', 'kite',
 'baseball bat','baseball glove','skateboard', 'surfboard','tennis racket','bottle', 
 'wine glass','cup','fork', 'knife','spoon','bowl', 'banana', 'apple','sandwich', 'orange', 
'broccoli','carrot','hot dog','pizza','donut','cake','chair', 'couch','potted plant','bed',
 'diningtable','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
 'oven','toaster','sink','refrigerator', 'book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush')


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',)
    ,'vgg16_coco': ('/media/gonthier/HDD/models/tf-faster-rcnn/vgg16/vgg16_faster_rcnn_iter_1190000.ckpt',)    
    ,'res101': ('res101_faster_rcnn_iter_110000.ckpt',)
    ,'res152' : ('res152_faster_rcnn_iter_1190000.ckpt',)}

DATASETS= {'coco': ('coco_2014_train+coco_2014_valminusminival',),'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

NETS_Pretrained = {'vgg16_VOC07' :'vgg16_faster_rcnn_iter_70000.ckpt',
                   'vgg16_VOC12' :'vgg16_faster_rcnn_iter_110000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt',
                   'res101_VOC07' :'res101_faster_rcnn_iter_70000.ckpt',
                   'res101_VOC12' :'res101_faster_rcnn_iter_110000.ckpt',
                   'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'
                   }
CLASSES_SET ={'VOC' : CLASSESVOC,
              'COCO' : CLASSESCOCO }

depicts_depictsLabel = {'Q942467_verif': 'Jesus_Child','Q235113_verif':'angel_Cupidon ','Q345_verif' :'Mary','Q109607_verif':'ruins','Q10791_verif': 'nudity'}


def VariationStudyPart1_forVOC07():
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    First Part Store thevectors computed
    '''
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    database_tab = ['VOC2007','PeopleArt','watercolor','WikiTenLabels']
#    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt']
    number_restarts = 100*12-1
    max_iters_all_base = 300
    
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    start_i = 0
    end_i = 1
    listi = range(start_i,end_i+1)
    listi = [5]
    seuil = 0.9 

    for i in listi:
        print('Scenario :',i)
        if i==0:
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==1:
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==2:
            loss_type='hinge_tanh'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==3:
            loss_type='hinge'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==4:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = False
            with_scores = True
            seuillage_by_score=False
        if i==5:
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False
        elif i==6:
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=False
        elif i==7:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.9
        elif i==8:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.5
        elif i==9:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.3
        elif i==10:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.1
        elif i==11:
            C_Searching = True
            CV_Mode = 'CV'
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = True
            seuillage_by_score=False   
        elif i==12:
            C_Searching = True
            CV_Mode = 'CV'
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False   
            
            
       # TODO rajouter ici un cas ou l on fait normalise les features
    
        for database in database_tab:
            ## Compte the vectors and bias W
            exportname,arrayParam = tfR_FRCNN(demonet = 'res101_VOC07',database = database,ReDo=True,
                                          verbose = False,testMode = False,jtest = 'cow',loss_type=loss_type,
                                          PlotRegions = False,saved_clf=False,RPN=False,
                                          CompBest=False,Stocha=True,k_per_bag=300,
                                          parallel_op=True,CV_Mode=CV_Mode,num_split=2,
                                          WR=WR,init_by_mean =None,seuil_estimation='',
                                          restarts=number_restarts,max_iters_all_base=max_iters_all_base,LR=0.01,with_tanh=True,
                                          C=1.0,Optimizer='GradientDescent',norm='',
                                          transform_output='tanh',with_rois_scores_atEnd=False,
                                          with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                                          Max_version='',w_exp=10.0,seuillage_by_score=seuillage_by_score,seuil=seuil,
                                          k_intopk=1,C_Searching=C_Searching,predict_with='MI_max',
                                          gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                          thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=AggregW
                                          ,proportionToKeep=proportionToKeep,storeVectors=True)
            tf.reset_default_graph()
            name_dict = path_data_output +database+'res101_VOC07_'+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            name_dict += '.pkl'
            copyfile(exportname,name_dict)
            print(name_dict,'copied')

def Study_eval_perf_onSplit_of_IconArt(computeMode=True):
    """
    The goal of this function is to proposed a splitting of the IconArt dataset
    to see if we have a lot of performance variation according to the train and 
    test set 
    @param : computeMode : we will compute the performance otherwise we will only show them
    """

    item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC =\
        get_database('IconArt_v1')
    
    nRep = 3
    nRestart_at_fixed_train_and_test_set = 10


    df_test = df_label[df_label['Anno']==1]
    df_train = df_label[df_label['Anno']==0]

    itera = 0
    max_iters_all_base = 300
    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'  
    
    multi = 100
    metric_tab = ['AP@.5','AP@.1','APClassif']
    for r in range(nRep):
        df_test = df_test.sample(frac=1,random_state=r) # To shuffle 
        df_train = df_train.sample(frac=1,random_state=r) # To shuffle 
        df_test_list =np.array_split(df_test,2)
        df_train_list =np.array_split(df_train,2)
        for df1, df2 in zip(df_test_list,df_train_list):
            if computeMode:
                df1['set'] = 'test'
                df2['set'] = 'train'
                df_out2 = df2[['item']].astype(str)
                df_out2.to_csv(path_data_csvfile+'train_'+str(itera)+'.txt',index=False,encoding='UTF_8', header=False)
                df_out = df1[['item']].astype(str)
                df_out.to_csv(path_data_csvfile+'test_'+str(itera)+'.txt',index=False,encoding='UTF_8', header=False)
                df = df1.append(df2)
                database = 'IconArt_v1_'+str(itera)
                df =df.astype(str)
                df.to_csv(path_data_csvfile+database+'.csv',index=False,encoding='UTF_8')
            for score in [True,False]:
                print('Iter :',itera,'score :',score)
                path_tmp = os.path.join(path_data,'SplitIconArt')    
                name = 'Perf_Split'+str(itera)
                if score:
                    name += '_withScore'
                name_dictAP = os.path.join(path_tmp,name)
                if computeMode:
                    perf05 = []
                    perf01 = []
                    perfC = []
                    DictAP = {}
                    for n in range(nRestart_at_fixed_train_and_test_set):
                        apsAt05,apsAt01,AP_per_class = tfR_FRCNN(demonet = 'res152_COCO',
                                                                 database = database, ReDo=True,
                          verbose = False,testMode = False,jtest = 'cow',
                          PlotRegions = False,saved_clf=False,RPN=False,
                          CompBest=False,Stocha=True,k_per_bag=300,
                          parallel_op=True,CV_Mode='',num_split=2,
                          WR=True,init_by_mean =None,seuil_estimation='',
                          restarts=11,max_iters_all_base=max_iters_all_base,LR=0.01,
                          C=1.0,Optimizer='GradientDescent',norm='',
                          transform_output='tanh',with_rois_scores_atEnd=False,
                          with_scores=score,epsilon=0.01,restarts_paral='paral',
                          predict_with='MI_max',
                          AggregW =None ,proportionToKeep=1.0,model='MI_max')
                        perf05 += [apsAt05]
                        perf01 += [apsAt01]
                        perfC += [AP_per_class]
                     
                    DictAP['AP@.5']= perf05
                    DictAP['AP@.1']=perf01
                    DictAP['APClassif']=perfC
                    with open(name_dictAP, 'wb') as f:
                        pickle.dump(DictAP, f, pickle.HIGHEST_PROTOCOL)
                else:
                    with open(name_dictAP, 'rb') as f:
                       DictAP =  pickle.load(f)
                    
                for metric in metric_tab:
                    #print(metric)
                    string_to_print =''
                    string_to_print += metric +' & '
                    ll_all = DictAP[metric]
                    mean_over_reboot = np.mean(ll_all,axis=1) # Moyenne par ligne / reboot 
    #                            print(mean_over_reboot.shape)
                    std_of_mean_over_reboot = np.std(mean_over_reboot)
                    mean_of_mean_over_reboot = np.mean(mean_over_reboot)
                    mean_over_class = np.mean(ll_all,axis=0) # Moyenne par column
                    std_over_class = np.std(ll_all,axis=0) # Moyenne par column 
    #                            print('ll_all.shape',ll_all.shape)
    #                            print(mean_over_class.shape)
    #                            print(std_over_class.shape)
    #                            input('wait')
                    for mean_c,std_c in zip(mean_over_class,std_over_class):
                        s =  "{0:.1f} ".format(mean_c*multi) + ' $\pm$ ' +  "{0:.1f}".format(std_c*multi)
                        string_to_print += s + ' & '
                    s =  "{0:.1f}  ".format(mean_of_mean_over_reboot*multi) + ' $\pm$ ' +  "{0:.1f}  ".format(std_of_mean_over_reboot*multi)
                    string_to_print += s + ' \\\  '
                    print(string_to_print)
#                mean_per_class = np.mean(perf05,axis=0)
#                mean = np.mean(perf05,axis=1)
#                meanOfmean = np.mean(mean,axis=0)
#                stdOfmean = np.std(mean,axis=0)
#                std_per_class = np.std(perf05,axis=0)
            itera += 1
                # a finir
                

def unefficient_way_MaxOfMax_evaluation(database='IconArt_v1'):
    """
    Compute the performance for the MaxOfMax model on 100 runs
    """
    num_rep = 100
    seuillage_by_score = False
    obj_score_add_tanh = False
    loss_type = ''
    seuil = 0
    C_Searching = False
    demonet = 'res152_COCO'
    layer = 'fc7'
    number_restarts = 11
    CV_Mode = ''
    AggregW = None
    proportionToKeep = [0.25,1.0]
    loss_type = ''
    WR = True
    with_scores = True
    seuillage_by_score=False
    obj_score_add_tanh=False
    lambdas = 0.5
    obj_score_mul_tanh = False
    PCAuse = False
    Max_version = None
    max_iters_all_base = 3000
    k_per_bag = 300
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    ReDo = True
    
    for with_scores in [False,True]:
        for loss_type in ['','hinge']:
            name_dict = path_data_output 
            if not(demonet== 'res152_COCO'):
                name_dict += demonet +'_'
            if not(layer== 'fc7'):
                name_dict += '_'+demonet
            name_dict +=  database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            if obj_score_add_tanh:
                name_dict += 'SAdd'+str(lambdas)
            if obj_score_mul_tanh:
                name_dict += 'SMul'    
            if PCAuse:
                name_dict +='_PCA09'
            if not(k_per_bag==300):
                name_dict +='_k'+str(k_per_bag)
            if not(Max_version=='' or Max_version is None) :
                name_dict +='_'+Max_version
            if not(max_iters_all_base==300) :
                name_dict +='_MIAB'+str(max_iters_all_base)
            name_dictAP = name_dict + '_MaxOfMax' + '_APscore.pkl'
            DictAP = {}
            ll = []
            l01 = []
            lclassif = []
            for r in range(num_rep):
                apsAt05,apsAt01,AP_per_class = tfR_FRCNN(demonet =demonet,database = database,ReDo=ReDo,
                                              verbose = False,testMode = False,jtest = 'cow',loss_type=loss_type,
                                              PlotRegions = False,saved_clf=False,RPN=False,
                                              CompBest=False,Stocha=True,k_per_bag=k_per_bag,
                                              parallel_op=True,CV_Mode=CV_Mode,num_split=2,
                                              WR=WR,init_by_mean =None,seuil_estimation='',
                                              restarts=number_restarts,max_iters_all_base=max_iters_all_base,LR=0.01,with_tanh=True,
                                              C=1.0,Optimizer='GradientDescent',norm='',
                                              transform_output='tanh',with_rois_scores_atEnd=False,
                                              with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                                              Max_version=Max_version,w_exp=10.0,seuillage_by_score=seuillage_by_score,seuil=seuil,
                                              k_intopk=1,C_Searching=C_Searching,predict_with='MI_max',
                                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=AggregW
                                              ,proportionToKeep=proportionToKeep,storeVectors=False,
                                              obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                                              obj_score_mul_tanh=obj_score_mul_tanh,PCAuse=PCAuse,
                                              layer=layer,MaxOfMax=True)
                ll += [apsAt05]
                l01 += [apsAt01]
                lclassif += [AP_per_class]
            # End of the 100 experiment for a specific AggreW
            ll_all = np.vstack(ll)
            l01_all = np.vstack(l01)
            apsClassif_all = np.vstack(lclassif)
    
            DictAP['AP@.5'] =  ll_all
            DictAP['AP@.1'] =  l01_all
            DictAP['APClassif'] =  apsClassif_all
        
            with open(name_dictAP, 'wb') as f:
                pickle.dump(DictAP, f, pickle.HIGHEST_PROTOCOL)
    
             
def VariationStudyPart1(database=None,scenarioSubset=None,demonet = 'res152_COCO',k_per_bag=300,layer='fc7'):
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    First Part Store thevectors computed
    '''
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    if database is None:
        database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
    else:
        database_tab = [database]
    start_i = 0
    end_i = 19
    if scenarioSubset is None:
        listi = np.arange(start_i,end_i)
    # Just to have with and without score : scenario 0 and 5 
    else:
        listi = scenarioSubset
#    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt']
    number_restarts = 100*12-1
    
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']

    seuil = 0.9 
    ReDo = False
    # TODO implement a ReDo that works !

    for i_scenario in listi:
        print('Scenario :',i_scenario)
        output = get_params_fromi_scenario(i_scenario)
        listAggregW,C_Searching,CV_Mode,AggregW,proportionToKeep,loss_type,WR,\
        with_scores,seuillage_by_score,obj_score_add_tanh,lambdas,obj_score_mul_tanh,\
        PCAuse,Max_version,max_iters_all_base = output   
            
            
       # TODO rajouter ici un cas ou l on fait normalise les features
    
        for database in database_tab:
            ## Compte the vectors and bias W
            exportname,arrayParam = tfR_FRCNN(demonet =demonet,database = database,ReDo=ReDo,
                                          verbose = False,testMode = False,jtest = 'cow',loss_type=loss_type,
                                          PlotRegions = False,saved_clf=False,RPN=False,
                                          CompBest=False,Stocha=True,k_per_bag=k_per_bag,
                                          parallel_op=True,CV_Mode=CV_Mode,num_split=2,
                                          WR=WR,init_by_mean =None,seuil_estimation='',
                                          restarts=number_restarts,max_iters_all_base=max_iters_all_base,LR=0.01,with_tanh=True,
                                          C=1.0,Optimizer='GradientDescent',norm='',
                                          transform_output='tanh',with_rois_scores_atEnd=False,
                                          with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                                          Max_version=Max_version,w_exp=10.0,seuillage_by_score=seuillage_by_score,seuil=seuil,
                                          k_intopk=1,C_Searching=C_Searching,predict_with='MI_max',
                                          gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                          thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=AggregW
                                          ,proportionToKeep=proportionToKeep,storeVectors=True,
                                          obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                                          obj_score_mul_tanh=obj_score_mul_tanh,PCAuse=PCAuse,
                                          layer=layer)
            tf.reset_default_graph()
            name_dict = path_data_output 
            if not(demonet== 'res152_COCO'):
                name_dict += demonet +'_'
            if not(layer== 'fc7'):
                name_dict += '_'+demonet
            name_dict +=  database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            if obj_score_add_tanh:
                name_dict += 'SAdd'+str(lambdas)
            if obj_score_mul_tanh:
                name_dict += 'SMul'
            if PCAuse:
                name_dict +='_PCA09'
            if not(k_per_bag==300):
                name_dict +='_k'+str(k_per_bag)
            if not(Max_version=='' or Max_version is None) :
                name_dict +='_'+Max_version
            if not(max_iters_all_base==300) :
                name_dict +='_MIAB'+str(max_iters_all_base)
            name_dict += '.pkl'
            copyfile(exportname,name_dict)
            print(name_dict,'copied')
            
def ComputationForLossPlot(database= 'PeopleArt'):
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    C_Searching = False
    loss_type = ''
    CV_Mode = ''
    seuillage_by_score = False
    seuil = 0.9
    with_scores_list = [True,False]
    WR = True
    for with_scores in with_scores_list:
        # A faire tourner plus tard pour tracer les decroissances des courbes 
        exportname,arrayParam =  tfR_FRCNN(demonet = 'res152_COCO',database=database,ReDo=True,
                                  verbose = False,testMode = False,jtest = 'cow',
                                  PlotRegions = False,saved_clf=False,RPN=False,
                                  CompBest=False,Stocha=True,k_per_bag=300,
                                  parallel_op=True,CV_Mode=CV_Mode,num_split=2,
                                  WR=WR,init_by_mean =None,seuil_estimation='',
                                  restarts=24,max_iters_all_base=300,LR=0.01,with_tanh=True,
                                  C=1.0,Optimizer='GradientDescent',norm='',
                                  transform_output='tanh',with_rois_scores_atEnd=False,
                                  with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                                  Max_version='',w_exp=10.0,seuillage_by_score=seuillage_by_score,seuil=seuil,
                                  k_intopk=1,C_Searching=C_Searching,predict_with='MI_max',
                                  gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                  thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=None,proportionToKeep=0.25,
                                  loss_type=loss_type,storeVectors=False,storeLossValues=True) 
        tf.reset_default_graph()
        name_dict = path_data_output+'storeLossValues_' +database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
        CV_Mode+'_'+str(loss_type)
        
        if not(WR):
            name_dict += '_withRegularisationTermInLoss'
        if with_scores:
            name_dict += '_WithScore'
        if seuillage_by_score:
            name_dict += 'SC'+str(seuil)
        name_dict += '.pkl'
        copyfile(exportname,name_dict)
        print(name_dict,'copied')
     
def get_params_fromi_scenario(i_scenario):
    """
    Number 0 : MIMAX-score
    Number 3 : hinge loss with score
    Number 5 : MIMAX without score
    Number 22 : hinge loss without score
    """
    listAggregW = [None]
    PCAuse = False
    Max_version = None
    max_iters_all_base = 300
    if i_scenario==0: # MIMAX-S : MILS
        listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = True
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==1:
        listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
        loss_type='MSE'
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = True
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==2:
        loss_type='hinge_tanh'
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = True
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==3:
        loss_type='hinge'
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = True
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==4:
        loss_type=''
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = False
        with_scores = True
        seuillage_by_score=False
    if i_scenario==5:  # MIMAX : MIL
        listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==6:
        loss_type='MSE'
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=False
    elif i_scenario==7:
        loss_type=''
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=True
        seuil=0.9
    elif i_scenario==8:
        loss_type=''
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=True
        seuil=0.5
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==9:
        loss_type=''
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=True
        seuil=0.3
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==10:
        loss_type=''
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=True
        seuil=0.1
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==11:
        listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
        C_Searching = True
        CV_Mode = 'CV'
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = True
        seuillage_by_score=False   
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==12:
        listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
        C_Searching = True
        CV_Mode = 'CV'
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==13:
        listAggregW = [None,'maxOfTanh','meanOfTanh','minOfTanh','AveragingW']
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = True
    elif i_scenario==14:
        listAggregW = [None,'maxOfTanh','meanOfTanh','minOfTanh','AveragingW']
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=True
        lambdas = 0.5
        obj_score_mul_tanh = False
    elif i_scenario==15:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=True
        lambdas = 0.9
        obj_score_mul_tanh = False
    elif i_scenario==16:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=True
        lambdas = 0.75
        obj_score_mul_tanh = False
    elif i_scenario==17:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = []
        loss_type = ''
        WR = True
        with_scores = True
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.0
        obj_score_mul_tanh = False
        PCAuse = True
    elif i_scenario==18:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = []
        loss_type = ''
        WR = True
        with_scores = False
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.0
        obj_score_mul_tanh = False
        PCAuse = True
    elif i_scenario==20:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = []
        loss_type = ''
        WR = True
        with_scores = True # !!
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.0
        obj_score_mul_tanh = False
        Max_version = 'MaxPlusMin' # !!
    elif i_scenario==21:
        listAggregW = [None]
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = []
        loss_type = ''
        WR = True
        with_scores = False # !!
        seuillage_by_score=False 
        obj_score_add_tanh=False
        lambdas = 0.0
        obj_score_mul_tanh = False
        Max_version = 'MaxPlusMin' # !!
    elif i_scenario==22:
        loss_type='hinge'
        C_Searching = False
        CV_Mode = ''
        AggregW = None
        proportionToKeep = [0.25,1.0]
        WR = True
        with_scores = False
        seuillage_by_score=False
        obj_score_add_tanh=False
        lambdas = 0.5
        obj_score_mul_tanh = False   
            
    output = listAggregW,C_Searching,CV_Mode,AggregW,proportionToKeep,loss_type,\
    WR,with_scores,seuillage_by_score,obj_score_add_tanh,lambdas,obj_score_mul_tanh,\
    PCAuse,Max_version,max_iters_all_base
    
    return(output)
        
def VariationStudyPart2(database=None,scenarioSubset=None,withoutAggregW=False,
                        demonet = 'res152_COCO',k_per_bag=300,layer='fc7'):
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    The second part compute the score in AP 
    '''
    print('========= Part 2 Variation Study ===========')
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    if database is None:
        database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
    else:
        database_tab = [database]
    start_i = 0
    end_i = 19
    if scenarioSubset is None:
        listi = np.arange(start_i,end_i)
    # Just to have with and without score : scenario 0 and 5 
    else:
        listi = scenarioSubset
#    database_tab = ['watercolor']
##    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt','watercolor']
    number_of_reboots = 100
    number_restarts = 100*12-1
    max_iters_all_base = 300
    k_per_bag = 300
    numberofW_to_keep_base = 12
#    numberofW_to_keep = 12
    eval_onk300 = True
    dont_use_07_metric  =True
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    seuil = 0.9 
    
    data_path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    
    for i_scenario in listi:
        print('Scenario :',i_scenario)
        output = get_params_fromi_scenario(i_scenario)
        listAggregW,C_Searching,CV_Mode,AggregW,proportionToKeepTab,loss_type,WR,\
        with_scores,seuillage_by_score,obj_score_add_tanh,lambdas,obj_score_mul_tanh,\
        PCAuse,Max_version,max_iters_all_base = output
        
        if withoutAggregW:
            listAggregW  = [None]

        if C_Searching:
            numberofW_to_keep = numberofW_to_keep_base*9 #Number of element in C
        else:
            numberofW_to_keep = numberofW_to_keep_base
        
        for database in database_tab:
            # Name of the vectors pickle
            name_dict = path_data_output 
            if not(demonet== 'res152_COCO'):
                name_dict += demonet +'_'
            if not(layer== 'fc7'):
                name_dict += '_'+demonet
            name_dict +=  database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            if obj_score_add_tanh:
                name_dict += 'SAdd'+str(lambdas)
            if obj_score_mul_tanh:
                name_dict += 'SMul'    
            if PCAuse:
                name_dict +='_PCA09'
            if not(k_per_bag==300):
                name_dict +='_k'+str(k_per_bag)
            if not(Max_version=='' or Max_version is None) :
                name_dict +='_'+Max_version
            if not(max_iters_all_base==300) :
                name_dict +='_MIAB'+str(max_iters_all_base)
            name_dictW = name_dict + '.pkl'
            

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
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
                raise NotImplementedError # TODO implementer cela !!! 
            elif(database=='IconArt_v1'):
                ext='.csv'
                item_name='item'
                classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
                'Mary','nudity', 'ruins','Saint_Sebastien']
                path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
            elif(database=='Wikidata_Paintings_miniset_verif'):
                item_name = 'image'
                path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
                classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
            else:
                raise NotImplementedError
            
            path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
            if database=='IconArt_v1':
                path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
            else:
                path_data_csvfile = path_data
            databasetxt =path_data_csvfile + database + ext
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
#                print(database,'with ',num_trainval_im,' images in the trainval set')
            N = 1
            extL2 = ''
            nms_thresh = 0.7
            savedstr = '_all'
            variance_thres = 0.9
            
            #    sLength_all = len(df_label[item_name])
            if demonet in ['vgg16_COCO','vgg16_VOC07','vgg16_VOC12']:
                num_features = 4096
            elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
                num_features = 2048
            
            if PCAuse:
                if variance_thres==0.9:
                    if database=='IconArt_v1':
                        number_composant=675
                    elif database=='watercolor':
                        number_composant=654    
                    else:
                        print('You have to add the value of  number_composant here !')
                else:
                    print('If you already have computed the PCA on the data you have to add the number_composant at the beginning of the tfR_FRCNN function')
                num_features =  number_composant
            
            sets = ['train','val','trainval','test']
            dict_name_file = {}
            data_precomputed= True
            if k_per_bag==300:
                k_per_bag_str = ''
            else:
                k_per_bag_str = '_k'+str(k_per_bag)
#            for set_str in sets:
#                name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
#                if not(k_per_bag==300) and eval_onk300 and set_str=='test': # We will evaluate on all the 300 regions and not only the k_per_bag ones
#                    name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
#                dict_name_file[set_str] = name_pkl_all_features
            metamodel = 'FasterRCNN'
            for set_str in sets:
                name_pkl_all_features = path_data+metamodel+'_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str
                if PCAuse:
                    name_pkl_all_features+='_PCAc'+str(number_composant)
                name_pkl_all_features+='_'+set_str+'.tfrecords'
                if not(k_per_bag==300) and eval_onk300 and set_str=='test': # We will evaluate on all the 300 regions and not only the k_per_bag ones
                    name_pkl_all_features = path_data+metamodel+'_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr
                    if PCAuse:
                        name_pkl_all_features+='_PCAc'+str(number_composant)
                    name_pkl_all_features+='_'+set_str+'.tfrecords'
                dict_name_file[set_str] = name_pkl_all_features
        

                   
            # Config param for TF session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  

            
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
            elif database=='IconArt_v1':
                imdb = get_imdb('IconArt_v1_test')
                imdb.set_force_dont_use_07_metric(dont_use_07_metric)
                num_images = len(df_label[df_label['set']=='test'][item_name])
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                imdb = get_imdb('WikiTenLabels_test')
                imdb.set_force_dont_use_07_metric(dont_use_07_metric)
                #num_images = len(imdb.image_index) 
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            else:
                num_images =  len(df_label[df_label['set']=='test'][item_name])
                
            with open(name_dictW, 'rb') as f:
                 Dict = pickle.load(f)
            Wstored = Dict['Wstored']
            Bstored =  Dict['Bstored']
            Lossstored = Dict['Lossstored']
            np_pos_value =  Dict['np_pos_value'] 
            np_neg_value =  Dict['np_neg_value']
    #            print(Wstored.shape)
            
            if not(list==type(proportionToKeepTab)):
                proportionToKeepTab = [proportionToKeepTab]
    
            for AggregW in listAggregW:
                print('Scenario',i_scenario,'AggregW',AggregW,'for ',database)
                if AggregW is None or AggregW=='':
                    proportionToKeepTabLocal = [0.]
                    name_dictAP = name_dict  + '_' +str(AggregW)  + '_APscore.pkl'
                else:
                    proportionToKeepTabLocal = proportionToKeepTab
                for proportionToKeep in proportionToKeepTabLocal:
                    name_dictAP = name_dict  + '_' +str(AggregW) 
                    if not(AggregW is None or AggregW==''):
                        name_dictAP += '_'+str(proportionToKeep)+ '_APscore.pkl'

                    ReDo  = False
                    if not os.path.isfile(name_dictAP) or ReDo:
    #                    print('name_dictAP',name_dictAP)
    #                    print('Wstored',Wstored.shape)
    #                    print('numberofW_to_keep',numberofW_to_keep)
    #                    input('wait')
                        DictAP = {}
                    
                        ll = []
                        l01 = []
                        lclassif = []
                        ## Create the model
                        modelcreator = ModelHyperplan(norm='',AggregW=AggregW,epsilon=0.01,mini_batch_size=1000,num_features=num_features,num_rois=k_per_bag,num_classes=num_classes,
                             with_scores=with_scores,seuillage_by_score=seuillage_by_score,
                             proportionToKeep=proportionToKeep,restarts=numberofW_to_keep-1,seuil=seuil,
                             obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                             obj_score_mul_tanh=obj_score_mul_tanh)
                        class_indice = -1
                        ## Compute the best vectors 
                        cachefile_model_base='WLS_run100'
                        for l in range(number_of_reboots): 
                            print('reboot :',l)
                            all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
                            Wstored_extract = Wstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
                            #print('Wstored_extract',Wstored_extract.shape)
                            #print('num_features',num_features)
                            W_tmp = np.reshape(Wstored_extract,(-1,num_features),order='F')
                            #print('W_tmp',W_tmp.shape)
                            b_tmp =np.reshape( Bstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep],(-1,1,1),order='F')
                            Lossstoredextract = Lossstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep]
                            loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
                            ## Creation of the model
                            export_dir =  modelcreator.createIt(data_path,class_indice,W_tmp,b_tmp,loss_value)
                            number_zone = 300
                            Number_of_positif_elt = 1
                            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
                            parameters=False,False,False,False
        #                    parameters=PlotRegions,RPN,Stocha,CompBest
                            param_clf = k_per_bag,1,num_features
        #                    param_clf = k_per_bag,Number_of_positif_elt,num_features
                            thresh_evaluation = 0.05
                            TEST_NMS = 0.3
                            predict_with= 'MI_max'
                            if  AggregW is None or AggregW =='':
                                transform_output = 'tanh'
                            elif 'Tanh' in AggregW:
                                transform_output = ''
                            else:
                                transform_output = 'tanh'
                            seuil_estimation= False
                            mini_batch_size = 1000
                            verbose = False
                            true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
                            ,all_boxes = \
                            tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
                                   export_dir,dict_name_file,mini_batch_size,config,
                                   path_to_img,path_data,param_clf,classes,parameters,verbose,
                                   seuil_estimation,thresh_evaluation,TEST_NMS,all_boxes=all_boxes,
                                   cachefile_model_base=cachefile_model_base,transform_output=transform_output,
                                   scoreInMI_max=(with_scores or seuillage_by_score or obj_score_add_tanh or obj_score_mul_tanh)
                                   ,AggregW=AggregW,obj_score_add_tanh=obj_score_add_tanh,
                                   obj_score_mul_tanh=obj_score_mul_tanh)
                            tf.reset_default_graph()
#                            print(export_dir)
#                            os.remove(export_dir)
                            # Classification Perf
                            AP_per_class = []
                            for j,classe in enumerate(classes):
                                AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
                                AP_per_class += [AP]    
                            
                            # Detection Perf 
    #                        det_file = os.path.join(path_data, 'detections_aux.pkl')
    #                        with open(det_file, 'wb') as f:
    #                            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
                            max_per_image = 100
                            num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels case
                            all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
                            number_im = 0
                            for idetect in range(num_images_detect):
                                name_img = imdb.image_path_at(idetect)
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
                                    all_boxes_order[j][idetect]  = all_boxes[j_minus_1][name_img_ind[0]]
                                if max_per_image > 0:
                                    image_scores = np.hstack([all_boxes_order[j][idetect][:, -1]
                                                for j in range(1, imdb.num_classes)])
                                    if len(image_scores) > max_per_image:
                                        image_thresh = np.sort(image_scores)[-max_per_image]
                                        for j in range(1, imdb.num_classes):
                                            keep = np.where(all_boxes_order[j][idetect][:, -1] >= image_thresh)[0]
                                            all_boxes_order[j][idetect] = all_boxes_order[j][idetect][keep, :]
                            assert (number_im==num_images_detect) # To check that we have the all the images in the detection prediction
                            det_file = os.path.join(path_data, 'detections.pkl')
                            with open(det_file, 'wb') as f:
                                pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
                            output_dir = path_data +'tmp/' + database+'_mAP.txt'
                            aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
                            apsAt05 = aps
    #                        print("Detection score (thres = 0.5): ",database)
    #                        print(arrayToLatex(aps,per=True))
                            ovthresh_tab = [0.1]
                            for ovthresh in ovthresh_tab:
                                aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
                                if ovthresh == 0.1:
                                    apsAt01 = aps
    #                            print("Detection score with thres at ",ovthresh)
    #                            print(arrayToLatex(aps,per=True))
    #                        imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb 
    #                        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
    #                        print("Detection score with the difficult element")
    #                        print(arrayToLatex(aps,per=True))
    #                        imdb.set_use_diff(False)
                            
                            print(apsAt05,apsAt01,AP_per_class)
                            tf.reset_default_graph()
                            # aps ne contient pas le mean sur les classes en fait
                            ll += [apsAt05]
                            l01 += [apsAt01]
                            lclassif += [AP_per_class]
                        # End of the 100 experiment for a specific AggreW
                        ll_all = np.vstack(ll)
                        l01_all = np.vstack(l01)
                        apsClassif_all = np.vstack(lclassif)
    
                        DictAP['AP@.5'] =  ll_all
                        DictAP['AP@.1'] =  l01_all
                        DictAP['APClassif'] =  apsClassif_all
                    
                        with open(name_dictAP, 'wb') as f:
                            pickle.dump(DictAP, f, pickle.HIGHEST_PROTOCOL)
                    else:
                        print('The files already exist we will not do it again')
def VariationStudyPart2_forVOC07():
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    The second part compute the score in AP 
    '''
    demonet = 'res101_VOC07'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
#    database_tab = ['watercolor']
##    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt','watercolor']
    number_of_reboots = 100
    k_per_bag = 300
    numberofW_to_keep_base = 12
#    numberofW_to_keep = 12
    
    dont_use_07_metric  =True
    Dict = {}

    seuil = 0.9 
    
    data_path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    listi = [0,5]
    seuil = 0.9 

    for i_scenario in listi:
        print('Scenario :',i_scenario)
        listAggregW = [None]
        if i_scenario==0:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = True
            seuillage_by_score=False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==1:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==2:
            loss_type='hinge_tanh'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==3:
            loss_type='hinge'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==4:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = False
            with_scores = True
            seuillage_by_score=False
        if i_scenario==5:
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==6:
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=False
        elif i_scenario==7:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.9
        elif i_scenario==8:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.5
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==9:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.3
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==10:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.1
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==11:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            C_Searching = True
            CV_Mode = 'CV'
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = True
            seuillage_by_score=False   
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==12:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            C_Searching = True
            CV_Mode = 'CV'
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False 
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==13:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False 
            obj_score_add_tanh=False
            lambdas = 0.5
            obj_score_mul_tanh = True
        elif i_scenario==14:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False 
            obj_score_add_tanh=True
            lambdas = 0.5
            obj_score_mul_tanh = False
        elif i_scenario==15:
            listAggregW = [None]
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False 
            obj_score_add_tanh=True
            lambdas = 0.9
            obj_score_mul_tanh = False
        elif i_scenario==16:
            listAggregW = [None]
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False 
            obj_score_add_tanh=True
            lambdas = 0.75
            obj_score_mul_tanh = False
       
        if C_Searching:
            numberofW_to_keep = numberofW_to_keep_base*9 #Number of element in C
        else:
            numberofW_to_keep = numberofW_to_keep_base
        
        for database in database_tab:
            # Name of the vectors pickle
            name_dict = path_data_output +database+'res101_VOC07_' + '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            if obj_score_add_tanh:
                name_dict += 'SAdd'+str(lambdas)
            if obj_score_mul_tanh:
                name_dict += 'SMul'    
            name_dictW = name_dict + '.pkl'
            

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
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
            else:
                raise NotImplemented
            
            path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
            databasetxt =path_data + database + ext
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
#                print(database,'with ',num_trainval_im,' images in the trainval set')
            N = 1
            extL2 = ''
            nms_thresh = 0.7
            savedstr = '_all'
            
            sets = ['train','val','trainval','test']
            dict_name_file = {}
            data_precomputeed= True
            eval_onk300 = True
            if k_per_bag==300:
                k_per_bag_str = ''
            else:
                k_per_bag_str = '_k'+str(k_per_bag)
            for set_str in sets:
                name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
                if not(k_per_bag==300) and eval_onk300 and set_str=='test': # We will evaluate on all the 300 regions and not only the k_per_bag ones
                    name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
                dict_name_file[set_str] = name_pkl_all_features
        
        #    sLength_all = len(df_label[item_name])
            if demonet in ['vgg16_COCO','vgg16_VOC07','vgg16_VOC12']:
                num_features = 4096
            elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
                num_features = 2048
                   
            # Config param for TF session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  

            
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
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                imdb = get_imdb('WikiTenLabels_test')
                imdb.set_force_dont_use_07_metric(dont_use_07_metric)
                #num_images = len(imdb.image_index) 
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            else:
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            
                           
               
                
            with open(name_dictW, 'rb') as f:
                 Dict = pickle.load(f)
            Wstored = Dict['Wstored']
            Bstored =  Dict['Bstored']
            Lossstored = Dict['Lossstored']
            np_pos_value =  Dict['np_pos_value'] 
            np_neg_value =  Dict['np_neg_value']
    #            print(Wstored.shape)

            for AggregW in listAggregW:
                
                name_dictAP = name_dict  + '_' +str(AggregW)  + '_APscore.pkl'
                ReDo  =False
                if not os.path.isfile(name_dictAP) or ReDo:
#                    print('name_dictAP',name_dictAP)
#                    print('Wstored',Wstored.shape)
#                    print('numberofW_to_keep',numberofW_to_keep)
#                    input('wait')
                    DictAP = {}
                
                    ll = []
                    l01 = []
                    lclassif = []
                    ## Create the model
                    modelcreator = ModelHyperplan(norm='',AggregW=AggregW,epsilon=0.01,mini_batch_size=1000,num_features=num_features,num_rois=k_per_bag,num_classes=num_classes,
                         with_scores=with_scores,seuillage_by_score=seuillage_by_score,
                         proportionToKeep=proportionToKeep,restarts=numberofW_to_keep-1,seuil=seuil,
                         obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                         obj_score_mul_tanh=obj_score_mul_tanh)
                    class_indice = -1
                    ## Compute the best vectors 
                    for l in range(number_of_reboots): 
                        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
                        Wstored_extract = Wstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
                        W_tmp = np.reshape(Wstored_extract,(-1,num_features),order='F')
                        b_tmp =np.reshape( Bstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep],(-1,1,1),order='F')
                        Lossstoredextract = Lossstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep]
                        loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
                        ## Creation of the model
                        export_dir =  modelcreator.createIt(data_path,class_indice,W_tmp,b_tmp,loss_value)
                        number_zone = 300
                        Number_of_positif_elt = 1
                        dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
                        parameters=False,False,False,False
    #                    parameters=PlotRegions,RPN,Stocha,CompBest
                        param_clf = k_per_bag,1,num_features
    #                    param_clf = k_per_bag,Number_of_positif_elt,num_features
                        thresh_evaluation = 0.05
                        TEST_NMS = 0.3
                        predict_with= 'MI_max'
                        transform_output = 'tanh'
                        seuil_estimation= False
                        mini_batch_size = 1000
                        verbose = False
                        true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
                        ,all_boxes = \
                        tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
                               export_dir,dict_name_file,mini_batch_size,config,
                               path_to_img,path_data,param_clf,classes,parameters,verbose,
                               seuil_estimation,thresh_evaluation,TEST_NMS,all_boxes=all_boxes,
                               cachefile_model_base='',transform_output=transform_output,
                               scoreInMI_max=(with_scores or seuillage_by_score)
                               ,AggregW=AggregW,obj_score_add_tanh=obj_score_add_tanh,obj_score_mul_tanh=obj_score_mul_tanh)
                        
                        # Classification Perf
                        AP_per_class = []
                        for j,classe in enumerate(classes):
                            AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
                            AP_per_class += [AP]    
                        
                        # Detection Perf 
#                        det_file = os.path.join(path_data, 'detections_aux.pkl')
#                        with open(det_file, 'wb') as f:
#                            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
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
                        apsAt05 = aps
#                        print("Detection score (thres = 0.5): ",database)
#                        print(arrayToLatex(aps,per=True))
                        ovthresh_tab = [0.1]
                        for ovthresh in ovthresh_tab:
                            aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
                            if ovthresh == 0.1:
                                apsAt01 = aps
#                            print("Detection score with thres at ",ovthresh)
#                            print(arrayToLatex(aps,per=True))
#                        imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb 
#                        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
#                        print("Detection score with the difficult element")
#                        print(arrayToLatex(aps,per=True))
#                        imdb.set_use_diff(False)
                        
                        print(apsAt05,apsAt01,AP_per_class)
                        tf.reset_default_graph()
                        # aps ne contient pas le mean sur les classes en fait
                        ll += [apsAt05]
                        l01 += [apsAt01]
                        lclassif += [AP_per_class]
                    # End of the 100 experiment for a specific AggreW
                    ll_all = np.vstack(ll)
                    l01_all = np.vstack(l01)
                    apsClassif_all = np.vstack(lclassif)

                    DictAP['AP@.5'] =  ll_all
                    DictAP['AP@.1'] =  l01_all
                    DictAP['APClassif'] =  apsClassif_all
                
                    with open(name_dictAP, 'wb') as f:
                        pickle.dump(DictAP, f, pickle.HIGHEST_PROTOCOL)
                        
def VariationStudyPart2bis():
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    The second part compute the score in AP 
    
    In this script we only consider keeping one vector to see if picking the best among 12 improve results
    
    '''
    demonet = 'res152_COCO'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
#    database_tab = ['watercolor']
##    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt','watercolor']
    number_of_reboots = 100
    number_restarts = 100*12-1
    max_iters_all_base = 300
    k_per_bag = 300
    numberofW_to_keep = 1
#    numberofW_to_keep = 12
    
    dont_use_07_metric  =True
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    start_i = 0
    end_i = 12
    seuil = 0.9 
    
    data_path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    
#    listi = range(start_i,end_i+1) 
    listi = [0,5]
    
    for i in listi:
        print('Scenario :',i)
        listAggregW = [None]
        if i==0:
            listAggregW = [None]
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==1:
            listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==2:
            loss_type='hinge_tanh'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==3:
            loss_type='hinge'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = True
            seuillage_by_score=False
        elif i==4:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = False
            with_scores = True
            seuillage_by_score=False
        if i==5:
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
            WR = True
            with_scores = False
            seuillage_by_score=False
        elif i==6:
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=False
        elif i==7:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.9
        elif i==8:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.5
        elif i==9:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.3
        elif i==10:
            loss_type=''
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            WR = True
            with_scores = False
            seuillage_by_score=True
            seuil=0.1
       
        
        for database in database_tab:
            # Name of the vectors pickle
            name_dict = path_data_output +database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'
                
            name_dictW = name_dict + '.pkl'
            

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
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
            else:
                raise NotImplemented
            
            path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
            databasetxt =path_data + database + ext
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
#                print(database,'with ',num_trainval_im,' images in the trainval set')
            N = 1
            extL2 = ''
            nms_thresh = 0.7
            savedstr = '_all'
            
            sets = ['train','val','trainval','test']
            dict_name_file = {}
            data_precomputeed= True
            eval_onk300 = True
            if k_per_bag==300:
                k_per_bag_str = ''
            else:
                k_per_bag_str = '_k'+str(k_per_bag)
            for set_str in sets:
                name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
                if not(k_per_bag==300) and eval_onk300 and set_str=='test': # We will evaluate on all the 300 regions and not only the k_per_bag ones
                    name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
                dict_name_file[set_str] = name_pkl_all_features
        
        #    sLength_all = len(df_label[item_name])
            if demonet in ['vgg16_COCO','vgg16_VOC07','vgg16_VOC12']:
                num_features = 4096
            elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
                num_features = 2048
                   
            # Config param for TF session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  

            
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
            elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                imdb = get_imdb('WikiTenLabels_test')
                imdb.set_force_dont_use_07_metric(dont_use_07_metric)
                #num_images = len(imdb.image_index) 
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            else:
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            
                           
               
                
            with open(name_dictW, 'rb') as f:
                 Dict = pickle.load(f)
            Wstored = Dict['Wstored']
            Bstored =  Dict['Bstored']
            Lossstored = Dict['Lossstored']
            np_pos_value =  Dict['np_pos_value'] 
            np_neg_value =  Dict['np_neg_value']
    #            print(Wstored.shape)
            
            for AggregW in listAggregW:
                
                name_dictAP = name_dict  + '_' +str(AggregW)  + '_APscore_WithOneVector.pkl'
                # Here we modify the name 
                ReDo  =False
                if not os.path.isfile(name_dictAP) or ReDo:
                    
                    DictAP = {}
                
                    ll = []
                    l01 = []
                    lclassif = []
                    ## Create the model
                    modelcreator = ModelHyperplan(norm='',AggregW=AggregW,epsilon=0.01,mini_batch_size=1000,num_features=num_features,num_rois=k_per_bag,num_classes=num_classes,
                         with_scores=with_scores,seuillage_by_score=seuillage_by_score,proportionToKeep=proportionToKeep,restarts=numberofW_to_keep-1,seuil=seuil)
                    class_indice = -1
                    ## Compute the best vectors 
                    for l in range(number_of_reboots): 
                        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
                        Wstored_extract = Wstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
                        W_tmp = np.reshape(Wstored_extract,(-1,num_features),order='F')
                        b_tmp =np.reshape( Bstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep],(-1,1,1),order='F')
                        Lossstoredextract = Lossstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep]
                        loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
                        ## Creation of the model
                        export_dir =  modelcreator.createIt(data_path,class_indice,W_tmp.astype(np.float32),b_tmp.astype(np.float32),loss_value)
                        number_zone = 300
                        Number_of_positif_elt = 1
                        dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
                        parameters=False,False,False,False
    #                    parameters=PlotRegions,RPN,Stocha,CompBest
                        param_clf = k_per_bag,1,num_features
    #                    param_clf = k_per_bag,Number_of_positif_elt,num_features
                        thresh_evaluation = 0.05
                        TEST_NMS = 0.3
                        predict_with= 'MI_max'
                        transform_output = 'tanh'
                        seuil_estimation= False
                        mini_batch_size = 1000
                        verbose = False
                        true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
                        ,all_boxes = \
                        tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
                               export_dir,dict_name_file,mini_batch_size,config,
                               path_to_img,path_data,param_clf,classes,parameters,verbose,
                               seuil_estimation,thresh_evaluation,TEST_NMS,all_boxes=all_boxes,
                               cachefile_model_base='',transform_output=transform_output,
                               scoreInMI_max=(with_scores or seuillage_by_score)
                               ,AggregW=AggregW)
                        
                        # Classification Perf
                        AP_per_class = []
                        for j,classe in enumerate(classes):
                            AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
                            AP_per_class += [AP]    
                        
                        # Detection Perf 
#                        det_file = os.path.join(path_data, 'detections_aux.pkl')
#                        with open(det_file, 'wb') as f:
#                            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
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
                        apsAt05 = aps
#                        print("Detection score (thres = 0.5): ",database)
#                        print(arrayToLatex(aps,per=True))
                        ovthresh_tab = [0.1]
                        for ovthresh in ovthresh_tab:
                            aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
                            if ovthresh == 0.1:
                                apsAt01 = aps
#                            print("Detection score with thres at ",ovthresh)
#                            print(arrayToLatex(aps,per=True))
#                        imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb 
#                        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
#                        print("Detection score with the difficult element")
#                        print(arrayToLatex(aps,per=True))
#                        imdb.set_use_diff(False)
                        
                        print(apsAt05,apsAt01,AP_per_class)
                        tf.reset_default_graph()
                        # aps ne contient pas le mean sur les classes en fait
                        ll += [apsAt05]
                        l01 += [apsAt01]
                        lclassif += [AP_per_class]
                    # End of the 100 experiment for a specific AggreW
                    ll_all = np.vstack(ll)
                    l01_all = np.vstack(l01)
                    apsClassif_all = np.vstack(lclassif)

                    DictAP['AP@.5'] =  ll_all
                    DictAP['AP@.1'] =  l01_all
                    DictAP['APClassif'] =  apsClassif_all
                
                    with open(name_dictAP, 'wb') as f:
                        pickle.dump(DictAP, f, pickle.HIGHEST_PROTOCOL)
      
def VariationStudyPart3(database=None,scenarioSubset=None,demonet = 'res152_COCO',onlyAP05=False,
                        withoutAggregW=False,k_per_bag=300,layer='fc7'):
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    The third part print the results 
    '''
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    if database is None:
        database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
    else:
        database_tab = [database]
    start_i = 0
    end_i = 19
    if scenarioSubset is None:
        listi = np.arange(start_i,end_i)
    # Just to have with and without score : scenario 0 and 5 
    else:
        listi = scenarioSubset
#    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt']
    number_restarts = 100*12-1
    max_iters_all_base = 300
    k_per_bag = 300
    numberofW_to_keep = 12
    
    dont_use_07_metric  =True
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    seuil = 0.9 
    listAggregW = [None,'maxOfTanh','meanOfTanh','minOfTanh','AveragingW']
    data_path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    
    for database  in database_tab:
        print('--------------------------------')
        print(database,demonet)
        for i_scenario in listi:
            output = get_params_fromi_scenario(i_scenario)
            listAggregW,C_Searching,CV_Mode,AggregW,proportionToKeepTab,loss_type,WR,\
            with_scores,seuillage_by_score,obj_score_add_tanh,lambdas,obj_score_mul_tanh,\
            PCAuse,Max_version,max_iters_all_base = output
            
            if withoutAggregW:
                listAggregW  = [None]

            name_dict = path_data_output 
            if not(demonet== 'res152_COCO'):
                name_dict += demonet +'_'
            if not(layer== 'fc7'):
                name_dict += '_'+demonet
            name_dict +=  database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'
            if seuillage_by_score:
                name_dict += 'SC'+str(seuil)
            if obj_score_add_tanh:
                name_dict += 'SAdd'+str(lambdas)
            if obj_score_mul_tanh:
                name_dict += 'SMul'    
            if PCAuse:
                name_dict +='_PCA09'
            if not(k_per_bag==300):
                name_dict +='_k'+str(k_per_bag)
            if not(Max_version=='' or Max_version is None) :
                name_dict +='_'+Max_version
            if not(max_iters_all_base==300) :
                name_dict +='_MIAB'+str(max_iters_all_base)

            for AggregW in listAggregW:
                if AggregW is None or AggregW=='':
                    proportionToKeepTabLocal = [0.]
                    name_dictAP = name_dict  + '_' +str(AggregW)  + '_APscore.pkl'
                else:
                    proportionToKeepTabLocal = proportionToKeepTab
                for proportionToKeep in proportionToKeepTabLocal:
                    name_dictAP = name_dict  + '_' +str(AggregW) 
                    if not(AggregW is None or AggregW==''):
                        name_dictAP += '_'+str(proportionToKeep)+ '_APscore.pkl'
                    multi = 100
                    try:
                        f= open(name_dictAP, 'rb')
                        print(name_dictAP)
                        DictAP = pickle.load(f)
                        for Metric in DictAP.keys():
                            string_to_print =  str(Metric) + ' & ' +'Mimax ' + str(loss_type) + ' ' 
                            if C_Searching:
                                string_to_print += 'C_Searching '
                            if CV_Mode=='CV':
                                string_to_print += 'CV '
                            if with_scores:
                                string_to_print += 'with_scores '
                            if not(WR):
                                string_to_print += 'with regularisation'
                            if seuillage_by_score:
                                string_to_print += 'seuillage score at ' +str(seuil)
                            if obj_score_mul_tanh :
                                string_to_print += 'obj score multi tanh '
                            if obj_score_add_tanh:
                                string_to_print += 'obj score add to tanh (lambda =' +str(lambdas)+')'
                            if not(Max_version=='' or Max_version is None) :
                                string_to_print += ' '+Max_version
                            string_to_print += ' & '
                            string_to_print += str(AggregW)
                            if AggregW is None or AggregW=='':
                                string_to_print += ' & '  
                            else:
                                 string_to_print +=  ' '+str(proportionToKeep)  +' & ' 
                            ll_all = DictAP[Metric] 
                            if database=='WikiTenLabels':
                                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
                            if not(database=='PeopleArt'):
                                mean_over_reboot = np.mean(ll_all,axis=1) # Moyenne par ligne / reboot 
    #                            print(mean_over_reboot.shape)
                                std_of_mean_over_reboot = np.std(mean_over_reboot)
                                mean_of_mean_over_reboot = np.mean(mean_over_reboot)
                                mean_over_class = np.mean(ll_all,axis=0) # Moyenne par column
                                std_over_class = np.std(ll_all,axis=0) # Moyenne par column 
    #                            print('ll_all.shape',ll_all.shape)
    #                            print(mean_over_class.shape)
    #                            print(std_over_class.shape)
    #                            input('wait')
                                for mean_c,std_c in zip(mean_over_class,std_over_class):
                                    s =  "{0:.1f} ".format(mean_c*multi) + ' $\pm$ ' +  "{0:.1f}".format(std_c*multi)
                                    string_to_print += s + ' & '
                                s =  "{0:.1f}  ".format(mean_of_mean_over_reboot*multi) + ' $\pm$ ' +  "{0:.1f}  ".format(std_of_mean_over_reboot*multi)
                                string_to_print += s + ' \\\  '
                            else:
                                std_of_mean_over_reboot = np.std(ll_all)
                                mean_of_mean_over_reboot = np.mean(ll_all)
                                s =  "{0:.1f} ".format(mean_of_mean_over_reboot*multi) + ' $\pm$ ' +  "{0:.1f} ".format(std_of_mean_over_reboot*multi)
                                string_to_print += s + ' \\\ '
                            string_to_print = string_to_print.replace('_','\_')
                            if not(onlyAP05):
                                print(string_to_print)
                            elif Metric=='AP@.5':
                                print(string_to_print)
                    except FileNotFoundError:
                        print(name_dictAP,'don t exist')
                    pass
    
def VariationStudyPart3bis():
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    The third part print the results 
    '''
    demonet = 'res152_COCO'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
#    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt']
    number_restarts = 100*12-1
    max_iters_all_base = 300
    k_per_bag = 300
    numberofW_to_keep = 12
    
    dont_use_07_metric  =True
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    start_i = 0
    end_i = 12
    seuil = 0.9 
    listAggregW = [None,'maxOfTanh','meanOfTanh','minOfTanh','AveragingW']
    data_path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    listi = [0,4]
    for database  in database_tab:
        print('--------------------------------')
        print(database)
        listi = [0,5]
    
        for i in listi:
            print('Scenario :',i)
            listAggregW = [None]
            if i==0:
                listAggregW = [None]
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                loss_type = ''
                WR = True
                with_scores = True
                seuillage_by_score=False
            elif i==1:
                listAggregW = ['maxOfTanh',None,'meanOfTanh','minOfTanh','AveragingW']
                loss_type='MSE'
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = True
                seuillage_by_score=False
            elif i==2:
                loss_type='hinge_tanh'
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = True
                seuillage_by_score=False
            elif i==3:
                loss_type='hinge'
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = True
                seuillage_by_score=False
            elif i==4:
                loss_type=''
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = False
                with_scores = True
                seuillage_by_score=False
            if i==5:
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                loss_type = ''
                WR = True
                with_scores = False
                seuillage_by_score=False
            elif i==6:
                loss_type='MSE'
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = False
                seuillage_by_score=False
            elif i==7:
                loss_type=''
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = False
                seuillage_by_score=True
                seuil=0.9
            elif i==8:
                loss_type=''
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = False
                seuillage_by_score=True
                seuil=0.5
            elif i==9:
                loss_type=''
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = False
                seuillage_by_score=True
                seuil=0.3
            elif i==10:
                loss_type=''
                C_Searching = False
                CV_Mode = ''
                AggregW = None
                proportionToKeep = [0.25,1.0]
                WR = True
                with_scores = False
                seuillage_by_score=True
                seuil=0.1        
                    
                
            name_dict = path_data_output +database+ '_Wvectors_C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+'_'+str(loss_type)
            if not(WR):
                name_dict += '_withRegularisationTermInLoss'
            if with_scores:
                 name_dict += '_WithScore'

            for AggregW in listAggregW:
                name_dictAP = name_dict +  '_' + str(AggregW)+ '_APscore_WithOneVector.pkl'
                multi = 100
                try:
                    f= open(name_dictAP, 'rb')
                    DictAP = pickle.load(f)
                    for Metric in DictAP.keys():
                        
                        string_to_print =  str(Metric) + ' & ' +'Mimax 1Vect ' + str(loss_type) + ' ' 
                        if C_Searching:
                            string_to_print += 'C_Searching '
                        if CV_Mode=='CV':
                            string_to_print += 'CV '
                        if with_scores:
                            string_to_print += 'with_scores '
                        if not(WR):
                            string_to_print += 'with regularisation'
                        if seuillage_by_score:
                            string_to_print += 'seuillage score at ' +str(seuil)
                        string_to_print += ' & '
                        string_to_print += str(AggregW) + ' & '  
                        ll_all = DictAP[Metric] 
                        if database=='WikiTenLabels':
                            ll_all = np.delete(ll_all, [1,2,9], axis=1)         
                        if not(database=='PeopleArt'):
                            mean_over_reboot = np.mean(ll_all,axis=1) # Moyenne par ligne / reboot 
#                            print(mean_over_reboot.shape)
                            std_of_mean_over_reboot = np.std(mean_over_reboot)
                            mean_of_mean_over_reboot = np.mean(mean_over_reboot)
                            mean_over_class = np.mean(ll_all,axis=0) # Moyenne par column
                            std_over_class = np.std(ll_all,axis=0) # Moyenne par column 
#                            print('ll_all.shape',ll_all.shape)
#                            print(mean_over_class.shape)
#                            print(std_over_class.shape)
#                            input('wait')
                            for mean_c,std_c in zip(mean_over_class,std_over_class):
                                s =  "{0:.1f} ".format(mean_c*multi) + ' $\pm$ ' +  "{0:.1f}".format(std_c*multi)
                                string_to_print += s + ' & '
                            s =  "{0:.1f}  ".format(mean_of_mean_over_reboot*multi) + ' $\pm$ ' +  "{0:.1f}  ".format(std_of_mean_over_reboot*multi)
                            string_to_print += s + ' \\\  '
                        else:
                            std_of_mean_over_reboot = np.std(ll_all)
                            mean_of_mean_over_reboot = np.mean(ll_all)
                            s =  "{0:.1f} ".format(mean_of_mean_over_reboot*multi) + ' $\pm$ ' +  "{0:.1f} ".format(std_of_mean_over_reboot*multi)
                            string_to_print += s + ' \\\ '
                        string_to_print = string_to_print.replace('_','\_')
                        print(string_to_print)

                except FileNotFoundError:
                    #print(name_dictAP,'don t exist')
                    pass    
        
        
        
        
def VariationStudy():
    '''
    The goal of this function is to study the variation of the performance of our 
    method
    '''
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    database_tab = ['PeopleArt','watercolor','WikiTenLabels','VOC2007']
#    database_tab = ['VOC2007','PeopleArt']
#    database_tab = ['PeopleArt','watercolor']
    number_restarts = 100
    max_iters_all_base = 300
    
    Dict = {}
    metric_tab = ['AP@.5','AP@.1','APClassif']
    
    for i in range(0,9):
        print('Scenario :',i)
        if i==0:
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
            loss_type = ''
        elif i==1:
            loss_type='MSE'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
        elif i==2:
            loss_type='hinge_tanh'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
        elif i==3:
            loss_type='hinge'
            C_Searching = False
            CV_Mode = ''
            AggregW = None
            proportionToKeep = [0.25,1.0]
        elif i==4:
            loss_type = ''
            C_Searching = False
            CV_Mode = ''
            AggregW = 'maxOfTanh'
            proportionToKeep = [0.25,1.0]
        elif i==5:
            loss_type = ''
            C_Searching = False
            CV_Mode = ''
            AggregW = 'meanOfTanh'
            proportionToKeep = [0.25,1.0]
        elif i==6:
            loss_type = ''
            C_Searching = False
            CV_Mode = ''
            AggregW = 'AveragingW'
            proportionToKeep = [0.25,1.0]
        elif i==7:
            loss_type = ''
            C_Searching = False
            CV_Mode = 'CVforCsearch'
            AggregW = None
            proportionToKeep = [0.25,1.0]
        elif i==8:
            loss_type = ''
            C_Searching = True
            CV_Mode = 'CV' 
            AggregW = None
            proportionToKeep = [0.25,1.0]
    
        for database in database_tab:
            ll = []
            l01 = []
            lclassif = []
            for i in range(number_restarts):
                aps,aps010,apsClassif = tfR_FRCNN(demonet = 'res152_COCO',database = database,ReDo=True,
                                          verbose = False,testMode = False,jtest = 'cow',loss_type=loss_type,
                                          PlotRegions = False,saved_clf=False,RPN=False,
                                          CompBest=False,Stocha=True,k_per_bag=300,
                                          parallel_op=True,CV_Mode=CV_Mode,num_split=2,
                                          WR=True,init_by_mean =None,seuil_estimation='',
                                          restarts=11,max_iters_all_base=max_iters_all_base,LR=0.01,with_tanh=True,
                                          C=1.0,Optimizer='GradientDescent',norm='',
                                          transform_output='tanh',with_rois_scores_atEnd=False,
                                          with_scores=True,epsilon=0.01,restarts_paral='paral',
                                          Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
                                          k_intopk=1,C_Searching=C_Searching,predict_with='MI_max',
                                          gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                          thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=AggregW,proportionToKeep=proportionToKeep)
                tf.reset_default_graph()
                # aps ne contient pas le mean sur les classes en fait
                ll += [aps]
                l01 += [aps010]
                lclassif += [apsClassif]
                
            ll_all = np.vstack(ll)
            l01_all = np.vstack(l01)
            apsClassif_all = np.vstack(lclassif)
            
            
            Dict[database] = {}
            Dict[database]['AP@.5'] =  ll_all
            Dict[database]['AP@.1'] =  l01_all
            Dict[database]['APClassif'] =  apsClassif_all
#    
#            if database=='WikiTenLabels':
#                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
#            if not(database=='PeopleArt'):
#                ll_all_mean = np.mean(ll_all,axis=0) # Moyenne par ligne / reboot
#            else:
#                ll_all_mean = ll_all.ravel()
#            std_ll_all_mean= np.std(ll_all_mean)
#            print('~~~~ !! ~~~~ ')
#            print(database,'with ',number_restarts,' reboot of the method')
#            print('For IuO >= 0.5')
#            print(ll_all_mean)
#            print(arrayToLatex(ll_all_mean,per=True))
#            print('std of the mean of mean :',100*std_ll_all_mean)
#            
#            print('~~~~')
#            print('All data :')
#            print(ll_all)
#            print('Mean :',np.mean(ll_all,axis=0))
#            print('Min :',np.min(ll_all,axis=0))
#            print('Max :',np.max(ll_all,axis=0))
#            print('Std :',np.std(ll_all,axis=0))
#            print('~~~~')
#            
#            ll_all = l01_all
#            if database=='WikiTenLabels':
#                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
#            if database=='WikiTenLabels':
#                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
#            if not(database=='PeopleArt'):
#                ll_all_mean = np.mean(ll_all,axis=0) # Moyenne par ligne / reboot
#            else:
#                ll_all_mean =ll_all.ravel()
#            std_ll_all_mean= np.std(ll_all_mean)
#            print('~~~~ !! ~~~~ ')
#            print('For IuO >= 0.1')
#            print(ll_all_mean)
#            print(arrayToLatex(ll_all_mean,per=True))
#            print('std of the mean of mean :',100*std_ll_all_mean)
#            print('~~~~')
#            print('All data :')
#            print(ll_all)
#            print('Mean :',np.mean(ll_all,axis=0))
#            print('Min :',np.min(ll_all,axis=0))
#            print('Max :',np.max(ll_all,axis=0))
#            print('Std :',np.std(ll_all,axis=0))
#            print('~~~~')
#            
#            ll_all = apsClassif_all
#            if database=='WikiTenLabels':
#                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
#            if database=='WikiTenLabels':
#                ll_all = np.delete(ll_all, [1,2,9], axis=1)         
#            if not(database=='PeopleArt'):
#                ll_all_mean = np.mean(ll_all,axis=0) # Moyenne par ligne / reboot
#            else:
#                ll_all_mean = ll_all.ravel()
#            std_ll_all_mean= np.std(ll_all_mean)
#            print('~~~~ !! ~~~~ ')
#            print('For Classification')
#            print(ll_all_mean)
#            print(arrayToLatex(ll_all_mean,per=True))
#            print('std of the mean of mean :',100*std_ll_all_mean)
#            print('~~~~')
#            print('All data :')
#            print(ll_all)
#            print('Mean :',np.mean(ll_all,axis=0))
#            print('Min :',np.min(ll_all,axis=0))
#            print('Max :',np.max(ll_all,axis=0))
#            print('Std :',np.std(ll_all,axis=0))
#            print('~~~~')
        name_dict = path_data_output + 'C_Searching'+str(C_Searching) + '_' +\
            CV_Mode+ str(AggregW) +str(proportionToKeep)+'_'+str(loss_type)+'.pkl'
        with open(name_dict, 'wb') as f:
            pickle.dump(Dict, f, pickle.HIGHEST_PROTOCOL)
        namebis = path_data_output + str(time.time()) + '.pkl'
        with open(namebis, 'wb') as f:
            pickle.dump(Dict, f, pickle.HIGHEST_PROTOCOL)
        print("Saved")  
#        
#        with open(name_dict, 'rb') as f:
#            Dict_loaded = pickle.load(f)
#            ll_all = Dict_loaded[database]['AP@.5']
#            print('Test mode')
#            print(ll_all)
#            ll_all_mean = np.mean(ll_all,axis=0)
#            ll_all_std = np.std(ll_all,axis=0)
#            mean_class_run = np.mean(ll_all,axis=1)
#            mean_mean_class_run = np.mean(mean_class_run)
#            std_mean_class_run = np.std(mean_class_run)
#            print(ll_all_mean)
#            print(ll_all_std)
#            print(mean_mean_class_run)
#            print(std_mean_class_run)
        
        
if __name__ == '__main__':
    #    VariationStudyPart1_forVOC07()
#    VariationStudyPart2_forVOC07()
    # Il faudra faire le part3 pour VOC07
#    VariationStudyPart1(database='IconArt_v1',scenarioSubset=[21])
#    VariationStudyPart2(database='IconArt_v1',scenarioSubset=[3,22,20,21],withoutAggregW=True)
#    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[3,22,20,21],withoutAggregW=True)
##    
    VariationStudyPart1(database='IconArt_v1',scenarioSubset=[0,5],demonet = 'vgg16_COCO')
    VariationStudyPart2(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO')
    VariationStudyPart1(database='IconArt_v1',scenarioSubset=[0,5],demonet = 'vgg16_COCO',layer='fc6')
    VariationStudyPart2(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO',layer='fc6')
    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO')
    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO',layer='fc6')
    #unefficient_way_MaxOfMax_evaluation()
#    # For Watercolor2k 
#    VariationStudyPart1(database='watercolor',scenarioSubset=[0,5,17,18])
#    VariationStudyPart2(database='watercolor',scenarioSubset=[0,5,17,18],withoutAggregW=True)
#    
#    # For PeopleArt
#    VariationStudyPart1(database='PeopleArt',scenarioSubset=[0,5])
#    VariationStudyPart2(database='PeopleArt',scenarioSubset=[0,5],withoutAggregW=True)
#    VariationStudyPart1(database='PeopleArt',scenarioSubset=[0,5],demonet = 'res101_VOC07')
#    VariationStudyPart2(database='PeopleArt',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'res101_VOC07')
    
#    # PASCAL
#    VariationStudyPart1(database='VOC2007',scenarioSubset=[5])
#    VariationStudyPart2(database='VOC2007',scenarioSubset=[5],withoutAggregW=True)
#    VariationStudyPart1(database='VOC2007',scenarioSubset=[5],demonet = 'res101_VOC07')
#    VariationStudyPart2(database='VOC2007',scenarioSubset=[5],withoutAggregW=True,demonet = 'res101_VOC07')
#    VariationStudyPart3(database='VOC2007',scenarioSubset=[5],withoutAggregW=True,demonet = 'res101_VOC07')
    
#    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5,17,18,11,12],withoutAggregW=True)
#    VariationStudyPart3(database='watercolor',scenarioSubset=[0,5,17,18],withoutAggregW=True)
#    VariationStudyPart3(database='PeopleArt',scenarioSubset=[0,5],withoutAggregW=True)
#    VariationStudyPart3(database='PeopleArt',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'res101_VOC07')
#    VariationStudyPart3(database='VOC2007',scenarioSubset=[0,5],withoutAggregW=True)
#    VariationStudyPart3(database='VOC2007',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'res101_VOC07')
#    
#    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5,17,18,11,12])
#    VariationStudyPart3(database='watercolor',scenarioSubset=[0,5,17,18,11,12])
#    VariationStudyPart3(database='watercolor',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'res101_VOC07')
#     VariationStudyPart3(demonet = 'res101_VOC07')
#    VariationStudyPart1()
##    VariationStudyPart2bis()
#    VariationStudyPart2()

#    VariationStudyPart3(onlyAP05=True)
#    VariationStudyPart3bis()
#    ComputationForLossPlot()
#    VariationStudyPart1_forVOC07()