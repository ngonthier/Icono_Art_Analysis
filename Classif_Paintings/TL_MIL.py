#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:45:35 2018

Script pour realiser du transfert d'apprentissage a partir de Faster RCNN

Il faut rajouter l elaboration de probabilite :
    https://stats.stackexchange.com/questions/55072/svm-confidence-according-to-distance-from-hyperline

Page utile sur VOC 2007 :
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html

@author: gonthier
"""

import time

import pickle
import gc
import tensorflow as tf
import csv
#from tensorflow.python.saved_model import tag_constants
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import IsolationForest
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.covariance import EllipticEnvelope
#from sklearn.linear_model import SGDClassifier
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
#from tf_faster_rcnn.lib.model.test import im_detect,TL_im_detect,TL_im_detect_end,get_blobs
from tf_faster_rcnn.lib.model.test import TL_im_detect,get_blobs
from tf_faster_rcnn.lib.model.nms_wrapper import nms
#from tf_faster_rcnn.lib.nms.py_cpu_nms import py_cpu_nms
import matplotlib.pyplot as plt
#from sklearn.svm import LinearSVC, SVC
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from nltk.classify.scikitlearn import SklearnClassifier
#from tf_faster_rcnn.tools.demo import vis_detections
import numpy as np
import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score,recall_score,precision_score,f1_score
from Custom_Metrics import ranking_precision_score
#from Classifier_Evaluation import Classification_evaluation
import os.path
import misvm # Library to do Multi Instance Learning with SVM
from sklearn.preprocessing import StandardScaler
from trouver_classes_parmi_K import MI_max,TrainClassif,tf_MI_max #ModelHyperplan
from trouver_classes_parmi_K_mi import tf_mi_model
from LatexOuput import arrayToLatex
from FasterRCNN import vis_detections_list,vis_detections,Compute_Faster_RCNN_features,\
    vis_GT_list,Save_TFRecords_PCA_features
from CNNfeatures import Compute_EdgeBoxesAndCNN_features
import pathlib
from milsvm import mi_linearsvm # Version de nicolas avec LinearSVC et TODO SGD 
from sklearn.externals import joblib # To save the classifier
from tool_on_Regions import reduce_to_k_regions
#from sklearn import linear_model
from tf_faster_rcnn.lib.datasets.factory import get_imdb
from Estimation_Param import kde_sklearn,findIntersection
from utils.save_param import create_param_id_file_and_dir,write_results,tabs_to_str
from Transform_Box import py_cpu_modif
#from hpsklearn import HyperoptEstimator,sgd
#from hyperopt import tpe
from random import uniform
#from shutil import copyfile
from IMDB import get_database

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

def parser_w_mei_reduce(record,num_rois=300,num_features=2048):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'score_mei': tf.FixedLenFeature([1], tf.float32),
                'mei': tf.FixedLenFeature([1], tf.int64),
                'rois': tf.FixedLenFeature([num_rois*5],tf.float32),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'fc7_selected': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([1],tf.float32),
                'name_img' : tf.FixedLenFeature([],tf.string)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
    label_300 = tf.tile(label,[num_rois])
    fc7_selected = parsed['fc7_selected']
    fc7_selected = tf.reshape(fc7_selected, [num_rois,num_features])         
    return fc7_selected,label_300

def parser_w_rois(record,classe_index=0,num_classes=10,num_rois=300,num_features=2048,
                  dim_rois=5):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'num_regions':  tf.FixedLenFeature([], tf.int64),
                'num_features':  tf.FixedLenFeature([], tf.int64),
                'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_classes],tf.float32),
                'name_img' : tf.FixedLenFeature([],tf.string)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
    name_img = parsed['name_img']
    label = tf.slice(label,[classe_index],[1])
    label = tf.squeeze(label) # To get a vector one dimension
    fc7 = parsed['fc7']
    fc7 = tf.reshape(fc7, [num_rois,num_features])
    rois = parsed['rois']
    rois = tf.reshape(rois, [num_rois,dim_rois])           
    return fc7,rois, label,name_img

def parser_w_rois_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            with_rois_scores=False,dim_rois=5):
        # Perform additional preprocessing on the parsed data.
        if not(with_rois_scores):
            keys_to_features={
                        'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}
        else:
            keys_to_features={
                        'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                        'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}
#        keys_to_features={
#                    'height': tf.FixedLenFeature([], tf.int64),
#                    'width': tf.FixedLenFeature([], tf.int64),
#                    'num_regions':  tf.FixedLenFeature([], tf.int64),
#                    'num_features':  tf.FixedLenFeature([], tf.int64),
#                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
#                    'rois': tf.FixedLenFeature([5*num_rois],tf.float32),
#                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
#                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
#                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
#                    'name_img' : tf.FixedLenFeature([],tf.string)}
            
        parsed = tf.parse_single_example(record, keys_to_features)
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [num_rois,num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [num_rois,dim_rois])    
        if not(with_rois_scores):
            return fc7,rois, label,name_img
        else:
            roi_scores = parsed['roi_scores'] 
            return fc7,rois,roi_scores,label,name_img
        
def parser_all_elt_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            dim_rois=5,noReshape=True):
    keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([dim_rois*num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
            
    parsed = tf.parse_single_example(record, keys_to_features)
      # Cast label data into int32
    list_elt = []
    for key in keys_to_features.keys():
          list_elt += [parsed[key]]
          print(key,parsed[key])
    if not(noReshape):
            list_elt[7] = tf.reshape(list_elt[7], [num_rois,num_features])
            list_elt[5] = tf.reshape(list_elt[5], [num_rois,dim_rois])  
        
    return(list_elt)


def rand_convex(n):
    rand = np.matrix([uniform(0.0, 1.0) for i in range(n)])
    return(rand / np.sum(rand))
        
def petitTestIllustratif():
    """
    We will try on 20 image from the Art UK Your paintings database and see what 
    we get as best zone with the MI_max de Said 
    """
    path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
    path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path + database + '.txt'
#    df_label = pd.read_csv(databasetxt,sep=",")
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}
    demonet = 'res152_COCO'
        #demonet = 'res101_COCO'
    tf.reset_default_graph() # Needed to use different nets one after the other
    print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32] # we  use  3  aspect  ratios  and  4  scales (adding 64**2)
    nbClasses = len(CLASSES)
    path_to_model = '/media/gonthier/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
    if 'vgg' in demonet:
        size_output = 4096
    elif 'res' in demonet :
        size_output = 2048
      
    path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
    symway = True
    if symway:
        path_to_output = '/media/gonthier/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MI_max/'
    else:
        path_to_output = '/media/gonthier/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MI_max_NotSymWay/'
    list_dog = ['ny_yag_yorag_326_624x544', 'dur_dbm_770_624x544', 'ntii_skh_1196043_624x544', 'nti_ldk_884912_624x544', 'syo_bha_90009742_624x544', 'tate_tate_t00888_10_624x544', 'ntii_lyp_500458_624x544', 'ny_yag_yorag_37_b_624x544', 'ngs_ngs_ng_1193_f_624x544', 'dur_dbm_533_624x544']
    list_not_dog = ['nid_qub_qub_264_624x544', 'gmiii_mosi_a1978_72_3_624x544', 'ny_nrm_1979_7964_624x544', 'che_crhc_pcf40_624x544', 'not_ntmag_1997_31_624x544', 'stf_strm_832_624x544', 'ny_nrm_1986_9418_624x544', 'ny_nrm_2004_7349_624x544', 'ny_nrm_1986_9421_624x544', 'ny_nrm_1996_7374_624x544', 'llr_rlrh_l_h38_1988_3_0_624x544', 'iwm_iwm_ld_5509_624x544', 'ny_nrm_1977_5834_624x544', 'cw_mte_45_624x544', 'ny_yam_260367_624x544', 'lne_rafm_fa03538_624x544', 'dur_dbm_769_624x544', 'ny_yag_yorag_66_624x544', 'lw_narm_131900_624x544', 'syo_cg_cp_tr_156_624x544']
    list_nms_thresh = [0.0,0.1,0.5,0.7]
    nms_thresh = list_nms_thresh[0]
    plt.ion()
    k_tab = [5,30,150,300]
    for k in k_tab:
        for nms_thresh in list_nms_thresh:
            sess = tf.Session(config=tfconfig)
            print("nms_thresh",nms_thresh,"k_per_bag",k)
            net.create_architecture("TEST", nbClasses,
                                          tag='default', anchor_scales=anchor_scales,
                                          modeTL= True,nms_thresh=nms_thresh)
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
            i=0
            
            k_per_bag = k
            k_max = 0
                
            pos_ex = np.zeros((len(list_dog),k_per_bag,size_output))
            neg_ex = np.zeros((len(list_not_dog),k_per_bag,size_output))
            
            dict_rois = {}
            dict_rois_score = {}
            
            for i,name_img in  enumerate(list_dog):
                #print(i,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)  # This call net.TL_image 
                
                k_max = np.max((k_max,len(fc7)))
                if(len(fc7) >= k_per_bag):
                    bag = np.expand_dims(fc7[0:k_per_bag,:],axis=0)
                else:
                    number_repeat = k_per_bag // len(fc7)  +1
                    f_repeat = np.repeat(fc7,number_repeat,axis=0)
                    bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
                pos_ex[i,:,:] = bag
                dict_rois[name_img] = rois
                dict_rois_score[name_img] = roi_scores
                
            for i,name_img in  enumerate(list_not_dog):
                #print(i,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)  # This call net.TL_image 
                k_max = np.max((k_max,len(fc7)))
                if(len(fc7) >= k_per_bag):
                    bag = np.expand_dims(fc7[0:k_per_bag,:],axis=0)
                else:
                    number_repeat = k_per_bag // len(fc7)  +1
                    f_repeat = np.repeat(fc7,number_repeat,axis=0)
                    bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
                neg_ex[i,:,:] = bag
                dict_rois[name_img] = rois
            
            print("k_max",k_max)
            
            tf.reset_default_graph()
            sess.close()
            
            # Train the MIL SVM 
            restarts = 20
            max_iters = 300
            classifierMI_max = MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                          max_iters=max_iters,symway=symway,n_jobs=-1,
                                          all_notpos_inNeg=False,gridSearch=False,
                                          verbose=False,final_clf='None')     
            classifierMI_max.fit(pos_ex, neg_ex)
            
            PositiveRegions = classifierMI_max.get_PositiveRegions()
            get_PositiveRegionsScore = classifierMI_max.get_PositiveRegionsScore()
        
            # Draw 
            for i,name_img in  enumerate(list_dog):
                #print(i,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                blobs, im_scales = get_blobs(im)
                rois = dict_rois[name_img]
                roi_with_object_of_the_class = PositiveRegions[i] % len(rois) # Because we have repeated some rois
                roi = rois[roi_with_object_of_the_class,:]
                roi_boxes =  roi[1:5] / im_scales[0]
                best_RPN_roi = rois[0,:]
                best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                best_RPN_roi_scores = [get_PositiveRegionsScore[0]]
                cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
        #            print(roi_boxes)
                
                roi_scores = [get_PositiveRegionsScore[i]]
        #            print(roi_scores)
                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MI_maxbestROI.jpg'
                plt.savefig(name_output)
            plt.close('all') 
            
def petitTestIllustratif_RefineRegions():
    """
    We will try on 20 image from the Art UK Your paintings database and see what 
    we get as best zone with the MI_max de Said 
    in this function we try to refine regions, ie remove not important regions
    """
    path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
    path = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}
    demonet = 'res152_COCO'
        #demonet = 'res101_COCO'
    tf.reset_default_graph() # Needed to use different nets one after the other
    print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32] # we  use  3  aspect  ratios  and  4  scales (adding 64**2)
    nbClasses = len(CLASSES)
    path_to_model = '/media/gonthier/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
      
    path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
    symway = True
    if symway:
        path_to_output = '/media/gonthier/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MI_max_Refine/'
    else:
        path_to_output = '/media/gonthier/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MI_max_NotSymWay_Refine/'
    list_dog = ['ny_yag_yorag_326_624x544', 'dur_dbm_770_624x544', 'ntii_skh_1196043_624x544', 'nti_ldk_884912_624x544', 'syo_bha_90009742_624x544', 'tate_tate_t00888_10_624x544', 'ntii_lyp_500458_624x544', 'ny_yag_yorag_37_b_624x544', 'ngs_ngs_ng_1193_f_624x544', 'dur_dbm_533_624x544']
    list_not_dog = ['nid_qub_qub_264_624x544', 'gmiii_mosi_a1978_72_3_624x544', 'ny_nrm_1979_7964_624x544', 'che_crhc_pcf40_624x544', 'not_ntmag_1997_31_624x544', 'stf_strm_832_624x544', 'ny_nrm_1986_9418_624x544', 'ny_nrm_2004_7349_624x544', 'ny_nrm_1986_9421_624x544', 'ny_nrm_1996_7374_624x544', 'llr_rlrh_l_h38_1988_3_0_624x544', 'iwm_iwm_ld_5509_624x544', 'ny_nrm_1977_5834_624x544', 'cw_mte_45_624x544', 'ny_yam_260367_624x544', 'lne_rafm_fa03538_624x544', 'dur_dbm_769_624x544', 'ny_yag_yorag_66_624x544', 'lw_narm_131900_624x544', 'syo_cg_cp_tr_156_624x544']
    
            
    # Now we try to quickly refine the considered regions 
    nms_thresh = 0.7
    k = 30
    new_nms_thresh = 0.0
    score_threshold = 0.1
    minimal_surface = 36*36
    sess = tf.Session(config=tfconfig)
    print("nms_thresh",nms_thresh,"k_per_bag",k)
    net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,
                                  modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    i=0
    
    k_per_bag = k
    k_max = 0
        
    pos_ex = np.zeros((len(list_dog),k_per_bag,size_output))
    neg_ex = np.zeros((len(list_not_dog),k_per_bag,size_output))
    
    dict_rois = {}
    dict_rois_score = {}
    
    for i,name_img in  enumerate(list_dog):
        print(i,name_img)
        complet_name = path_to_img + name_img + '.jpg'
        im = cv2.imread(complet_name)
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)  # This call net.TL_image 
        
        rois,roi_scores, fc7= reduce_to_k_regions(k,rois,roi_scores, fc7,
                                                  new_nms_thresh,score_threshold,
                                                  minimal_surface)
        
        print(len(fc7))
        
        k_max = np.max((k_max,len(fc7)))
        if(len(fc7) >= k_per_bag):
            bag = np.expand_dims(fc7[0:k_per_bag,:],axis=0)
        else:
            number_repeat = k_per_bag // len(fc7)  +1
            f_repeat = np.repeat(fc7,number_repeat,axis=0)
            bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
        pos_ex[i,:,:] = bag
        dict_rois[name_img] = rois
        dict_rois_score[name_img] = roi_scores

    for i,name_img in  enumerate(list_not_dog):
        #print(i,name_img)
        complet_name = path_to_img + name_img + '.jpg'
        im = cv2.imread(complet_name)
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)  # This call net.TL_image 
        k_max = np.max((k_max,len(fc7)))
        rois,roi_scores, fc7= reduce_to_k_regions(k,rois,roi_scores, fc7,
                                                  new_nms_thresh,score_threshold,
                                                  minimal_surface)

        
        if(len(fc7) >= k_per_bag):
            bag = np.expand_dims(fc7[0:k_per_bag,:],axis=0)
        else:
            number_repeat = k_per_bag // len(fc7)  +1
            f_repeat = np.repeat(fc7,number_repeat,axis=0)
            bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
        neg_ex[i,:,:] = bag
        dict_rois[name_img] = rois
    
    print("k_max",k_max)
    
    tf.reset_default_graph()
    sess.close()
    
    # Train the MIL SVM 
    restarts = 20
    max_iters = 300
    classifierMI_max = MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                  max_iters=max_iters,symway=symway,n_jobs=-1,
                                  all_notpos_inNeg=False,gridSearch=False,
                                  verbose=False,final_clf='None')     
    print("Start Learning MI_max")
    classifierMI_max.fit(pos_ex, neg_ex)
    print("End Learning MI_max")
    PositiveRegions = classifierMI_max.get_PositiveRegions()
    get_PositiveRegionsScore = classifierMI_max.get_PositiveRegionsScore()

    # Draw 
    for i,name_img in  enumerate(list_dog):
        #print(i,name_img)
        complet_name = path_to_img + name_img + '.jpg'
        im = cv2.imread(complet_name)
        blobs, im_scales = get_blobs(im)
        rois = dict_rois[name_img]
        roi_with_object_of_the_class = PositiveRegions[i] % len(rois) # Because we have repeated some rois
        roi = rois[roi_with_object_of_the_class,:]
        roi_boxes =  roi[1:5] / im_scales[0]
        best_RPN_roi = rois[0,:]
        best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
        best_RPN_roi_scores = [get_PositiveRegionsScore[0]]
        cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
#            print(roi_boxes)
        
        roi_scores = [get_PositiveRegionsScore[i]]
#            print(roi_scores)
        best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
        roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
        roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
        name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MI_maxbestROI.jpg'
        plt.savefig(name_output)
    plt.close('all')
    
    

def old_FasterRCNN_TL_MI_max_newVersion():
    """ Function to test if you can refind the same AP metric by reading the saved 
    CNN features 
    Older version of the function than FasterRCNN_TL_MI_max_ClassifOutMI_max
    """
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    N = 1
    extL2 = ''
    nms_thresh = 0.0
    demonet = 'res152_COCO'
    item_name = 'name_img'
    name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
    features_resnet_dict = {}
    sLength_all = len(df_label['name_img'])
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
    
    with open(name_pkl, 'rb') as pkl:
        for i,name_img in  enumerate(df_label[item_name]):
            if i%1000==0 and not(i==0):
                print(i,name_img)
                features_resnet_dict_tmp = pickle.load(pkl)
                if i==1000:
                    features_resnet_dict = features_resnet_dict_tmp
                else:
                    features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
        features_resnet_dict_tmp = pickle.load(pkl)
        features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
    print("Data loaded",len(features_resnet_dict))
    
    
    k_per_bag = 5
    features_resnet = np.ones((sLength_all,k_per_bag,size_output))
    classes_vectors = np.zeros((sLength_all,10))
    f_test = {}
    index_test = 0
    Test_on_k_bag = False
    normalisation= True
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0 and not(i==0):
            print(i,name_img)
        fc7 = features_resnet_dict[name_img]
        if(len(fc7) >= k_per_bag):
            bag = np.expand_dims(fc7[0:k_per_bag,:],axis=0)
        else:
            number_repeat = k_per_bag // len(fc7)  +1
            f_repeat = np.repeat(fc7,number_repeat,axis=0)
            bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
        
        features_resnet[i,:,:] = np.array(bag)
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if(classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
        InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
        if InSet and not(Test_on_k_bag):
            f_test[index_test] = fc7
            index_test += 1
            
    
    # TODO : keep the info of the repeat feature to remove them in the LinearSVC !! 
    
    print("End data processing")
    restarts = 20
    max_iters = 300
    n_jobs = -1
    #from trouver_classes_parmi_K import MI_max
    X_train = features_resnet[df_label['set']=='train',:,:]
    y_train = classes_vectors[df_label['set']=='train',:]
    X_test= features_resnet[df_label['set']=='test',:,:]
    y_test = classes_vectors[df_label['set']=='test',:]
    X_val = features_resnet[df_label['set']=='validation',:,:]
    y_val = classes_vectors[df_label['set']=='validation',:]
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
    
    if normalisation == True:
        mean_training_ex = np.mean(X_trainval,axis=(0,1))
        std_training_ex = np.std(X_trainval,axis=(0,1))
        X_trainval = (X_trainval - mean_training_ex)/std_training_ex
        X_test = (X_test - mean_training_ex)/std_training_ex
    else:
        mean_training_ex = 0.
        std_training_ex = 1.
    
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    final_clf = 'LinearSVC'
    #final_clf = 'defaultSGD'
    #final_clf = 'SGDsquared_hinge'
    for j,classe in enumerate(classes):
        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
        classifierMI_max = MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                      max_iters=max_iters,symway=True,n_jobs=n_jobs,
                                      all_notpos_inNeg=False,gridSearch=True,
                                      verbose=False,final_clf=final_clf)     
        classifier = classifierMI_max.fit(pos_ex, neg_ex)
        #print("End training the MI_max")
        y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j])
        labels_test_predited = np.zeros_like(y_test[:,j])
        
        for k in range(len(X_test)): 
            if Test_on_k_bag: 
                decision_function_output = classifier.decision_function(X_test[k,:,:])
            else:
                elt_k = (f_test[k] - mean_training_ex)/std_training_ex
                decision_function_output = classifier.decision_function(elt_k)
            y_predict_confidence_score_classifier[k]  = np.max(decision_function_output)
            if np.max(decision_function_output) > 0:
                labels_test_predited[k] = 1 
            else: 
                labels_test_predited[k] =  0 # Label of the class 0 or 1
        AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
        print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
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
    
def Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'Paintings',Test_on_k_bag = False,
                             normalisation= False,baseline_kind = 'MAX1',
                             verbose = True,gridSearch=False,k_per_bag=300,jtest=0,testMode=False,
                             n_jobs=-1,clf='LinearSVC',restarts = 0,max_iter_MI_max=500):
    """ 
    18 juin 2018 ==> voir le fichier Baseline_script.py
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
    @param : restarts = 0, le nombre de restarts pour les algos de MISVM et miSVM
    @param : max_iter_MI_max : nombre d iterations maximales dans le cadre des algos MISVM et miSVM
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
        elif database=='PeopleArt':
            ext = '.csv'
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/PeopleArt/JPEGImages/'
            classes =  ['person']
        elif(database=='WikiTenLabels'):
            ext='.csv'
            item_name='item'
            classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien','turban']
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        elif database=='watercolor':
            ext = '.csv'
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
            classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
        elif(database=='Wikidata_Paintings'):
            item_name = 'image'
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
            raise NotImplementedError # TODO implementer cela !!! 
        elif(database=='Wikidata_Paintings_miniset_verif'):
            item_name = 'image'
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
            classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
        else:
            print(database,'is unknown')
            raise NotImplementedError
        
        if(jtest>len(classes)) and testMode:
           print("We are in test mode but jtest>len(classes), we will use jtest =0" )
           jtest =0
        
        path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
        databasetxt =path_data + database + ext    
        if database=='VOC2007' or database=='watercolor' or database=='PeopleArt':
            dtypes = {0:str,'name_img':str,'aeroplane':int,'bicycle':int,'bird':int, \
                      'boat':int,'bottle':int,'bus':int,'car':int,'cat':int,'cow':int,\
                      'dinningtable':int,'dog':int,'horse':int,'motorbike':int,'person':int,\
                      'pottedplant':int,'sheep':int,'sofa':int,'train':int,'tvmonitor':int,'set':str}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
            df_label[classes] = df_label[classes].apply(lambda x: np.floor((x + 1.0) /2.0))
        elif database=='WikiTenLabels':
            dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                      'turban':int,'set':str,'Anno':int}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
        elif database=='PeopleArt':
            dtypes = {'name_img':str,'person':int,'set':str}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
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
        filesave = 'pkl'
        if not(os.path.isfile(name_pkl)):
            # Compute the features
            if verbose: print("We will computer the CNN features")
            Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                         database=database,augmentation=False,L2 =False,
                                         saved='all',verbose=verbose,filesave=filesave)
        
        if baseline_kind == 'MAX1' or baseline_kind == 'MEAN' or baseline_kind in ['miSVM','MISVM']:
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
        if database=='Wikidata_Paintings_miniset_verif' or database=='VOC2007' or database=='watercolor' or database=='WikiTenLabels' or database=='PeopleArt':
            classes_vectors = df_label.as_matrix(columns=classes).astype(np.float32)

        f_test = {}

        # Parameters important
        new_nms_thresh = 0.0
        score_threshold = 0.1
        minimal_surface = 36*36
        # In the case of Wikidata
        if database=='Wikidata_Paintings_miniset_verif':
            random_state = 0
            index = np.arange(0,len(features_resnet_dict))
            index_trainval, index_test = train_test_split(index, test_size=0.6, random_state=random_state)
            index_trainval = np.sort(index_trainval)
            index_test = np.sort(index_test)
    
        if database=='VOC2007'  or database=='watercolor' or database=='clipart' or database=='WikiTenLabels' or database=='PeopleArt':
            if database=='VOC2007' : imdb = get_imdb('voc_2007_test')
            if database=='PeopleArt' : imdb = get_imdb('PeopleArt_test')
            if database=='watercolor' : imdb = get_imdb('watercolor_test')
            if database=='clipart' : imdb = get_imdb('clipart_test')
            if database=='WikiTenLabels' :imdb = get_imdb('WikiTenLabels_test')
            if database=='WikiTenLabels':
                num_images =  len(df_label[df_label['set']=='test'][item_name])
            else:
                num_images = len(imdb.image_index)
            imdb.set_force_dont_use_07_metric(True)
        else:
            num_images = len(df_label[df_label['set']=='test'])
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
#        if database=='WikiTenLabels' : 
#            all_boxes_other_images = [[[] for _ in range(num_images)] for _ in range(len(df_label[df_label['set']=='test']) -num_classes)]
    
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
        if database=='VOC12' or database=='Paintings' or database=='VOC2007'  or database=='watercolor' or database=='WikiTenLabels' or database=='PeopleArt':
            if database=='VOC2007'  or database=='watercolor' or database=='WikiTenLabels' or database=='PeopleArt':
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
            names = df_label.as_matrix(columns=[item_name])
            name_train = names[df_label['set']=='train']
            name_val = names[df_label['set']==str_val]
            name_all_test =  names[df_label['set']=='test']
            name_trainval = np.append(name_train,name_val,axis=0)
        elif database=='Wikidata_Paintings_miniset_verif' :
            name = df_label.as_matrix(columns=[item_name])
            name_trainval = name[index_trainval]
            #name_test = name[index_test]
#            X_test= features_resnet[index_test,:,:]
            y_test = classes_vectors[index_test,:]
#            X_trainval =features_resnet[index_trainval,:,:]
            y_trainval =  classes_vectors[index_trainval,:]
        
        name_trainval = name_trainval.ravel()

        # Then we run on the different classes
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        if baseline_kind == 'MAX1':
            number_neg = 1
        elif baseline_kind == 'MAXA' or baseline_kind in ['miSVM','MISVM']:
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
                            if y_trainval[index_nav,j] == 1: # Positive exemple
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
                            index_nav += 1
                    del features_resnet_dict
                X_trainval_select = np.concatenate(X_trainval_select,axis=0).astype(np.float32)
                y_trainval_select = np.array(y_trainval_select,dtype=np.float32)
            if verbose: 
                try:
                    print("Shape X and y",X_trainval_select.shape,y_trainval_select.shape)
                except UnboundLocalError:
                    if not( baseline_kind in ['miSVM','MISVM']):
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
                    n_jobs=n_jobs,C_finalSVM=1,cskind='small')
                dict_clf[j] = classifier_trained
            elif baseline_kind=='MISVM':
                ## Implementation of the MISVM of Andrews 2006
                #Initialisation  
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
#                    max_iter = 10
                    iteration = 0
                    SelectirVar_haveChanged = True
                    while((iteration < max_iter_MI_max) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
                        X_trainval_select = np.vstack((X_trainval_select_neg,X_pos))
                        clf_MISVM = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1,cskind='small')
                        
                        S_I_old = S_I
                        S_I = []
                        for k in range(len(X_trainval_select_pos)):
                            argmax_k = np.argmax(clf_MISVM.decision_function(X_trainval_select_pos[k]))
                            S_I += [argmax_k]
                            X_S_I = X_trainval_select_pos[k][argmax_k,:]
                            X_pos[k,:] = X_S_I
                        if S_I==S_I_old:
                            SelectirVar_haveChanged=False
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                    if verbose: print("End after ",iteration,"iterations on",max_iter_MI_max)
                # Sur Watercolor avec LinearSVC et sans GridSearch on a 7 iterations max
                # Training ended
                dict_clf[j] = clf_MISVM   
                del X_trainval_select
                
            elif baseline_kind=='miSVM':
                ## Implementation of the MISVM of Andrews 2006
                #Initialisation
                X_pos =np.concatenate(X_trainval_select_pos,axis=0).astype(np.float32)
                X_trainval_select = np.vstack((X_trainval_select_neg,X_pos)).astype(np.float32)
                size_X_pos = len(X_pos)
                size_X_trainval_select_neg = len(X_trainval_select_neg)
                del X_pos
                for rr in range(restarts+1):
                    y_pos = np.ones((size_X_pos,)).astype(np.float32)
#                    max_iter = 10
                    iteration = 0
                    SelectirVar_haveChanged = True
    #                clf = LinearSVC(penalty='l2',class_weight='balanced', 
    #                            loss='squared_hinge',max_iter=1000,dual=True)
                    while((iteration < max_iter_MI_max) and SelectirVar_haveChanged):
                        iteration +=1
                        t0=time.time()
#                        print(y_pos.shape)
#                        print(np.zeros(len(X_trainval_select_neg),).shape)
                        y_trainval_select = np.hstack((np.zeros((size_X_trainval_select_neg,)).astype(np.float32),y_pos))
    #                    clf.fit(X_trainval_select,y_trainval_select)
                        
                        clf_misvm = TrainClassif(X_trainval_select,y_trainval_select,clf=clf,
                                     class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,
                                     C_finalSVM=1,cskind='small')
                        index_k = 0
                        changed= False
                        for k in range(len(X_trainval_select_pos)):
                            predictions_k = clf_misvm.predict(X_trainval_select_pos[k])
                            size_k = len(X_trainval_select_pos[k])
                            if not(np.sum(predictions_k) >= 1.):
                                predictions_k_decision_function = clf_misvm.decision_function(X_trainval_select_pos[k])
                                predictions_k[np.argmax(predictions_k_decision_function)] = 1.
                            if not((predictions_k==y_pos[index_k:index_k+size_k]).all()):
                                changed = True # Il y a eu un changement
                            y_pos[index_k:index_k+size_k] = predictions_k.astype(np.float32)
                            index_k +=size_k
                        if not(changed):
                            SelectirVar_haveChanged=False
                        t1=time.time()
                        if verbose: print("Duration of one iteration :",str(t1-t0),"s")
                    if verbose: print("End after ",iteration,"iterations on",max_iter_MI_max)
                # Sur Watercolor avec LinearSVC et sans GridSearch on tape tout le temps dans les 10 iterations
                
                # Training ended
                dict_clf[j] = clf_misvm   
                del X_trainval_select
            if verbose: print("End learning for class",j)
            
        gc.collect()
        
        #Load test set 
        if baseline_kind == 'MAXA':
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
        key_annotated = []        
        for i,name_img in  enumerate(df_label[item_name]):
            if i%1000==0 and not(i==0):
                if verbose: print(i,name_img)
            if database in ['VOC2007','VOC12','Paintings','watercolor','clipart','WikiTenLabels','PeopleArt']:         
                InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
            elif database=='Wikidata_Paintings_miniset_verif':
                InSet = (i in index_test)
            if database=='WikiTenLabels':
#                print(df_label.sum())
#                print(df_label.loc[df_label[item_name]==name_img])
#                print(df_label.loc[df_label[item_name]==name_img]['Anno'])
#                print((df_label.loc[df_label[item_name]==name_img]['Anno'].value==1.0))
#                print((df_label.loc[df_label[item_name]==name_img]['Anno'].value==1.0).any())
                if (df_label.loc[df_label[item_name]==name_img]['Anno']==1.0).any() :
#                    print('key_annotated')
                    key_annotated += [key_test]
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
#        print('key_annotated',key_annotated)
#        print('len(f_test)',len(f_test))
#        print('jtest',jtest,'testMode',testMode)
#        print('database',database)

        for j,classe in enumerate(classes):
            if testMode and not(j==jtest):
                continue
            else:
                classifier_trained = dict_clf[j]
            if database=='WikiTenLabels':
                name_all_test = []
            y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j],dtype=np.float32)
            labels_test_predited = np.zeros_like(y_test[:,j],dtype=np.float32)
            
            # Test Time
            k_local = 0
            for k in range(len(f_test)): 
                if Test_on_k_bag: 
                    raise(NotImplementedError)
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
                if database in ['VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt']:
                    thresh = 0.05 # Threshold score or distance MI_max
                    TEST_NMS = 0.3 # Recouvrement entre les classes
                    if not(database=='PeopleArt'):
                        complet_name = path_to_img + str(name_test[k]) + '.jpg'
                    else:
                        complet_name = path_to_img + str(name_test[k])
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
                    
                    if database=='WikiTenLabels':
                        if k in key_annotated:
                            all_boxes[j][k_local] = cls_dets
                            k_local +=1
                            name_all_test += [name_test[k]]
                        # else : il faudrait peut etre sauvegarder les donnees ailleurs pour les autres images
                    else:
                        all_boxes[j][k] = cls_dets
                        
                if np.max(decision_function_output) > 0:
                    labels_test_predited[k] = 1 
                else: 
                    labels_test_predited[k] =  0 # Label of the class 0 or 1

            print(y_predict_confidence_score_classifier)
            print(y_test[:,j])
            AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                print("Baseline SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("Baseline SVM version Average Precision for",classes[j]," =",AP)
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
        print('~~~~ Classification ~~~~')
        print(AP_per_class)
        print(arrayToLatex(AP_per_class,per=True))
        print('~~~~~~~~')

        if database in ['VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt']:
#            print('name_all_test',name_all_test)
            if testMode:
                for j in range(0, imdb.num_classes-1):
                    if not(j==jtest):
                        #print(all_boxes[jtest])
                        all_boxes[j] = all_boxes[jtest]
            det_file = os.path.join(path_data, 'detections_aux.pkl')
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            max_per_image = 100
            num_images_detect =  len(imdb.image_index)
            all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
            number_im = 0
            for i in range(num_images_detect):
                name_img = imdb.image_path_at(i)
                if database=='PeopleArt':
                    name_img_wt_ext = name_img.split('/')[-2] +'/' +name_img.split('/')[-1]
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
            if verbose: print("Number of images in the test set",number_im)
            assert(number_im==num_images_detect)
            det_file = os.path.join(path_data, 'detections.pkl')
            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
            output_dir = path_data +'tmp/' + database + '/'
            aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
            print("Detection scores for Baseline algo ",baseline_kind)
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

    except KeyboardInterrupt:
        gc.collect()
        tf.reset_default_graph()

def FasterRCNN_TL_MI_max_ClassifOutMI_max(demonet = 'res152_COCO',database = 'Paintings', 
                                          verbose = True,testMode = True,jtest = 0,
                                          PlotRegions = True,saved_clf=False,RPN=False,
                                          CompBest=True,Stocha=False,k_per_bag=30,
                                          WR=True):
    """ 
    15 mars 2017
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    This one use pickle precomputed saved data
    In this function we train an SVM only on the positive element returned by 
    the algo
    
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
    TODO : mine hard negative exemple ! 
    
    Cette fonction permet de calculer les performances AP pour les differents dataset 
    Wikidata et Your Paintings avec l'algo de selection de Said et l'entrainement du SVM final 
    en dehors du code de Said trouver_classes_parmi_k
    
    FasterRCNN_TL_MISVM est sense etre la meme chose avec en utilisant les algos MISVM et miSVm de Andrews
    
    
    """
    # TODO be able to train on background 
    try:
        if demonet == 'vgg16_COCO':
            num_features = 4096
        elif demonet in ['res101_COCO','res152_COCO','res101_VOC07']:
            num_features = 2048
        ext = '.txt'
        dtypes = str
        if database=='Paintings':
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        elif database=='VOC12':
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        elif database=='VOC2007':
            ext = '.csv'
            isVOC = True
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
            classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
        elif database=='WikiTenLabels':
            ext = '.csv'
            item_name = 'item'
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
            classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                        'Mary','nudity', 'ruins','Saint_Sebastien','turban']
        elif database=='watercolor':
            ext = '.csv'
            item_name = 'name_img'
            path_to_img = '/media/gonthier/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
            classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
        elif(database=='Wikidata_Paintings'):
            item_name = 'image'
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
            raise NotImplementedError # TODO implementer cela !!! 
        elif(database=='Wikidata_Paintings_miniset_verif'):
            item_name = 'image'
            path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
            classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
        
        if(jtest>len(classes)) and testMode:
           print("We are in test mode but jtest>len(classes), we will use jtest =0" )
           jtest =0
        
        path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
        databasetxt =path_data + database + ext    
        if database=='VOC2007' or database=='watercolor':
            dtypes = {0:str,'name_img':str,'aeroplane':int,'bicycle':int,'bird':int, \
                      'boat':int,'bottle':int,'bus':int,'car':int,'cat':int,'cow':int,\
                      'dinningtable':int,'dog':int,'horse':int,'motorbike':int,'person':int,\
                      'pottedplant':int,'sheep':int,'sofa':int,'train':int,'tvmonitor':int,'set':str}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
            df_label[classes] = df_label[classes].apply(lambda x: np.floor((x + 1.0) /2.0))
        elif database=='WikiTenLabels':
            dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                      'turban':int,'set':str,'Anno':int}
            df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)    
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
        filesave = 'pkl'
        if not(os.path.isfile(name_pkl)):
            # Compute the features
            if verbose: print("We will computer the CNN features")
            Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                         database=database,augmentation=False,L2 =False,
                                         saved='all',verbose=verbose,filesave=filesave)
            
        
        if verbose: print("Start loading data",name_pkl)
            
        if (k_per_bag*len(df_label[item_name]) > 30*10000):
            print("You risk Memory Error, good luck. You should use tfrecord")
    
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
        
        
        features_resnet = np.empty((sLength_all,k_per_bag,size_output),dtype=np.float32)  
        classes_vectors = np.zeros((sLength_all,num_classes),dtype=np.float32)
        if database=='Wikidata_Paintings_miniset_verif' or database=='VOC2007' or database=='watercolor' or database=='WikiTenLabels':
            classes_vectors = df_label.as_matrix(columns=classes)
        else:
            raise(NotImplementedError)
        f_test = {}
        
        Test_on_k_bag = False
        normalisation= False
        
        # Parameters important
        new_nms_thresh = 0.0
        score_threshold = 0.1
        minimal_surface = 36*36
        # In the case of Wikidata
        if database=='Wikidata_Paintings_miniset_verif':
            random_state = 0
            index = np.arange(0,len(features_resnet_dict))
            index_trainval, index_test = train_test_split(index, test_size=0.6, random_state=random_state)
            index_trainval = np.sort(index_trainval)
            index_test = np.sort(index_test)
    
        len_fc7 = []
        roi_test = {}
        roi_train = {}
        name_test = {}
        key_test = 0
        for i,name_img in  enumerate(df_label[item_name]):
            if i%1000==0 and not(i==0):
                if verbose: print(i,name_img)
            rois,roi_scores,fc7 = features_resnet_dict[name_img]
            #print(rois.shape,roi_scores.shape)
            rois_reduce,roi_scores,fc7_reduce =  reduce_to_k_regions(k_per_bag,rois, \
                                                       roi_scores, fc7,new_nms_thresh, \
                                                       score_threshold,minimal_surface)
            len_fc7 += [len(fc7_reduce)]
            if(len(fc7_reduce) >= k_per_bag):
                bag = np.expand_dims(fc7_reduce[0:k_per_bag,:],axis=0)
            else:
                number_repeat = k_per_bag // len(fc7_reduce)  +1
                f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
                bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)  
                
            features_resnet[i,:,:] = np.array(bag)
            
            if database=='VOC12' or database=='Paintings':
                for j in range(num_classes):
                    if(classes[j] in df_label['classe'][i]):
                        classes_vectors[i,j] = 1
            if database=='VOC2007' or database=='VOC12' or database=='Paintings'  or database=='watercolor' or database=='WikiTenLabels':          
                InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
            elif database=='Wikidata_Paintings_miniset_verif':
                InSet = (i in index_test)
            
            if InSet: 
                if not(Test_on_k_bag):
                    f_test[key_test] = fc7
                    roi_test[key_test] = rois
                    name_test[key_test] = name_img
                    key_test += 1 
            else:
                roi_train[name_img] = rois_reduce  
        
        if verbose: 
            print('len(fc7), max',np.max(len_fc7),'mean',np.mean(len_fc7),'min',np.min(len_fc7))
            print('But we only keep k_per_bag =',k_per_bag)
        
        # TODO : keep the info of the repeat feature to remove them in the LinearSVC !! 
        
        if verbose: print("End data processing")
    
        if database=='VOC2007'  or database=='watercolor' or database=='clipart':
            if database=='VOC2007' : imdb = get_imdb('voc_2007_test')
            if database=='watercolor' : imdb = get_imdb('watercolor_test')
            if database=='clipart' : imdb = get_imdb('clipart_test')
            num_images = len(imdb.image_index)
        elif database=='WikiTenLabels':
            imdb = get_imdb('WikiTenLabels_test')
            imdb.set_force_dont_use_07_metric(True)
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        else:
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        if CompBest: all_boxes_bS = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    
        if testMode:
            verboseMIL = True
            restarts = 0
            max_iters = 1
        else:
            verboseMIL = False
            restarts = 19
            max_iters = 300
        print('restarts',restarts,'max_iters',max_iters)
        n_jobs = 1
        #from trouver_classes_parmi_K import MI_max
        
    #    del features_resnet_dict
        gc.collect()
        
        if database=='VOC12' or database=='Paintings' or database=='VOC2007'  or database=='watercolor'or database=='WikiTenLabels':
            if database=='VOC2007'  or database=='watercolor': 
                str_val ='val' 
            else: 
                str_val='validation'
            X_train = features_resnet[df_label['set']=='train',:,:]
            y_train = classes_vectors[df_label['set']=='train',:]
            X_test= features_resnet[df_label['set']=='test',:,:]
            y_test = classes_vectors[df_label['set']=='test',:]
            X_val = features_resnet[df_label['set']==str_val,:,:]
            y_val = classes_vectors[df_label['set']==str_val,:]
            X_trainval = np.append(X_train,X_val,axis=0)
            y_trainval = np.append(y_train,y_val,axis=0)
            names = df_label.as_matrix(columns=['name_img'])
            name_train = names[df_label['set']=='train']
            name_val = names[df_label['set']==str_val]
            name_all_test =  names[df_label['set']=='test']
            name_trainval = np.append(name_train,name_val,axis=0)
        elif database=='Wikidata_Paintings_miniset_verif' :
            name = df_label.as_matrix(columns=[item_name])
            name_trainval = name[index_trainval]
            #name_test = name[index_test]
            X_test= features_resnet[index_test,:,:]
            y_test = classes_vectors[index_test,:]
            X_trainval =features_resnet[index_trainval,:,:]
            y_trainval =  classes_vectors[index_trainval,:]
        if normalisation == True:
            if verbose: print('Normalisation')
            scaler = StandardScaler()
            scaler.fit(X_trainval.reshape(-1,size_output))
            X_trainval = scaler.transform(X_trainval.reshape(-1,size_output))
            X_trainval = X_trainval.reshape(-1,k_per_bag,size_output)
            X_test = scaler.transform(X_test.reshape(-1,size_output))
            X_test = X_test.reshape(-1,k_per_bag,size_output)
        
    #    del features_resnet,X_train,X_val
        gc.collect()
            
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        AP_per_classbS = []
        final_clf = None
        class_weight = None
        for j,classe in enumerate(classes):
            gc.collect()
            if testMode and not(j==jtest):
                continue
            if verbose : print(j,classes[j])
            if PlotRegions:
                if Stocha:
                    extensionStocha = 'Stocha/'
                else:
                    extensionStocha = ''
                if database=='Wikidata_Paintings_miniset_verif':
                    path_to_output2  = path_data + '/MI_maxRegion/'+extensionStocha+depicts_depictsLabel[classes[j]]
                else:
                    path_to_output2  = path_data + '/MI_maxRegion/'+extensionStocha+classes[j]
                if RPN:
                    path_to_output2 += '_RPNetMISVM/'
                elif CompBest:
                     path_to_output2 += '_BestObject/'
                else:
                    path_to_output2 += '/'
                path_to_output2_bis = path_to_output2 + 'Train'
                path_to_output2_ter = path_to_output2 + 'Test'
                pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
                pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True) 
                
            neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
            pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
            pos_name = name_trainval[y_trainval[:,j]==1]
            
            if verbose: print("Start train the MI_max")
    
            
            if Stocha:
                # TODO Cela ne marche pas encore !
                raise(NotImplementedError)
                bags = np.vstack((neg_ex,pos_ex))
                y_pos = np.ones((len(neg_ex),1))
                y_neg = np.zeros((len(pos_ex),1))
                labels = np.vstack((y_pos,y_neg)).ravel()
                max_iters_wt_minibatch = 300
                max_supported_by_gpu = 30*9000
                N,k,d = bags.shape
                mini_batch_size = max_supported_by_gpu // k
                n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
                max_iters = n_batch*max_iters_wt_minibatch
                Optimizer='GradientDescent'
                optimArg=None
                classifierMI_max = tf_MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,
                   all_notpos_inNeg=False,gridSearch=True,
                   verbose=verboseMIL,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                   mini_batch_size=mini_batch_size,WR=True) 
                classifierMI_max.fit_Stocha(bags,labels,shuffle=True)
            else:
                classifierMI_max = MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,
                   all_notpos_inNeg=False,gridSearch=True,
                   verbose=verboseMIL,final_clf=final_clf,WR=True)   
                classifierMI_max.fit(pos_ex, neg_ex)
                #durations : between 26 and durations : 8 for Paintings
            
            PositiveRegions = classifierMI_max.get_PositiveRegions()
            get_PositiveRegionsScore = classifierMI_max.get_PositiveRegionsScore()
            PositiveExScoreAll =  classifierMI_max.get_PositiveExScoreAll()
            
            if PlotRegions:
                # Just des verifications
                a = np.argmax(PositiveExScoreAll,axis=1)
                assert((a==PositiveRegions).all())
                assert(len(pos_name)==len(PositiveRegions))
            
            if verbose: print("End training the MI_max")
            
            pos_ex_after_MI_max = np.zeros((len(pos_ex),size_output))
            neg_ex_keep = np.zeros((len(neg_ex),num_features))
            for k,name_imgtab in enumerate(pos_name):
                pos_ex_after_MI_max[k,:] = pos_ex[k,PositiveRegions[k],:] # We keep the positive exemple according to the MI_max from Said
                
                if PlotRegions:
                    if verbose: print(k,name_img)
                    name_img = name_imgtab[0]
                    if database=='VOC2007' :
                        name_sans_ext =  str(name_img.decode("utf-8"))
                        complet_name = path_to_img + str(name_img.decode("utf-8")) + '.jpg'
                    if database=='VOC12' or database=='Paintings'  or database=='watercolor'or database=='WikiTenLabels':
                        complet_name = path_to_img + name_img + '.jpg'
                        name_sans_ext = name_img
                    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                        name_sans_ext = os.path.splitext(name_img)[0]
                        complet_name = path_to_img +name_sans_ext + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    rois = roi_train[name_img]
                    roi_with_object_of_the_class = PositiveRegions[k] % len(rois) # Because we have repeated some rois
                    roi = rois[roi_with_object_of_the_class,:]
                    roi_scores = [get_PositiveRegionsScore[k]]
                    roi_boxes =  roi[1:5] / im_scales[0]   
                    roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                    if RPN:
                        best_RPN_roi = rois[0,:]
                        best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                        best_RPN_roi_scores = [PositiveExScoreAll[k,0]]
                        assert((get_PositiveRegionsScore[k] >= PositiveExScoreAll[k,0]).all())
                        cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                        best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                        roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                    else:
                        cls = ['MI_max']
                        roi_boxes_and_score = roi_boxes_score
                    vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                    name_output = path_to_output2 +'Train/' + name_sans_ext + '_Regions.jpg'
                    plt.savefig(name_output)
                    plt.close()
            
            neg_ex_keep = neg_ex.reshape(-1,num_features)
            
            X = np.vstack((pos_ex_after_MI_max,neg_ex_keep))
            y_pos = np.ones((len(pos_ex_after_MI_max),1))
            y_neg = np.zeros((len(neg_ex_keep),1))
            y = np.vstack((y_pos,y_neg)).ravel()
            if verbose: print('Start Learning Final Classifier X.shape,y.shape',X.shape,y.shape)
            classifier = TrainClassif(X,y,clf='LinearSVC',class_weight=class_weight,
                                      gridSearch=True,n_jobs=-1)
            
            if saved_clf:
                name_clf_pkl = path_data+'clf_FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_'+str(classe)+'.pkl'
                joblib.dump(classifier,name_clf_pkl) 
            
            if CompBest:
                if verbose : print("Start Learning the Best Score object classif")
                pos_ex_bestScore = pos_ex[:,0,:]
                XbestScore = np.vstack((pos_ex_bestScore,neg_ex_keep))
                classifierBest = TrainClassif(XbestScore,y,clf='LinearSVC',class_weight=class_weight,
                                      gridSearch=True,n_jobs=1)
                name_clf_pkl = path_data+'clf_FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_bestObject_'+str(nms_thresh)+'_'+str(classe)+'.pkl'
                joblib.dump(classifierBest,name_clf_pkl) 
            
            if verbose: print("End training the SVM")
            
            y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j])
            y_predict_confidence_score_classifierbS = np.zeros_like(y_test[:,j])
            labels_test_predited = np.zeros_like(y_test[:,j])
            
            for k in range(len(X_test)): 
                if Test_on_k_bag: 
                    decision_function_output = classifier.decision_function(X_test[k,:,:])
                else:
                    if normalisation:
                        elt_k =  scaler.transform(f_test[k])
                    else:
                        elt_k = f_test[k]
                    decision_function_output = classifier.decision_function(elt_k)
                    if CompBest:
                        decision_function_output_bS = classifierBest.decision_function(elt_k)
                        y_predict_confidence_score_classifierbS[k]  = np.max(decision_function_output_bS)
                        
                y_predict_confidence_score_classifier[k]  = np.max(decision_function_output)
                roi_with_object_of_the_class = np.argmax(decision_function_output)
                
                # For detection 
                if database=='VOC2007'  or database=='watercolor'or database=='WikiTenLabels':
                    thresh = 0.05 # Threshold score or distance MI_max
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
                    if CompBest:
                        inds = np.where(decision_function_output_bS > thresh)[0]
                        cls_scores = decision_function_output_bS[inds]
                        roi = roi_test[k]
                        roi_boxes =  roi[:,1:5] / im_scales[0] 
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        all_boxes_bS[j][k] = cls_dets
                
                if np.max(decision_function_output) > 0:
                    labels_test_predited[k] = 1 
                    if PlotRegions: # We predict a n element of the class we will plot  
                        name_img = name_test[k]
                        if database=='VOC2007' or database=='WikiTenLabels':
                            name_sans_ext =  str(name_img)
                            complet_name = path_to_img + str(name_img) + '.jpg'
                        if database=='VOC12' or database=='Paintings':
                            complet_name = path_to_img + name_img + '.jpg'
                            name_sans_ext = name_img
                        elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                            name_sans_ext = os.path.splitext(name_img)[0]
                            complet_name = path_to_img +name_sans_ext + '.jpg'
                        if verbose: print(k,name_sans_ext)
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        rois = roi_test[k]
                        roi = rois[roi_with_object_of_the_class,:]
                        roi_boxes =  roi[1:5] / im_scales[0]
                        roi_scores =  [np.max(decision_function_output)]
                        roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                        if RPN:
                            best_RPN_roi = rois[0,:]
                            best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                            best_RPN_roi_scores = [decision_function_output[0]]
                            assert((np.max(decision_function_output) >= decision_function_output[0]).all())
                            cls = ['RPN','Classif']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                            best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                            roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                        elif CompBest:
                            roi_with_object_of_the_class = np.argmax(decision_function_output_bS)
                            roi2 = rois[roi_with_object_of_the_class,:]
                            roi_boxes2 =  roi2[1:5] / im_scales[0]
                            roi_scores2 =  [np.max(decision_function_output_bS)]
                            roi_boxes_score2 = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes2,roi_scores2)),axis=0),axis=0)
                            cls = ['BestObject','Classif']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                            roi_boxes_and_score = np.vstack((roi_boxes_score2,roi_boxes_score))
                        else:
                            cls = ['Classif']
                            roi_boxes_and_score = roi_boxes_score
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                        name_output = path_to_output2 +'Test/' + name_sans_ext  + '_Regions.jpg'
                        plt.savefig(name_output)
                        plt.close()
                else: 
                    labels_test_predited[k] =  0 # Label of the class 0 or 1
            AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
            if CompBest: APbS = average_precision_score(y_test[:,j],y_predict_confidence_score_classifierbS,average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                print("MIL-SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
            if CompBest and (database=='Wikidata_Paintings_miniset_verif'):  print("Best Score version Average Precision for",depicts_depictsLabel[classes[j]]," = ",APbS)
            test_precision = precision_score(y_test[:,j],labels_test_predited)
            test_recall = recall_score(y_test[:,j],labels_test_predited)
            F1 = f1_score(y_test[:,j],labels_test_predited)
            print("Test on all the data precision = {0:.2f}, recall = {1:.2f},F1 = {2:.2f}".format(test_precision,test_recall,F1))
            precision_at_k = ranking_precision_score(np.array(y_test), y_predict_confidence_score_classifier,20)
            P20_per_class += [precision_at_k]
            AP_per_class += [AP]
            R_per_class += [test_recall]
            P_per_class += [test_precision]
            if CompBest: AP_per_classbS += [APbS]
        print('~~~~~ Classification ~~~~~~')
        print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
        if CompBest: print("mean Average Precision for BEst Score = {0:.3f}".format(np.mean(AP_per_classbS))) 
        print("mean Precision for all the data = {0:.3f}".format(np.mean(P_per_class)))  
        print("mean Recall for all the data = {0:.3f}".format(np.mean(R_per_class)))  
        print("mean Precision @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
        
        print(AP_per_class)
        print(arrayToLatex(AP_per_class,per=True))
        print('~~~~~~~~~~~')
        if database=='VOC2007'  or database=='watercolor'or database=='WikiTenLabels':
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
            print("~~~~ Detection scores ~~~~")
            print(arrayToLatex(aps,per=True))
        
        plot_Test_illust_bol = False
        if plot_Test_illust_bol:
            
            dict_clf = {}
            classe_str = []
            for classe in classes:
                name_clf_pkl = path_data+'clf_FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_'+str(classe)+'.pkl'
                classifier = joblib.load(name_clf_pkl) 
                dict_clf[classe] = classifier
                classe_str += depicts_depictsLabel[classe]
                
            path_to_output2  = path_data + '/MI_maxRegion/TestIllust/'
            pathlib.Path(path_to_output2).mkdir(parents=True, exist_ok=True)
            CONF_THRESH = 0.7
            NMS_THRESH = 0.3 # non max suppression
            for k in range(len(X_test)):
                name_img = name_test[k]
                if database=='VOC12' or database=='Paintings'or database=='WikiTenLabels' or database=='VOC2007':
                    complet_name = path_to_img + name_img + '.jpg'
                    name_sans_ext = name_img
                elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                    name_sans_ext = os.path.splitext(name_img)[0]
                    complet_name = path_to_img +name_sans_ext + '.jpg'
                im = cv2.imread(complet_name)
                blobs, im_scales = get_blobs(im)
                rois = roi_test[k]
                #print(rois.shape)
                cls_boxes =  rois[:,1:5] / im_scales[0]
                #print(cls_boxes.shape)
                elt_k = f_test[k]
                cls_list = []
                dets_list= []
                for j,classe in enumerate(classes):
                    classifier = dict_clf[classe]
                    decision_function_output = classifier.decision_function(elt_k)
                    # TODO Il faudra changer cela pour mettre une proba et non une distance 
                    # TODO gerer le cas ou l on normalise les donnees
                    cls_scores = decision_function_output
                    dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    cls_dets = dets[keep, :]
                    #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                    cls_list += [depicts_depictsLabel[classe]]
                    dets_list += [cls_dets]
                #print(dets_list[0].shape)
                vis_detections_list(im, cls_list, dets_list, thresh=CONF_THRESH)
                name_output = path_to_output2  + name_sans_ext  + '_NMSRegions.jpg'
                plt.savefig(name_output)
                plt.close()
                
                path_to_output2  = path_data + '/MI_maxRegion/TestIllust2/'
                pathlib.Path(path_to_output2).mkdir(parents=True, exist_ok=True)
                CONF_THRESH = 0.7
                NMS_THRESH = 0.3 # non max suppression
                for k in range(len(X_test)):
                    name_img = name_test[k]
                    if database=='VOC12' or database=='Paintings':
                        complet_name = path_to_img + name_img + '.jpg'
                        name_sans_ext = name_img
                    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                        name_sans_ext = os.path.splitext(name_img)[0]
                        complet_name = path_to_img +name_sans_ext + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    rois = roi_test[k]
                    #print(rois.shape)
                    cls_boxes =  rois[:,1:5] / im_scales[0]
                    #print(cls_boxes.shape)
                    elt_k = f_test[k]
                    cls_list = []
                    dets_list= []
                    for j,classe in enumerate(classes):
                        classifier = dict_clf[classe]
                        decision_function_output = classifier.decision_function(elt_k)
                        index = np.argmax(decision_function_output)
                        cls_scores = np.array([np.max(decision_function_output)])
                        dets = np.hstack((cls_boxes[np.newaxis,index,:],
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                        #keep = nms(dets, NMS_THRESH)
                        #dets = dets[keep, :]
                        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                        cls_list += [depicts_depictsLabel[classe]]
                        dets_list += [dets]
                    #print(dets_list[0].shape)
                    vis_detections_list(im, cls_list, dets_list, thresh=0.0)
                    name_output = path_to_output2  + name_sans_ext  + '_1RegionsPerClass.jpg'
                    plt.savefig(name_output)
                    plt.close()
    except KeyboardInterrupt:
        gc.collect()
        tf.reset_default_graph()


    
def tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo = False,
                                  model='MI_max',
                                  verbose = True,testMode = False,jtest = 0,
                                  PlotRegions = False,saved_clf=False,RPN=False,
                                  CompBest=False,Stocha=True,k_per_bag=300,
                                  parallel_op =True,CV_Mode=None,num_split=2,
                                  WR=True,init_by_mean=None,seuil_estimation=None,
                                  restarts=11,max_iters_all_base=300,LR=0.01,
                                  with_tanh=True,C=1.0,Optimizer='GradientDescent',norm=None,
                                  transform_output=None,with_rois_scores_atEnd=False,
                                  with_scores=False,epsilon=0.0,restarts_paral='paral',
                                  Max_version=None,w_exp=1.0,seuillage_by_score=False,
                                  seuil=0.5,k_intopk=3,C_Searching=False,gridSearch=False,n_jobs=1,
                                  predict_with='MI_max',thres_FinalClassifier=0.5,
                                  thresh_evaluation=0.05,TEST_NMS=0.3,eval_onk300=False,
                                  optim_wt_Reg=False,AggregW=None,proportionToKeep=0.25,
                                  plot_onSubSet=None,loss_type=None,storeVectors=False,
                                  storeLossValues=False,obj_score_add_tanh=False,lambdas=0.5,
                                  obj_score_mul_tanh=False,metamodel='FasterRCNN',
                                  PCAuse=False,variance_thres=0.9,trainOnTest=False,
                                  AddOneLayer=False,exp=10,MaxOfMax=False,debug = False,alpha=0.7,
                                  layer='fc7',MaxMMeanOfMax=False,Cosine_ofW_inLoss=False,Coeff_cosine=1.):
    """ 
    10 avril 2017
    This function used TFrecords file 
    
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    
    In this function we train an SVM only on the positive element returned by 
    the algo
    
    Note : with a features maps of 2048, k_bag =300 and a batchsize of 1000 we can 
    train up to 1200 W vectors in parallel at the same time on a NVIDIA 1080 Ti
    
    @param : demonet : the kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : database : the database used for the classification task
        For OIV5 : OIV5_small_3135 and  OIV5_small_30001
    @param : verbose : Verbose option classical
    @param : ReDo = False : Erase the former computation
    @param : model : kind of model use for the optimization a instance based one (mi_model) 
        or a bag based one as MI_max (default)
    @param : testMode : boolean True we only run on one class
    @param : jtest : the class on which we run the test
    @param : PlotRegions : plot the regions used for learn and the regions in 
        the positive output response
    @param : saved_clf : [default : True] Too sva ethe classifier 
    @param : RPN=False trace la boite autour de l'element ayant le plus haut score du RPN object
    @param : CompBest : Comparaison with the CompBest classifier trained
    @param : Stocha : Use of a SGD for the MIL SVM SAID [default : False]
    @param : k_per_bag : number of element per batch in the slection phase [defaut : 300] 
    !!!!! for the moment it is not possible to use something else than 300 if the dataset is not 
    records with the selection of the regions already !!!! TODO change that
    @param : parallel_op : use of the parallelisation version of the MI_max 
        for the all classes same time
    @param : CV_Mode : cross validation mode in the MI_max : possibility ; 
        None, CV in k split or LA for Leave apart one of the split
    @param : num_split  : Number of split for the CV or LA
    @param : WR   :  use of not of the regularisation term in the evaluation of 
        the final classifier ; loss on train set, if True we don't use it [default=True]
    @param : init_by_mean   :  use of an initialisation of the vecteur W and 
        bias b by a optimisation on a classification task on the mean on all 
        the regions of the image
    @param : seuil_estimation : ByHist or MaxDesNeg :  Estimation of the seuil 
        for the prediction detection 
    @param : restarts  :  number of restart in the MI_max [default=11]
    @param : max_iters_all_base  :  number of maximum iteration on the going on 
        the full database 
    @param : LR  :  Learning rate for the optimizer in the MI_max 
    @param : C  :  Regularisation term for the optimizer in the MI_max 
    @param : Optimizer  : Optimizer for the MI_max GradientDescent or Adam
    @param : norm : normalisation of the data or not : possible : None or ''
            'L2' : normalisation L2 or 'STDall' : Standardisation on all data 
            'STD' standardisation by feature maps : not implemented
    @param : restarts_paral : run several W vecteur optimisation in parallel 
            two versions exist 'Dim','paral'
    @param : transform_output : Transformation for the final estimation can be 
        sigmoid or tanh (string)
    @param : with_rois_scores_atEnd : Multiplication of the final result by the 
        object score
    @param Max_version : default None : Different max that can be used in the 
        optimisation :
        Choice : 'max', None or '' for a reduce max 
        'softmax' : a softmax witht the product multiplied by w_exp
        'sparsemax' : use a sparsemax
        'mintopk' : use the min of the top k_intopk regions 
        'maxByPow': use the approximation of the max by the exposant sum
        'LogSumExp' : use the approximation of the max by the LogSumExp formula 
        'MaxPlusMin' : mean of k top + alpha * mean of k min
    @param : k_intopk
    @param w_exp : default 1.0 : weight in the softmax 
    @param seuillage_by_score : default False : remove the region with a score under seuil
    @param seuil : used to eliminate some regions : it remove all the image with an objectness score under seuil
    @param : gridSearch=False, use a grid search on the C parameter of the final classifier if predict_with is LeanarSVC
    @param : n_jobs=1, number of jobs for the grid search hyperparamter optimisation
    @param : predict_with='MI_max', final classifier 
    @param : thres_FinalClassifier : parameter to choose some of the exemple before feeding the final classifier LinearSVC
            if you add a final classifier after the MIL or MILS hyperplan separation
    @param : thresh_evaluation : 0.05 : seuillage avant de fournir les boites a l evaluation de detections
    @param : TEST_NMS : 0.3 : recouvrement autorise avant le NMS avant l evaluation de detections
    @param proportionToKeep : proportion to keep in the votingW or votingWmedian case
    @param AggregW (default =None ) The different solution for the aggregation 
    of the differents W vectors :
        Choice : None or '', we only take the vector that minimize the loss (with ou without CV)
        'AveragingW' : that the average of the proportionToKeep*numberW first vectors
        'meanOfProd' : Take the mean of the product the first vectors
        'medianOfProd' : Take the meadian of the product of the first vectors
        'maxOfProd' : Take the max of the product of the first vectors
        'meanOfTanh' : Take the mean of the tanh of the product of the first vectors
        'medianOfTanh' : Take the meadian of the tanh of product of the first vectors
        'maxOfTanh' : Take the max of the tanh of product of the first vectors
   @param obj_score_add_tanh : the objectness_score is add to the tanh of the dot product
   @param obj_score_mul_tanh : the objectness_score is multiply to the tanh of the dot product
   @param lambdas : the lambda ratio between the tanh scalar product and the objectness score
   @param metamodel : default : FasterRCNN it is the meta algorithm use to get bouding boxes and features
       metamodel in ['FasterRCNN','EdgeBoxes']
   @param PCAuse=False
   @param variance_thres=0.9 variance keep to the PCA
       number of component keeped in the PCA 
       If variance_thres=0.9 for IconArt_v1 we have numcomp=675 and for watercolor=654
   @param exp: exposant pour maxByPow
       
   @param trainOnTest : default False, if True, the model is learn on the test set
   @param AddOneLayer : default False, if True, we add one layer on the model
   @param : MaxOfMax use the max of the max of product and keep all the (W,b) learnt
            (default False)
   @param : MaxMMeanOfMax use the max minues the mean of the max of product
            and keep all the (W,b) learnt (default False)
   @param : alpha factor for the 'MaxPlusMin' pooling 
   @param : layer that we get in the pretrained net for ResNet only fc7 possible for 
        vgg16 : fc6 or fc7 but in both case it will be saved in the fc7 name
        in the tfrecords dataset !
   @param : Cosine_ofW_inLoss : We the mean of the cosine of the vectors in the loss function 
            Have to be used with MaxOfMax of MaxMMeanOfMax (default = False)
   @param : Coeff_cosine : weight in front the cosine loss (default = 1.)
         
    The idea of this algo is : 
        1/ Compute CNN features
        2/ Do NMS on the regions 
    
    option to train on background part also
    option on  scaling : sklearn.preprocessing.StandardScaler
    option : add a wieghted balanced of the SVM because they are really unbalanced classes
    TODO : mine hard negative exemple ! 
    
    Cette fonction permet de calculer les performances AP pour les differents dataset 
    Wikidata et Your Paintings avec l'algo de selection de Said et l'entrainement du SVM final 
    en dehors du code de Said trouver_classes_parmi_k
    
    FasterRCNN_TL_MISVM est sense etre la meme chose avec en utilisant les algos MISVM et miSVm de Andrews
    
    
    """
    if trainOnTest:
        print('/!\ you will train the model on the test data be careful with the conclusion')
    
    if not(metamodel in ['FasterRCNN','EdgeBoxes']):
        print(metamodel,' is unknown')
        raise(NotImplementedError)
    list_net_Edgeboxes = ['res152']
    if metamodel=='EdgeBoxes' and not(demonet in list_net_Edgeboxes):
        print('The pretrained nets are :',list_net_Edgeboxes)
        print(demonet,' is unknown')
        raise(NotImplementedError)
    
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
    
    
    print('==========')
    # TODO be able to train on background 
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC = get_database(database)
    
    if testMode and not(type(jtest)==int):
        assert(type(jtest)==str)
        jtest = int(np.where(np.array(classes)==jtest)[0][0])# Conversion of the jtest string to the value number
        assert(type(jtest)==int)
        
    if testMode and (jtest>len(classes)) :
       print("We are in test mode but jtest>len(classes), we will use jtest =0" )
       jtest =0


    if not(database=='RMN'):       
        num_trainval_im = len(df_label[df_label['set']=='train'][item_name]) + len(df_label[df_label['set']==str_val][item_name])
    else:
        num_trainval_im = len(df_label[item_name])
        output = get_database('IconArt_v1')
        item_name,path_to_img,_,_,_,_,df_label,_,_ = output
 
        
    print(database,'with ',num_trainval_im,' images in the trainval set')
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    if layer=='fc6':
        savedstr+='_fc6'

    sets = ['train','val','trainval','test']
    dict_name_file = {}
    data_precomputeed= True
    if k_per_bag==300:
        k_per_bag_str = ''
    else:
        k_per_bag_str = '_k'+str(k_per_bag)
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
        if set_str in ['trainval','test'] and not(os.path.isfile(name_pkl_all_features)):
            data_precomputeed = False
    
    if database=='RMN':
        data_precomputeed = True
        for set_str in sets:
            if set_str=='train':
                dict_name_file[set_str] = dict_name_file['trainval']
            if set_str=='test' or set_str=='val':
                dict_name_file[set_str] = dict_name_file[set_str].replace('RMN','IconArt_v1')
            if not(os.path.isfile(dict_name_file[set_str])):
                data_precomputeed = False
                print("Warning in the case of RMN database you have to precompute for RMN and for IconArt_v1")
    
    if 'OIV5' in database:
        data_precomputeed = True
        for set_str in sets:
            if set_str=='train' or set_str=='val':
                dict_name_file[set_str] = dict_name_file['trainval']
            if set_str=='test':
                dict_name_file[set_str] = dict_name_file[set_str].replace(database,'OIV5')
            if not(os.path.isfile(dict_name_file[set_str])):
                data_precomputeed = False
    
    print('data_precomputeed',data_precomputeed)
    input('wait')
    
    if demonet in ['vgg16_COCO','vgg16_VOC07','vgg16_VOC12']:
        num_features = 4096
    elif demonet in ['res101_COCO','res152_COCO','res101_VOC07','res152']:
        num_features = 2048
    
    if not(data_precomputeed):
        # Compute the features
        if verbose: print("We will computer the CNN features")
        if metamodel=='FasterRCNN':
            if not(PCAuse):
                Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                             database=database,augmentation=False,L2 =False,
                                             saved='all',verbose=verbose,filesave='tfrecords',
                                             k_regions=k_per_bag,layer=layer)
            else:
                number_composant = Save_TFRecords_PCA_features(demonet=demonet,nms_thresh =nms_thresh,database=database,
                                 augmentation=False,L2 =False,
                                 saved='all',verbose=verbose,k_regions=k_per_bag,
                                 variance_thres=variance_thres,layer=layer)
        elif metamodel=='EdgeBoxes':
            Compute_EdgeBoxesAndCNN_features(demonet=demonet,nms_thresh =nms_thresh,database=database,
                                 augmentation=False,L2 =False,
                                 saved='all',verbose=verbose,filesave='tfrecords',k_regions=k_per_bag)
          
    if PCAuse:    
        num_features = number_composant
    else:
        number_composant  = num_features
        
    # Config param for TF session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
            
    # Data for the MI_max Latent SVM
    # All those parameter are design for my GPU 1080 Ti memory size 
    performance = False
    if parallel_op:
        sizeMax = 30*10000 // (k_per_bag*num_classes) 
    else:
        sizeMax = 30*10000 // k_per_bag
    if restarts_paral=='Dim': # It will create a new dimension
        restarts_paral_str = '_RP'
        sizeMax //= max(int((restarts+1)//2),1) # To avoid division by zero
        # it seems that using a different size batch drasticly change the results
    elif restarts_paral=='paral': # Version 2 of the parallelisation
        restarts_paral_str = '_RPV2'
        sizeMax = 30*200000 // (k_per_bag*20)
    else:
        restarts_paral_str=''
    if not(init_by_mean is None) and not(init_by_mean==''):
        if not(CV_Mode=='CV' and num_split==2):
            sizeMax //= 2
     # boolean paralleliation du W
    if CV_Mode == 'CVforCsearch':
        sizeMax //= 2
    if num_features > 2048:
        sizeMax //= (num_features//2048)
#    elif num_features < 2048:
#        sizeMax //= (2048//num_features)
    # InternalError: Dst tensor is not initialized. can mean that you are running out of GPU memory
    
    if model=='MI_max' or model=='':
        model_str = 'MI_max'
        if k_per_bag==300:
            buffer_size = 10000
        else:
            buffer_size = 5000*300 // k_per_bag
        if AddOneLayer:
            sizeMax //= 2
    elif model=='mi_model':
        model_str ='mi_model'
        buffer_size = 10000*300 // k_per_bag
        if not (database in ['watercolor','IconArt_v1']):
            sizeMax //= 2
        if AddOneLayer:
            print('AddOneLayer is not implemented in mi_model')
            raise(NotImplementedError)
    else:
        print(model,' is unknown')
        raise(NotImplementedError)

    mini_batch_size = min(sizeMax,num_trainval_im)
    if (k_per_bag > 300 or num_trainval_im > 5000) and not(Not_on_NicolasPC): # We do the assumption that you are on a cluster with a big RAM (>50Go)
        usecache = False
    else:
        usecache = True

    if verbose : print('usecache',usecache,mini_batch_size,buffer_size)

    if CV_Mode=='1000max':
        mini_batch_size = min(sizeMax,1000)
    
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
    
    if C == 1.0:
        C_str=''
    else:
        C_str = '_C'+str(C) # regularisation term 
    if C_Searching:
        C_Searching_str ='_Csearch'
        C_str = ''
    else:
        C_Searching_str = ''
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
        raise(NotImplementedError)
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
        extCV = '_cv'+str(num_split)
    elif CV_Mode == '1000max':
        extCV = '_1000max'
    elif CV_Mode == 'CVforCsearch':
        extCV = '_CVforCsearch'
    elif CV_Mode is None or CV_Mode=='':
        extCV =''
    else:
        raise(NotImplementedError)
    if WR: extCV += '_wr'

    if Optimizer=='Adam':
        opti_str=''
    elif Optimizer=='GradientDescent':
        opti_str='_gd'
    elif Optimizer=='lbfgs':
        opti_str='_lbfgs'
    else:
        raise(NotImplementedError)
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
    
    if Max_version=='max' or Max_version=='' or Max_version is None:
        Max_version_str =''
    elif Max_version=='softmax':
        Max_version_str ='_MVSF'
        if not(w_exp==1.0): Max_version_str+=str(w_exp)
    elif Max_version=='sparsemax':
        Max_version_str ='_MVSM'
    elif Max_version=='mintopk':
        Max_version_str ='_MVMT'+str(k_intopk)
    elif Max_version=='MaxPlusMin':
        Max_version_str ='_MaxPlusMin'+str(k_intopk)+'_'+str(alpha)
    elif Max_version=='LogSumExp':
        Max_version_str ='_MLogSumExp'
    elif Max_version=='maxByPow':
        Max_version_str ='_maxByPow'+str(exp)
    optimArg = None
    #optimArg = {'learning_rate':LR,'beta1':0.9,'beta2':0.999,'epsilon':1}
    if optimArg== None or Optimizer=='GradientDescent':
        optimArg_str = ''
    else:
        if  Optimizer=='Adam' and str(optimArg).replace(' ','_')=="{'learning_rate':_0.01,_'beta1':_0.9,_'beta2':_0.999,_'epsilon':_1e-08}":
            optimArg_str = ''
        else:
            optimArg_str =  str(optimArg).replace(' ','_')
    verboseMI_max = verbose
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
    if AggregW is None or AggregW=='':
        AggregW_str =''
    elif AggregW=='AveragingWportion' and not(proportionToKeep==1.0):
        AggregW_str = '_AvW'+str(proportionToKeep) 
    elif AggregW=='AveragingW' or (AggregW=='AveragingWportion' and (proportionToKeep==1.0)):
        AggregW_str = '_AveragingW'
    elif AggregW=='meanOfProd':
        AggregW_str = '_VW'+str(proportionToKeep)
    elif AggregW=='medianOfProd':
        AggregW_str = '_VMedW'+str(proportionToKeep)
    elif AggregW=='maxOfProd':
        AggregW_str = '_VMaxW'+str(proportionToKeep)
    elif AggregW=='minOfProd':
        AggregW_str = '_VMinW'+str(proportionToKeep)
    elif AggregW=='meanOfTanh':
        AggregW_str = '_VTanh'+str(proportionToKeep)
    elif AggregW=='medianOfTanh':
        AggregW_str = '_VMedTanh'+str(proportionToKeep)
    elif AggregW=='maxOfTanh':
        AggregW_str = '_VMaxTanh'+str(proportionToKeep)
    elif AggregW=='minOfTanh':
        AggregW_str = '_VMinTanh'+str(proportionToKeep)
    elif AggregW=='meanOfSign':
        AggregW_str = '_VMeanSign'+str(proportionToKeep)
    
    if loss_type is None or loss_type=='':
        loss_type_str =''
    elif loss_type=='MSE':
        loss_type_str = 'LossMSE'
    elif loss_type=='hinge':
        loss_type_str = 'Losshinge'
    elif loss_type=='hinge_tanh':
        loss_type_str = 'LosshingeTanh'
    elif loss_type=='log':
        loss_type_str = 'LossLog'
    else:
        raise(NotImplementedError)
        
    if obj_score_add_tanh:
        str_obj_score_add_tanh = '_ScoreAdd'+str(lambdas)
    else:
        str_obj_score_add_tanh = ''
    if obj_score_mul_tanh:
        str_obj_score_mul_tanh = '_SMulTanh'
    else:
        str_obj_score_mul_tanh = ''
        
    if MaxOfMax:
        str_MaxOfMax = '_MaxOfMax'
    elif MaxMMeanOfMax :
        str_MaxOfMax = '_MaxMMeanOfMax'
    else:
        str_MaxOfMax =''
    if Cosine_ofW_inLoss:
        str_Cosine_ofW_inLoss = '_Cosine_ofW_inLoss'+str(Coeff_cosine)
    else:
        str_Cosine_ofW_inLoss = ''
        
    Number_of_positif_elt = 1 
    number_zone = k_per_bag
    
#    thresh_evaluation,TEST_NMS = 0.05,0.3
    dont_use_07_metric = True
    symway = True
    

        
    if predict_with=='':
        predict_with = 'MI_max' 
    if not(PCAuse):
        PCAusestr =''
    else:
        PCAusestr= '_PCAc'+str(number_composant)
     
        
    if metamodel=='FasterRCNN':
        metamodelstr=''
    else:
        metamodelstr ='_'+ metamodel
        
    if metamodel=='FasterRCNN':
        dim_rois = 5
    if metamodel=='EdgeBoxes':
        dim_rois = 4

    arrayParam = [demonet,database,N,extL2,nms_thresh,savedstr,mini_batch_size,
                  performance,buffer_size,predict_with,shuffle,C,testMode,restarts,max_iters_all_base,
                  max_iters,CV_Mode,num_split,parallel_op,WR,norm,Optimizer,LR,optimArg,
                  Number_of_positif_elt,number_zone,seuil_estimation,thresh_evaluation,
                  TEST_NMS,init_by_mean,transform_output,with_rois_scores_atEnd,
                  with_scores,epsilon,restarts_paral,Max_version,w_exp,seuillage_by_score,seuil,
                  k_intopk,C_Searching,gridSearch,thres_FinalClassifier,optim_wt_Reg,AggregW,
                  proportionToKeep,loss_type,storeVectors,obj_score_add_tanh,lambdas,obj_score_mul_tanh,
                  model,metamodel,PCAuse,number_composant,AddOneLayer,exp,MaxOfMax,MaxMMeanOfMax,
                  alpha,layer,Cosine_ofW_inLoss,Coeff_cosine]
    arrayParamStr = ['demonet','database','N','extL2','nms_thresh','savedstr',
                     'mini_batch_size','performance','buffer_size','predict_with',
                     'shuffle','C','testMode','restarts','max_iters_all_base','max_iters','CV_Mode',
                     'num_split','parallel_op','WR','norm','Optimizer','LR',
                     'optimArg','Number_of_positif_elt','number_zone','seuil_estimation'
                     ,'thresh_evaluation','TEST_NMS','init_by_mean','transform_output','with_rois_scores_atEnd',
                     'with_scores','epsilon','restarts_paral','Max_version','w_exp','seuillage_by_score',
                     'seuil','k_intopk','C_Searching','gridSearch','thres_FinalClassifier','optim_wt_Reg',
                     'AggregW','proportionToKeep','loss_type','storeVectors','obj_score_add_tanh','lambdas',
                     'obj_score_mul_tanh','model','metamodel','PCAuse','number_composant',\
                     'AddOneLayer','exp','MaxOfMax','MaxMMeanOfMax','alpha','layer',\
                     'Cosine_ofW_inLoss','Coeff_cosine']
    assert(len(arrayParam)==len(arrayParamStr))
    print(tabs_to_str(arrayParam,arrayParamStr))
#    print('database',database,'mini_batch_size',mini_batch_size,'max_iters',max_iters,'norm',norm,\
#          'parallel_op',parallel_op,'CV_Mode',CV_Mode,'WR',WR,'restarts',restarts,'demonet',demonet,
#          'Optimizer',Optimizer,'init_by_mean',init_by_mean,'with_tanh',with_tanh)
    
    cachefilefolder = os.path.join(path_data,'cachefile')
    if not(layer=='fc7'):
        layerStr = '_'+layer
    else:
        layerStr = ''
    cachefile_model_base='WLS_'+ database+metamodelstr+ '_'+demonet+layerStr+'_r'+str(restarts)+'_s' \
        +str(mini_batch_size)+'_k'+str(k_per_bag)+'_m'+str(max_iters)+extNorm+extPar+\
        extCV+ext_test+opti_str+LR_str+C_str+init_by_mean_str+with_scores_str+restarts_paral_str\
        +Max_version_str+seuillage_by_score_str+shuffle_str+C_Searching_str+optim_wt_Reg_str+optimArg_str\
        + AggregW_str + loss_type_str+str_obj_score_add_tanh+str_obj_score_mul_tanh \
        + PCAusestr+str_MaxOfMax+str_Cosine_ofW_inLoss
    if trainOnTest:
        cachefile_model_base += '_trainOnTest'
    if AddOneLayer:
        cachefile_model_base += '_AddOneLayer'
    pathlib.Path(cachefilefolder).mkdir(parents=True, exist_ok=True)
    cachefile_model = os.path.join(cachefilefolder,cachefile_model_base+'_'+model_str+'.pkl')
#    if os.path.isfile(cachefile_model_old):
#        print('Do you want to erase the model or do a new one ?')
#        input_str = input('Answer yes or not')
#    cachefile_model = path_data + param_name + '.pkl'
    if verbose: print("cachefile name",cachefile_model)
    if not os.path.isfile(cachefile_model) or ReDo:
        name_milsvm = {}
        if verbose: print("The cachefile doesn t exist or we will erase it.")    
    else:
        with open(cachefile_model, 'rb') as f:
            name_milsvm = pickle.load(f)
            if verbose: print("The cachefile exists")
    
    usecache_eval = True
    boxCoord01 =False
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
    elif database=='IconArt_v1' or database=='RMN':
        imdb = get_imdb('IconArt_v1_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif 'IconArt_v1' in database and not('IconArt_v1' ==database):
        imdb = get_imdb('IconArt_v1_test',ext=database.split('_')[-1])
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
#        num_images = len(imdb.image_index) 
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        imdb = get_imdb('WikiTenLabels_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        #num_images = len(imdb.image_index) 
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif 'OIV5' in database: # For OIV5 for instance !
        num_images =  len(df_label[df_label['set']=='test'][item_name])
        usecache_eval  = False
        boxCoord01 = True
    else:
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
   
    data_path_train= dict_name_file['trainval']
    if trainOnTest:
        data_path_train= dict_name_file['test']
    
    if parallel_op:
        # For Pascal VOC2007 pour les 20 classes cela prend environ 2500s par iteration 
        if not os.path.isfile(cachefile_model) or ReDo:
             if verbose: t0 = time.time()
             
             if model=='MI_max' or model=='':
             
                 classifierMI_max = tf_MI_max(LR=LR,C=C,C_finalSVM=1.0,restarts=restarts,num_rois=k_per_bag,
                       max_iters=max_iters,symway=symway,n_jobs=n_jobs,buffer_size=buffer_size,
                       verbose=verboseMI_max,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                       mini_batch_size=mini_batch_size,num_features=num_features,debug=debug,
                       num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode,with_scores=with_scores,epsilon=epsilon,
                       Max_version=Max_version,seuillage_by_score=seuillage_by_score,w_exp=w_exp,seuil=seuil,
                       k_intopk=k_intopk,optim_wt_Reg=optim_wt_Reg,AggregW=AggregW,proportionToKeep=proportionToKeep,
                       loss_type=loss_type,obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                       obj_score_mul_tanh=obj_score_mul_tanh,AddOneLayer=AddOneLayer,exp=exp,\
                       MaxOfMax=MaxOfMax,MaxMMeanOfMax=MaxMMeanOfMax,usecache=usecache,\
                       alpha=alpha,Cosine_ofW_inLoss=Cosine_ofW_inLoss,Coeff_cosine=Coeff_cosine)
                 export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=-1,shuffle=shuffle,init_by_mean=init_by_mean,norm=norm,
                       WR=WR,performance=performance,restarts_paral=restarts_paral,
                       C_Searching=C_Searching,storeVectors=storeVectors,storeLossValues=storeLossValues)  
             elif model=='mi_model':
                 classifierMI_max = tf_mi_model(LR=LR,C=C,C_finalSVM=1.0,restarts=restarts,num_rois=k_per_bag,
                       max_iters=max_iters,symway=symway,n_jobs=n_jobs,buffer_size=buffer_size,
                       verbose=verboseMI_max,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                       mini_batch_size=mini_batch_size,num_features=num_features,debug=debug,
                       num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode,with_scores=with_scores,epsilon=epsilon,
                       Max_version=Max_version,seuillage_by_score=seuillage_by_score,w_exp=w_exp,seuil=seuil,
                       k_intopk=k_intopk,optim_wt_Reg=optim_wt_Reg,AggregW=AggregW,proportionToKeep=proportionToKeep,
                       loss_type=loss_type,obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                       obj_score_mul_tanh=obj_score_mul_tanh)
                 # TODO ajouter MaxOfMax
                 export_dir = classifierMI_max.fit_mi_model_tfrecords(data_path=data_path_train, \
                       class_indice=-1,shuffle=shuffle,init_by_mean=init_by_mean,norm=norm,
                       WR=WR,performance=performance,restarts_paral=restarts_paral,
                       C_Searching=C_Searching,storeVectors=storeVectors,storeLossValues=storeLossValues)  
             else:
                print('Model unknown')
                raise(NotImplementedError)
                 
             if verbose: 
                 t1 = time.time() 
                 print('Total duration training part :',str(t1-t0))
                 
             if storeVectors or storeLossValues:
                 print(export_dir)
                 return(export_dir,arrayParam)
                  
             np_pos_value,np_neg_value = classifierMI_max.get_porportions()
             name_milsvm =export_dir,np_pos_value,np_neg_value
             with open(cachefile_model, 'wb') as f:
                 pickle.dump(name_milsvm, f)
        else:
            export_dir,np_pos_value,np_neg_value= name_milsvm
#        plot_onSubSet =  ['angel','Child_Jesus', 'crucifixion_of_Jesus','Mary','nudity', 'ruins','Saint_Sebastien'] 
#        
        dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
        parameters=PlotRegions,RPN,Stocha,CompBest
        param_clf = k_per_bag,Number_of_positif_elt,num_features

        if database=='RMN': 
            database = 'IconArt_v1'
            item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC = get_database(database)

        true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
        ,all_boxes = \
        tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,
               path_to_img,path_data,param_clf,classes,parameters,verbose,
               seuil_estimation,thresh_evaluation,TEST_NMS,all_boxes=all_boxes,
               cachefile_model_base=cachefile_model_base,transform_output=transform_output,
               with_rois_scores_atEnd=with_rois_scores_atEnd,scoreInMI_max=(with_scores or seuillage_by_score),
               gridSearch=gridSearch,n_jobs=n_jobs,thres_FinalClassifier=thres_FinalClassifier,
               k_per_bag=k_per_bag,eval_onk300=eval_onk300,plot_onSubSet=plot_onSubSet,AggregW=AggregW,
               obj_score_add_tanh=obj_score_add_tanh,obj_score_mul_tanh=obj_score_mul_tanh,dim_rois=dim_rois,
               trainOnTest=trainOnTest,usecache=usecache_eval,boxCoord01=boxCoord01)
   
        for j,classe in enumerate(classes):
            AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                print("MIL-SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
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
    
    else:
        for j,classe in enumerate(classes):
            if testMode and not(j==jtest):
                continue
            if verbose : print(j,classes[j])       
            if verbose: print("Start train the MI_max")

            #data_path_train=  '/home/gonthier/Data_tmp/FasterRCNN_res152_COCO_Paintings_N1_TLforMIL_nms_0.7_all_trainval.tfrecords'
            needToDo = False
            if not(ReDo):
                try:
                    export_dir,np_pos_value,np_neg_value= name_milsvm[j]
                    print('The model of MI_max exists')
                except(KeyError):
                    print('The model of MI_max doesn t exist')
                    needToDo = True
            if ReDo or needToDo:
                classifierMI_max = tf_MI_max(LR=LR,C=C,C_finalSVM=1.0,restarts=restarts,num_rois=k_per_bag,
                       max_iters=max_iters,symway=symway,n_jobs=n_jobs,buffer_size=buffer_size,
                       verbose=verboseMI_max,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                       mini_batch_size=mini_batch_size,num_features=num_features,debug=debug,
                       num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode,with_scores=with_scores,epsilon=epsilon,
                       Max_version=Max_version,seuillage_by_score=seuillage_by_score,w_exp=w_exp,seuil=seuil,
                       k_intopk=k_intopk,optim_wt_Reg=optim_wt_Reg,AggregW=AggregW,proportionToKeep=proportionToKeep,
                       loss_type=loss_type,obj_score_add_tanh=obj_score_add_tanh,lambdas=lambdas,
                       obj_score_mul_tanh=obj_score_mul_tanh,AddOneLayer=AddOneLayer) 
                export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=j,shuffle=shuffle,init_by_mean=init_by_mean,norm=norm,
                       WR=WR,performance=performance,restarts_paral=restarts_paral,
                       C_Searching=C_Searching,storeVectors=storeVectors,storeLossValues=storeLossValues)
                np_pos_value,np_neg_value = classifierMI_max.get_porportions()
                name_milsvm[j]=export_dir,np_pos_value,np_neg_value
                with open(cachefile_model, 'wb') as f:
                    pickle.dump(name_milsvm, f)
    
            Number_of_positif_elt = 1 
            number_zone = k_per_bag
            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
            #print(export_dir)
           
            ## Predicition with the MI_max
            parameters=PlotRegions,RPN,Stocha,CompBest
            param_clf = k_per_bag,Number_of_positif_elt,num_features
            true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited,all_boxes = \
                tfR_evaluation(database=database,j=j,dict_class_weight=dict_class_weight,num_classes=num_classes,predict_with=predict_with,
                               export_dir=export_dir,dict_name_file=dict_name_file,mini_batch_size=mini_batch_size,config=config,
                               PlotRegions=PlotRegions,path_to_img=path_to_img,path_data=path_data,param_clf=param_clf,classes=classes,parameters=parameters,verbose=verbose,
                               seuil_estimation=seuil_estimation,thresh_evaluation=thresh_evaluation,TEST_NMS=TEST_NMS,
                               all_boxes=all_boxes,with_tanh=with_tanh,dim_rois=dim_rois)
                              
            # Regroupement des informations    
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training',\
                            'IconArt_v1'] \
                        or 'IconArt_v1' in database:
                name_all_test = df_label[df_label['set']=='test'].as_matrix([item_name])
                #print(name_all_test)
                #print(name_all_test.shape)
        
            AP = average_precision_score(true_label_all_test,predict_label_all_test,average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                print("MIL-model version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("MIL-model version Average Precision for",classes[j]," = ",AP)
            test_precision = precision_score(true_label_all_test,labels_test_predited)
            test_recall = recall_score(true_label_all_test,labels_test_predited)
            F1 = f1_score(true_label_all_test,labels_test_predited)
            print("Test on all the data precision = {0:.2f}, recall = {1:.2f},F1 = {2:.2f}".format(test_precision,test_recall,F1))
            precision_at_k = ranking_precision_score(np.array(true_label_all_test), predict_label_all_test,20)
            P20_per_class += [precision_at_k]
            AP_per_class += [AP]
            R_per_class += [test_recall]
            P_per_class += [test_precision]
    # End of the loop on the different class
    with open(cachefile_model, 'wb') as f:
        pickle.dump(name_milsvm, f)
    
    # Detection evaluation
    if database in ['RMN','VOC2007','watercolor','clipart','WikiTenLabels',\
                    'PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training','IconArt_v1']\
                    or 'IconArt_v1' in database:
        if testMode:
            for j in range(0, imdb.num_classes-1):
                if not(j==jtest):
                    #print(all_boxes[jtest])
                    all_boxes[j] = all_boxes[jtest]
        det_file = os.path.join(path_data, 'detections_aux.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        max_per_image = 100
        num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels or IconArt_v1 case
        all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
        number_im = 0
        name_all_test = name_all_test.astype(str)
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
        print("Detection score (thres = 0.5): ",database,'with ',model,'with score =',with_scores)
        if not(AggregW is None or AggregW==''):
            print('AggregW :',AggregW,proportionToKeep,'of',str(restarts+1),'vectors')
        print(arrayToLatex(aps,per=True))
        ovthresh_tab = [0.3,0.1,0.]
        for ovthresh in ovthresh_tab:
            aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
            if ovthresh == 0.1:
                apsAt01 = aps
            print("Detection score with thres at ",ovthresh,'with ',model,'with score =',with_scores)
            print(arrayToLatex(aps,per=True))
        imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb 
        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
        print("Detection score with the difficult elementwith ",model)
        print(arrayToLatex(aps,per=True))
        imdb.set_use_diff(False)
    
    elif 'OIV5' in database:
        OIV5_csv_file = cachefile_model_base  + '_boxes_predited.csv'
        with open(OIV5_csv_file, 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            first_line = ['ImageID','LabelName','XMin','XMax','YMin','YMax','IsGroupOf','Score']
            filewriter.writerow(first_line)
            for i_name, name in enumerate(name_all_test):
                for j in range(num_classes):
                    list_boxes = all_boxes[j][i_name]
                    classe_str = classes[j]
                    for box in list_boxes:
                        line = [name,classe_str,box[0],box[1],box[2],box[3],0,box[5]]
                        filewriter.writerow(line)
    
    
        
    print('~~~~~~~~')        
    print("mean Average Precision Classification for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
    if CompBest: print("mean Average Precision for BEst Score = {0:.3f}".format(np.mean(AP_per_classbS))) 
    print("mean Precision Classification for all the data = {0:.3f}".format(np.mean(P_per_class)))  
    print("mean Recall Classification for all the data = {0:.3f}".format(np.mean(R_per_class)))  
    print("mean Precision Classification @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    print('Mean Average Precision Classification with ',model,'with score =',with_scores,' : ')
    print(AP_per_class)
    print(arrayToLatex(AP_per_class,per=True))
    
    param_name,path_data_file,file_param = \
    create_param_id_file_and_dir(path_data+'/SauvParam/',arrayParam,arrayParamStr)
    
    if database in ['RMN','VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt',\
                    'MiniTrain_WikiTenLabels','WikiLabels1000training','IconArt_v1']\
                    or 'IconArt_v1' in database:
        write_results(file_param,[classes,AP_per_class,np.mean(AP_per_class),aps,np.mean(aps)],
                      ['classes','AP_per_class','mAP Classif','AP detection','mAP detection'])
        return(apsAt05,apsAt01,AP_per_class)
    else:
        write_results(file_param,[classes,AP_per_class,np.mean(AP_per_class)],
                      ['classes','AP_per_class','mAP Classif'])
        return(AP_per_class) 

def plot_Correct_Incorrect_Images(all_boxes,imbd,database):
    """
    Fonction jamais finie en fait
    """
    # imbd = get_imdb('watercolor_test')
    GT= imbd.gt_roidb()
    ii = 0 
    thresh = 0.75
    for k,boxGT in enumerate(GT):
        complet_name = imbd.image_path_at(k)
        im = cv2.imread(complet_name)
        blobs, im_scales = get_blobs(im)
        roi_boxes_and_score = []
        local_cls = []
        Correct = 'Correct'
        for j in range(imbd.num_classes):
            cls_dets = all_boxes[j][ii] # Here we have #classe x box dim + score
            if len(cls_dets) > 0:
                roi_boxes_score = cls_dets
                inds = np.where(roi_boxes_score[:, -1] >= thresh)[0]
                if not(len(inds) == 0):
                    roi_boxes_score =  roi_boxes_score[inds,:]
                    local_cls += [imbd.classes[j]]
                    if roi_boxes_and_score is None:
                        roi_boxes_and_score = [roi_boxes_score]
                    else:
                        roi_boxes_and_score += [roi_boxes_score]
                    if not(j in boxGT['gt_classes']):
                         Correct = 'Incorrect'   
                    else:
#                        get_iou
                        print('pas fini')
            else:
                # This class is not predected
                if j in boxGT['gt_classes']:
                    if not(Correct=='Incorrect'):
                        Correct = 'ClassMissing'
                    
                    
                
        ii += 1        

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,
               path_to_img,path_data,param_clf,classes,parameters,verbose,
               seuil_estimation,thresh_evaluation,TEST_NMS,all_boxes=None,
               cachefile_model_base='',number_im=np.inf,transform_output=None,
               with_rois_scores_atEnd=False,scoreInMI_max=False,seuillage_by_score=False,
               gridSearch=False,n_jobs=1,thres_FinalClassifier=0.5,k_per_bag=300,
               eval_onk300=False,plot_onSubSet=None,AggregW=None,obj_score_add_tanh=False,
               obj_score_mul_tanh=False,dim_rois=5,trainOnTest=False,usecache=True,
               boxCoord01=False):
     """
     @param : seuil_estimation : ByHist or MaxDesNeg
     @param : number_im : number of image plot at maximum
     @param : transform_output : use of a softmax or a 
     @ use the with_rois_scores_atEnd to pondare the final results
     @param : boxCoord01 : convert the box coordinates between 0 and 1
     """

     # TODO : predict_with LinearSVC_Extremboth : use the most discriminative regions for each images
     # LinearSVC_Sign: use the negative examples for negatives and positive for positives
     # LinearSVC_MAXPos : use all the regions of negatives examples and the max for positives
     # LinearSVC_Seuil : select of the regions for the negative and with a threshold for the positive
     PlotRegions,RPN,Stocha,CompBest=parameters # TOFO modify that you have a redundancy for PlotRegions

     if predict_with=='mi_model':
         predict_with='MI_max'

     if obj_score_add_tanh or obj_score_mul_tanh:
        scoreInMI_max =True

     if not(plot_onSubSet is None):
         PlotRegions = True
         index_SubSet = []
         for elt in plot_onSubSet:
             index_SubSet += [np.where(np.array(classes)==elt)[0][0]]   
    
     if seuil_estimation in ['ByHist','MinDesPos','MaxDesNeg','byHistOnPos']:
         seuil_estimation_bool = True
     else:
         seuil_estimation_bool = False
    
     if verbose: print('thresh_evaluation',thresh_evaluation,'TEST_NMS',TEST_NMS,'seuil_estimation',seuil_estimation)
     
     k_per_bag,positive_elt,num_features = param_clf
     thresh = thresh_evaluation # Threshold score or distance MI_max
     #TEST_NMS = 0.7 # Recouvrement entre les classes
     thres_max = False
     load_model = False
     with_softmax,with_tanh,with_softmax_a_intraining = False,False,False
     if transform_output=='tanh':
         with_tanh=True
     elif transform_output=='softmax':
         with_softmax=True
         if seuil_estimation_bool: print('It may cause problem of doing softmax and tangent estimation')
     elif  transform_output=='softmaxTraining':
         with_softmax_a_intraining = True
     with_tanh_alreadyApplied = False
     if not(AggregW is None): 
        if'Tanh' in AggregW or 'Sign' in AggregW: 
            # The tanh transformation have already be done
            with_tanh = False
            with_tanh_alreadyApplied = True
     if obj_score_add_tanh or obj_score_mul_tanh:
        with_tanh = False
        with_tanh_alreadyApplied = True
     seuil_estimation_debug = False
     plot_hist = False

     if PlotRegions or (seuil_estimation_bool and plot_hist):
         extensionStocha = cachefile_model_base 
         if not(plot_onSubSet is None):
             extensionStocha += 'ForIllustraion'
         path_to_output2  = path_data + '/tfMI_maxRegion_paral/'+database+'/'+extensionStocha
         if RPN:
             path_to_output2 += '/RPNetMISVM/'
         elif CompBest:
              path_to_output2 += '/BestObject/'
         elif  trainOnTest:
            path_to_output2 += '_trainOnTest/'
         else:
             path_to_output2 += '/'
         path_to_output2_bis = path_to_output2 + 'Train'
         path_to_output2_ter = path_to_output2 + 'Test'
         path_to_output2_q = path_to_output2 + 'Hist/'
         pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
         pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True)
         pathlib.Path(path_to_output2_q).mkdir(parents=True, exist_ok=True)
         
     export_dir_path = ('/').join(export_dir.split('/')[:-1])
     name_model_meta = export_dir + '.meta'
     
#     if 'LinearSVC' in predict_with:
#         raise(NotImplementedError)
#         length_matrix = dict_class_weight[0] + dict_class_weight[1]
#         if length_matrix>17500*300:
#             print('Not enough memory on Nicolas Computer ! use an other classifier than LinearSVC')
#             raise(MemoryError)
#         X_array = np.empty((length_matrix,num_features),dtype=np.float32)
#         y_array =  np.empty((num_classes,length_matrix),dtype=np.float32)
#         x_array_ind = 0
     
     if seuil_estimation_bool:
         if seuil_estimation_debug and not(seuil_estimation=='byHistOnPos'):
             top_k = 3
         else:
             top_k =1
         list_arrays = ['prod_neg_ex','prod_pos_ex_topk','prod_pos_ex']
         dict_seuil_estim = {}
         for i in range(num_classes):
             dict_seuil_estim[i] = {}
             for str_name in list_arrays:
                 dict_seuil_estim[i][str_name] = []  # Array of the scalar product of the negative examples  
     get_roisScore = (with_rois_scores_atEnd or scoreInMI_max)

     if (PlotRegions or seuil_estimation_bool) and not('LinearSVC' in predict_with):
        index_im = 0
        if verbose: print("Start ploting Regions selected by the MI_max in training phase")
        if trainOnTest:
            train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
        else:
            train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,\
            num_rois=k_per_bag,dim_rois=dim_rois))
        dataset_batch = train_dataset.batch(mini_batch_size)
        if usecache:
            dataset_batch.cache()
        iterator = dataset_batch.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        with tf.Session(config=config) as sess:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            load_model = True
            graph= tf.get_default_graph()
            if not(k_per_bag==300) and eval_onk300:
                print('Que fais tu la ?')
                X = tf.placeholder(tf.float32, shape=(None,300,num_features),name='X')
                y = tf.placeholder(tf.float32, shape=(None,num_classes),name='y')
                if scoreInMI_max:
                    scores_tf = tf.placeholder(tf.float32, shape=(None,),name='scores')
            else:
                X = get_tensor_by_nameDescendant(graph,"X")
                y = get_tensor_by_nameDescendant(graph,"y")
            if scoreInMI_max: 
                scores_tf = get_tensor_by_nameDescendant(graph,"scores")
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best = get_tensor_by_nameDescendant(graph,"ProdScore")
            else:
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best =  get_tensor_by_nameDescendant(graph,"Prod")
            if with_tanh:
                if verbose: print('use of tanh')
                Tanh = tf.tanh(Prod_best)
                mei = tf.argmax(Tanh,axis=2)
                score_mei = tf.reduce_max(Tanh,axis=2)
            elif with_softmax:
                if verbose: print('use of softmax')
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
#                    print(sess.run(next_element))
                    next_element_value = sess.run(next_element)
#                    print(len(next_element_value))
                    if not(with_rois_scores_atEnd) and not(scoreInMI_max):
                        fc7s,roiss, labels,name_imgs = next_element_value
                    else:
                        fc7s,roiss,rois_scores,labels,name_imgs = next_element_value
                    if scoreInMI_max:
                        feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                    else:
                        feed_dict_value = {X: fc7s, y: labels}
                    if with_tanh:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                        sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                    elif with_softmax or with_softmax_a_intraining:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                        sess.run([mei,score_mei,Softmax], feed_dict=feed_dict_value)
                    else:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = \
                        sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)
                    if with_rois_scores_atEnd:
                        PositiveExScoreAll = PositiveExScoreAll*rois_scores
                        score_mei = np.max(PositiveExScoreAll,axis=2)
                        mei = np.amax(PositiveExScoreAll,axis=2)
#                    if with_tanh: assert(np.max(PositiveExScoreAll) <= 1.)
                                                    
                    if seuil_estimation_bool:
                        for k in range(len(fc7s)):
                            for l in range(num_classes):
                                label_i = labels[k,l]
                                if label_i ==0:
                                    dict_seuil_estim[l]['prod_neg_ex'] += [PositiveExScoreAll[l,k,:]]
                                else:
                                    dict_seuil_estim[l]['prod_pos_ex'] += [PositiveExScoreAll[l,k,:]]
                                    a= PositiveExScoreAll[l,k,:][np.argsort(PositiveExScoreAll[l,k,:])[-top_k:]]
                                    # Shape 3 
                                    dict_seuil_estim[l]['prod_pos_ex_topk'] += [a]
                            
                                
                    if PlotRegions:
                        print('Start plotting Training exemples')
                        for k in range(len(labels)):
                            if index_im > number_im:
                                continue
                            if database in ['IconArt_v1','VOC2007','watercolor','Paintings','clipart','WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                                name_img = str(name_imgs[k].decode("utf-8") )
                            else:
                                name_img = name_imgs[k]
                            rois = roiss[k,:]
                            #if verbose: print(name_img)
                            if database in ['IconArt_v1','VOC12','Paintings','VOC2007','clipart','watercolor','WikiTenLabels','PeopleArt','WikiLabels1000training','MiniTrain_WikiTenLabels']:
                                complet_name = path_to_img + name_img + '.jpg'
                                name_sans_ext = name_img
                            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                                name_sans_ext = os.path.splitext(name_img)[0]
                                complet_name = path_to_img +name_sans_ext + '.jpg'
                            im = cv2.imread(complet_name)
                            blobs, im_scales = get_blobs(im)
                            scores_all = PositiveExScoreAll[:,k,:]
                            roi = roiss[k,:]
                            if dim_rois==5:
                                roi_boxes =  roi[:,1:5] / im_scales[0] 
                            else:
                                roi_boxes =  roi / im_scales[0] 
                            roi_boxes_and_score = None
                            local_cls = []
                            for j in range(num_classes):
                                if (not(plot_onSubSet is None) and (j in index_SubSet)) or (plot_onSubSet is None):
                                    if labels[k,j] == 1:
                                        local_cls += [classes[j]]
                                        roi_with_object_of_the_class = PositiveRegions[j,k] % len(rois) # Because we have repeated some rois
                                        roi = rois[roi_with_object_of_the_class,:]
                                        roi_scores = [get_PositiveRegionsScore[j,k]]
                                        if dim_rois==5:
                                            roi_boxes =  roi[1:5] / im_scales[0]
                                        else:
                                            roi_boxes =  roi / im_scales[0]   
                                        roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                                        if roi_boxes_and_score is None:
                                            roi_boxes_and_score = roi_boxes_score
                                        else:
                                            roi_boxes_and_score= \
                                            np.vstack((roi_boxes_and_score,roi_boxes_score))

                            if RPN:
                                best_RPN_roi = rois[0,:]
                                best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                                best_RPN_roi_scores = [PositiveExScoreAll[j,k,0]]
                                cls = local_cls + ['RPN']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                roi_boxes_and_score = np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                            else:
                                cls = local_cls
                            if roi_boxes_and_score is None:
                                # Pour les images qui ne contiennent aucune classe du tout
                                roi_boxes_and_score = []
                             
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Train/' + name_sans_ext + '_Regions.jpg'
                            if database=='PeopleArt':
                                path_tmp = '/'.join(name_output.split('/')[0:-1])
                                pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                            plt.savefig(name_output)
                            plt.close()
                            index_im +=1
                except tf.errors.OutOfRangeError:
                    break
       
     # Training the differents SVC for each class 
     # Training Time !! 
     if 'LinearSVC' in predict_with:
        if not(AggregW is None or AggregW==''): raise(NotImplementedError)
        if verbose: print('predict_with',predict_with)
        classifier_trained_dict = {}
        load_model = True
        train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,
            num_rois=k_per_bag,dim_rois=dim_rois))
        dataset_batch = train_dataset.batch(mini_batch_size)
        if usecache:
            dataset_batch.cache()
        dataset_batch = dataset_batch.prefetch(1)
        iterator = dataset_batch.make_initializable_iterator()
        next_element = iterator.get_next()
        
        with tf.Session(config=config) as sess:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            load_model = True
            graph= tf.get_default_graph()
            if not(k_per_bag==300) and eval_onk300:
                print('Que fais tu la ?')
                X = tf.placeholder(tf.float32, shape=(None,300,num_features),name='X')
                y = tf.placeholder(tf.float32, shape=(None,num_classes),name='y')
                if scoreInMI_max:
                    scores_tf = tf.placeholder(tf.float32, shape=(None,),name='scores')
            else:
                X = get_tensor_by_nameDescendant(graph,"X")
                y = get_tensor_by_nameDescendant(graph,"y")
            if scoreInMI_max: 
                scores_tf = get_tensor_by_nameDescendant(graph,"scores")
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best = get_tensor_by_nameDescendant(graph,"ProdScore")
            else:
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best =  get_tensor_by_nameDescendant(graph,"Prod")
            if with_tanh:
                print('use of tanh')
                Tanh = tf.tanh(Prod_best)
                mei = tf.argmax(Tanh,axis=2)
                score_mei = tf.reduce_max(Tanh,axis=2)
            elif with_softmax:
                print('use of softmax')
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
            for j in range(num_classes):
                y_trainval_select = []
                X_trainval_select = []
                sess.run(iterator.initializer)
                while True:
                    try:
                        next_element_value = sess.run(next_element)
                        if not(with_rois_scores_atEnd) and not(scoreInMI_max):
                            fc7s,roiss, labels,name_imgs = next_element_value
                        else:
                            fc7s,roiss,rois_scores,labels,name_imgs = next_element_value
                        if scoreInMI_max:
                            feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                        else:
                            feed_dict_value = {X: fc7s, y: labels}
                        if with_tanh:
                            PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                            sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                        elif with_softmax or with_softmax_a_intraining:
                            PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                            sess.run([mei,score_mei,Softmax], feed_dict=feed_dict_value)
                        else:
                            PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = \
                            sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)

                        for i in range(len(fc7s)):
                            if predict_with=='LinearSVC_Extremboth' or predict_with=='LinearSVC_MAXPos':
                                if labels[i,j] == 1: # Positive exemple
                                    if not(len(X_trainval_select)==0):
                                        if predict_with=='LinearSVC_Extremboth' or predict_with=='LinearSVC_MAXPos':
                                            y_trainval_select+= [1]
                                            X_trainval_select += [np.expand_dims(fc7s[i,PositiveRegions[j,i],:],axis=0)]
                                    else:
                                        if predict_with=='LinearSVC_Extremboth' or predict_with=='LinearSVC_MAXPos':
                                            y_trainval_select = [1]
                                            X_trainval_select = [np.expand_dims(fc7s[i,PositiveRegions[j,i],:],axis=0)] # The roi_scores vector is sorted
                                else:
                                    if not(len(X_trainval_select)==0):
                                        if predict_with=='LinearSVC_Extremboth':
                                            y_trainval_select+= [0]
                                            X_trainval_select += [np.expand_dims(fc7s[i,np.argmin(PositiveExScoreAll[j,i,:]),:],axis=0)]
                                        elif  predict_with=='LinearSVC_MAXPos':
                                            y_trainval_select+= [0]*len(fc7s[i,:,:])
                                            X_trainval_select += [fc7s[i,:,:]]
                                    else:
                                        if predict_with=='LinearSVC_Extremboth':
                                            y_trainval_select= [0]
                                            X_trainval_select = [np.expand_dims(fc7s[i,np.argmin(PositiveExScoreAll[j,i,:]),:],axis=0)]
                                        elif predict_with=='LinearSVC_MAXPos':
                                            y_trainval_select= [0]*len(fc7s[i,:,:])
                                            X_trainval_select = [fc7s[i,:,:]]
                            elif predict_with=='LinearSVC_Sign':
                                # TODO : this is not optimal at all to recompu
                                index_pos = np.where(PositiveExScoreAll[j,i,:] > 0)
                                y_trainval_select_tmp = np.zeros((len(PositiveExScoreAll[j,i,:],)),dtype=np.float32)
                                y_trainval_select_tmp[index_pos] = 1  
                                if not(len(X_trainval_select)==0):
                                    y_trainval_select = np.hstack((y_trainval_select,y_trainval_select_tmp))
                                    X_trainval_select += [fc7s[i,:,:]]
                                else:
                                    X_trainval_select = [fc7s[i,:,:]]
                                    y_trainval_select = y_trainval_select_tmp
                            elif predict_with=='LinearSVC_Seuil':
                                if labels[i,j] == 1:
                                    index_pos = np.where(PositiveExScoreAll[j,i,:] > thres_FinalClassifier)

                                    y_trainval_select_tmp = np.ones((len(index_pos[0]),),dtype=np.float32) 
                                    if not(len(X_trainval_select)==0):
                                        y_trainval_select = np.hstack((y_trainval_select,y_trainval_select_tmp))
                                        X_trainval_select += [fc7s[i,index_pos[0],:]]
                                    else:
                                        X_trainval_select = [fc7s[i,index_pos[0],:]]
                                        y_trainval_select = y_trainval_select_tmp
                                else:
                                    y_trainval_select_tmp = np.zeros((len(PositiveExScoreAll[j,i,:],)),dtype=np.float32)
                                    if not(len(X_trainval_select)==0):
                                        y_trainval_select = np.hstack((y_trainval_select,y_trainval_select_tmp))
                                        X_trainval_select += [fc7s[i,:,:]]
                                    else:
                                        X_trainval_select = [fc7s[i,:,:]]
                                        y_trainval_select = y_trainval_select_tmp
                                
                    except tf.errors.OutOfRangeError:
                        break
                        
#                    print(X_trainval_select)
#                    print(X_trainval_select[0].shape)
#                    print(X_trainval_select[-1].shape)
                X_trainval_select = np.array(np.concatenate(X_trainval_select,axis=0),dtype=np.float32)
                y_trainval_select = np.array(y_trainval_select,dtype=np.float32)
                if verbose: print("Shape X and y",X_trainval_select.shape,y_trainval_select.shape)
                if verbose: print("Start learning for class",j)
                classifier_trained = TrainClassif(X_trainval_select,y_trainval_select,
                   clf='LinearSVC',class_weight='balanced',gridSearch=gridSearch,n_jobs=n_jobs,C_finalSVM=1)
                if verbose: print("End learning for class",j)
                classifier_trained_dict[j] = classifier_trained

     
     # Parameter Evaluation Time !
     if seuil_estimation=='byHist':
         list_thresh = []
         plt.ion()
         if verbose: print('Seuil Estimation Time')
         num_bins = 100
         # Concatenation of the element
         for l in range(num_classes):
             dontPlot = False
             dontPlot2 = False
             prod_neg_ex = np.concatenate(dict_seuil_estim[l]['prod_neg_ex'],axis=0)
             prod_pos_ex = np.concatenate(dict_seuil_estim[l]['prod_pos_ex'],axis=0)
             prod_pos_ex_topk= np.vstack(dict_seuil_estim[l]['prod_pos_ex_topk'])
             
             if plot_hist:
                 plt.figure()
                 plt.hist(prod_neg_ex,alpha=0.5,bins=num_bins,label='hist_neg',density=True)
             
             e_max = max(np.max(prod_neg_ex),np.max(prod_pos_ex_topk))
             e_min = min(np.min(prod_neg_ex),np.min(prod_pos_ex_topk))
             x_axis = np.linspace(e_min,e_max, num_bins)
             kdeA, pdfA = kde_sklearn(prod_neg_ex, x_axis, bandwidth=0.25)
             funcA = lambda x: np.exp(kdeA.score_samples([x][0]))
             if seuil_estimation_debug:
                 kdeB, pdfB = kde_sklearn(prod_pos_ex, x_axis, bandwidth=0.25)
                 funcB = lambda x: np.exp(kdeB.score_samples([x][0]))
                 try:
                     result_full = findIntersection(funcA, funcB,e_min,e_max)
                     print('Seuil if we consider all the element of each image',result_full,'for class ',l)
                 except(ValueError):
                     dontPlot =True
                     print('seuil not found in the case of full data')
             for kk in range(top_k):
                 kdeB, pdfB = kde_sklearn(prod_pos_ex_topk[:,kk], x_axis, bandwidth=0.25)
                 funcB = lambda x: np.exp(kdeB.score_samples([x][0]))
                 try:
                     result = findIntersection(funcA, funcB,e_min,e_max)
                 except(ValueError):
                     dontPlot2 =True
                     result = thresh_evaluation
                     print('Intersection not found in the case',kk)
                 if kk==0:
                     if thres_max:
                         list_thresh += [max(result,thresh_evaluation)]
                     else:
                         list_thresh += [result]
                     if not(dontPlot2): plt.axvline(result, color='red')
                 if seuil_estimation_debug: print(kk,result)
                 if plot_hist:
                     label_str = 'hist_pos '+str(kk)
                     plt.hist(prod_pos_ex_topk[:,kk],alpha=0.3,bins=num_bins,label=label_str,density=True)
#                plt.xlim(min(bin_edges), max(bin_edges))
             if plot_hist:
                 plt.legend(loc='best')
                 plt.title('Histogram of the scalar product for class '+str(l))
                 name_output = path_to_output2_q + 'Hist_top_'+ str(top_k) + '_class'+str(l)+'.jpg'
                 plt.savefig(name_output)
                 plt.close()
                 if seuil_estimation_debug:
                     plt.figure()
                     plt.hist(prod_neg_ex,alpha=0.5,bins=num_bins,label='hist_neg',density=True)
                     plt.hist(prod_pos_ex,alpha=0.5,bins=num_bins,label='hist_pos',density=True)
                     if not(dontPlot) : plt.axvline(result_full, color='red')
                     plt.legend(loc='best')
                     plt.title('Histogram of the scalar product for class '+str(l)+' for all element')
                     name_output = path_to_output2_q + 'Hist_all_class'+str(l)+'.jpg'
                     plt.savefig(name_output)
                     plt.close() 
                     
     if seuil_estimation=='MaxDesNeg':
         # On prend le maximum des exemples negatifs 
         list_thresh = np.zeros((num_classes,))
         for l in range(num_classes):
             prod_neg_ex = np.concatenate(dict_seuil_estim[l]['prod_neg_ex'],axis=0)
             seuil_max_negative = np.max(prod_neg_ex)
             list_thresh[l] = seuil_max_negative  
             
     if seuil_estimation=='MinDesPos':
         # On prend le minimum des max des exemples positifs 
         list_thresh = np.zeros((num_classes,))
         for l in range(num_classes):
             prod_pos_ex_topk= np.vstack(dict_seuil_estim[l]['prod_pos_ex_topk'])
             prod_pos_ex_topk = prod_pos_ex_topk[np.where(prod_pos_ex_topk>0)]
             seuil_estim =np.min(prod_pos_ex_topk)   
             list_thresh[l] = seuil_estim 
             
     if seuil_estimation=='byHistOnPos':
         list_thresh = []
         plt.ion()
         num_bins = 100
         # Concatenation of the element
         for l in range(num_classes):
             dontPlot = False
             dontPlot2 = False
             prod_neg_ex = np.concatenate(dict_seuil_estim[l]['prod_neg_ex'],axis=0)
             prod_pos_ex = np.concatenate(dict_seuil_estim[l]['prod_pos_ex'],axis=0)
             prod_pos_ex_topk= np.vstack(dict_seuil_estim[l]['prod_pos_ex_topk'])
             prod_neg_ex = prod_neg_ex[np.where(prod_neg_ex>0)]
             prod_pos_ex_topk = prod_pos_ex_topk[np.where(prod_pos_ex_topk>0)][:,np.newaxis]
             if plot_hist:
                 plt.figure()
                 plt.hist(prod_neg_ex,alpha=0.5,bins=num_bins,label='hist_neg',density=True)
             e_max = max(np.max(prod_neg_ex),np.max(prod_pos_ex_topk))
             e_min = min(np.min(prod_neg_ex),np.min(prod_pos_ex_topk))
             x_axis = np.linspace(e_min,e_max, num_bins)
             kdeA, pdfA = kde_sklearn(prod_neg_ex, x_axis, bandwidth=0.25)
             funcA = lambda x: np.exp(kdeA.score_samples([x][0]))
             if seuil_estimation_debug:
                 kdeB, pdfB = kde_sklearn(prod_pos_ex, x_axis, bandwidth=0.25)
                 funcB = lambda x: np.exp(kdeB.score_samples([x][0]))
                 try:
                     result_full = findIntersection(funcA, funcB,e_min,e_max)
                     print('Seuil if we consider all the element of each image',result_full,'for class ',l)
                 except(ValueError):
                     dontPlot =True
                     print('seuil not found in the case of full data')
             for kk in range(top_k):
                 kdeB, pdfB = kde_sklearn(prod_pos_ex_topk[:,kk], x_axis, bandwidth=0.25)
                 funcB = lambda x: np.exp(kdeB.score_samples([x][0]))
                 try:
                     result = findIntersection(funcA, funcB,e_min,e_max)
                 except(ValueError):
                     dontPlot2 =True
                     result = thresh_evaluation
                     print('Intersection not found in the case',kk)
                 if kk==0:
                     if thres_max:
                         list_thresh += [max(result,thresh_evaluation)]
                     else:
                         list_thresh += [result]
                     if not(dontPlot2): plt.axvline(result, color='red')
                 if seuil_estimation_debug: print('seuil estime : ',kk,result)
                 if plot_hist:
                     label_str = 'hist_pos '+str(kk)
                     plt.hist(prod_pos_ex_topk[:,kk],alpha=0.3,bins=num_bins,label=label_str,density=True)
#                plt.xlim(min(bin_edges), max(bin_edges))
                 if plot_hist:
                     plt.legend(loc='best')
                     plt.title('Histogram of the scalar product for class '+str(l))
                     name_output = path_to_output2_q + 'Hist_top_'+ str(top_k) + '_class'+str(l)+'.jpg'
                     plt.savefig(name_output)
                     plt.close()
                     if seuil_estimation_debug:
                         plt.figure()
                         plt.hist(prod_neg_ex,alpha=0.5,bins=num_bins,label='hist_neg',density=True)
                         plt.hist(prod_pos_ex,alpha=0.5,bins=num_bins,label='hist_pos',density=True)
                         if not(dontPlot) : plt.axvline(result_full, color='red')
                         plt.legend(loc='best')
                         plt.title('Histogram of the scalar product for class '+str(l)+' for all element')
                         name_output = path_to_output2_q + 'Hist_all_class'+str(l)+'.jpg'
                         plt.savefig(name_output)
                         plt.close() 
                         
     if verbose: print("Testing Time")            
     # Testing time !
     train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
     if not(k_per_bag==300) and eval_onk300:
         train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,
            num_rois=300,dim_rois=dim_rois))
     else:
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,
            num_rois=k_per_bag,dim_rois=dim_rois))
     dataset_batch = train_dataset.batch(mini_batch_size)
     if usecache:
         dataset_batch.cache()
     iterator = dataset_batch.make_one_shot_iterator()
     next_element = iterator.get_next()
     true_label_all_test =  []
     predict_label_all_test =  []
     name_all_test =  []
     FirstTime= True
     i = 0
     ii = 0
     with tf.Session(config=config) as sess:
        if load_model==False:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            graph= tf.get_default_graph()
            if not(k_per_bag==300) and eval_onk300:
                print('Que fais tu la ?')
                X = tf.placeholder(tf.float32, shape=(None,300,num_features),name='X')
                y = tf.placeholder(tf.float32, shape=(None,num_classes),name='y')
                if scoreInMI_max:
                    scores_tf = tf.placeholder(tf.float32, shape=(None,),name='scores')
            else:
                X = get_tensor_by_nameDescendant(graph,"X")
                y = get_tensor_by_nameDescendant(graph,"y")
            if scoreInMI_max: 
                scores_tf = get_tensor_by_nameDescendant(graph,"scores")
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best = get_tensor_by_nameDescendant(graph,"ProdScore")
            else:
                if with_tanh_alreadyApplied:
                    Prod_best = get_tensor_by_nameDescendant(graph,"Tanh")
                else:
                    Prod_best =  get_tensor_by_nameDescendant(graph,"Prod")
            if with_tanh:
                print('We add the tanh in the test fct')
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
                mei = tf.argmax(Prod_best,axis=-1)
                score_mei = tf.reduce_max(Prod_best,axis=-1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        # Evaluation Test : Probleme ici souvent 
        while True:
            try:
                if not(with_rois_scores_atEnd) and not(scoreInMI_max):
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                else:
                    fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                if predict_with=='MI_max':
                    if scoreInMI_max:
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
#                    if with_tanh or with_tanh_alreadyApplied: 
#                        print('np.max(PositiveExScoreAll)',np.max(PositiveExScoreAll))
#                        assert(np.max(PositiveExScoreAll) <= 1.)

                true_label_all_test += [labels]
                
                if predict_with=='MI_max':
                    predict_label_all_test +=  [get_RegionsScore] # For the classification task
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
                    
                #print(PositiveExScoreAll.shape)
                for k in range(len(labels)):
                    if database in ['IconArt_v1','VOC2007','watercolor','Paintings','clipart',\
                                    'WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels',\
                                    'WikiLabels1000training'] or 'IconArt_v1' in database :
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    else:
                        complet_name = path_to_img + name_imgs[k] + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    if predict_with=='MI_max':
                        scores_all = PositiveExScoreAll[:,k,:]
                    elif 'LinearSVC' in predict_with:
                        scores_all = predict_label_all_test_batch[:,k,:]

                    roi = roiss[k,:]
                    if dim_rois==5:
                        roi_boxes =  roi[:,1:5] / im_scales[0] 
                    else:
                        roi_boxes =  roi / im_scales[0]
                    if boxCoord01:
                        roi_boxes[:,0:2] =  roi_boxes[:,0:2] / im.shape[0]
                        roi_boxes[:,2:4] =  roi_boxes[:,2:4] / im.shape[1]
                    
                    for j in range(num_classes):
                        scores = scores_all[j,:]
                        #print(scores.shape)
                        if seuil_estimation_bool:
                            inds = np.where(scores > list_thresh[j])[0]
                        else:
                            inds = np.where(scores > thresh)[0]
#                        print(inds)
#                        print(roi_boxes.shape)
                        cls_scores = scores[inds]
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                        
                        modif_box = '' # Possibility : SumPond, Inter
                        if(not(modif_box=='') and not(modif_box is None)):
                            # Modification of the bounding box 
                            cls_dets = py_cpu_modif(cls_dets,kind=modif_box)
#                        print(cls_dets)
                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        
                        all_boxes[j][i] = cls_dets
                    i+=1
    
                for l in range(len(name_imgs)): 
                    if database in ['IconArt_v1','VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                        name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                    else:
                        name_all_test += [[name_imgs[l]]]
                
                if PlotRegions and predict_with=='MI_max':
                    if verbose and (ii%1000==0):
                        print("Plot the images :",ii)
                    if verbose and FirstTime: 
                        FirstTime = False
                        print("Start ploting Regions on test set")
                    for k in range(len(labels)):   
                        if ii > number_im:
                            continue
                        if  database in ['IconArt_v1','VOC2007','Paintings','watercolor','clipart','WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                            name_img = str(name_imgs[k].decode("utf-8") )
                        else:
                            name_img = name_imgs[k]
                        rois = roiss[k,:]
                        if database in ['IconArt_v1','VOC12','Paintings','VOC2007','clipart','watercolor','WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training']:
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
                            if (not(plot_onSubSet is None) and (j in index_SubSet)) or (plot_onSubSet is None):
                            
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
                            print('You can have a problem here with the EdgeBoxes model or other model right number of dim_rois')
                            best_RPN_roi = rois[0,:]
                            best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                            best_RPN_roi_scores = [PositiveExScoreAll[j,k,0]]
                            cls = local_cls + ['RPN']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                            #best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                            best_RPN_roi_boxes_score =  np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0)
#                            print(best_RPN_roi_boxes_score.shape)
                            #roi_boxes_and_score = np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                            roi_boxes_and_score += [best_RPN_roi_boxes_score] #np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                        else:
                            cls = local_cls
                        #print(len(cls),len(roi_boxes_and_score))
                        if not(plot_onSubSet is None):
                            # In this case we will plot several version of the image
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions.jpg'
                            if database=='PeopleArt':
                                path_tmp = '/'.join(name_output.split('/')[0:-1])
                                pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                            plt.savefig(name_output)
                            plt.close()
                            
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.25,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions_over025.jpg'
                            plt.savefig(name_output)
                            plt.close()
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.5,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions_over05.jpg'
                            plt.savefig(name_output)
                            plt.close()
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.75,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions_over075.jpg'
                            plt.savefig(name_output)
                            plt.close()
                            
                            for j_subset in range(len(plot_onSubSet)):
                                j_class = np.where(np.array(classes)==plot_onSubSet[j_subset])[0][0]
                                cls_dets = all_boxes[j_class][ii-1] # Here we have #classe x box dim + score
                                class_name = plot_onSubSet[j_subset]
                                if not(len(cls_dets)==0):
                                    vis_detections(im, class_name, cls_dets, thresh=0.25,with_title=False)
                                    name_output = path_to_output2 +'Test/' + name_sans_ext + '_'+class_name+'_'+'_over25.jpg'
                                    plt.savefig(name_output)
                                    plt.close()
                                    vis_detections(im, class_name, cls_dets, thresh=0.5,with_title=False)
                                    name_output = path_to_output2 +'Test/' + name_sans_ext + '_'+class_name+'_'+'_over25.jpg'
                                    plt.savefig(name_output)
                                    plt.close()
                                    vis_detections(im, class_name, cls_dets, thresh=0.75,with_title=False)
                                    name_output = path_to_output2 +'Test/' + name_sans_ext + '_'+class_name+'_'+'_over25.jpg'
                                    plt.savefig(name_output)
                                    plt.close()
                            
                            
                        else:
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf,list_class=plot_onSubSet)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions.jpg'
                            if database=='PeopleArt':
                                path_tmp = '/'.join(name_output.split('/')[0:-1])
                                pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                            plt.savefig(name_output)
                            plt.close()
            except tf.errors.OutOfRangeError:
                break
     tf.reset_default_graph()
     true_label_all_test = np.concatenate(true_label_all_test)
     predict_label_all_test = np.transpose(np.concatenate(predict_label_all_test,axis=1))
     name_all_test = np.concatenate(name_all_test)
     labels_test_predited = (np.sign(predict_label_all_test) +1.)/2
     labels_test_predited[np.where(labels_test_predited==0.5)] = 0 # To deal with the case where predict_label_all_test == 0 
     return(true_label_all_test,predict_label_all_test,name_all_test,
            labels_test_predited,all_boxes)

def get_tensor_by_nameDescendant(graph,name):
    """
    This function is a very bad way to get the tensor by name from the graph
    because it will test the different possibility in a ascending way starting 
    by none and stop when it get the highest
    """
    complet_name = name + ':0'
    tensor = graph.get_tensor_by_name(complet_name)
    for i in range(100):
        try:
            complet_name = name + '_'+str(i+1)+':0'
            tensor = graph.get_tensor_by_name(complet_name)
        except KeyError:
            return(tensor)
    print("We only test the 100 possible tensor, we will return the 101st tensor")
    return(tensor)
    
def tfR_evaluation(database,j,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,PlotRegions,
               path_to_img,path_data,param_clf,classes,parameters,
               seuil_estimation,thresh_evaluation,TEST_NMS,verbose,
               all_boxes=None,dim_rois=5,with_tanh=False):
    
     PlotRegions,RPN,Stocha,CompBest=parameters
     k_per_bag,positive_elt,size_output = param_clf
     thresh = thresh_evaluation # Threshold score or distance MI_max
#     TEST_NMS = 0.7 # Recouvrement entre les classes
     
     load_model = False
     
     if PlotRegions:
         if Stocha:
             extensionStocha = 'Stocha/'
         else:
             extensionStocha = ''
         if database=='Wikidata_Paintings_miniset_verif':
             path_to_output2  = path_data + '/tfMI_maxRegion/'+database+'/'+extensionStocha+depicts_depictsLabel[classes[j]]
         else:
             path_to_output2  = path_data + '/tfMI_maxRegion/'+database+'/'+extensionStocha+classes[j]
         if RPN:
             path_to_output2 += '_RPNetMISVM/'
         elif CompBest:
              path_to_output2 += '_BestObject/'
         else:
             path_to_output2 += '/'
         path_to_output2_bis = path_to_output2 + 'Train'
         path_to_output2_ter = path_to_output2 + 'Test'
         pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
         pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True)
         
     export_dir_path = ('/').join(export_dir.split('/')[:-1])
     name_model_meta = export_dir + '.meta'
     
     if predict_with=='LinearSVC':
         length_matrix = dict_class_weight[0] + dict_class_weight[1]
         if length_matrix>17500*300:
             print('Not enough memory on Nicolas Computer ! use an other classifier than LinearSVC')
             raise(MemoryError)
         X_array = np.empty((length_matrix,size_output),dtype=np.float32)
         y_array =  np.empty((length_matrix,),dtype=np.float32)
         x_array_ind = 0
     
     if PlotRegions or predict_with=='LinearSVC':
        if verbose: print("Start ploting Regions selected by the MI_max in training phase")
        train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois(r,classe_index=j,num_classes=num_classes,dim_rois=dim_rois))
        dataset_batch = train_dataset.batch(mini_batch_size)
        dataset_batch.cache()
        iterator = dataset_batch.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session(config=config) as sess:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            load_model = True
            graph= tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            y = graph.get_tensor_by_name("y:0")
            Prod_best = graph.get_tensor_by_name("Prod:0")
            if with_tanh:
                Tanh = tf.tanh(Prod_best)
            mei = tf.argmax(Prod_best,axis=1)
            score_mei = tf.reduce_max(Prod_best,axis=1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                    if with_tanh:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                        sess.run([mei,score_mei,Tanh], feed_dict={X: fc7s, y: labels})
                    else:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                        sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
                    if predict_with=='LinearSVC' and k_per_bag==300  and positive_elt==1:
                        for k in range(len(labels)):
                            label_i = labels[k]
                            if label_i ==0:
                                X_array[x_array_ind:x_array_ind+300,:] = fc7s[k,:]
                                y_array[x_array_ind:x_array_ind+300] = 0
                                x_array_ind += 300
                            else:
                                X_array[x_array_ind,:] = fc7s[k,PositiveRegions[k]]
                                y_array[x_array_ind] = 1
                                x_array_ind += 1
                                # TODO need to be finished and tested 
                    if PlotRegions:
                        for k in range(len(PositiveRegions)):                          
                            if labels[k] == 1:
                                name_img = str(name_imgs[k].decode("utf-8") )
                                rois = roiss[k,:]
                                #if verbose: print(name_img)
                                if database in ['VOC12','Paintings','VOC2007','watercolor','IconArt_v1']:
                                    complet_name = path_to_img + name_img + '.jpg'
                                    name_sans_ext = name_img
                                elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                                    name_sans_ext = os.path.splitext(name_img)[0]
                                    complet_name = path_to_img +name_sans_ext + '.jpg'
                                im = cv2.imread(complet_name)
                                blobs, im_scales = get_blobs(im)
                                roi_with_object_of_the_class = PositiveRegions[k] % len(rois) # Because we have repeated some rois
                                roi = rois[roi_with_object_of_the_class,:]
                                roi_scores = [get_PositiveRegionsScore[k]]
                                roi_boxes =  roi[1:5] / im_scales[0]   
                                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                                if RPN:
                                    best_RPN_roi = rois[0,:]
                                    best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                                    best_RPN_roi_scores = [PositiveExScoreAll[k,0]]
                                    cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                                    best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                    roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                                else:
                                    cls = ['MI_max']
                                    roi_boxes_and_score = roi_boxes_score
                                vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                                name_output = path_to_output2 +'Train/' + name_sans_ext + '_Regions.jpg'
                                plt.savefig(name_output)
                                plt.close()
                except tf.errors.OutOfRangeError:
                    break
        #tf.reset_default_graph()
     
     # Training time !
     if predict_with=='LinearSVC':
         if verbose: print('Start training LiearSVC')
         clf =  TrainClassif(X_array,y_array,clf='LinearSVC',
                             class_weight=dict_class_weight,gridSearch=True,
                             n_jobs=1,C_finalSVM=1)
         if verbose: print('End training LiearSVC')
             
     # Testing time !
     train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
     train_dataset = train_dataset.map(lambda r: parser_w_rois(r,classe_index=j,num_classes=num_classes,dim_rois=dim_rois))
     dataset_batch = train_dataset.batch(mini_batch_size)
     dataset_batch.cache()
     iterator = dataset_batch.make_one_shot_iterator()
     next_element = iterator.get_next()
     true_label_all_test =  []
     predict_label_all_test =  []
     name_all_test =  []
     FirstTime= True
     i = 0
    
     with tf.Session(config=config) as sess:
        if load_model==False:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            graph= tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            y = graph.get_tensor_by_name("y:0")
            Prod_best = graph.get_tensor_by_name("Prod:0")
            if with_tanh:
                Tanh = tf.tanh(Prod_best)
            mei = tf.argmax(Prod_best,axis=1)
            score_mei = tf.reduce_max(Prod_best,axis=1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        while True:
            try:
                fc7s,roiss, labels,name_imgs = sess.run(next_element)
                if with_tanh:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Tanh], feed_dict={X: fc7s, y: labels})
                else:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
                true_label_all_test += [labels]
                if predict_with=='MI_max':
                    predict_label_all_test +=  [get_RegionsScore]
#                if predict_with=='LinearSVC':
                for k in range(len(labels)):
                    if database in['IconArt_v1','VOC12','Paintings','VOC2007','watercolor','WikiTenLabels']:
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                        name_sans_ext = os.path.splitext(name_img)[0]
                        complet_name = path_to_img +name_sans_ext + '.jpg'
                    else:
                        print(database,' is not known !')
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    if predict_with=='MI_max':
                        scores = PositiveExScoreAll[k,:]
                    elif predict_with=='LinearSVC':
                        scores = clf.decision_function(fc7s[k,:])

                    inds = np.where(scores > float(thresh))[0]
                    cls_scores = scores[inds]
                    roi = roiss[k,:]
                    roi_boxes =  roi[:,1:5] / im_scales[0] 
                    cls_boxes = roi_boxes[inds,:]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(cls_dets, TEST_NMS)
                    cls_dets = cls_dets[keep, :]
                    all_boxes[j][i] = cls_dets
                    i+=1
                
                    
                for l in range(len(name_imgs)): 
                    name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                if PlotRegions and predict_with=='MI_max':
                    if verbose and FirstTime: 
                        FirstTime = False
                        print("Start ploting Regions selected")
                    for k in range(len(PositiveRegions)):                          
                        if labels[k] == 1:
                            name_img = str(name_imgs[k].decode("utf-8") )
                            rois = roiss[k,:]
                            #if verbose: print(name_img)
                            if database in['IconArt_v1','VOC12','Paintings','VOC2007','watercolor','WikiTenLabels']:
                                complet_name = path_to_img + name_img + '.jpg'
                                name_sans_ext = name_img
                            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                                name_sans_ext = os.path.splitext(name_img)[0]
                                complet_name = path_to_img +name_sans_ext + '.jpg'
                            im = cv2.imread(complet_name)
                            blobs, im_scales = get_blobs(im)
                            roi_with_object_of_the_class = PositiveRegions[k] % len(rois) # Because we have repeated some rois
                            roi = rois[roi_with_object_of_the_class,:]
                            roi_scores = [get_RegionsScore[k]]
                            roi_boxes =  roi[1:5] / im_scales[0]   
                            roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                            if RPN:
                                best_RPN_roi = rois[0,:]
                                best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                                best_RPN_roi_scores = [PositiveExScoreAll[k,0]]
                                cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                            else:
                                cls = ['MI_max']
                                roi_boxes_and_score = roi_boxes_score
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                            name_output = path_to_output2 +'Test/' + name_sans_ext + '_Regions.jpg'
                            plt.savefig(name_output)
                            plt.close()
            except tf.errors.OutOfRangeError:
                break
     tf.reset_default_graph()
#        if predict_with=='LinearSVC':
#            
#            export_dir_path = ('/').join(export_dir.split('/')[:-1])
#            name_model_meta = export_dir + '.meta'  
#            train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
#            train_dataset = train_dataset.map(lambda r: parser_w_rois(r,classe_index=j))
#            dataset_batch = train_dataset.batch(mini_batch_size)
#            dataset_batch.cache()
#            iterator = dataset_batch.make_one_shot_iterator()
#            next_element = iterator.get_next()
#            true_label_all_test =  []
#            predict_label_all_test =  []
#            name_all_test =  []
#            length_matrix = np_neg_value*number_zone + np_pos_value* Number_of_positif_elt
#            X_array = np.empty((length_matrix,size_output))
#            FirstTime = True
#            with tf.Session(config=config) as sess:
#                new_saver = tf.train.import_meta_graph(name_model_meta)
#                new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
#                graph= tf.get_default_graph()
#                X = graph.get_tensor_by_name("X:0")
#                y = graph.get_tensor_by_name("y:0")
#                Prod_best = graph.get_tensor_by_name("Prod:0")
#                mei = tf.argmax(Prod_best,axis=1)
#                score_mei = tf.reduce_max(Prod_best,axis=1)
#                sess.run(tf.global_variables_initializer())
#                sess.run(tf.local_variables_initializer())
#                while True:
#                    try:
#                        fc7s,roiss, labels,name_imgs = sess.run(next_element)
#                        PositiveRegions,get_RegionsScore,PositiveExScoreAll = sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
#                        for label in labels:
#                            if label == 1:
#                                true_label_all_test += [labels]
#                            else:
#                                true_label_all_test += [label]*300
#                                
#            
#            return(0)
#        elif predict_with =='SGDClassif_sk': # SGD Classifier of scikit learn
##            np_pos_value,np_neg_value = classifierMI_max.get_porportions()
##            Number_of_positif_elt = 1 
##            number_zone = k_per_bag
##            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
##            classifier_sgd = SGDClassifier(loss='hinge',class_weight=dict_class_weight,n_jobs=-1)
##            # For best results using the default learning rate schedule, the data should have zero mean and unit variance.
##            # TODO !! 
#            return(0)
#            
#        elif predict_with =='hpsklearn_sgd': # SGD Classifier of scikit learn
##            # Use of the hyperopt optimisation 
##            np_pos_value,np_neg_value = classifierMI_max.get_porportions()
##            Number_of_positif_elt = 1 
##            number_zone = k_per_bag
##            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
##            classifier_sgd = sgd(loss='hinge',class_weight=dict_class_weight)
##            estim = HyperoptEstimator(classifier=classifier_sgd,  
##                            algo=tpe.suggest, trial_timeout=300)
##            #estim.fit_ # fit partial ??? 
##            
##                        
#            
#            return(0)
            
     true_label_all_test =np.concatenate(true_label_all_test)
     predict_label_all_test =np.concatenate(predict_label_all_test)
     name_all_test =np.concatenate(name_all_test)
     labels_test_predited = (np.sign(predict_label_all_test) +1.)/2
     return(true_label_all_test,predict_label_all_test,name_all_test,
            labels_test_predited,all_boxes)

def detectionOnOtherImages(demonet = 'res152_COCO',database = 'Wikidata_Paintings_miniset_verif'):
    if database=='Paintings':
#        item_name = 'name_img'
#        path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
#    elif database=='VOC12':
#        item_name = 'name_img'
#        path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
#        item_name = 'image'
#        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        raise NotImplementedError # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
#        item_name = 'image'
#        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']    
    elif(database=='IconArt_v1'):
        classes = ['angel','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien']    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
#    databasetxt =path_data + database + '.txt'
#    df_label = pd.read_csv(databasetxt,sep=",")
#    if database=='Wikidata_Paintings_miniset_verif':
#        df_label = df_label[df_label['BadPhoto'] <= 0.0]
    
    DATA_DIR =  '/media/gonthier/HDD/data/Fondazione_Zeri/Selection_Olivier/'
    output_DIR = '/media/gonthier/HDD/output_exp/ClassifPaintings/Zeri/'
    pathlib.Path(output_DIR).mkdir(parents=True, exist_ok=True)
        
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    
    # Load model :
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32] # we  use  3  aspect  ratios  and  4  scales (adding 64**2)
    nbClasses = len(CLASSES)
    path_to_model = '/media/gonthier/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
    
    net.create_architecture("TEST", nbClasses,
                          tag='default', anchor_scales=anchor_scales,
                          modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    
    dict_clf = {}
    classe_str = []
    for classe in classes:
        name_clf_pkl = path_data+'clf_FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_'+str(classe)+'.pkl'
        classifier = joblib.load(name_clf_pkl) 
        dict_clf[classe] = classifier
        classe_str += depicts_depictsLabel[classe]
        
    CONF_THRESH = 0.0
    NMS_THRESH = 0.0 # non max suppression
    dirs = os.listdir(DATA_DIR)
    for im_name in dirs:
        im_name_wt_ext, _ = im_name.split('.')
        imfile = os.path.join(DATA_DIR, im_name)
        im = cv2.imread(imfile)
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)
        blobs, im_scales = get_blobs(im)
        cls_boxes =  rois[:,1:5] / im_scales[0]
        #print(cls_boxes.shape)
        elt_k = fc7
        cls_list = []
        dets_list= []
        for j,classe in enumerate(classes):
            classifier = dict_clf[classe]
            decision_function_output = classifier.decision_function(elt_k)
            # TODO Il faudra changer cela pour mettre une proba et non une distance 
            # TODO gerer le cas ou l on normalise les donnees
            cls_scores = decision_function_output
            dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            cls_list += [depicts_depictsLabel[classe]]
            dets_list += [dets]
        #print(dets_list[0].shape)
        vis_detections_list(im, cls_list, dets_list, thresh=CONF_THRESH)
        name_output = output_DIR  + im_name_wt_ext  + '_NMSRegions.jpg'
        plt.savefig(name_output)
        plt.close()
    
    sess.close()
        
    
def FasterRCNN_TL_MISVM(demonet = 'res152_COCO',database = 'Paintings', 
                                          verbose = True,testMode = True,jtest = 0,
                                          PlotRegions = True,misvm_type = 'MISVM'):
    """ 
    15 mars 2017
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    
    Le but de cette fonction est de tester pour voir si l on peut utiliser les algos 
    MISVM et miSVM
    
    In this function we train an SVM only on the positive element returned by 
    the algo
    
    @param : demonet : the kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : database : the database used for the classification task
    @param : verbose : Verbose option classical
    @param : testMode : boolean True we only run on one class
    @param : jtest : the class on which we run the test
    @param : PlotRegions : plot the regions used for learn and the regions in the positive output response
    @param : Type of MI_max used or not : choice :  ['MISVM','miSVM','LinearMISVC','LinearmiSVC'] # TODO in the future also SGD
    The idea of thi algo is : 
        1/ Compute CNN features
        2/ Do NMS on the regions 
    
    option to train on background part also
    option on  scaling : sklearn.preprocessing.StandardScaler
    option : add a wieghted balanced of the SVM because they are really unbalanced classes
    TODO : mine hard negative exemple ! 
    """
    # TODO be able to train on background 
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        raise NotImplementedError # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    
    if(jtest>len(classes)) and testMode:
       print("We are in test mode but jtest>len(classes), we will use jtest =0" )
       jtest =0
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    databasetxt =path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_all.pkl'
    features_resnet_dict = {}
    sLength_all = len(df_label[item_name])
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
    
    if not(os.path.isfile(name_pkl)):
        # Compute the features
        if verbose: print("We will computer the CNN features")
        Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                     database=database,augmentation=False,L2 =False,
                                     saved='all',verbose=verbose)
        
    
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
    
    
    k_per_bag = 2
    features_resnet = np.ones((sLength_all,k_per_bag,size_output))
    classes_vectors = np.zeros((sLength_all,10))
    if database=='Wikidata_Paintings_miniset_verif':
        classes_vectors = df_label.as_matrix(columns=classes)
        # Convert 0/1 labels to -1/1 labels
        classes_vectors = 2 * classes_vectors - 1
    f_test = {}
    
    Test_on_k_bag = False
    normalisation= False
    
    # Parameters important
    new_nms_thresh = 0.0
    score_threshold = 0.1
    minimal_surface = 36*36
    
    # In the case of Wikidata
    if database=='Wikidata_Paintings_miniset_verif':
        random_state = 0
        index = np.arange(0,len(features_resnet_dict))
        index_trainval, index_test = train_test_split(index, test_size=0.6, random_state=random_state)
        index_trainval = np.sort(index_trainval)
        index_test = np.sort(index_test)

    len_fc7 = []
    roi_test = {}
    roi_train = {}
    name_test = {}
    train_bags = []
    test_bags = []
    key_test = 0
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0 and not(i==0):
            if verbose: print(i,name_img)
        rois,roi_scores,fc7 = features_resnet_dict[name_img]
        #print(rois.shape,roi_scores.shape)
        
        rois_reduce,roi_scores,fc7_reduce =  reduce_to_k_regions(k_per_bag,rois,roi_scores, fc7,
                                                   new_nms_thresh,
                                                   score_threshold,minimal_surface)
    
        len_fc7 += [len(fc7_reduce)]
        
        # bags : a sequence of n bags; each bag is an m-by-k array-like
        #               object containing m instances with k features
        
        
#        if(len(fc7_reduce) >= k_per_bag):
#            bag = np.expand_dims(fc7_reduce[0:k_per_bag,:],axis=0)
#        else:
#            number_repeat = k_per_bag // len(fc7_reduce)  +1
#            f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
#            bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
#        
#        features_resnet[i,:,:] = np.array(bag)
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if(classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
                else:
                    classes_vectors[i,j] = -1
        if database=='VOC12' or database=='Paintings':          
            InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
        elif database=='Wikidata_Paintings_miniset_verif':
            InSet = (i in index_test)
        
        if InSet: 
            if not(Test_on_k_bag):
                test_bags += [fc7]
            else:
                test_bags += [fc7_reduce]
            roi_test[key_test] = rois
            name_test[key_test] = name_img
            key_test += 1 
        else:
            train_bags += [fc7_reduce]
            roi_train[name_img] = rois_reduce  
    
    if verbose: 
        print('len(fc7), max',np.max(len_fc7),'mean',np.mean(len_fc7),'min',np.min(len_fc7))
        print('But we only keep k_per_bag =',k_per_bag)
    
    # TODO : keep the info of the repeat feature to remove them in the LinearSVC !! 
    
    if verbose: print("End data processing")
    if database=='VOC12' or database=='Paintings':
        #X_train = features_resnet[df_label['set']=='train',:,:]
        y_train = classes_vectors[df_label['set']=='train',:]
        #X_test= features_resnet[df_label['set']=='test',:,:]
        y_test = classes_vectors[df_label['set']=='test',:]
        #X_val = features_resnet[df_label['set']=='validation',:,:]
        y_val = classes_vectors[df_label['set']=='validation',:]
        #X_trainval = np.append(X_train,X_val,axis=0)
        y_trainval = np.append(y_train,y_val,axis=0)
        names = df_label.as_matrix(columns=['name_img'])
        name_train = names[df_label['set']=='train']
        name_val = names[df_label['set']=='validation']
        name_trainval = np.append(name_train,name_val,axis=0)
    elif database=='Wikidata_Paintings_miniset_verif':
        name = df_label.as_matrix(columns=[item_name])
        #name_trainval = name[index_trainval]
        #name_test = name[index_test]
        X_test= features_resnet[index_test,:,:]
        y_test = classes_vectors[index_test,:]
        #X_trainval =features_resnet[index_trainval,:,:]
        y_trainval =  classes_vectors[index_trainval,:]
#    
#    if normalisation == True:
#        if verbose: print('Normalisation')
#        scaler = StandardScaler()
#        scaler.fit(X_trainval.reshape(-1,size_output))
#        X_trainval = scaler.transform(X_trainval.reshape(-1,size_output))
#        X_trainval = X_trainval.reshape(-1,k_per_bag,size_output)
#        X_test = scaler.transform(X_test.reshape(-1,size_output))
#        X_test = X_test.reshape(-1,k_per_bag,size_output)
        
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    final_clf = None
    class_weight = None
    for j,classe in enumerate(classes):
        if testMode and not(j==jtest):
            continue
        if verbose : print(j,classes[j])
        if PlotRegions:
            if database=='Wikidata_Paintings_miniset_verif':
                path_to_output2  = path_data + '/MI_maxRegion/'+depicts_depictsLabel[classes[j]] + '/'
            else:
                path_to_output2  = path_data + '/MI_maxRegion/'+classes[j] + '/'
            path_to_output2_bis = path_to_output2 + 'Train'
            path_to_output2_ter = path_to_output2 + 'Test'
            pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
            pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True) 
            
#        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
#        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
#        pos_name = name_trainval[y_trainval[:,j]==1]
        
        if verbose: print("Start train the MI_max")
        
        if misvm_type=='miSVM':
            classifierMI_max = misvm.miSVM(kernel='linear', C=1.0, max_iters=10)
        elif misvm_type=='MISVM':
            classifierMI_max = misvm.MISVM(kernel='linear', C=1.0, max_iters=10,verbose=True,restarts=0)
        elif misvm_type=='LinearMISVC':
            classifierMI_max = mi_linearsvm.MISVM(C=1.0, max_iters=10,verbose=True,restarts=0)
        elif misvm_type=='LinearmiSVC':
            classifierMI_max = mi_linearsvm.miSVM(C=1.0, max_iters=10,verbose=True,restarts=0)
        
        classifierMI_max.fit(train_bags, y_trainval)
        classifier = classifierMI_max

        
        if verbose: print("End training the MI_max")
        
       
        
#        if verbose: print('Start Learning Final Classifier X.shape,y.shape',X.shape,y.shape)
#        classifier = TrainClassif(X,y,clf='LinearSVC',class_weight=class_weight,
#                                  gridSearch=True,n_jobs=1)
        
        if verbose: print("End training the SVM")
        
        y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j])
        labels_test_predited = np.zeros_like(y_test[:,j])

       
        for k in range(len(test_bags)): 
            prediction = classifier.predict(test_bags[k])
            y_predict_confidence_score_classifier[k]  = np.max(prediction)
            print(np.max(prediction))
            roi_with_object_of_the_class = np.argmax(prediction) 
            if np.max(prediction) > 0:
                print("We have positive element !!!")
                labels_test_predited[k] = 1
            else:
                labels_test_predited[k] = -1
        # Could add the plotting of the image
        AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
        if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
            print("MIL-SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
        else:
            print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
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
    
    
def PlotRegionsLearnByMI_max():
    """ 
   This function will plot the regions considered as dog by the MI_max of Said
   during the training and after during the testing
    """
    verbose = True
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
#    path_to_output2 = '/media/gonthier/HDD/output_exp/ClassifPaintings/dogRegion/'
#    path_to_output2 = '/media/gonthier/HDD/output_exp/ClassifPaintings/aeroplaneRegion/'
#    path_to_output2 = '/media/gonthier/HDD/output_exp/ClassifPaintings/chairRegion/'
#    path_to_output2 = '/media/gonthier/HDD/output_exp/ClassifPaintings/boatRegion/'
#    path_to_output2 = '/media/gonthier/HDD/output_exp/ClassifPaintings/birdRegion/'
    database = 'Paintings'
    databasetxt =path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    demonet = 'res152_COCO'
    item_name = 'name_img'
    name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_all.pkl'
    features_resnet_dict = {}
    sLength_all = len(df_label['name_img'])
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
    
    if not(os.path.isfile(name_pkl)):
        # Compute the features
        if verbose: print("We will computer the CNN features")
        Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                     database=database,augmentation=False,L2 =False,
                                     saved='all',verbose=verbose)
        
    
    if verbose: print("Start loading data")
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
    
    
    k_per_bag = 30
    features_resnet = np.ones((sLength_all,k_per_bag,size_output))
    classes_vectors = np.zeros((sLength_all,10))
    f_test = {}
    index_test = 0
    Test_on_k_bag = False
    normalisation= False
    
    # Parameters important
    new_nms_thresh = 0.0
    score_threshold = 0.1
    minimal_surface = 36*36
    
    len_fc7 = []
    roi_test = {}
    roi_train = {}
    name_test = {}
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0 and not(i==0):
            if verbose: print(i,name_img)
        rois,roi_scores,fc7 = features_resnet_dict[name_img]
        #print(rois.shape,roi_scores.shape)
        
        rois_reduce,roi_scores_reduce,fc7_reduce =  reduce_to_k_regions(k_per_bag,rois,roi_scores, fc7,
                                                   new_nms_thresh,
                                                   score_threshold,minimal_surface)
    

        len_fc7 += [len(fc7_reduce)]
        
        if(len(fc7_reduce) >= k_per_bag):
            bag = np.expand_dims(fc7_reduce[0:k_per_bag,:],axis=0)
        else:
            number_repeat = k_per_bag // len(fc7_reduce)  +1
            f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
            bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
        
        features_resnet[i,:,:] = np.array(bag)
        InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
        
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if(classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
        else:
            raise NotImplementedError
        
        if InSet: 
            if not(Test_on_k_bag):
                f_test[index_test] = fc7
                roi_test[index_test] = rois
                name_test[index_test] = name_img
                index_test += 1 
        else:
            roi_train[name_img] = rois_reduce
    
    #del features_resnet_dict
    if verbose: 
        print('len(fc7), max',np.max(len_fc7),'mean',np.mean(len_fc7),'min',np.min(len_fc7))
        print('But we only keep k_per_bag =',k_per_bag)
    
    # TODO : keep the info of the repeat feature to remove them in the LinearSVC !! 
    
    if verbose: print("End data processing")
    restarts = 20
    max_iters = 300
    n_jobs = -1
    #from trouver_classes_parmi_K import MI_max
    X_train = features_resnet[df_label['set']=='train',:,:]
    y_train = classes_vectors[df_label['set']=='train',:]
    X_test= features_resnet[df_label['set']=='test',:,:]
    y_test = classes_vectors[df_label['set']=='test',:]
    X_val = features_resnet[df_label['set']=='validation',:,:]
    y_val = classes_vectors[df_label['set']=='validation',:]
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
    
    
    names = df_label.as_matrix(columns=['name_img'])
    name_train = names[df_label['set']=='train']
    name_val = names[df_label['set']=='validation']
    name_trainval = np.append(name_train,name_val,axis=0)
    
    if normalisation == True:
        if verbose: print('Normalisation')
        scaler = StandardScaler()
        scaler.fit(X_trainval.reshape(-1,size_output))
        X_trainval = scaler.transform(X_trainval.reshape(-1,size_output))
        X_trainval = X_trainval.reshape(-1,k_per_bag,size_output)
        X_test = scaler.transform(X_test.reshape(-1,size_output))
        X_test = X_test.reshape(-1,k_per_bag,size_output)
        
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    final_clf = None
    class_weight = 'balanced'
    class_weight = None
    testMode = True
    jtest = 9
    for j,classe in enumerate(classes):
        if testMode and not(j==jtest):
            continue
        print(j,classes[j])
        path_to_output2  = path_data + '/MI_maxRegion/'+classes[j] + '/'
        path_to_output2_bis = path_to_output2 + 'Train'
        path_to_output2_ter = path_to_output2 + 'Test'
        pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True) 
        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
        pos_name = name_trainval[y_trainval[:,j]==1]
        #print(pos_name)
        classifierMI_max = MI_max(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
               max_iters=max_iters,symway=True,n_jobs=n_jobs,
               all_notpos_inNeg=False,gridSearch=True,
               verbose=False,final_clf=final_clf)     
        classifierMI_max.fit(pos_ex, neg_ex)
        PositiveRegions = classifierMI_max.get_PositiveRegions()
        get_PositiveRegionsScore = classifierMI_max.get_PositiveRegionsScore()
        PositiveExScoreAll =  classifierMI_max.get_PositiveExScoreAll()
        
        a = np.argmax(PositiveExScoreAll,axis=1)
        assert((a==PositiveRegions).all())
        assert(len(pos_name)==len(PositiveRegions))
        
#        get_PositiveRegionsScore = classifierMI_max.get_PositiveRegionsScore()
        
        if verbose: print("End training the MI_max")
        
        pos_ex_after_MI_max = np.zeros((len(pos_ex),size_output))
        neg_ex_keep = np.zeros((len(neg_ex),size_output))
        for k,name_imgtab in enumerate(pos_name):
            pos_ex_after_MI_max[k,:] = pos_ex[k,PositiveRegions[k],:] # We keep the positive exemple according to the MI_max from Said
            
            if verbose: print(k,name_img)
            name_img = name_imgtab[0]
            # Plot the regions considered as dog
            complet_name = path_to_img + name_img + '.jpg'
            im = cv2.imread(complet_name)
            blobs, im_scales = get_blobs(im)
            rois = roi_train[name_img]
            roi_with_object_of_the_class = PositiveRegions[k] % len(rois) # Because we have repeated some rois
            roi = rois[roi_with_object_of_the_class,:]
            roi_scores = [get_PositiveRegionsScore[k]]
            roi_boxes =  roi[1:5] / im_scales[0]            
            best_RPN_roi = rois[0,:]
            best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
            best_RPN_roi_scores = [PositiveExScoreAll[k,0]]
            assert((get_PositiveRegionsScore[k] >= PositiveExScoreAll[k,0]).all())
            cls = ['RPN','MI_max']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
            best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
            roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
            roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
            name_output = path_to_output2 +'Train/' + name_img + '_Regions.jpg'
            #+ '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MI_maxbestROI.jpg'
            plt.savefig(name_output)
            plt.close()
        
        neg_ex_keep = neg_ex.reshape(-1,size_output)
        
        X = np.vstack((pos_ex_after_MI_max,neg_ex_keep))
        y_pos = np.ones((len(pos_ex_after_MI_max),1))
        y_neg = np.zeros((len(neg_ex_keep),1))
        y = np.vstack((y_pos,y_neg)).ravel()
        if verbose: print('X.shape,y.shape',X.shape,y.shape)
        classifier = TrainClassif(X,y,clf='LinearSVC',class_weight=class_weight,
                                  gridSearch=True,n_jobs=1)
        
        if verbose: print("End training the SVM")
        
        y_predict_confidence_score_classifier = np.zeros_like(y_test[:,j])
        labels_test_predited = np.zeros_like(y_test[:,j])
        
        for k in range(len(X_test)): 
            if Test_on_k_bag: 
                decision_function_output = classifier.decision_function(X_test[k,:,:])
            else:
                if normalisation:
                    elt_k =  scaler.transform(f_test[k])
                else:
                    elt_k = f_test[k]
                decision_function_output = classifier.decision_function(elt_k)
            y_predict_confidence_score_classifier[k]  = np.max(decision_function_output)
            roi_with_object_of_the_class = np.argmax(decision_function_output)
            if np.max(decision_function_output) > 0:
                labels_test_predited[k] = 1 
                # We predict a dog 
                name_img = name_test[k]
                if verbose: print(k,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                blobs, im_scales = get_blobs(im)
                rois = roi_test[k]
                roi = rois[roi_with_object_of_the_class,:]
                roi_boxes =  roi[1:5] / im_scales[0]
                best_RPN_roi = rois[0,:]
                best_RPN_roi_boxes =  best_RPN_roi[1:5] / im_scales[0]
                best_RPN_roi_scores = [decision_function_output[0]]
                assert((np.max(decision_function_output) >= decision_function_output[0]).all())
                cls = ['RPN','Classif']  # Comparison of the best region according to the faster RCNN and according to the MI_max de Said
                roi_scores =  [np.max(decision_function_output)]
                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                name_output = path_to_output2 +'Test/' + name_img  + '_Regions.jpg'
                #'_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MI_maxbestROI.jpg'
                plt.savefig(name_output)
                plt.close()
            else: 
                labels_test_predited[k] =  0 # Label of the class 0 or 1
        AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
        print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
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
 
def plotGT(name):
     plot_onSubSet =  ['angel','Child_Jesus', 'crucifixion_of_Jesus','Mary','nudity', 'ruins','Saint_Sebastien'] 
     imbd = get_imdb('WikiTenLabels_test')
     complet_name = imbd.image_path_from_index(name)
     complet_name_tab = ('.'.join(complet_name.split('.')[0:-1])).split('/')
     complet_name_tab[-2] = 'Annotations'
     complet_name_xml = '/'.join(complet_name_tab) + '.xml'
     im = cv2.imread(complet_name)
     blobs, im_scales = get_blobs(im)
     import voc_eval
     read_file = voc_eval.parse_rec(complet_name_xml)
     cls = []
     dict_bbox = {}
     for element in read_file:
        classe_elt_xml = element['name']
        if classe_elt_xml in plot_onSubSet:
            bbox = element['bbox']
            if not(classe_elt_xml in cls):
                cls += [classe_elt_xml]
                dict_bbox[classe_elt_xml]= np.array(bbox).reshape(1,4)
            else:
                bboxs = np.vstack((dict_bbox[classe_elt_xml],bbox))
                dict_bbox[classe_elt_xml] = bboxs
     dets = []
     for classe_elt_xml in cls:
        dets += [dict_bbox[classe_elt_xml]] 
    
     vis_GT_list(im, cls, dets,list_class=plot_onSubSet)
     plt.show()
     input('close ?')
 

            
    
    
if __name__ == '__main__':
#    FasterRCNN_TL_MI_max_newVersion()
#    petitTestIllustratif()
#    FasterRCNN_TL_MI_max_ClassifOutMI_max()
#    petitTestIllustratif_RefineRegions()
#    PlotRegionsLearnByMI_max()
#    FasterRCNN_TL_MI_max_ClassifOutMI_max(demonet = 'res152_COCO',database = 'Paintings', 
    # Baseline on PeopleArt
#    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database = 'watercolor',Test_on_k_bag=False,
#                 normalisation= False,baseline_kind = 'miSVM',verbose = True,
#                 gridSearch=False,k_per_bag=300,n_jobs=1,clf='LinearSVC',testMode=False,restarts = 0,max_iter_MI_max=50) # defaultSGD or LinearSVC

#    VariationStudy()
    
#    listAggregOnProdorTanh = ['AveragingW','meanOfProd','medianOfProd','maxOfProd','maxOfTanh',\
#                                       'meanOfTanh','medianOfTanh','minOfTanh','minOfProd']
#    listAggregOnProdorTanh = ['meanOfSign']
#    with_scores = True
#    for elt in listAggregOnProdorTanh:
#        print(elt)
#        tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,
#                                  verbose = True,testMode = False,jtest = 'cow',
#                                  PlotRegions = False,saved_clf=False,RPN=False,
#                                  CompBest=False,Stocha=True,k_per_bag=300,
#                                  parallel_op=True,CV_Mode='',num_split=2,
#                                  WR=True,init_by_mean =None,seuil_estimation='',
#                                  restarts=11,max_iters_all_base=300,LR=0.01,with_tanh=False,
#                                  C=1.0,Optimizer='GradientDescent',norm='',
#                                  transform_output='tanh',with_rois_scores_atEnd=False,
#                                  with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
#                                  Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                                  k_intopk=1,C_Searching=False,predict_with='MI_max',
#                                  gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                                  thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=elt,proportionToKeep=0.25) 
#        tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,
#                                  verbose = True,testMode = False,jtest = 'cow',
#                                  PlotRegions = False,saved_clf=False,RPN=False,
#                                  CompBest=False,Stocha=True,k_per_bag=300,
#                                  parallel_op=True,CV_Mode='',num_split=2,
#                                  WR=True,init_by_mean =None,seuil_estimation='',
#                                  restarts=11,max_iters_all_base=300,LR=0.01,with_tanh=False,
#                                  C=1.0,Optimizer='GradientDescent',norm='',
#                                  transform_output='tanh',with_rois_scores_atEnd=False,
#                                  with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
#                                  Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                                  k_intopk=1,C_Searching=False,predict_with='MI_max',
#                                  gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                                  thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=elt,proportionToKeep=1.0) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,
#                              verbose = True,testMode = False,jtest = 'cow',
#                              PlotRegions = True,saved_clf=False,RPN=False,
#                              CompBest=False,Stocha=True,k_per_bag=300,
#                              parallel_op=True,CV_Mode='',num_split=2,
#                              WR=True,init_by_mean =None,seuil_estimation='',
#                              restarts=11,max_iters_all_base=300,LR=0.01,with_tanh=True,
#                              C=1.0,Optimizer='GradientDescent',norm='',
#                              transform_output='tanh',with_rois_scores_atEnd=False,
#                              with_scores=True,epsilon=0.01,restarts_paral='paral',
#                              Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                              k_intopk=1,C_Searching=False,predict_with='MI_max',
#                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=None,proportionToKeep=0.25,
#                              loss_type='',storeVectors=False,storeLossValues=False,
#                              plot_onSubSet=['bicycle', 'bird', 'car', 'cat', 'dog', 'person'])


# Test PCA data

#    for model in  ['MI_max']:
#        for with_scores in [False,True]:
##            for AggregW  in ['maxOfTanh','maxOfProd']:
#            for AggregW  in ['maxOfTanh','meanOfTanh']:
#                tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                              verbose = True,testMode = False,jtest = 'cow',
#                              PlotRegions = False,saved_clf=False,RPN=False,
#                              CompBest=False,Stocha=True,k_per_bag=300,
#                              parallel_op=True,CV_Mode='',num_split=2,
#                              WR=True,init_by_mean =None,seuil_estimation='',
#                              restarts=11,max_iters_all_base=300,LR=0.01,
#                              C=1.0,Optimizer='GradientDescent',norm='',
#                              transform_output='tanh',with_rois_scores_atEnd=False,
#                              with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
#                              predict_with='MI_max',
#                              AggregW =AggregW ,proportionToKeep=1.0,model=model) 

#    tfR_FRCNN(demonet = 'res152_COCO',database = 'RMN', ReDo=False,
#              verbose = True,testMode = False,jtest = 'cow',
#              PlotRegions = False,saved_clf=False,RPN=False,
#              CompBest=False,Stocha=True,k_per_bag=300,
#              parallel_op=True,CV_Mode='',num_split=2,
#              WR=True,init_by_mean =None,seuil_estimation='',
#              restarts=11,max_iters_all_base=3000,LR=0.01,
#              C=1.0,Optimizer='GradientDescent',norm='',
#              transform_output='tanh',with_rois_scores_atEnd=False,
#              with_scores=True,epsilon=0.01,restarts_paral='paral',
#              predict_with='MI_max',
#              AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'RMN', ReDo=False,
#              verbose = True,testMode = False,jtest = 'cow',
#              PlotRegions = False,saved_clf=False,RPN=False,
#              CompBest=False,Stocha=True,k_per_bag=300,
#              parallel_op=True,CV_Mode='',num_split=2,
#              WR=True,init_by_mean =None,seuil_estimation='',
#              restarts=11,max_iters_all_base=3000,LR=0.01,
#              C=1.0,Optimizer='GradientDescent',norm='',
#              transform_output='tanh',with_rois_scores_atEnd=False,
#              with_scores=False,epsilon=0.01,restarts_paral='paral',
#              predict_with='MI_max',
#              AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=False,
#          verbose = True,testMode = False,jtest = 'cow',
#          PlotRegions = False,saved_clf=False,RPN=False,
#          CompBest=False,Stocha=True,k_per_bag=2000,
#          parallel_op=True,CV_Mode='',num_split=2,
#          WR=True,init_by_mean =None,seuil_estimation='',
#          restarts=49,max_iters_all_base=3000,LR=0.01,
#          C=1.0,Optimizer='GradientDescent',norm='',
#          transform_output='tan',with_rois_scores_atEnd=False,
#          with_scores=True,epsilon=0.01,restarts_paral='paral',
#          predict_with='MI_max',
#          AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=False,
#          verbose = True,testMode = False,jtest = 'cow',
#          PlotRegions = False,saved_clf=False,RPN=False,
#          CompBest=False,Stocha=True,k_per_bag=2000,
#          parallel_op=True,CV_Mode='',num_split=2,
#          WR=True,init_by_mean =None,seuil_estimation='',
#          restarts=49,max_iters_all_base=3000,LR=0.01,
#          C=1.0,Optimizer='GradientDescent',norm='',
#          transform_output='tan',with_rois_scores_atEnd=False,
#          with_scores=False,epsilon=0.01,restarts_paral='paral',
#          predict_with='MI_max',
#          AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False) 
    
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#              verbose = True,testMode = False,jtest = 'cow',
#              PlotRegions = False,saved_clf=False,RPN=False,
#              CompBest=False,Stocha=True,k_per_bag=300,
#              parallel_op=True,CV_Mode='',num_split=2,
#              WR=True,init_by_mean =None,seuil_estimation='',
#              restarts=11,max_iters_all_base=300,LR=0.01,
#              C=1.0,Optimizer='GradientDescent',norm='',
#              transform_output='tanh',with_rois_scores_atEnd=False,
#              with_scores=True,epsilon=0.01,restarts_paral='paral',
#              predict_with='MI_max',
#              AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=True) 

# Test EdgeBoxes 
    for k_per_bag in [300]:
#        for database in ['watercolor','IconArt_v1','VOC2007']:
        for database in ['IconArt_v1','VOC2007']:
            for model in ['MI_max','mi_model']:
                tfR_FRCNN(demonet = 'res152',database = database, ReDo=True,
                                          verbose = True,testMode = False,jtest = 'cow',
                                          PlotRegions = False,saved_clf=False,RPN=False,
                                          CompBest=False,Stocha=True,k_per_bag=k_per_bag,
                                          parallel_op=True,CV_Mode='',num_split=2,
                                          WR=True,init_by_mean =None,seuil_estimation='',
                                          restarts=11,max_iters_all_base=300,LR=0.01,with_tanh=True,
                                          C=1.0,Optimizer='GradientDescent',norm='',
                                          transform_output='tanh',with_rois_scores_atEnd=False,
                                          with_scores=False,epsilon=0.01,restarts_paral='paral',
                                          Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
                                          k_intopk=1,C_Searching=False,predict_with='',
                                          gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                                          thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
                                          loss_type='',storeVectors=False,storeLossValues=False,
                                          metamodel='EdgeBoxes',model=model)
#    tfR_FRCNN(demonet = 'res152',database = 'VOC2007', ReDo=True,
#                              verbose = True,testMode = False,jtest = 'cow',
#                              PlotRegions = False,saved_clf=False,RPN=False,
#                              CompBest=False,Stocha=True,k_per_bag=300,
#                              parallel_op=True,CV_Mode='',num_split=2,
#                              WR=True,init_by_mean =None,seuil_estimation='',
#                              restarts=11,max_iters_all_base=300,LR=0.001,with_tanh=True,
#                              C=1.0,Optimizer='GradientDescent',norm='',
#                              transform_output='tanh',with_rois_scores_atEnd=False,
#                              with_scores=False,epsilon=0.01,restarts_paral='paral',
#                              Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                              k_intopk=1,C_Searching=False,predict_with='',
#                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
#                              loss_type='',storeVectors=False,storeLossValues=False,
#                              metamodel='EdgeBoxes',model='mi_model')
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = False,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=True,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max',
#                          PCAuse=False,variance_thres=0.9) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'PeopleArt', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = False,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=False,epsilon=0.01,restarts_paral='paral',
#                          predict_with='mi_model',
#                          PCAuse=False,trainOnTest=False,AddOneLayer=False,
#                          Max_version='')  # Not parall computation at all
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = False,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=False,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max')  # Not parall computation at all
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = False,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=False,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=True,epsilon=0.01,restarts_paral='',
#                          predict_with='MI_max',
#                          PCAuse=False,trainOnTest=False,AddOneLayer=True)  # Not parall computation at all
#

#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = True,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=False,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max',
#                          PCAuse=False,trainOnTest=True,AddOneLayer=False) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = True,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=True,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max',
#                          PCAuse=False,trainOnTest=True,AddOneLayer=False) 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = True,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=False,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max',
#                          PCAuse=False,trainOnTest=False,AddOneLayer=False,model='mi_model') 
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,
#                          verbose = True,testMode = False,jtest = 'cow',
#                          PlotRegions = False,saved_clf=False,RPN=False,
#                          CompBest=False,Stocha=True,k_per_bag=300,
#                          parallel_op=True,CV_Mode='',num_split=2,
#                          WR=True,init_by_mean =None,seuil_estimation='',
#                          restarts=11,max_iters_all_base=300,LR=0.01,
#                          C=1.0,Optimizer='GradientDescent',norm='',
#                          transform_output='tanh',with_rois_scores_atEnd=False,
#                          with_scores=True,epsilon=0.01,restarts_paral='paral',
#                          predict_with='MI_max',
#                          PCAuse=False,trainOnTest=False,AddOneLayer=False) 

    
    ## Test of mi_model ! mi_model a finir !! 
    # A faire : faire en sorte que les npos et nneg soit recalcules pour chaque batch. 
    # Il faut aussi regarder a quoi ressemble la loss pour SGD pour CNN
#    tfR_FRCNN(demonet = 'res152_COCO',database = 'watercolor', ReDo=True,model='mi_model',
#                              verbose = True,testMode = False,jtest = 'cow',
#                              PlotRegions = False,saved_clf=False,RPN=False,
#                              CompBest=False,Stocha=True,k_per_bag=300,
#                              parallel_op=True,CV_Mode='',num_split=2,
#                              WR=True,init_by_mean =None,seuil_estimation='',
#                              restarts=12,max_iters_all_base=300,LR=0.01,with_tanh=True,
#                              C=1.0,Optimizer='GradientDescent',norm='',
#                              transform_output='tanh',with_rois_scores_atEnd=False,
#                              with_scores=True,epsilon=0.01,restarts_paral='paral',
#                              Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                              k_intopk=1,C_Searching=False,predict_with='MI_max',
#                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW=None,proportionToKeep=0.25,
#                              loss_type='',storeVectors=False,storeLossValues=False,
#                              obj_score_add_tanh=False,lambdas=0.5,obj_score_mul_tanh=False) 




   # A comparer avec du 93s par restart pour les 6 classes de watercolor

    ## TODO : tester avec une image constante en entrée et voir ce que cela donne de couleur différentes
    # Peut etre a rajouter dans les exemples negatifs 
#    plotGT('Q28810789')
#    plotGT('Q3213763')
#    
