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
#from hpsklearn import HyperoptEstimator,sgd
#from hyperopt import tpe


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
    ,'vgg16_coco': ('/media/HDD/models/tf-faster-rcnn/vgg16/vgg16_faster_rcnn_iter_1190000.ckpt',)    
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
num_features = 2048 # TODO change thath !!!

def parser_w_mei_reduce(record):
    num_rois = 300
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

def parser_w_rois(record,classe_index=0,num_classes=10):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'num_regions':  tf.FixedLenFeature([], tf.int64),
                'num_features':  tf.FixedLenFeature([], tf.int64),
                'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                'rois': tf.FixedLenFeature([300*5],tf.float32),
                'roi_scores':tf.FixedLenFeature([300],tf.float32),
                'fc7': tf.FixedLenFeature([300*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_classes],tf.float32),
                'name_img' : tf.FixedLenFeature([],tf.string)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
    name_img = parsed['name_img']
    label = tf.slice(label,[classe_index],[1])
    label = tf.squeeze(label) # To get a vector one dimension
    fc7 = parsed['fc7']
    fc7 = tf.reshape(fc7, [300,num_features])
    rois = parsed['rois']
    rois = tf.reshape(rois, [300,5])           
    return fc7,rois, label,name_img

def parser_w_rois_all_class(record,num_classes=10,num_rois=300,num_features=2048):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([5*num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [num_rois,num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [num_rois,5])           
        return fc7,rois, label,name_img

def petitTestIllustratif():
    """
    We will try on 20 image from the Art UK Your paintings database and see what 
    we get as best zone with the MILSVM de Said 
    """
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
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
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
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
      
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    symway = True
    if symway:
        path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MILSVM/'
    else:
        path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MILSVM_NotSymWay/'
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
            classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                          max_iters=max_iters,symway=symway,n_jobs=-1,
                                          all_notpos_inNeg=False,gridSearch=False,
                                          verbose=False,final_clf='None')     
            classifierMILSVM.fit(pos_ex, neg_ex)
            
            PositiveRegions = classifierMILSVM.get_PositiveRegions()
            get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()
        
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
                cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
        #            print(roi_boxes)
                
                roi_scores = [get_PositiveRegionsScore[i]]
        #            print(roi_scores)
                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
                plt.savefig(name_output)
            plt.close('all') 
            
def petitTestIllustratif_RefineRegions():
    """
    We will try on 20 image from the Art UK Your paintings database and see what 
    we get as best zone with the MILSVM de Said 
    in this function we try to refine regions, ie remove not important regions
    """
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
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
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
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
      
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    symway = True
    if symway:
        path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MILSVM_Refine/'
    else:
        path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/MILSVM_NotSymWay_Refine/'
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
    classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                  max_iters=max_iters,symway=symway,n_jobs=-1,
                                  all_notpos_inNeg=False,gridSearch=False,
                                  verbose=False,final_clf='None')     
    print("Start Learning MILSVM")
    classifierMILSVM.fit(pos_ex, neg_ex)
    print("End Learning MILSVM")
    PositiveRegions = classifierMILSVM.get_PositiveRegions()
    get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()

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
        cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
#            print(roi_boxes)
        
        roi_scores = [get_PositiveRegionsScore[i]]
#            print(roi_scores)
        best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
        roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
        roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
        name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
        plt.savefig(name_output)
    plt.close('all')
    
    

def FasterRCNN_TL_MILSVM_newVersion():
    """ Function to test if you can refind the same AP metric by reading the saved 
    CNN features 
    Older version of the function than FasterRCNN_TL_MILSVM_ClassifOutMILSVM
    """
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
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
    #from trouver_classes_parmi_K import MILSVM
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
        classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                      max_iters=max_iters,symway=True,n_jobs=n_jobs,
                                      all_notpos_inNeg=False,gridSearch=True,
                                      verbose=False,final_clf=final_clf)     
        classifier = classifierMILSVM.fit(pos_ex, neg_ex)
        #print("End training the MILSVM")
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
    
def FasterRCNN_TL_MILSVM_ClassifOutMILSVM(demonet = 'res152_COCO',database = 'Paintings', 
                                          verbose = True,testMode = True,jtest = 0,
                                          PlotRegions = True,saved_clf=False,RPN=False,
                                          CompBest=True,Stocha=False,k_per_bag=30):
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
            isVOC = True
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
            classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
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
        if database=='Wikidata_Paintings_miniset_verif' or database=='VOC2007' or database=='watercolor':
            classes_vectors = df_label.as_matrix(columns=classes)
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
            if database=='VOC2007' or database=='VOC12' or database=='Paintings'  or database=='watercolor':          
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
    
        if database=='VOC2007'  or database=='watercolor':
            if database=='VOC2007' : imdb = get_imdb('voc_2007_test')
            if database=='watercolor' : imdb = get_imdb('watercolor_test')
            imdb.set_force_dont_use_07_metric(True)
            num_images = len(imdb.image_index)
            all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
            if CompBest: all_boxes_bS = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        else:
            all_boxes = None
    
        if testMode:
            verboseMIL = True
            restarts = 0
            max_iters = 1
        else:
            verboseMIL = False
            restarts = 19
            max_iters = 300
        print('restarts',restarts,'max_iters',max_iters)
        n_jobs = -1
        #from trouver_classes_parmi_K import MILSVM
        
    #    del features_resnet_dict
        gc.collect()
        
        if database=='VOC12' or database=='Paintings' or database=='VOC2007'  or database=='watercolor':
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
                    path_to_output2  = path_data + '/MILSVMRegion/'+extensionStocha+depicts_depictsLabel[classes[j]]
                else:
                    path_to_output2  = path_data + '/MILSVMRegion/'+extensionStocha+classes[j]
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
            
            if verbose: print("Start train the MILSVM")
    
            
            if Stocha:
                # Cela ne marche pas encore !
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
                classifierMILSVM = tf_MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,
                   all_notpos_inNeg=False,gridSearch=True,
                   verbose=verboseMIL,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                   mini_batch_size=mini_batch_size) 
                classifierMILSVM.fit_Stocha(bags,labels,shuffle=True)
            else:
                classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,
                   all_notpos_inNeg=False,gridSearch=True,
                   verbose=verboseMIL,final_clf=final_clf)   
                classifierMILSVM.fit(pos_ex, neg_ex)
                #durations : between 26 and durations : 8 for Paintings
            
            PositiveRegions = classifierMILSVM.get_PositiveRegions()
            get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()
            PositiveExScoreAll =  classifierMILSVM.get_PositiveExScoreAll()
            
            if PlotRegions:
                # Just des verifications
                a = np.argmax(PositiveExScoreAll,axis=1)
                assert((a==PositiveRegions).all())
                assert(len(pos_name)==len(PositiveRegions))
            
            if verbose: print("End training the MILSVM")
            
            pos_ex_after_MILSVM = np.zeros((len(pos_ex),size_output))
            neg_ex_keep = np.zeros((len(neg_ex),size_output))
            for k,name_imgtab in enumerate(pos_name):
                pos_ex_after_MILSVM[k,:] = pos_ex[k,PositiveRegions[k],:] # We keep the positive exemple according to the MILSVM from Said
                
                if PlotRegions:
                    if verbose: print(k,name_img)
                    name_img = name_imgtab[0]
                    if database=='VOC2007' :
                        name_sans_ext =  str(name_img.decode("utf-8"))
                        complet_name = path_to_img + str(name_img.decode("utf-8")) + '.jpg'
                    if database=='VOC12' or database=='Paintings'  or database=='watercolor':
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
                        cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                        best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                        roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                    else:
                        cls = ['MILSVM']
                        roi_boxes_and_score = roi_boxes_score
                    vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                    name_output = path_to_output2 +'Train/' + name_sans_ext + '_Regions.jpg'
                    plt.savefig(name_output)
                    plt.close()
            
            neg_ex_keep = neg_ex.reshape(-1,size_output)
            
            X = np.vstack((pos_ex_after_MILSVM,neg_ex_keep))
            y_pos = np.ones((len(pos_ex_after_MILSVM),1))
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
                if database=='VOC2007'  or database=='watercolor':
                    thresh = 0.0 # Threshold score or distance MILSVM
                    TEST_NMS = 0.7 # Recouvrement entre les classes
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
                        if database=='VOC2007' :
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
                            cls = ['RPN','Classif']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                            best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                            roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                        elif CompBest:
                            roi_with_object_of_the_class = np.argmax(decision_function_output_bS)
                            roi2 = rois[roi_with_object_of_the_class,:]
                            roi_boxes2 =  roi2[1:5] / im_scales[0]
                            roi_scores2 =  [np.max(decision_function_output_bS)]
                            roi_boxes_score2 = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes2,roi_scores2)),axis=0),axis=0)
                            cls = ['BestObject','Classif']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
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
        print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
        if CompBest: print("mean Average Precision for BEst Score = {0:.3f}".format(np.mean(AP_per_classbS))) 
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
            print("Detection scores")
            print(arrayToLatex(aps))
        
        plot_Test_illust_bol = False
        if plot_Test_illust_bol:
            
            dict_clf = {}
            classe_str = []
            for classe in classes:
                name_clf_pkl = path_data+'clf_FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'_'+str(classe)+'.pkl'
                classifier = joblib.load(name_clf_pkl) 
                dict_clf[classe] = classifier
                classe_str += depicts_depictsLabel[classe]
                
            path_to_output2  = path_data + '/MILSVMRegion/TestIllust/'
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
                
                path_to_output2  = path_data + '/MILSVMRegion/TestIllust2/'
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
    
def tfRecords_FasterRCNN(demonet = 'res152_COCO',database = 'Paintings', 
                                  verbose = True,testMode = True,jtest = 0,
                                  PlotRegions = True,saved_clf=False,RPN=False,
                                  CompBest=True,Stocha=True,k_per_bag=300,
                                  parallel_op =True,CV_Mode=None,num_split=2):
    """ 
    10 avril 2017
    This function used TFrecords file 
    
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    
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
    @param : k_per_bag : number of element per batch in the slection phase [defaut : 300] 
    !!!!! for the moment it is not possible to use something else than 300 if the dataset is not 
    records with the selection of the regions already !!!! TODO change that
    @param : parallel_op : use of the parallelisation version of the MILSVM for the all classes same time
    @param : CV_Mode : cross validation mode in the MILSVM : possibility ; None, CV in k split or LA for Leave apart one of the split
    @param : num_split  : Number of split for the CV or LA
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
    ext = '.txt'
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
    
    if testMode and not(type(jtest)==int):
        assert(type(jtest)==str)
        jtest = int(np.where(np.array(classes)==jtest)[0][0])# Conversion of the jtest string to the value number
        assert(type(jtest)==int)
        
    if(jtest>len(classes)) and testMode:
       print("We are in test mode but jtest>len(classes), we will use jtest =0" )
       jtest =0
    
        
    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    databasetxt =path_data + database + ext
    df_label = pd.read_csv(databasetxt,sep=",")
    str_val = 'val'
    if database=='Wikidata_Paintings_miniset_verif' or database=='Paintings':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
        str_val = 'validation'
    if database=='VOC2007' or database=='watercolor':
        str_val = 'val'
        df_label[classes] = df_label[classes].apply(lambda x:(x + 1.0)/2.0)
    num_trainval_im = len(df_label[df_label['set']=='train'][item_name]) + len(df_label[df_label['set']==str_val][item_name])
    num_classes = len(classes)
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    
    sets = ['train','val','trainval','test']
    dict_name_file = {}
    data_precomputeed= True
    for set_str in sets:
        name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
        dict_name_file[set_str] = name_pkl_all_features
        if not(os.path.isfile(name_pkl_all_features)):
            data_precomputeed = False

    sLength_all = len(df_label[item_name])
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
    
    if not(data_precomputeed):
        # Compute the features
        if verbose: print("We will computer the CNN features")
        Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                     database=database,augmentation=False,L2 =False,
                                     saved='all',verbose=verbose,filesave='tfrecords')
           
    # Config param for TF session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
            
    # Data for the MILSVM Latent SVM
    n_jobs = -1
    performance = False
    if parallel_op:
        sizeMax = 30*20000 // (k_per_bag*num_classes)
    else:
        sizeMax = 30*10000 // k_per_bag
    mini_batch_size = sizeMax
    buffer_size = 10000
    if testMode:
        restarts = 0
        restarts = 19
        max_iters = 300
        max_iters =  (num_trainval_im //mini_batch_size)*300
        ext_test = '_Test_Mode'
    else:
        ext_test= ''
        restarts = 19
        max_iters = (num_trainval_im //mini_batch_size)*300
    print('mini_batch_size',mini_batch_size,'max_iters',max_iters)
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    AP_per_classbS = []
    final_clf = None
    class_weight = None
    ReDo = False
    norm=False
    if norm:
        extNorm = '_N'
    else:
        extNorm = ''
    if parallel_op:
        extPar = '_p'
    else:
        extPar =''
    if CV_Mode=='CV':
        extCV = '_cv'+str(num_split)
    elif CV_Mode=='LA':
        extCV = '_la'+str(num_split)
    elif CV_Mode is None:
        extCV =''
    cachefile_model = path_data + database +'_'+demonet+'_r'+str(restarts)+'_s' \
        +str(mini_batch_size)+'_k'+str(k_per_bag)+'_m'+str(max_iters)+extNorm+extPar+extCV+ext_test+'_MILSVM.pkl'
    if verbose: print("cachefile name",cachefile_model)
    if not os.path.isfile(cachefile_model) or ReDo:
        name_milsvm = {}
        if verbose: print("The cachefile doesn t exist")
    else:
        with open(cachefile_model, 'rb') as f:
            name_milsvm = pickle.load(f)
            if verbose: print("The cachefile exists")

    if database=='VOC2007':
        imdb = get_imdb('voc_2007_test')
        imdb.set_force_dont_use_07_metric(True)
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    elif database=='watercolor':
        imdb = get_imdb('watercolor_test')
        imdb.set_force_dont_use_07_metric(True)
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    else:
        all_boxes = None

    Optimizer = 'GradientDescent'
    Optimizer = 'Adam'
    LR=0.01
    optimArg= None
    verboseMILSVM = True
    data_path_train= dict_name_file['trainval']
    
    if parallel_op:
        # For Pascal VOC2007 pour les 20 classes cela prend environ 2500s par iteration 
        if not os.path.isfile(cachefile_model) or ReDo:
             classifierMILSVM = tf_MILSVM(LR=LR,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,buffer_size=buffer_size,
                   verbose=verboseMILSVM,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                   mini_batch_size=mini_batch_size,num_features=size_output,debug=False,
                   num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode) 
             export_dir = classifierMILSVM.fit_MILSVM_tfrecords(data_path=data_path_train, \
                   class_indice=-1,shuffle=True,performance=performance)
             np_pos_value,np_neg_value = classifierMILSVM.get_porportions()
             name_milsvm =export_dir,np_pos_value,np_neg_value
             with open(cachefile_model, 'wb') as f:
                 pickle.dump(name_milsvm, f)
        else:
            export_dir,np_pos_value,np_neg_value= name_milsvm
        
        Number_of_positif_elt = 1 
        number_zone = k_per_bag
        dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
        parameters=PlotRegions,RPN,Stocha,CompBest
        param_clf = k_per_bag,Number_of_positif_elt,size_output
        predict_with='MILSVM'
        true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
        ,all_boxes = \
        tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,PlotRegions,
               path_to_img,path_data,param_clf,classes,parameters,verbose,all_boxes=all_boxes)
   
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
            if verbose: print("Start train the MILSVM")

            
            #data_path_train=  '/home/gonthier/Data_tmp/FasterRCNN_res152_COCO_Paintings_N1_TLforMIL_nms_0.7_all_trainval.tfrecords'
            needToDo = False
            if not(ReDo):
                try:
                    export_dir,np_pos_value,np_neg_value= name_milsvm[j]
                    print('The model of MILSVM exists')
                except(KeyError):
                    print('The model of MILSVM doesn t exist')
                    needToDo = True
            if ReDo or needToDo:
                classifierMILSVM = tf_MILSVM(LR=LR,C=1.0,C_finalSVM=1.0,restarts=restarts,
                   max_iters=max_iters,symway=True,n_jobs=n_jobs,buffer_size=buffer_size,
                   verbose=verboseMILSVM,final_clf=final_clf,Optimizer=Optimizer,optimArg=optimArg,
                   mini_batch_size=mini_batch_size,num_features=size_output,debug=False,
                   num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode) 
                export_dir = classifierMILSVM.fit_MILSVM_tfrecords(data_path=data_path_train,
                                                      class_indice=j,shuffle=True,
                                                      performance=False)
                np_pos_value,np_neg_value = classifierMILSVM.get_porportions()
                name_milsvm[j]=export_dir,np_pos_value,np_neg_value
                with open(cachefile_model, 'wb') as f:
                    pickle.dump(name_milsvm, f)
    
            Number_of_positif_elt = 1 
            number_zone = k_per_bag
            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
            #print(export_dir)
           
            ## Predicition with the MILSVM
            parameters=PlotRegions,RPN,Stocha,CompBest
            param_clf = k_per_bag,Number_of_positif_elt,size_output
            predict_with='LinearSVC'
            predict_with='MILSVM'
            true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited,all_boxes = \
                tfR_evaluation(database,j,dict_class_weight,num_classes,predict_with,
                               export_dir,dict_name_file,mini_batch_size,config,
                               PlotRegions,path_to_img,path_data,param_clf,classes,parameters,verbose,
                               all_boxes=all_boxes)
                  
            # Regroupement des informations     
           
            AP = average_precision_score(true_label_all_test,predict_label_all_test,average=None)
            if (database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                print("MIL-SVM version Average Precision for",depicts_depictsLabel[classes[j]]," = ",AP)
            else:
                print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
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
    
    if database=='VOC2007' or database=='watercolor':
        # DEtection evaluation 
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
        output_dir = path_data +'tmp/' + 'VOC2007_mAP.txt'
        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
        print("Detection score : ",database)
        print(arrayToLatex(aps))
        
           
    print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
    if CompBest: print("mean Average Precision for BEst Score = {0:.3f}".format(np.mean(AP_per_classbS))) 
    print("mean Precision for all the data = {0:.3f}".format(np.mean(P_per_class)))  
    print("mean Recall for all the data = {0:.3f}".format(np.mean(R_per_class)))  
    print("mean Precision @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    
    print(AP_per_class)
    print(arrayToLatex(AP_per_class))


def tfR_evaluation_parall(database,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,PlotRegions,
               path_to_img,path_data,param_clf,classes,parameters,verbose,all_boxes=None):
       
     PlotRegions,RPN,Stocha,CompBest=parameters
     k_per_bag,positive_elt,size_output = param_clf
     thresh = 0.0 # Threshold score or distance MILSVM
     TEST_NMS = 0.7 # Recouvrement entre les classes
     
     load_model = False
     
     if PlotRegions:
         if Stocha:
             extensionStocha = 'Stocha/'
         else:
             extensionStocha = ''
         if database=='Wikidata_Paintings_miniset_verif':
             path_to_output2  = path_data + '/tfMILSVMRegion_paral/'+database+'/'+extensionStocha+'/All'
         else:
             path_to_output2  = path_data + '/tfMILSVMRegion_paral/'+database+'/'+extensionStocha+'/All'
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
         raise(NotImplemented)
         length_matrix = dict_class_weight[0] + dict_class_weight[1]
         if length_matrix>17500*300:
             print('Not enough memory on Nicolas Computer ! use an other classifier than LinearSVC')
             raise(MemoryError)
         X_array = np.empty((length_matrix,size_output),dtype=np.float32)
         y_array =  np.empty((num_classes,length_matrix),dtype=np.float32)
         x_array_ind = 0
     
     if (PlotRegions or predict_with=='LinearSVC'):
        if verbose: print("Start ploting Regions selected by the MILSVM in training phase")
        train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
            num_classes=num_classes))
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
            mei = tf.argmax(Prod_best,axis=2)
            score_mei = tf.reduce_max(Prod_best,axis=2)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                    PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
                    #print(PositiveExScoreAll.shape)
                    if predict_with=='LinearSVC' and k_per_bag==300  and positive_elt==1:
                        raise(NotImplemented)
                        for k in range(len(fc7s)):
                            for l in range(num_classes):
                                label_i = labels[k,l]
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
                        for k in range(len(labels)):                          
                            if database=='VOC2007':
                                name_img = str(name_imgs[k].decode("utf-8") )
                            else:
                                name_img = name_imgs[k]
                            rois = roiss[k,:]
                            #if verbose: print(name_img)
                            if database=='VOC12' or database=='Paintings' or database=='VOC2007' or database =='watercolor':
                                complet_name = path_to_img + name_img + '.jpg'
                                name_sans_ext = name_img
                            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                                name_sans_ext = os.path.splitext(name_img)[0]
                                complet_name = path_to_img +name_sans_ext + '.jpg'
                            im = cv2.imread(complet_name)
                            blobs, im_scales = get_blobs(im)
                            scores_all = PositiveExScoreAll[:,k,:]
                            roi = roiss[k,:]
                            roi_boxes =  roi[:,1:5] / im_scales[0] 
                            roi_boxes_and_score = None
                            local_cls = []
                            for j in range(num_classes):
                                if labels[k,j] == 1:
                                    local_cls += [classes[j]]
                                    roi_with_object_of_the_class = PositiveRegions[j,k] % len(rois) # Because we have repeated some rois
                                    roi = rois[roi_with_object_of_the_class,:]
                                    roi_scores = [get_PositiveRegionsScore[j,k]]
                                    roi_boxes =  roi[1:5] / im_scales[0]   
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
                                cls = local_cls + ['RPN']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                roi_boxes_and_score = np.vstack((roi_boxes_and_score,best_RPN_roi_boxes_score))
                            else:
                                cls = local_cls
                            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                            name_output = path_to_output2 +'Train/' + name_sans_ext + '_Regions.jpg'
                            plt.savefig(name_output)
                            plt.close()
                except tf.errors.OutOfRangeError:
                    break
        #tf.reset_default_graph()
     
     print("Testing Time")
     # Training time !
     if predict_with=='LinearSVC':
         if verbose: print('Start training LiearSVC')
         clf =  TrainClassif(X_array,y_array,clf='LinearSVC',
                             class_weight=dict_class_weight,gridSearch=True,
                             n_jobs=1,C_finalSVM=1)
         if verbose: print('End training LiearSVC')
             
     # Testing time !
     train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
     train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
                                                    num_classes=num_classes))
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
     with tf.Session(config=config) as sess:
        if load_model==False:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            graph= tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            y = graph.get_tensor_by_name("y:0")
            Prod_best = graph.get_tensor_by_name("Prod:0")
            mei = tf.argmax(Prod_best,axis=2)
            score_mei = tf.reduce_max(Prod_best,axis=2)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        while True:
            try:
                fc7s,roiss, labels,name_imgs = sess.run(next_element)
                PositiveRegions,get_RegionsScore,PositiveExScoreAll = \
                    sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
                true_label_all_test += [labels]
                if predict_with=='MILSVM':
                    predict_label_all_test +=  [get_RegionsScore]
#                if predict_with=='LinearSVC':
                    
                for k in range(len(labels)):
                    if database=='VOC2007' :
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    else:
                         complet_name = path_to_img + name_imgs[k] + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    if predict_with=='MILSVM':
                        scores_all = PositiveExScoreAll[:,k,:]
                    elif predict_with=='LinearSVC':
                        scores = clf.decision_function(fc7s[k,:])

                    roi = roiss[k,:]
                    roi_boxes =  roi[:,1:5] / im_scales[0] 
                    
                    for j in range(num_classes):
                        scores = scores_all[j,:]
                        inds = np.where(scores > thresh)[0]
                        cls_scores = scores[inds]
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        all_boxes[j][i] = cls_dets
                    i+=1
    
                for l in range(len(name_imgs)): 
                    if database=='VOC2007' :
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
                        if  database=='VOC2007':
                            name_img = str(name_imgs[k].decode("utf-8") )
                        else:
                            name_img = name_imgs[k]
                        rois = roiss[k,:]
                        if database=='VOC12' or database=='Paintings' or database=='VOC2007' or database=='watercolor':
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
                        plt.savefig(name_output)
                        plt.close()
            except tf.errors.OutOfRangeError:
                break
     tf.reset_default_graph()
     true_label_all_test = np.concatenate(true_label_all_test)
     predict_label_all_test = np.transpose(np.concatenate(predict_label_all_test,axis=1))
     name_all_test = np.concatenate(name_all_test)
     labels_test_predited = (np.sign(predict_label_all_test) +1.)/2
     return(true_label_all_test,predict_label_all_test,name_all_test,
            labels_test_predited,all_boxes)
      
def tfR_evaluation(database,j,dict_class_weight,num_classes,predict_with,
               export_dir,dict_name_file,mini_batch_size,config,PlotRegions,
               path_to_img,path_data,param_clf,classes,parameters,verbose,all_boxes=None):
    
     PlotRegions,RPN,Stocha,CompBest=parameters
     k_per_bag,positive_elt,size_output = param_clf
     thresh = 0.0 # Threshold score or distance MILSVM
     TEST_NMS = 0.7 # Recouvrement entre les classes
     
     load_model = False
     
     if PlotRegions:
         if Stocha:
             extensionStocha = 'Stocha/'
         else:
             extensionStocha = ''
         if database=='Wikidata_Paintings_miniset_verif':
             path_to_output2  = path_data + '/tfMILSVMRegion/'+database+'/'+extensionStocha+depicts_depictsLabel[classes[j]]
         else:
             path_to_output2  = path_data + '/tfMILSVMRegion/'+database+'/'+extensionStocha+classes[j]
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
        if verbose: print("Start ploting Regions selected by the MILSVM in training phase")
        train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois(r,classe_index=j,num_classes=num_classes))
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
            mei = tf.argmax(Prod_best,axis=1)
            score_mei = tf.reduce_max(Prod_best,axis=1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                    PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
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
                                if database=='VOC12' or database=='Paintings' or database=='VOC2007' or database =='watercolor':
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
                                    cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                                    best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                    roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                                else:
                                    cls = ['MILSVM']
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
     train_dataset = train_dataset.map(lambda r: parser_w_rois(r,classe_index=j,num_classes=num_classes))
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
            mei = tf.argmax(Prod_best,axis=1)
            score_mei = tf.reduce_max(Prod_best,axis=1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        while True:
            try:
                fc7s,roiss, labels,name_imgs = sess.run(next_element)
                PositiveRegions,get_RegionsScore,PositiveExScoreAll = sess.run([mei,score_mei,Prod_best], feed_dict={X: fc7s, y: labels})
                #print(PositiveExScoreAll.shape)
                true_label_all_test += [labels]
                if predict_with=='MILSVM':
                    predict_label_all_test +=  [get_RegionsScore]
#                if predict_with=='LinearSVC':
                    
                for k in range(len(labels)):
                    if database=='VOC2007' or database=='watercolor':
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    if predict_with=='MILSVM':
                        scores = PositiveExScoreAll[k,:]
                    elif predict_with=='LinearSVC':
                        scores = clf.decision_function(fc7s[k,:])
                    inds = np.where(scores > thresh)[0]
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
                if PlotRegions and predict_with=='MILSVM':
                    if verbose and FirstTime: 
                        FirstTime = False
                        print("Start ploting Regions selected")
                    for k in range(len(PositiveRegions)):                          
                        if labels[k] == 1:
                            name_img = str(name_imgs[k].decode("utf-8") )
                            rois = roiss[k,:]
                            #if verbose: print(name_img)
                            if database=='VOC12' or database=='Paintings' or database=='VOC2007' or database =='watercolor':
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
                                cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                            else:
                                cls = ['MILSVM']
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
##            np_pos_value,np_neg_value = classifierMILSVM.get_porportions()
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
##            np_pos_value,np_neg_value = classifierMILSVM.get_porportions()
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
#        path_to_img = '/media/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
#    elif database=='VOC12':
#        item_name = 'name_img'
#        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
#        item_name = 'image'
#        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
        raise NotImplemented # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
#        item_name = 'image'
#        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
#    databasetxt =path_data + database + '.txt'
#    df_label = pd.read_csv(databasetxt,sep=",")
#    if database=='Wikidata_Paintings_miniset_verif':
#        df_label = df_label[df_label['BadPhoto'] <= 0.0]
    
    DATA_DIR =  '/media/HDD/data/Fondazione_Zeri/Selection_Olivier/'
    output_DIR = '/media/HDD/output_exp/ClassifPaintings/Zeri/'
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
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
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
    @param : Type of MILSVM used or not : choice :  ['MISVM','miSVM','LinearMISVC','LinearmiSVC'] # TODO in the future also SGD
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
        path_to_img = '/media/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
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
                path_to_output2  = path_data + '/MILSVMRegion/'+depicts_depictsLabel[classes[j]] + '/'
            else:
                path_to_output2  = path_data + '/MILSVMRegion/'+classes[j] + '/'
            path_to_output2_bis = path_to_output2 + 'Train'
            path_to_output2_ter = path_to_output2 + 'Test'
            pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
            pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True) 
            
#        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
#        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
#        pos_name = name_trainval[y_trainval[:,j]==1]
        
        if verbose: print("Start train the MILSVM")
        
        if misvm_type=='miSVM':
            classifierMILSVM = misvm.miSVM(kernel='linear', C=1.0, max_iters=10)
        elif misvm_type=='MISVM':
            classifierMILSVM = misvm.MISVM(kernel='linear', C=1.0, max_iters=10,verbose=True,restarts=0)
        elif misvm_type=='LinearMISVC':
            classifierMILSVM = mi_linearsvm.MISVM(C=1.0, max_iters=10,verbose=True,restarts=0)
        elif misvm_type=='LinearmiSVC':
            classifierMILSVM = mi_linearsvm.miSVM(C=1.0, max_iters=10,verbose=True,restarts=0)
        
        classifierMILSVM.fit(train_bags, y_trainval)
        classifier = classifierMILSVM

        
        if verbose: print("End training the MILSVM")
        
       
        
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
    
    
def PlotRegionsLearnByMILSVM():
    """ 
   This function will plot the regions considered as dog by the MILSVM of Said
   during the training and after during the testing
    """
    verbose = True
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    path_to_img = '/media/HDD/data/Painting_Dataset/'
#    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/dogRegion/'
#    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/aeroplaneRegion/'
#    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/chairRegion/'
#    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/boatRegion/'
#    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/birdRegion/'
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
    #from trouver_classes_parmi_K import MILSVM
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
        path_to_output2  = path_data + '/MILSVMRegion/'+classes[j] + '/'
        path_to_output2_bis = path_to_output2 + 'Train'
        path_to_output2_ter = path_to_output2 + 'Test'
        pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True) 
        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
        pos_name = name_trainval[y_trainval[:,j]==1]
        #print(pos_name)
        classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
               max_iters=max_iters,symway=True,n_jobs=n_jobs,
               all_notpos_inNeg=False,gridSearch=True,
               verbose=False,final_clf=final_clf)     
        classifierMILSVM.fit(pos_ex, neg_ex)
        PositiveRegions = classifierMILSVM.get_PositiveRegions()
        get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()
        PositiveExScoreAll =  classifierMILSVM.get_PositiveExScoreAll()
        
        a = np.argmax(PositiveExScoreAll,axis=1)
        assert((a==PositiveRegions).all())
        assert(len(pos_name)==len(PositiveRegions))
        
#        get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()
        
        if verbose: print("End training the MILSVM")
        
        pos_ex_after_MILSVM = np.zeros((len(pos_ex),size_output))
        neg_ex_keep = np.zeros((len(neg_ex),size_output))
        for k,name_imgtab in enumerate(pos_name):
            pos_ex_after_MILSVM[k,:] = pos_ex[k,PositiveRegions[k],:] # We keep the positive exemple according to the MILSVM from Said
            
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
            cls = ['RPN','MILSVM']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
            best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
            roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
            roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
            vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
            name_output = path_to_output2 +'Train/' + name_img + '_Regions.jpg'
            #+ '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
            plt.savefig(name_output)
            plt.close()
        
        neg_ex_keep = neg_ex.reshape(-1,size_output)
        
        X = np.vstack((pos_ex_after_MILSVM,neg_ex_keep))
        y_pos = np.ones((len(pos_ex_after_MILSVM),1))
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
                cls = ['RPN','Classif']  # Comparison of the best region according to the faster RCNN and according to the MILSVM de Said
                roi_scores =  [np.max(decision_function_output)]
                best_RPN_roi_boxes_score =  np.expand_dims(np.expand_dims(np.concatenate((best_RPN_roi_boxes,best_RPN_roi_scores)),axis=0),axis=0)
                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                roi_boxes_and_score = np.vstack((best_RPN_roi_boxes_score,roi_boxes_score))
                vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                name_output = path_to_output2 +'Test/' + name_img  + '_Regions.jpg'
                #'_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
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
        
if __name__ == '__main__':
#    FasterRCNN_TL_MILSVM_newVersion()
#    petitTestIllustratif()
#    FasterRCNN_TL_MILSVM_ClassifOutMILSVM()
#    petitTestIllustratif_RefineRegions()
#    PlotRegionsLearnByMILSVM()
#    FasterRCNN_TL_MILSVM_ClassifOutMILSVM(demonet = 'res152_COCO',database = 'Paintings', 
#                                          verbose = True,testMode = True,jtest = 6,
#                                          PlotRegions = False)
#    FasterRCNN_TL_MILSVM_ClassifOutMILSVM(demonet = 'res152_COCO',
#                                          database = 'Paintings', 
#                                          verbose = True,testMode = True,jtest = 6,
#                                          PlotRegions = False,RPN=False,Stocha=False,
#                                          k_per_bag=30)
#    FasterRCNN_TL_MILSVM_ClassifOutMILSVM(demonet = 'res152_COCO',
#                                          database = 'VOC12', 
#                                          verbose = True,testMode = True,jtest = 0,
#                                          PlotRegions = False,RPN=False,Stocha=False,
#                                          k_per_bag=30)

#    FasterRCNN_TL_MILSVM_ClassifOutMILSVM(demonet = 'res152_COCO',
#                                          database = 'VOC2007', 
#                                          verbose = True,testMode = False,jtest = 1,
#                                          PlotRegions = False,RPN=False,CompBest=False)
#    FasterRCNN_TL_MISVM(demonet = 'res152_COCO',database = 'Paintings', 
#                                          verbose = True,testMode = True,jtest = 0,
#                                          PlotRegions = False,misvm_type='LinearMISVC')
#    detectionOnOtherImages(demonet = 'res152_COCO',database = 'Wikidata_Paintings_miniset_verif')
    tfRecords_FasterRCNN(demonet = 'res152_COCO',database = 'VOC2007', 
                                  verbose = True,testMode = True,jtest = 'cow',
                                  PlotRegions = False,saved_clf=False,RPN=True,
                                  CompBest=False,Stocha=True,k_per_bag=300,
                                  parallel_op=False,CV_Mode='LA',num_split=2)
#    tfRecords_FasterRCNN(demonet = 'res152_COCO',database = 'Paintings', 
#                                  verbose = True,testMode = False,jtest = 0,
#                                  PlotRegions = False,saved_clf=False,RPN=True,
#                                  CompBest=False,Stocha=True,k_per_bag=300,parallel_op=True)
#    tfRecords_FasterRCNN(demonet = 'res152_COCO',database = 'Paintings', 
#                                  verbose = False,testMode = False,jtest = 6,
#                                  PlotRegions = False,saved_clf=False,RPN=True,
#                                  CompBest=False,Stocha=True,k_per_bag=300)
#    tfRecords_FasterRCNN(demonet = 'res152_COCO',database = 'watercolor', 
#                                  verbose = True,testMode = True,jtest =0,
#                                  PlotRegions = False,saved_clf=False,RPN=True,
#                                  CompBest=False,Stocha=True,k_per_bag=300,parallel_op=False)
