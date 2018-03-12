#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:45:35 2018

@author: gonthier
"""
import pickle
import tensorflow as tf
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
from sklearn.model_selection import PredefinedSplit
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
from trouver_classes_parmi_K import MILSVM,TrainClassif
from LatexOuput import arrayToLatex
from FasterRCNN import vis_detections_list,Compute_Faster_RCNN_features

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
                   'res101_VOC12' :'res101_faster_rcnn_iter_110000.ckpt',
                   'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'
                   }
CLASSES_SET ={'VOC' : CLASSESVOC,
              'COCO' : CLASSESCOCO }

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
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
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
                vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
                plt.savefig(name_output)
            plt.close('all')

def reduce_to_k_regions(k,rois,roi_scores, fc7,new_nms_thresh,score_threshold,minimal_surface):
    """ Reduce the number of region to k or less """
    
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)
        
    keep = np.where(roi_scores> score_threshold)
    rois = rois[keep[0], :]
    roi_scores = roi_scores[keep]
    fc7 = fc7[keep[0],:]
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)

    width = rois[:,2] - rois[:,0] +1
    height = rois[:,3] - rois[:,1] +1
    surface = width*height
    keep = np.where(surface > minimal_surface)
    rois = rois[keep[0], :]
    roi_scores = roi_scores[keep]
    fc7 = fc7[keep[0],:]
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)
        
    #new_nms_thresh = 0.0
    keep_all = []
    for i in range(7):
        rois_plus_scores = np.hstack((rois[:,1:5],roi_scores.reshape(-1,1)))
        tmp_keep = nms(rois_plus_scores,new_nms_thresh)
        
        keep_new = np.setdiff1d(tmp_keep,keep_all) # Nouveau index
        
        keep_all2 = np.union1d(keep_all,tmp_keep) # sorted 
        if len(keep_all2) > k:
            keep = np.union1d(keep_all,keep_new[0:k-len(keep_all)]).astype(int)
            rois = rois[keep, :]
            roi_scores = roi_scores[keep]
            fc7 = fc7[keep,:]
            assert(0 in keep)
            return(rois,roi_scores, fc7)
        else: 
            keep_all = keep_all2
            
        new_nms_thresh += 0.1
    
    return(rois,roi_scores, fc7)
    
            
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
        vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.0)
        name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'k_'+str(k_per_bag)+'_MILSVMbestROI.jpg'
        plt.savefig(name_output)
    plt.close('all')
    
    

def FasterRCNN_TL_MILSVM_newVersion():
    """ Function to test if you can refind the same AP metric by reading the saved CNN features """
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
    
def FasterRCNN_TL_MILSVM_ClassifOutMILSVM():
    """ 
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    
    In this function we train an SVM only on the positive element returned by 
    the algo
    
    The idea of thi algo is : 
        1/ Compute CNN features
        2/ Do NMS on the regions 
    
    option to train on background part also
    option on  scaling : sklearn.preprocessing.StandardScaler
    option : add a wieghted balanced of the SVM because they are really unbalanced classes
    TODO : mine hard negative exemple ! 
    """
    # TODO be able to train on background 
    verbose = False
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
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
    
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0 and not(i==0):
            if verbose: print(i,name_img)
        rois,roi_scores,fc7 = features_resnet_dict[name_img]
        #print(rois.shape,roi_scores.shape)
        
        rois,roi_scores,fc7_reduce =  reduce_to_k_regions(k_per_bag,rois,roi_scores, fc7,
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
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if(classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
        else:
            raise NotImplementedError
        InSet = (df_label.loc[df_label[item_name]==name_img]['set']=='test').any()
        if InSet: 
            if not(Test_on_k_bag):
                f_test[index_test] = fc7
                index_test += 1   
    
    del features_resnet_dict
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
    testMode = False
    jtest = 5
    for j,classe in enumerate(classes):
        if testMode and j<jtest:
            continue
        neg_ex = X_trainval[y_trainval[:,j]==0,:,:]
        pos_ex =  X_trainval[y_trainval[:,j]==1,:,:]
        classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
               max_iters=max_iters,symway=True,n_jobs=n_jobs,
               all_notpos_inNeg=False,gridSearch=True,
               verbose=False,final_clf=final_clf)     
        classifierMILSVM.fit(pos_ex, neg_ex)
        PositiveRegions = classifierMILSVM.get_PositiveRegions()
#        get_PositiveRegionsScore = classifierMILSVM.get_PositiveRegionsScore()
        
        if verbose: print("End training the MILSVM")
        
        pos_ex_after_MILSVM = np.zeros((len(pos_ex),size_output))
        neg_ex_keep = np.zeros((len(neg_ex),size_output))
        for k in range(len(pos_ex)):
            pos_ex_after_MILSVM[k,:] = pos_ex[k,PositiveRegions[k],:] # We keep the positive exemple according to the MILSVM from Said
        
        NegativeRegions = classifierMILSVM.get_NegativeRegions()
        
#        neg_ex_keep = np.zeros((len(pos_ex),size_output))
#        for k in range(len(neg_ex)):
#            neg_ex_keep[k,:] = neg_ex[k,NegativeRegions[k],:]
        
        #neg_ex_keep= neg_ex[:,0,:] # shape = (4237, 2048)
        
        neg_ex_keep = neg_ex.reshape(-1,size_output)
        
        #neg_ex_keep = neg_ex_keep.reshape(-1,size_output)
        #print(neg_ex_keep.shape)
        # TODO Here we only keep the best score element but it would be nice to used other think 
        X = np.vstack((pos_ex_after_MILSVM,neg_ex_keep))
        y_pos = np.ones((len(pos_ex_after_MILSVM),1))
        y_neg = np.zeros((len(neg_ex_keep),1))
        y = np.vstack((y_pos,y_neg)).ravel()
        if verbose: print(X.shape,y.shape)
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
        
if __name__ == '__main__':
#    FasterRCNN_TL_MILSVM_newVersion()
#    petitTestIllustratif()
    FasterRCNN_TL_MILSVM_ClassifOutMILSVM()
#    petitTestIllustratif_RefineRegions()