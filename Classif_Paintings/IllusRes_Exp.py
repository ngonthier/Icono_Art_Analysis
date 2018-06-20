#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:22:22 2018

The goal of this file is to plot the results from the CP_Sauvegarde/Experience_bizarre_1 pour l instant

@author: gonthier
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from TL_MILSVM import parser_w_rois_all_class
import cv2
from tf_faster_rcnn.lib.model.test import get_blobs
import numpy as np
import os
from tf_faster_rcnn.lib.model.nms_wrapper import nms
from FasterRCNN import vis_detections_list

def plot_Train_Test_Regions(database,number_im,dict_name_file,path_to_output2,
                            export_dir,problem_class=[],RPN=False,water_mark='',
                            transform_output=None,with_rois_scores_atEnd=False,scoreInMILSVM=False):
    verbose =True 
#    k_per_bag,positive_elt,size_output = param_clf
    thresh = 0.0 # Threshold score or distance MILSVM
    TEST_NMS = 0.3 # Recouvrement entre les classes
    mini_batch_size = 100
    
    load_model = False
    if database=='VOC2007':
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
        classes =  ['bicycle', 'bird','car', 'cat','dog', 'person']
    else:
        raise(NotImplemented)
    if len(problem_class) > 0:
        label_problem = []
        for c in problem_class:
            label_problem += [np.where(np.array(classes)==c)[0][0]]
    else:
        label_problem = np.arange(len(classes))
    num_classes = len(classes)
    num_images = 10000
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    path_to_output2_bis = path_to_output2 +  'Train/'
    path_to_output2_ter = path_to_output2 + 'Test/'
    print(path_to_output2_bis,path_to_output2_ter)
    pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True)
         
    export_dir_path = ('/').join(export_dir.split('/')[:-1])
    name_model_meta = export_dir + '.meta'
    get_roisScore = (with_rois_scores_atEnd or scoreInMILSVM)
    if verbose: print("Start ploting Regions selected by the MILSVM in training phase")
    if transform_output=='tanh':
         with_tanh=True
         with_softmax=False
    elif transform_output=='softmax':
         with_softmax=True
         with_tanh = False
#    if seuil_estimation: print('It may cause problem of doing softmax and tangent estimation')
    else:
         with_softmax,with_tanh = False,False
    train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
    train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
        num_classes=num_classes,with_rois_scores=get_roisScore))
    dataset_batch = train_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(name_model_meta)
        new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
        load_model = True
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
        else:
            mei = tf.argmax(Prod_best,axis=2)
            score_mei = tf.reduce_max(Prod_best,axis=2)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        index_im = 0
        while True:
            try:
                next_element_value = sess.run(next_element)
#                    print(len(next_element_value))
                if not(with_rois_scores_atEnd) and not(scoreInMILSVM):
                    fc7s,roiss, labels,name_imgs = next_element_value
                else:
                    fc7s,roiss,rois_scores,labels,name_imgs = next_element_value
                if scoreInMILSVM:
                    feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                else:
                    feed_dict_value = {X: fc7s, y: labels}
                if with_tanh:
                    PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                elif with_softmax:
                    PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Softmax], feed_dict=feed_dict_value)
                else:
                    PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = \
                    sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)
                for k in range(len(labels)):
                    label_tmp = np.array(labels[k,:])
                    label_tmp_index = np.where(label_tmp==1)
                    intersec = np.intersect1d(label_tmp_index,label_problem)
#                    print(intersec)
                    if len(intersec)==0:
                        continue
                    
                    if index_im > number_im:
                        continue                          
                    if database=='VOC2007' or database =='watercolor':
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
                    name_output = path_to_output2_bis + name_sans_ext +water_mark+ '_Regions.jpg'
                    plt.savefig(name_output)
                    plt.close()
                    index_im += 1
            except tf.errors.OutOfRangeError:
                break
        #tf.reset_default_graph()
     
    print("Testing Time")
     # Training time !
     
     # Testing time !
    train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
    train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
        num_classes=num_classes,with_rois_scores=get_roisScore))
    dataset_batch = train_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()
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
                if scoreInMILSVM:
                    feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                else:
                    feed_dict_value = {X: fc7s, y: labels}
                if with_tanh:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                elif with_softmax:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Softmax], feed_dict=feed_dict_value)
                else:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll = \
                    sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)
                if with_rois_scores_atEnd:
                    PositiveExScoreAll = PositiveExScoreAll*rois_scores
                    get_RegionsScore = np.max(PositiveExScoreAll,axis=2)
                    PositiveRegions = np.amax(PositiveExScoreAll,axis=2)
#                if predict_with=='LinearSVC':              
                for k in range(len(labels)):
                    label_tmp = np.array(labels[k,:])
                    label_tmp_index = np.where(label_tmp==1)
                    intersec = np.intersect1d(label_tmp_index,label_problem)
                    if len(intersec)==0:
                        continue
                    if database=='VOC2007'or database =='watercolor' :
                        complet_name = path_to_img + str(name_imgs[k].decode("utf-8")) + '.jpg'
                    else:
                         complet_name = path_to_img + name_imgs[k] + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    scores_all = PositiveExScoreAll[:,k,:]
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
    
#                for l in range(len(name_imgs)): 
#                    if database=='VOC2007' :
#                        name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
#                    else:
#                        name_all_test += [[name_imgs[l]]]
                
                if verbose and (ii%1000==0):
                    print("Plot the images :",ii)
                if verbose and FirstTime: 
                    FirstTime = False
                    print("Start ploting Regions on test set")
                for k in range(len(labels)): 
                    if ii > number_im:
                        continue
                    if  database=='VOC2007' or database =='watercolor':
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
                    # Attention you use the 0.5 threshold
                    vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.0)
                    name_output = path_to_output2_ter + name_sans_ext +water_mark+ '_Regions.jpg'
                    plt.savefig(name_output)
                    plt.close()
            except tf.errors.OutOfRangeError:
                break
    tf.reset_default_graph()

def plotEXP1():
    """ Difference of result with and without right ratio """
    cache_model_right_ratio = '/media/HDD/output_exp/ClassifPaintings/MILSVM1527210600.5675514/model'
    cache_model_false_ratio = '/media/HDD/output_exp/ClassifPaintings/MILSVM1527118925.7643065/model'
    database = 'VOC2007'
    number_im = 250
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    sets = ['train','val','trainval','test']
    dict_name_file = {}
    demonet = 'res152_COCO'
    problem_class = ['boat','chair','cow']
    path_data =  '/media/HDD/output_exp/ClassifPaintings/'
    for set_str in sets:
        name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
        dict_name_file[set_str] = name_pkl_all_features
    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/Illus_Exp1/'
    RPN = False
    water_mark = 'TrueRatio'
    plot_Train_Test_Regions(database,number_im,dict_name_file,path_to_output2,\
                            cache_model_right_ratio,problem_class,RPN,water_mark)
    water_mark = 'FalseRatio'
    plot_Train_Test_Regions(database,number_im,dict_name_file,path_to_output2,\
                            cache_model_false_ratio,problem_class,RPN,water_mark)

def plotEXP2():
    """ Illustration of the use of score and not """
     # watercolor_res152_COCO_r10_s166_k300_m900_p_wr_gd_MILSVM
     
    cache_model_wt_score = '/media/HDD/output_exp/ClassifPaintings/MILSVM/1527793922.5090935/model'
    cache_model_w_score = '/media/HDD/output_exp/ClassifPaintings/MILSVM/1528134380.9923458/model'
    database = 'watercolor'
    number_im = 250
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    sets = ['train','val','trainval','test']
    dict_name_file = {}
    demonet = 'res152_COCO'
    problem_class = ['person','cat','bicycle']
    path_data =  '/media/HDD/output_exp/ClassifPaintings/'
    for set_str in sets:
        name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'_'+set_str+'.tfrecords'
        dict_name_file[set_str] = name_pkl_all_features
    path_to_output2 = '/media/HDD/output_exp/ClassifPaintings/Illus_Exp2/'
    RPN = False
    water_mark = 'WithoutScore'
    plot_Train_Test_Regions(database,number_im,dict_name_file,path_to_output2,\
                            cache_model_wt_score,problem_class,RPN,water_mark,
                            transform_output='tanh',scoreInMILSVM=False)
    water_mark = 'WithScore'
    plot_Train_Test_Regions(database,number_im,dict_name_file,path_to_output2,\
                            cache_model_w_score,problem_class,RPN,water_mark,
                            transform_output='tanh',scoreInMILSVM=True)
        
if __name__ == '__main__':
    plotEXP2()