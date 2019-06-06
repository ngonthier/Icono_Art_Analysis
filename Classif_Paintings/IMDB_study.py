#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:12:22 2019

Le but de ce script est de verifier differentes questions que l on se pose sur 
notre dataset IconArt : 
    
1/ Est ce que les boites des GT sont contenues dans les boites proposees par 
Faster RCNN : Test_GT_inProposals
 : quel est le score maximal que l'on peut theoriquement obtenir avec ce jeu de boites 
 issues du Faster RCNN ?

2/ Faire defiler les boites sur une image donnee

3/ Quel est le score avec des boites aléatoires @ 0.1

4/ A quoi ressemble l'ensemble des points en TSNE ?

@author: gonthier
"""

import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle
from shutil import copyfile
from LatexOuput import arrayToLatex

import voc_eval

from tf_faster_rcnn.lib.datasets.factory import get_imdb
from tf_faster_rcnn.lib.model.test import get_blobs

from TL_MIL import parser_w_rois_all_class,parser_all_elt_all_class
from FasterRCNN import vis_detections

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from matplotlib import offsetbox

from time import time

from tensorflow.contrib.tensorboard.plugins import projector
import os

from FasterRCNN import _int64_feature,_bytes_feature,_floats_feature
from IMDB import get_database

def getDictFeaturesFasterRCNN(database,k_per_bag = 300,demonet='res152_COCO',metamodel = 'FasterRCNN'):
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    
    #demonet = 'res152_COCO'
    #metamodel = 'FasterRCNN'
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    PCAuse = False
    eval_onk300 = False
    number_composant = 0.
    
    sets = ['train','val','trainval','test']
    dict_name_file = {}
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
    return(dict_name_file)

def getTFRecordDataset(name_file,k_per_bag = 300,num_features = 2048,num_classes = 7,dim_rois = 5,allelt=False):
    
    
    get_roisScore = True
    
    mini_batch_size = 256
    train_dataset = tf.data.TFRecordDataset(name_file)
    if not(allelt):
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,\
            num_rois=k_per_bag,dim_rois=dim_rois))
    else:
        train_dataset = train_dataset.map(lambda r: parser_all_elt_all_class(r, \
            num_classes=num_classes,num_features=num_features,\
            num_rois=k_per_bag,dim_rois=dim_rois,noReshape=False))
    print(train_dataset)
    dataset_batch = train_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    return(next_element)

def bb_intersection_over_union(boxA, boxB):
    """
    Boxes must be x1,y1,x2,y2
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    print(boxAArea,boxBArea,interArea)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def plotBoxesWithinImage(number_Im_plot=None):
    """
    Number of images we save with the boxes
    """
    
    database='IconArt_v1'
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    databasetxt =path_data_csvfile + database + ext

    df_label = pd.read_csv(databasetxt,sep=",")
    
    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    
    imdb = get_imdb('IconArt_v1_test')

    k_per_bag = 300
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag)
    name_file = dict_name_file['test']
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag)

    # Load the Faster RCNN proposals
    dict_rois = {}
    sess = tf.Session()
    sum_of_classes = []
    
    list_im_with_classes = []
    while True:
        try:
            fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
            for k in range(len(labels)):
                name_im = name_imgs[k].decode("utf-8")
                if name_im in list_im_withanno: 
                    complet_name = path_to_img + str(name_im) + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    roi = roiss[k,:]
                    roi_boxes =  roi[:,1:5] / im_scales[0] 
                    dict_rois[name_im] = roi_boxes
                    sum_labels = np.sum(labels[k,:])
                    sum_of_classes += [sum_labels]
                    if sum_labels >0.:
                        list_im_with_classes += [name_im]
        except tf.errors.OutOfRangeError:
            break

    sess.close()
    print('End read the boxes proposals')
    
    # Plot all the boxes on all the images
    base_plot = '/media/gonthier/HDD/output_exp/ClassifPaintings/PlotBoxesIconArt_v1_Test'
    if number_Im_plot is None:
        list_im_with_classes_loc = list_im_with_classes
    else:
        list_im_with_classes_loc = list_im_with_classes[0:number_Im_plot]
    for name_im in list_im_with_classes_loc:
        path_tmp = base_plot +'/' + name_im
        pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
        boxes = dict_rois[name_im]
        complet_name = path_to_img + str(name_im) + '.jpg'
        im = cv2.imread(complet_name)
        for i in range(len(boxes)):
            dets = np.hstack((boxes[i,:],[1.])).reshape(1,-1)
            class_name ='object'
            vis_detections(im, class_name, dets, thresh=0.5,with_title=True,draw=False)
#            plt.show()
            name_output = path_tmp + '/' +name_im +'_'+str(i)+'jpg'
            plt.savefig(name_output)
            plt.close()
    
    ## This doesn t work
#    print('Please provide the name of the image, to quit right quit and rand for random image')
#    name = input('Name of the image :') 
#    while not(name=='quit'):
#        if name=='rand' or name in list_im_with_classes:
#            if name=='rand' :
#                name_im = np.random.choice(list_im_with_classes,[1])[0]
#            else:
#                name_im = name
#            boxes = dict_rois[name_im]
#            plotBoxesIm(name_im,boxes,path_to_img=path_to_img)
#        else:
#            print(name,'is not in the test image with a classes of interest')
#            name = input('Name of the image or rand or quit :')

def plotBoxesIm(name_im,boxes,path_to_img=''):
    complet_name = path_to_img + str(name_im) + '.jpg'
    im = cv2.imread(complet_name)
    for i in range(len(boxes)):
        dets = np.hstack((boxes[i,:],[1.])).reshape(1,-1)
        class_name ='object'
        vis_detections(im, class_name, dets, thresh=0.5,with_title=True)
        plt.show()
#        plt.savefig(name_output)
        plt.close()
#        input("Press Enter to continue...")
        
def modify_EdgeBoxesWrongBoxes(database='IconArt_v1',k_per_bag = 300,\
                               metamodel = 'EdgeBoxes',demonet='res152'):    
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    sess = tf.Session()
    dim_rois = 4
    #num_classes = 7
    num_features = 2048
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC =get_database(database)
    for key in dict_name_file.keys():
        print('=========',key,'==========')
        name_file = dict_name_file[key]
        dst = name_file.replace('.tfrecords','_old.tfrecords')
        name_file_new = name_file.replace('.tfrecords','_new.tfrecords')
        copyfile(name_file, dst)   
        next_element = getTFRecordDataset(dst,k_per_bag =k_per_bag,\
                                          dim_rois = dim_rois,allelt=True,num_classes=num_classes)
        writer = tf.python_io.TFRecordWriter(name_file_new)
        while True:
            try:
                heights,widths,num_regionss,num_featuress,dim1_roiss,roiss,roi_scoress,\
                fc7s,classes_vectorss,name_sans_exts = sess.run(next_element)
                
                for k in range(len(classes_vectorss)):
                    height,width,num_regions,num_features,dim1_rois,rois,roi_scores,\
                    fc7,classes_vectors,name_sans_ext = heights[k],widths[k],num_regionss[k],\
                    num_featuress[k],dim1_roiss[k],roiss[k,:],roi_scoress[k],\
                    fc7s[k,:],classes_vectorss[k,:],name_sans_exts[k]
                    rois = rois[:,[0,2,1,3]] # Modification of the elements
                    # We change from x,x+w,y,y+h to x,y,x+w,y+h
                    feature={
                                'height': _int64_feature(height),
                                'width': _int64_feature(width),
                                'num_regions': _int64_feature(num_regions),
                                'num_features': _int64_feature(num_features),
                                'dim1_rois': _int64_feature(dim1_rois),
                                'rois': _floats_feature(rois),
                                'roi_scores': _floats_feature(roi_scores),
                                'fc7': _floats_feature(fc7),
                                'label' : _floats_feature(classes_vectors),
                                'name_img' : _bytes_feature(name_sans_ext)} # str.encode(
                    features=tf.train.Features(feature=feature)
                    example = tf.train.Example(features=features)    
                    writer.write(example.SerializeToString())
            except tf.errors.OutOfRangeError:
                break
        writer.close()
        
def modify_RMNAddClasses(database='RMN',k_per_bag = 300,\
                               metamodel = 'FasterRCNN',demonet='res152_COCO'):    
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    sess = tf.Session()
    dim_rois = 5
    num_classes_old = 1
    num_features = 2048
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC =get_database(database)
    name_file = dict_name_file['trainval']
    dst = name_file.replace('.tfrecords','_old.tfrecords')
    name_file_new = name_file.replace('.tfrecords','_new.tfrecords')
    #copyfile(name_file, dst)   
    next_element = getTFRecordDataset(dst,k_per_bag =k_per_bag,\
                                          dim_rois = dim_rois,allelt=True,num_classes=num_classes_old)
    writer = tf.python_io.TFRecordWriter(name_file_new)
    while True:
        try:
            heights,widths,num_regionss,num_featuress,dim1_roiss,roiss,roi_scoress,\
            fc7s,classes_vectorss,name_sans_exts = sess.run(next_element)
                
            for k in range(len(classes_vectorss)):
                height,width,num_regions,num_features,dim1_rois,rois,roi_scores,\
                fc7,classes_vectors,name_sans_ext = heights[k],widths[k],num_regionss[k],\
                num_featuress[k],dim1_roiss[k],roiss[k,:],roi_scoress[k],\
                fc7s[k,:],classes_vectorss[k,:],name_sans_exts[k]
                rois = rois[:,[0,2,1,3]] # Modification of the elements
                classes_vectors = np.zeros((num_classes,1))
                for j in range(num_classes):
                    value = df_label[df_label['item']==str(name_sans_ext.decode("utf-8"))][classes[j]]
                    value = int(value)
                    classes_vectors[j] = value
                
                feature={
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'num_regions': _int64_feature(num_regions),
                        'num_features': _int64_feature(num_features),
                        'dim1_rois': _int64_feature(dim1_rois),
                        'rois': _floats_feature(rois),
                        'roi_scores': _floats_feature(roi_scores),
                        'fc7': _floats_feature(fc7),
                        'label' : _floats_feature(classes_vectors),
                        'name_img' : _bytes_feature(name_sans_ext)} # str.encode(
                features=tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)    
                writer.write(example.SerializeToString())
        except tf.errors.OutOfRangeError:
            break
    writer.close()
    

        
def Test_GT_inProposals(database='IconArt_v1',k_per_bag = 300,metamodel = 'FasterRCNN',demonet='res152_COCO'):
    
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    databasetxt =path_data_csvfile + database + ext

    df_label = pd.read_csv(databasetxt,sep=",")
    
    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    
    imdb = get_imdb('IconArt_v1_test')

    
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    name_file = dict_name_file['test']
    if metamodel=='EdgeBoxes':
        dim_rois = 4
    else:
        dim_rois = 5
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,dim_rois = dim_rois)

    # Load the Faster RCNN proposals
    dict_rois = {}
    sess = tf.Session()
    sum_of_classes = []
    while True:
        try:
            fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
            for k in range(len(labels)):
                name_im = name_imgs[k].decode("utf-8")
                if name_im in list_im_withanno: 
                    complet_name = path_to_img + str(name_im) + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    roi = roiss[k,:]
                    if metamodel=='EdgeBoxes':
                        roi_boxes =  roi / im_scales[0] 
                    else:
                        roi_boxes =  roi[:,1:5] / im_scales[0] 
                    dict_rois[name_im] = roi_boxes
                    sum_of_classes += [np.sum(labels[k,:])]
        except tf.errors.OutOfRangeError:
            break

    sess.close()
    print('End read the boxes proposals')
    
#    # Read the GT boxes from imdb : DON T Work !
#    print('With read from imdb')
#    gt_roidb = imdb.gt_roidb()
#
#    list_gt_boxes_best_iou = []
#    list_gt_boxes_classes = []
#    gt_roidb_i = 0
#    for i in range(imdb.num_images):
#        if sum_of_classes[i] > 0.:
#            im_path = imdb.image_path_at(i)
#            name_im = im_path.split('/')[-1]
#            name_im = name_im.split('.')[0]
#            gt_roi = gt_roidb[gt_roidb_i]
#            gt_roidb_i += 1
#            gt_boxes = gt_roi['boxes']
#            gt_classes = gt_roi['gt_classes']
#            print(gt_classes)
#            proposals_boxes = dict_rois[name_im]
#            
#            for j in range(len(gt_boxes)):
#                best_iou = 0.
#                for k in range(len(proposals_boxes)):    
#                    IoU = bb_intersection_over_union(proposals_boxes[k],gt_boxes[j,:])
#                    assert(IoU>=0.)
#                    assert(IoU<=1.)
#                    if IoU > best_iou:
#                        best_iou = IoU
#                list_gt_boxes_best_iou += [best_iou]
#                list_gt_boxes_classes += [gt_classes[j]]
#      
#    np_gt_boxes_classes = np.array(list_gt_boxes_classes)
#    np_gt_boxes_best_iou = np.array(list_gt_boxes_best_iou)
        
    num_classes = 7

    list_gt_boxes_best_iou = []
    list_gt_boxes_classes = []
    all_boxes = [[[] for _ in range(imdb.num_images)] for _ in range(imdb.num_classes+1)]
    number_gt_boxes = 0
    for i in range(imdb.num_images):
        complet_name = imdb.image_path_at(i)
        complet_name_tab = ('.'.join(complet_name.split('.')[0:-1])).split('/')
        complet_name_tab[-2] = 'Annotations'
        complet_name_xml = '/'.join(complet_name_tab) + '.xml'
        read_file = voc_eval.parse_rec(complet_name_xml)
        im_path = imdb.image_path_at(i)
        name_im = im_path.split('/')[-1]
        name_im = name_im.split('.')[0]
        proposals_boxes = dict_rois[name_im]

        for element in read_file:
#                print(element)
            number_gt_boxes += 1
            classe_elt_xml = element['name']
            c = classes.index(classe_elt_xml)
            bbox = element['bbox']
            best_iou = 0.
#            best_boxes = np.hstack((bbox,[1.]))

            for k in range(len(proposals_boxes)):    
                #print(proposals_boxes[k])
                #print(bbox) # A retirer
                IoU = bb_intersection_over_union(proposals_boxes[k],bbox)
                #print(IoU)
                assert(IoU>=0.)
                assert(IoU<=1.)
                if IoU > best_iou:
                    best_iou = IoU
                    best_boxes = np.hstack((proposals_boxes[k],[1.]))
                    
#                        print(best_boxes)
            list_gt_boxes_best_iou += [best_iou]
            list_gt_boxes_classes += [c]
            all_boxes[c+1][i] += [best_boxes] 
      
    np_gt_boxes_classes = np.array(list_gt_boxes_classes)
    np_gt_boxes_best_iou = np.array(list_gt_boxes_best_iou)
    assert(len(list_gt_boxes_best_iou)==number_gt_boxes)
                
    for c in range(num_classes):
        # We will compute the statistics for each classes
        index_c = np.where(np_gt_boxes_classes==c)[0]
        np_gt_boxes_best_iou_c = np_gt_boxes_best_iou[index_c]
        number_of_regions = len(np_gt_boxes_best_iou_c)
        number_sup_05 = len(np.where(np_gt_boxes_best_iou_c>=0.5)[0])
        max_iou = np.max(np_gt_boxes_best_iou_c)
        mean_iou = np.mean(np_gt_boxes_best_iou_c)
        min_iou = np.min(np_gt_boxes_best_iou_c)
        std_iou = np.std(np_gt_boxes_best_iou_c)
        print('For class ',classes[c],', IoU max : {0:.2f} min : {1:.2f}, mean : {2:.2f}, std : {3:.2f}'.format(max_iou,min_iou,mean_iou,std_iou))
        print('Number of boxes with IoU with the GT superior to 0.5',number_sup_05,' for ',number_of_regions,' GT boxes, soit {0:.2f} %'.format(number_sup_05*100./number_of_regions))
    # Evalution of the best AP scores that we can obtain on this dataset with those boxes
    
    for i in range(imdb.num_images):
        for j in range(imdb.num_classes):
            all_boxes[j+1][i] = np.array(all_boxes[j+1][i])
    
    imdb.set_force_dont_use_07_metric(True)
    output_dir = path_data +'tmp/' + database+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes, output_dir)
    print("Detection score (thres = 0.5): ",database,'with k_per_bag =',k_per_bag)
    print(arrayToLatex(aps,per=True))
    
    ovthresh = 0.1
    aps = imdb.evaluate_localisation_ovthresh(all_boxes, output_dir,ovthresh)
    print("Detection score (thres = 0.1): ",database,'with k_per_bag =',k_per_bag)
    print(arrayToLatex(aps,per=True))
    
    tf.reset_default_graph()

def draw_random_bbbox(h,w):
    x = np.random.randint(0,h-60)
    y = np.random.randint(0,w-60)
    x2 = np.random.randint(x,h-30)
    y2 = np.random.randint(y,w-30)
    bbox = [x,y,x2,y2]
    return bbox
    
def RandomBoxes_withTrueGT(database='IconArt_v1',withGT=True):
    """
    This function will compute the performance with random boxes 
    @param : withGT : with the Ground truth instance
    """
    
    if(database=='IconArt_v1'):
#        ext='.csv'
#        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
#    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'s
#    databasetxt =path_data_csvfile + database + ext

#    df_label = pd.read_csv(databasetxt,sep=",")
    
#    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    
    imdb = get_imdb('IconArt_v1_test')
#    list_gt_boxes_classes = []
    all_boxes = [[[] for _ in range(imdb.num_images)] for _ in range(imdb.num_classes+1)]
    number_gt_boxes = 0
    for i in range(imdb.num_images):
        complet_name = imdb.image_path_at(i)
        complet_name_tab = ('.'.join(complet_name.split('.')[0:-1])).split('/')
        complet_name_tab[-2] = 'Annotations'
        complet_name_xml = '/'.join(complet_name_tab) + '.xml'
        read_file = voc_eval.parse_rec(complet_name_xml)
        im_path = imdb.image_path_at(i)
        im = cv2.imread(complet_name)
        h,w,c = im.shape
        blobs, im_scales = get_blobs(im)
        name_im = im_path.split('/')[-1]
        name_im = name_im.split('.')[0]

        if withGT:

            for element in read_file:
                # For each instance we will draw a random boxes
                number_gt_boxes += 1
                classe_elt_xml = element['name']
                c = classes.index(classe_elt_xml)
                bbox = draw_random_bbbox(h,w)
                all_boxes[c+1][i] += [bbox]
        else:
            number_of_class = np.random.poisson(2, 1)[0]
            cs = np.random.choice(6,number_of_class)
            for c in cs:
                bbox = draw_random_bbbox(h,w)
                all_boxes[c+1][i] += [bbox]
            
    for i in range(imdb.num_images):
        for j in range(imdb.num_classes):
            all_boxes[j+1][i] = np.array(all_boxes[j+1][i])
    
    imdb.set_force_dont_use_07_metric(True)
    output_dir = path_data +'tmp/' + database+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes, output_dir)
    print("Detection score with random boxes prediction but GT (thres = 0.5): ",database)
    print(arrayToLatex(aps,per=True))
    
    ovthresh = 0.1
    aps = imdb.evaluate_localisation_ovthresh(all_boxes, output_dir,ovthresh)
    print("Detection score with random boxes prediction but GT  (thres = 0.1): ",database)
    print(arrayToLatex(aps,per=True))

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

#    if hasattr(offsetbox, 'AnnotationBbox'):
#        # only print thumbnails with matplotlib > 1.0
#        shown_images = np.array([[1., 1.]])  # just something big
#        for i in range(X.shape[0]):
#            dist = np.sum((X[i] - shown_images) ** 2, 1)
#            if np.min(dist) < 4e-3:
#                # don't show points that are too close
#                continue
#            shown_images = np.r_[shown_images, [X[i]]]
#            imagebox = offsetbox.AnnotationBbox(
#                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                X[i])
#            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def prepareData_to_TSNE(IoUValid=True):
    """
    Goal to prepare data for a TSNE tensorboard
    We assign the label of the class if the IoU is superior to 0.5
    """
    database='IconArt_v1'
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
        path_to_xml = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/Annotations/'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    databasetxt =path_data_csvfile + database + ext

    df_label = pd.read_csv(databasetxt,sep=",")
    
    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    set = 'test'
    name_imdb = database + '_' + set
    imdb = get_imdb(name_imdb)

    k_per_bag = 300
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag)
    name_file = dict_name_file['test']
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag)

    # Load the Faster RCNN proposals
    dict_rois = {}
#    sess = tf.Session()
    sum_of_classes = []
    
    list_im_with_classes = []
    num_features = 2048
    if set=='test':
#        num_ex = 857
        num_ex = 1480
    else:
        num_ex = 2978
    num_classes = 7
    k_per_bag = 300
    k_per_im = 5
    X = np.empty(shape=(num_ex*k_per_im,num_features),dtype=np.float32)
    y = np.empty(shape=(num_ex*k_per_im,num_classes),dtype=np.float32)
    with tf.Session() as sess:
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                for k in range(len(labels)): # Loop on the images of the batch
                    name_im = name_imgs[k].decode("utf-8")
                    if name_im in list_im_withanno: 
                        labels_k = np.zeros((k_per_bag,num_classes),dtype=np.float32)
                        complet_name =path_to_img + name_im + '.jpg'
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        fc7 = fc7s[k,:].reshape((-1,num_features))
                        if IoUValid: # We will assign the label to the bbox with a IoU sup 0.5
                            list_index = list(np.arange(k_per_bag,dtype=np.int))
                            labels_k = np.zeros((k_per_bag,num_classes),dtype=np.float32)
                            roi = roiss[k,:]
                            roi_boxes =  roi[:,1:5] / im_scales[0]
                            complet_name_xml =path_to_xml + name_im + '.xml'
                            read_file = voc_eval.parse_rec(complet_name_xml)
                            list_kb_pos = []
                            for kb in range(k_per_bag): # Loop on the boxes of the image
                                for element in read_file:
                                    classe_elt_xml = element['name']
                                    c = classes.index(classe_elt_xml)
                                    bbox = element['bbox']
                                    IoU = bb_intersection_over_union(roi_boxes[kb,:],bbox)
                                    if IoU >= 0.5:
                                        labels_k[kb,c] = 1.
                                        list_kb_pos += [kb]
                                        try: 
                                            list_index.remove(kb)
                                        except ValueError:
                                            pass
                            list_kb_pos = np.unique(np.array(list_kb_pos,dtype=np.int))
                            if len(list_kb_pos) < k_per_im:   
                                other_index = np.random.choice(list_index,k_per_im-len(list_kb_pos),replace=False)
                                index_selected =  np.concatenate((list_kb_pos,other_index))
                            elif len(list_kb_pos) == k_per_im: 
                                index_selected = list_kb_pos
                            else:
                                index_selected = np.random.choice(list_kb_pos,k_per_im,replace=False)
                                
                        else:
                             labels_k = np.reshape(np.tile(labels[k,:],k_per_bag),(k_per_bag,num_classes))
                             index_selected = np.random.choice(300,(200,),replace=False)
                             
                             
                        X[k:k+k_per_im,:] = fc7[index_selected,:]
                        y[k:k+k_per_im,:] =  labels_k[index_selected,:]
            except tf.errors.OutOfRangeError:
                break
    PATH = os.getcwd()
    # Path to save the embedding and checkpoints generated
    LOG_DIR = PATH + '/data/'
    # Write the metadata 
    if IoUValid:
        nametsv =  PATH + '/data/metadata_tsneIoU05_'+str(k_per_im)+'.tsv'
    else:
        nametsv =  PATH + '/data/metadata_tsneLabelPerIm_'+str(k_per_im)+'.tsv'
    ypd = pd.DataFrame(y,columns=classes)
    ypd.to_csv(nametsv,sep='\t',index=False,header=classes)
    
    metadata = nametsv
    tf_data = tf.Variable(X)

    ## Running TensorFlow Session
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()
    
    # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name
    
        # Link this tensor to its metadata(Labels) file
        embedding.metadata_path = metadata
    
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        
        
def prepareData_to_Pickle(IoUValid=True,demonet='res152_COCO'):
    """
    Goal to prepare data for a study of the features vectors
    We assign the label of the class if the IoU is superior to 0.5
    @param demonet : aussi possible vgg16_COCO
    """
    database='IconArt_v1'
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
        path_to_xml = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/Annotations/'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    databasetxt =path_data_csvfile + database + ext

    df_label = pd.read_csv(databasetxt,sep=",")
    
    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    set = 'test'
    name_imdb = database + '_' + set
    imdb = get_imdb(name_imdb)

    k_per_bag = 300
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag,demonet=demonet)
    name_file = dict_name_file['test']
    print(name_file)
   

    # Load the Faster RCNN proposals
    dict_rois = {}
#    sess = tf.Session()
    sum_of_classes = []
    
    list_im_with_classes = []
    
    if  'vgg16' in demonet:
      num_features = 4096
    elif 'res101' in demonet:
      num_features = 2048
    elif 'res152' in demonet:
      num_features = 2048
    else:
      raise NotImplementedError
    
    if demonet=='res152_COCO':
        base = ''
    else:
        base=demonet
    
    if set=='test':
#        num_ex = 857
        num_ex = 1480
    else:
        num_ex = 2978
        
    num_classes = 7
    k_per_bag = 300
    k_per_im = 300   
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,num_features=num_features)
    
    X = np.empty(shape=(num_ex*k_per_im,num_features),dtype=np.float32)
    y = np.empty(shape=(num_ex*k_per_im,num_classes),dtype=np.float32)
    with tf.Session() as sess:
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                for k in range(len(labels)): # Loop on the images of the batch
                    name_im = name_imgs[k].decode("utf-8")
                    if name_im in list_im_withanno: 
                        labels_k = np.zeros((k_per_bag,num_classes),dtype=np.float32)
                        complet_name =path_to_img + name_im + '.jpg'
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        fc7 = fc7s[k,:].reshape((-1,num_features))
                        if IoUValid: # We will assign the label to the bbox with a IoU sup 0.5
                            list_index = list(np.arange(k_per_bag,dtype=np.int))
                            labels_k = np.zeros((k_per_bag,num_classes),dtype=np.float32)
                            roi = roiss[k,:]
                            roi_boxes =  roi[:,1:5] / im_scales[0]
                            complet_name_xml =path_to_xml + name_im + '.xml'
                            read_file = voc_eval.parse_rec(complet_name_xml)
                            list_kb_pos = []
                            for kb in range(k_per_bag): # Loop on the boxes of the image
                                for element in read_file:
                                    classe_elt_xml = element['name']
                                    c = classes.index(classe_elt_xml)
                                    bbox = element['bbox']
                                    IoU = bb_intersection_over_union(roi_boxes[kb,:],bbox)
                                    if IoU >= 0.5:
                                        labels_k[kb,c] = 1.
                                        list_kb_pos += [kb]
                                        try: 
                                            list_index.remove(kb)
                                        except ValueError:
                                            pass
                            list_kb_pos = np.unique(np.array(list_kb_pos,dtype=np.int))
                            if len(list_kb_pos) < k_per_im:   
                                other_index = np.random.choice(list_index,k_per_im-len(list_kb_pos),replace=False)
                                index_selected =  np.concatenate((list_kb_pos,other_index))
                            elif len(list_kb_pos) == k_per_im: 
                                index_selected = list_kb_pos
                            else:
                                index_selected = np.random.choice(list_kb_pos,k_per_im,replace=False)
                                
                        else:
                             labels_k = np.reshape(np.tile(labels[k,:],k_per_bag),(k_per_bag,num_classes))
                             index_selected = np.random.choice(300,(200,),replace=False)
                             
                             
                        X[k:k+k_per_im,:] = fc7[index_selected,:]
                        y[k:k+k_per_im,:] =  labels_k[index_selected,:]
            except tf.errors.OutOfRangeError:
                break
    PATH = os.getcwd()
    
    if IoUValid:
        namepkl =  PATH + '/data/'+base+'IconArt_v1_test_set_'+str(k_per_im)+'.pkl'
    else:
        namepkl =  PATH + '/data/'+base+'IconArt_v1_test_set_LabelPerIm_'+str(k_per_im)+'.pkl'
    
    data = [X,y,classes]
    
    with open(namepkl, 'wb') as pkl_file:
        pickle.dump(data,pkl_file, protocol=4)
    del data
    del X,y
        
    # Test :
    with open(namepkl,'rb') as rfp: 
        [X,y,classes] = pickle.load(rfp)
        print('X shape',X.shape)
        print('y shape',y.shape)
        print('classes :',classes)

