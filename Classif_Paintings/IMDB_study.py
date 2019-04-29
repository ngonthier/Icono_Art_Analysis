#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:12:22 2019

Le but de ce script est de verifier differentes questions que l on se pose sur 
notre dataset IconArt : 
    
1/ Est ce que les boites des GT sont contenues dans les boites proposees par 
Faster RCNN : Test_GT_inProposals

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

from LatexOuput import arrayToLatex

import voc_eval

from tf_faster_rcnn.lib.datasets.factory import get_imdb
from tf_faster_rcnn.lib.model.test import get_blobs

from TL_MIL import parser_w_rois_all_class
from FasterRCNN import vis_detections

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from matplotlib import offsetbox

from time import time

def getDictFeaturesFasterRCNN(database,k_per_bag = 300):
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    
    demonet = 'res152_COCO'
    metamodel = 'FasterRCNN'
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

def getTFRecordDataset(name_file,k_per_bag = 300):
    dim_rois = 5
    num_features = 2048
    get_roisScore = True
    
    num_classes = 7
    
    
    
    mini_batch_size = 256
    train_dataset = tf.data.TFRecordDataset(name_file)
    train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
        num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,\
        num_rois=k_per_bag,dim_rois=dim_rois))
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
        path_to_img = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
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
    base_plot = '/media/HDD/output_exp/ClassifPaintings/PlotBoxesIconArt_v1_Test'
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
        
def Test_GT_inProposals(database='IconArt_v1',k_per_bag = 300):
    
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    databasetxt =path_data_csvfile + database + ext

    df_label = pd.read_csv(databasetxt,sep=",")
    
    list_im_withanno = list(df_label[df_label['Anno']==1][item_name].values)
    # List of images with Bounding boxes GT annotations
    
    imdb = get_imdb('IconArt_v1_test')

    
    dict_name_file = getDictFeaturesFasterRCNN(database,k_per_bag=k_per_bag)
    name_file = dict_name_file['test']
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag)

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
#                    iuo = bb_intersection_over_union(proposals_boxes[k],gt_boxes[j,:])
#                    assert(iuo>=0.)
#                    assert(iuo<=1.)
#                    if iuo > best_iou:
#                        best_iou = iuo
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
                iuo = bb_intersection_over_union(proposals_boxes[k],bbox)
                assert(iuo>=0.)
                assert(iuo<=1.)
                if iuo > best_iou:
                    best_iou = iuo
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
        print('For class ',classes[c],', IuO max : {0:.2f} min : {1:.2f}, mean : {2:.2f}, std : {3:.2f}'.format(max_iou,min_iou,mean_iou,std_iou))
        print('Number of boxes with IuO with the GT superior to 0.5',number_sup_05,' for ',number_of_regions,' GT boxes, soit {0:.2f} %'.format(number_sup_05*100./number_of_regions))
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
    
def RandomBoxes_withTrueGT(database='IconArt_v1'):
    """
    This function will compute the performance with random boxes 
    """
    
    if(database=='IconArt_v1'):
#        ext='.csv'
#        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
#    path_data_csvfile = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'s
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

        for element in read_file:
            # For each instance we will draw a random boxes
            number_gt_boxes += 1
            classe_elt_xml = element['name']
            c = classes.index(classe_elt_xml)
#            bbox = element['bbox']
            x = np.random.randint(0,h-60)
            y = np.random.randint(0,w-60)
            x2 = np.random.randint(x,h-30)
            y2 = np.random.randint(y,w-30)
            bbox = [x,y,x2,y2]
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

def plotTSNE():

    database='IconArt_v1'
    if(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    path_data_csvfile = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
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
        num_ex = 857
    else:
        num_ex = 2978
    num_classes = 7
    k_per_bag = 300
    X = np.empty(shape=(num_ex*k_per_bag,num_features),dtype=np.float32)
    y = np.empty(shape=(num_ex*k_per_bag,num_classes),dtype=np.float32)
    indexX = 0
    with tf.Session() as sess:
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                for k in range(len(labels)):
                    name_im = name_imgs[k].decode("utf-8")
                    if name_im in list_im_withanno: 
                        sum_labels = np.sum(labels[k,:])
                        if sum_labels >0.:
                            X[indexX:(k_per_bag+indexX),:]  =  fc7s[k,:]
                            y[indexX:(k_per_bag+indexX),:]  = labels[k,:]
                            indexX += k_per_bag
    #                    complet_name = path_to_img + str(name_im) + '.jpg'
    #                    im = cv2.imread(complet_name)
    #                    blobs, im_scales = get_blobs(im)
    #                    roi = roiss[k,:]
    #                    roi_boxes =  roi[:,1:5] / im_scales[0] 
    #                    dict_rois[name_im] = roi_boxes
    #                    sum_labels = np.sum(labels[k,:])
    #                    sum_of_classes += [sum_labels]
    #                    if sum_labels >0.:
    #                        list_im_with_classes += [name_im]
            except tf.errors.OutOfRangeError:
                break
     
    ## TensorFlow Variable from data
    from tensorflow.contrib.tensorboard.plugins import projector
    import os
    metadata = y
    tf_data = tf.Variable(X)
    ## Get working directory
    PATH = os.getcwd()
    
    ## Path to save the embedding and checkpoints generated
    LOG_DIR = PATH + '/data/'
    print('LOG_DIR',LOG_DIR)
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
    
    # After run : tensorboard --logdir=/home/gonthier/Travail_Local/Icono_Art/Icono_Art_Analysis/Classif_Paintings/data/ --port=6006
        
#        
#    from tensorflow.contrib.tensorboard.plugins import projector
#    embedding_var = tf.Variable(X, name='embedding')
#    # Create summary writer.
#    writer = tf.summary.FileWriter('./graphs/embedding_test', sess.graph)
#    # Initialize embedding_var
#    sess.run(embedding_var.initializer)
#    # Create Projector config
#    config = projector.ProjectorConfig()
#    # Add embedding visualizer
#    embedding = config.embeddings.add()
#    # Attache the name 'embedding'
#    embedding.tensor_name = embedding_var.name
#    # Metafile which is described later
#    embedding.metadata_path = './100_vocab.csv'
#    # Add writer and config to Projector
#    projector.visualize_embeddings(writer, config)
#    # Save the model
#    saver_embed = tf.train.Saver([embedding_var])
#    saver_embed.save(sess, './graphs/embedding_test/embedding_test.ckpt', 1)
#
#    writer.close()
        
#    #----------------------------------------------------------------------
#    # t-SNE embedding of the digits dataset
#    print("Computing t-SNE embedding")
#    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#    t0 = time()
#    X_tsne = tsne.fit_transform(X)
#    
#    for i in range(num_classes):
#        y_i = y[:,i]
#        plot_embedding(X_tsne,y_i,
#                       "t-SNE embedding of the digits (time %.2fs)" %
#                       (time() - t0))
#        
#        plt.show()
