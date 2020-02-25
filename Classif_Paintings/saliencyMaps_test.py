#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:30:54 2020

@author: gonthier
"""

import os
import numpy as np
from matplotlib import pylab as P
import tensorflow as tf
import cv2
import pickle
import time

from StatsConstr_ClassifwithTL import learn_and_eval
from keras_resnet_utils import getBNlayersResNet50
from IMDB import get_database
from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator
from Stats_Fcts import load_resize_and_process_img
from saliencyMaps import GetSmoothedMask,GetMask_IntegratedGradients,\
    GetMask_RandomBaseline_IntegratedGradients,GetMask_IntegratedGradients_noisyImage,\
    SmoothedMask
from tf_faster_rcnn.lib.model.nms_wrapper import nms
from LatexOuput import arrayToLatex

from IMDB_study import getDictFeaturesPrecomputed,getTFRecordDataset,getDictBoxesProposals,\
    get_imdb_and_listImagesInTestSet
import voc_eval

from tf_faster_rcnn.lib.datasets.factory import get_imdb
from tf_faster_rcnn.lib.model.test import get_blobs

from TL_MIL import parser_w_rois_all_class,parser_all_elt_all_class,parser_minimal_elt_all_class
from FasterRCNN import vis_detections

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def to_01(image):
    return((image - np.min(image))/ (np.max(image)  - np.min(image) ))

def saliencyMap_ImageSize():
    
    target_dataset = 'IconArt_v1'
    style_layers = getBNlayersResNet50()
    features = 'activation_48'
    normalisation = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    transformOnFinalLayer='GlobalAveragePooling2D'

    final_clf = 'MLP2'
    epochs = 20
    optimizer = 'SGD'
    return_best_model = True
    batch_size= 16
    dropout=None
    regulOnNewLayer=None
    nesterov=False
    SGDmomentum=0.9
    decay=1e-4
    cropCenter = False
    # Load ResNet50 normalisation statistics
    
    opt_option = [0.1,0.01]
    pretrainingModif = True
    kind_method = 'FT'
    computeGlobalVariance = False
    constrNet = 'ResNet50'
    #list_bn_layers = getBNlayersResNet50()

    Model_dict = {}
    list_markers = ['o','s','X','*','v','^','<','>','d','1','2','3','4','8','h','H','p','d','$f$','P']
    alpha = 0.7
    sizeIm = 224
    Net = constrNet
    
    
    print('loading :',constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option)         
    model = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers=style_layers,
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                           batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                           returnStatistics=True,cropCenter=cropCenter,\
                           computeGlobalVariance=computeGlobalVariance,\
                           epochs=epochs,optimizer=optimizer,opt_option=opt_option,
                           return_best_model=return_best_model,\
                           batch_size=batch_size,gridSearch=False,verbose=True)
    # Performance : & 54.4 & 76.3 & 60.7 & 82.1 & 74.3 & 70.6 & 11.0 & 61.3 \\ 
    
        
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(target_dataset)
        
    images_in_set = df_label[df_label['set']=='test'][item_name].values
    
    images_in_set = [images_in_set[0]]
    for image in images_in_set:
        #image = images_in_set[20]
        image_path = os.path.join(path_to_img,image+'.jpg')           
        if cropCenter:
            image_array= load_and_crop_img(path=image_path,Net=Net,target_size=sizeIm,
                                    crop_size=sizeIm,interpolation='lanczos:center')
            # For VGG or ResNet with classification head size == 224
        else:
            image_array = load_resize_and_process_img(image_path,Net=Net,max_dim=sizeIm)
        
        predictions = model.predict(image_array)
        df_label[df_label[item_name]==image][classes].values
        c_i = 0
        saliencyMap = SmoothedMask(model,c_i,stdev_spread=.15,\
                                               nsamples=25,magnitude=False)
        smooth_grad_of_image = saliencyMap.GetMask(image_array)
        smooth_grad_of_image_scaled = to_01(smooth_grad_of_image)
        ShowGrayscaleImage(smooth_grad_of_image_scaled[0,:,:,:])
        
        integrated_grad_of_image = GetMask_IntegratedGradients(image_array,model,c_i,
                                               x_steps=50)
        integrated_grad_of_image_scaled = to_01(integrated_grad_of_image) 
        ShowGrayscaleImage(integrated_grad_of_image_scaled[0,:,:,:])
        
        # Dans ce cas là on a un gradient selon les 3 canaux couleurs
        integrated_grad_randBaseline_of_image = GetMask_RandomBaseline_IntegratedGradients(image_array,model,c_i,
                                               x_steps=50,num_random_trials=10)
        integrated_grad_randBaseline_of_image_scaled = to_01(integrated_grad_randBaseline_of_image) 
        ShowGrayscaleImage(integrated_grad_randBaseline_of_image_scaled[0,:,:,:])
        
        integrated_grad_noisy_image = GetMask_IntegratedGradients_noisyImage(image_array,model,c_i,
                                               x_steps=50,num_random_trials=10,stdev_spread=.15)
        integrated_grad_noisy_image_scaled = to_01(integrated_grad_noisy_image) 
        ShowGrayscaleImage(integrated_grad_noisy_image_scaled[0,:,:,:])
        
def getSaliencyMap(image_array,model,c_i,method,norm=True,stdev_spread=.15,nsamples=20,x_steps=50,\
                   ):
    
    if method=='SmoothGrad':
        saliencyMap = GetSmoothedMask(image_array,model,c_i,stdev_spread=stdev_spread,\
                                               nsamples=nsamples,magnitude=False)

    elif method=='IntegratedGrad':
        saliencyMap = GetMask_IntegratedGradients(image_array,model,c_i,
                                               x_steps=x_steps)
        
    elif method=='IntegratedGrad_RandBaseline':
        saliencyMap = GetMask_RandomBaseline_IntegratedGradients(image_array,model,c_i,
                                               x_steps=x_steps,num_random_trials=nsamples)
        
    elif method=='IntegratedGrad_NoisyInput':
        saliencyMap = GetMask_IntegratedGradients_noisyImage(image_array,model,c_i,
                                               x_steps=x_steps,num_random_trials=nsamples,stdev_spread=stdev_spread)
        
    else:
        print(method,'is unknown !')
        raise(NotImplementedError)
    if norm:
        saliencyMap = to_01(saliencyMap)
    return(saliencyMap)

def getSaliencyMapClass(model,c_i,method,stdev_spread=.15,nsamples=20,x_steps=50,\
                   ):
    
    if method=='SmoothGrad':
        saliencyMap = SmoothedMask(model,c_i,stdev_spread=stdev_spread,\
                                   nsamples=nsamples,magnitude=False)

    elif method=='IntegratedGrad':
        raise(NotImplementedError)
        # saliencyMap = GetMask_IntegratedGradients(image_array,model,c_i,
        #                                        x_steps=x_steps)
        
    elif method=='IntegratedGrad_RandBaseline':
        raise(NotImplementedError)
        # saliencyMap = GetMask_RandomBaseline_IntegratedGradients(image_array,model,c_i,
        #                                        x_steps=x_steps,num_random_trials=nsamples)
        
    elif method=='IntegratedGrad_NoisyInput':
        raise(NotImplementedError)
        # saliencyMap = GetMask_IntegratedGradients_noisyImage(image_array,model,c_i,
        #                                        x_steps=x_steps,num_random_trials=nsamples,stdev_spread=stdev_spread)
        
    else:
        print(method,'is unknown !')
        raise(NotImplementedError)
    
    return(saliencyMap)
        
database='IconArt_v1'
metamodel='FasterRCNN'
demonet='res152_COCO'
k_per_bag=300
SaliencyMethod='SmoothGrad'   
def eval_MAP_SaliencyMethods(database='IconArt_v1',metamodel='FasterRCNN',demonet='res152_COCO',
                      k_per_bag=300,SaliencyMethod='SmoothGrad'):
    """
    The goal of this function is to compute the mAP of the saliency method for 
    classification ResNet 
    """
    TEST_NMS = 0.01
    thresh_classif = 0.1
    
    # Parameter for the classification network 
    target_dataset = 'IconArt_v1'
    style_layers = []
    features = 'activation_48'
    normalisation = False
    final_clf= 'LinearSVC' # Don t matter
    source_dataset=  'ImageNet'
    transformOnFinalLayer='GlobalAveragePooling2D'

    final_clf = 'MLP2'
    epochs = 20
    optimizer = 'SGD'
    return_best_model = True
    batch_size= 16
    dropout=None
    regulOnNewLayer=None
    nesterov=False
    SGDmomentum=0.9
    decay=1e-4
    cropCenter = False
    # Load ResNet50 normalisation statistics
    
    opt_option = [0.1,0.01]
    pretrainingModif = True
    kind_method = 'FT'
    computeGlobalVariance = False
    constrNet = 'ResNet50'

    sizeIm = 224
    Net = constrNet

    # Load the box proosals
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC = get_database(database)
    imdb,list_im_withanno = get_imdb_and_listImagesInTestSet(database)
    num_images_detect = len(list_im_withanno)
    dict_rois = getDictBoxesProposals(database=target_dataset,k_per_bag=k_per_bag,\
                                      metamodel=metamodel,demonet=demonet)
    
    # Load Classification model 
    print('loading :',constrNet,computeGlobalVariance,kind_method,pretrainingModif,opt_option)         
    model = learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers=style_layers,
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,
                           batch_size_RF=16,epochs_RF=20,momentum=0.9,ReDo=False,
                           returnStatistics=True,cropCenter=cropCenter,\
                           computeGlobalVariance=computeGlobalVariance,\
                           epochs=epochs,optimizer=optimizer,opt_option=opt_option,
                           return_best_model=return_best_model,\
                           batch_size=batch_size,gridSearch=False,verbose=True)
        
    SaliencyMapClass_tab = []
    for j in range(num_classes):
        SaliencyMapClass=getSaliencyMapClass(model,c_i=j,method=SaliencyMethod,\
                                         stdev_spread=.15,nsamples=20,x_steps=50)    
        SaliencyMapClass_tab +=[SaliencyMapClass]
        
#    list_gt_boxes_classes = []
    candidate_boxes = [[] for _ in range(imdb.num_images)]
    all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
#    number_gt_boxes = 0
    itera = 20
    norm = True
    t0 = time.time()
    # Un peu plus de 1440 images
    for i in range(imdb.num_images):
#        complet_name_tab = ('.'.join(complet_name.split('.')[0:-1])).split('/')
        im_path = imdb.image_path_at(i)
        name_im = im_path.split('/')[-1]
        if i%itera==0:
            t1 = time.time()
            print(i,name_im,'duration for ',itera,'iterations = ',str(t1-t0),'s')
            t0 = time.time()
        im = cv2.imread(im_path)
        largeur, hauteur,_ = im.shape
        blobs, im_scales = get_blobs(im)
        
        if database=='PeopleArt':
            name_im =  '/'.join(im_path.split('/')[-2:])
        if database=='PeopleArt':
            name_im= '.'.join(name_im.split('.')[0:-1])
        else:
            name_im = name_im.split('.')[0]
        proposals_boxes = dict_rois[name_im]
        
        if cropCenter:
            image_array= load_and_crop_img(path=im_path,Net=Net,target_size=sizeIm,
                                    crop_size=sizeIm,interpolation='lanczos:center')
            # For VGG or ResNet with classification head size == 224
        else:
            image_array = load_resize_and_process_img(im_path,Net=Net,max_dim=sizeIm)
        
        predictions = model.predict(image_array)[0]
        inds = np.where(predictions > thresh_classif)[0]
        for ind in inds:
            candidate_boxes = []
            j = ind +1  # the class index for the evaluation part
            Smap=SaliencyMapClass_tab[ind].GetMask(image_array)
            if norm:
                Smap = to_01(Smap)
            Smap_grey = np.mean(Smap,axis=-1)
            prediction = predictions[ind]
            Smap_grey_time_score = prediction*Smap_grey
            # attention truc super contre intuitif dans le resize c'est hauteur largeur alors que 
            # la focntion size retourne largeur hauteur
            Smap_grey_time_score_resized =  cv2.resize(Smap_grey_time_score, (hauteur, largeur),interpolation=cv2.INTER_LINEAR) 

            for k in range(len(proposals_boxes)):
                box = proposals_boxes[k]
                x1,y1,x2,y2 = box
                x1_int = int(np.round(x1))
                x2_int = int(np.round(x2))
                y1_int = int(np.round(y1))
                y2_int = int(np.round(y2))
                Smap_grey_time_score_resized_crop = Smap_grey_time_score_resized[x1_int:x2_int,y1_int:y2_int,:]
                # because bbox = dets[i, :4] # Boxes are score, x1,y1,x2,y2
                Smap_grey_time_score_resized_crop_score = np.mean(Smap_grey_time_score_resized_crop)
                box_with_scores = np.append(box,[Smap_grey_time_score_resized_crop_score])
                
                candidate_boxes += [box_with_scores]
            candidate_boxes_NP = np.array(candidate_boxes)
            keep = nms(candidate_boxes_NP, TEST_NMS)
            cls_dets = candidate_boxes_NP[keep, :]
            all_boxes_order[j][i]  = cls_dets
            
    # for i in range(imdb.num_images):     
    #     candidate_boxes[i] = np.array(candidate_boxes[i])
    
    imdb.set_force_dont_use_07_metric(True)
    det_file = os.path.join(path_data, 'detectionsSaliencyMap.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
    output_dir = path_data +'tmp/' + database+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes_order, output_dir) # AP at O.5 
    print("===> Detection score (thres = 0.5): ",database,'with Saliency map from',SaliencyMethod)
    print(arrayToLatex(aps,per=True))
    ovthresh_tab = [0.3,0.1,0.]
    for ovthresh in ovthresh_tab:
        aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
        print("Detection score with thres at ",ovthresh)
        print(arrayToLatex(aps,per=True))
        
if __name__ == '__main__':  
    eval_MAP_SaliencyMethods()
    