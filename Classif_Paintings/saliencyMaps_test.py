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
import pathlib
import matplotlib.pyplot as plt
import matplotlib

from StatsConstr_ClassifwithTL import learn_and_eval
from keras_resnet_utils import getBNlayersResNet50
from IMDB import get_database
from preprocess_crop import load_and_crop_img,load_and_crop_img_forImageGenerator
from Stats_Fcts import load_resize_and_process_img
from saliencyMaps import GetSmoothedMask,GetMask_IntegratedGradients,\
    GetMask_RandomBaseline_IntegratedGradients,GetMask_IntegratedGradients_noisyImage,\
    SmoothedMask,IntegratedGradient
from tf_faster_rcnn.lib.model.nms_wrapper import nms
from LatexOuput import arrayToLatex

from IMDB_study import getDictFeaturesPrecomputed,getTFRecordDataset,getDictBoxesProposals,\
    get_imdb_and_listImagesInTestSet
import voc_eval

from tf_faster_rcnn.lib.datasets.factory import get_imdb
from tf_faster_rcnn.lib.model.test import get_blobs

from TL_MIL import parser_w_rois_all_class,parser_all_elt_all_class,parser_minimal_elt_all_class
from FasterRCNN import vis_detections,vis_detections_list

from scipy import ndimage
import matplotlib.cm as cm

def MorphologicalCleanup(attributions, structure=np.ones((4,4))):
  closed = ndimage.grey_closing(attributions, structure=structure)
  opened = ndimage.grey_opening(closed, structure=structure)
  
  return opened

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def to_01(image):
    return(np.nan_to_num((image - np.min(image))/ (np.max(image)  - np.min(image))))

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
        assert(nsamples>0)
        saliencyMap = SmoothedMask(model,c_i,stdev_spread=stdev_spread,\
                                   nsamples=nsamples,magnitude=False)

    elif method=='IntegratedGrad':
        assert(x_steps>0)
        saliencyMap =  IntegratedGradient(model,c_i,x_steps=x_steps,x_baseline=None)
        
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
    
    @param : SaliencyMethod : IntegratedGrad ou SmoothGrad pour le moment
    """
    matplotlib.use('Agg') 
    save_data = False
    
    ReDo = True
    plot = False    
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
        
    for_data_output = os.path.join(path_data,'dataSaliencyMap',SaliencyMethod)
    im_with_boxes_output = os.path.join(path_data,'SaliencyMapImagesBoxes',SaliencyMethod)
    print('===',im_with_boxes_output)
    pathlib.Path(for_data_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(im_with_boxes_output).mkdir(parents=True, exist_ok=True)
    
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
    stdev_spread = 0.1
    nsamples = 50
    x_steps = 50
    
    for j in range(num_classes):
        SaliencyMapClass=getSaliencyMapClass(model,c_i=j,method=SaliencyMethod,\
                                         stdev_spread=stdev_spread,nsamples=nsamples,x_steps=x_steps)    
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
        hauteur, largeur ,_ = im.shape
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
        
        #print(np.max(image_array),np.min(image_array),np.mean(image_array),np.median(image_array))
        #input('wait')
        
        dict_sensitivity = {}
        dict_sensitivity_path = os.path.join(for_data_output,name_im+'_dict_SaliencyMap'+SaliencyMethod+'_std'+str(stdev_spread)+'_n'+str(nsamples)+'_steps'+str(x_steps)+'.pkl')
        if not(os.path.exists(dict_sensitivity_path)) or ReDo:
            predictions = model.predict(image_array)[0]
            dict_sensitivity['predictions'] = predictions
            inds = np.where(predictions > thresh_classif)[0]
            for ind in inds:
                prediction = predictions[ind]
                if np.isnan(prediction):
                    print('Prediction of ',name_im,'is nan !!!')
                    input('wait')
                candidate_boxes = []
                j = ind +1  # the class index for the evaluation part
                Smap=SaliencyMapClass_tab[ind].GetMask(image_array)
                #print('before normalisation',np.max(Smap),np.min(Smap),np.mean(Smap),np.median(Smap))
                if save_data: dict_sensitivity[j] = Smap
                
                if SaliencyMethod=='SmoothGrad':
                    #Smap_grey = np.mean(Smap,axis=-1,keepdims=True)
                    Smap_grey = np.mean(np.abs(Smap),axis=-1,keepdims=True)
                    #print('after grey',np.max(Smap_grey),np.min(Smap_grey),np.mean(Smap_grey),np.median(Smap_grey))
                    if norm:
                        Smap_grey = to_01(Smap_grey)
                    #print('after normalisation',np.max(Smap_grey),np.min(Smap_grey),np.mean(Smap_grey),np.median(Smap_grey))
                    
                    Smap_grey_time_score = prediction*Smap_grey
                    
                else: # In the case of Integrated Gradient
                    
                    # Sur conseil d Antoine Pirovano
                    ptile= 99
                    # Sum for grayscale of the absolute value
                    pixel_attrs = np.sum(np.abs(Smap), axis=-1,keepdims=True)
                    pixel_attrs = np.clip(pixel_attrs / np.percentile(pixel_attrs, ptile), 0, 1)
                    
                    Smap_grey_time_score = prediction * pixel_attrs
                    
                #print('after mul score',np.max(Smap_grey_time_score),np.min(Smap_grey_time_score),np.mean(Smap_grey_time_score),np.median(Smap_grey_time_score))
                # attention truc super contre intuitif dans le resize c'est hauteur largeur alors que 
                # la fonction size retourne largeur hauteur
                Smap_grey_time_score = Smap_grey_time_score[0]
                #Smap_grey_time_score_resized =  cv2.resize(Smap_grey_time_score, (hauteur, largeur),interpolation=cv2.INTER_NEAREST) 
                Smap_grey_time_score_resized =  cv2.resize(Smap_grey_time_score, (largeur,hauteur),interpolation=cv2.INTER_NEAREST) 
                #print('Smap_grey_time_score_resized',Smap_grey_time_score_resized.shape,im.shape)
                #print('after resize',np.max(Smap_grey_time_score_resized),np.min(Smap_grey_time_score_resized),np.mean(Smap_grey_time_score_resized),np.median(Smap_grey_time_score_resized))
                
                if plot:
                    name_output = name_im+'_'+SaliencyMethod+'_std'+str(stdev_spread)+'_n'+str(nsamples)+'_steps'+str(x_steps)+ '_'+str(j)+'.jpg'
                    name_output_path = os.path.join(im_with_boxes_output,name_output)
                    Smap_grey_time_score_resized_01 = to_01(Smap_grey_time_score_resized)
                    plt.imshow(Smap_grey_time_score_resized_01, cmap=cm.gray)
                    plt.title(classes[j-1]+' : '+str(prediction))
                    plt.savefig(name_output_path)
                    plt.close()
                
                for k in range(len(proposals_boxes)):
                    box = proposals_boxes[k]
                    x1,y1,x2,y2 = box # x : largeur et y en hauteur
                    x1_int = int(np.round(x1))
                    x2_int = int(np.round(x2))
                    y1_int = int(np.round(y1))
                    y2_int = int(np.round(y2))
                    #print(name_im,'Smap_grey_time_score_resized',Smap_grey_time_score_resized.shape,im.shape)
                    #print(x1_int,x2_int,y1_int,y2_int)
                    assert(x2_int<=largeur)
                    assert(y2_int<=hauteur)
                    Smap_grey_time_score_resized_crop = Smap_grey_time_score_resized[y1_int:y2_int,x1_int:x2_int]
                    
                    # because bbox = dets[i, :4] # Boxes are score, x1,y1,x2,y2
                    Smap_grey_time_score_resized_crop_score = np.mean(Smap_grey_time_score_resized_crop)
                    # if k < 3:
                    #     print('Smap_grey_time_score_resized_crop',Smap_grey_time_score_resized_crop.shape)
                    #     print(x1_int,x2_int,y1_int,y2_int)
                    #     print('j',j,'k',k,',score',Smap_grey_time_score_resized_crop_score)
                    if not(np.isnan(Smap_grey_time_score_resized_crop_score)):
                        box_with_scores = np.append(box,[Smap_grey_time_score_resized_crop_score])
                        candidate_boxes += [box_with_scores]
                    else:
                        box_with_scores = np.append(box,[0.0])
                        candidate_boxes += [box_with_scores]
                        
                    # if np.isnan(Smap_grey_time_score_resized_crop_score):
                    #     print('!!! score is nan')
                    #     print(x1,y1,x2,y2)
                    #     print(x1_int,x2_int,y1_int,y2_int)
                    #     print(Smap_grey_time_score_resized_crop.shape)
                    #     print(name_im,'Smap_grey_time_score_resized',Smap_grey_time_score_resized.shape,im.shape)
                    #     print(prediction)
                    #     print('after resize',np.max(Smap_grey_time_score_resized),np.min(Smap_grey_time_score_resized),np.mean(Smap_grey_time_score_resized),np.median(Smap_grey_time_score_resized))
                    #     print(Smap_grey_time_score_resized_crop_score)
                    #     input('wait')
                    
                #print(candidate_boxes)
                if len(candidate_boxes)>0:
                    candidate_boxes_NP = np.array(candidate_boxes)
                    
                    candidate_boxes_NP[:,-1] = candidate_boxes_NP[:,-1] -np.max(candidate_boxes_NP[:,-1]) + prediction 
                    keep = nms(candidate_boxes_NP, TEST_NMS)
                    cls_dets = candidate_boxes_NP[keep, :]
                    all_boxes_order[j][i]  = cls_dets
                
            if plot:
                roi_boxes_and_score = []
                local_cls = []
                for j in range(num_classes):
                    cls_dets = all_boxes_order[j+1][i] 
                    if len(cls_dets) > 0:
                        local_cls += [classes[j]]
                        roi_boxes_score = cls_dets
                        if roi_boxes_and_score is None:
                            roi_boxes_and_score = [roi_boxes_score]
                        else:
                            roi_boxes_and_score += [roi_boxes_score] 
                if roi_boxes_and_score is None: roi_boxes_and_score = [[]]
                #print(name_im,roi_boxes_and_score,local_cls)
                vis_detections_list(im, local_cls, roi_boxes_and_score, thresh=-np.inf)
                name_output = name_im+'_'+SaliencyMethod+'_std'+str(stdev_spread)+'_n'+str(nsamples)+'_steps'+str(x_steps)+ '_Regions.jpg'
                name_output_path = os.path.join(im_with_boxes_output,name_output)
                #input("wait")
                plt.savefig(name_output_path)
                plt.close()
            
            if save_data:
                with open(dict_sensitivity_path, 'wb') as f:
                    pickle.dump(dict_sensitivity, f, pickle.HIGHEST_PROTOCOL)
            
    # for i in range(imdb.num_images):     
    #     candidate_boxes[i] = np.array(candidate_boxes[i])
    
    imdb.set_force_dont_use_07_metric(True)
    det_file = os.path.join(path_data, 'detectionsSaliencyMap'+SaliencyMethod+'_std'+str(stdev_spread)+'_n'+str(nsamples)+'_steps'+str(x_steps)+'.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
    output_dir = path_data +'tmp/' + database+'_'+SaliencyMethod+'_std'+str(stdev_spread)+'_n'+str(nsamples)+'_steps'+str(x_steps)+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes_order, output_dir) # AP at O.5 
    print("===> Detection score (thres = 0.5): ",database,'with Saliency map from',SaliencyMethod,'with std =',stdev_spread,'nsamples = ',nsamples,'x_steps =',x_steps)
    print(arrayToLatex(aps,per=True))
    ovthresh_tab = [0.3,0.1,0.]
    for ovthresh in ovthresh_tab:
        aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
        print("Detection score with thres at ",ovthresh)
        print(arrayToLatex(aps,per=True))
        
if __name__ == '__main__':  
    eval_MAP_SaliencyMethods(SaliencyMethod='IntegratedGrad')
    #eval_MAP_SaliencyMethods(SaliencyMethod='SmoothGrad')
    