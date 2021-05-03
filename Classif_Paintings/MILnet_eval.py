#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:02:50 2019

The goal of this script is to evaluate the Revisiting MIL network on our case of 
WSOD

@author: gonthier
"""

from MINNpy3.mil_nets.WSOD_datasets import load_dataset
from MINNpy3.MI_Net import MI_Net_WSOD,MI_Max_AddOneLayer_Keras
from MINNpy3.MI_Net_with_DS import MI_Net_with_DS_WSOD
from MINNpy3.MI_Net_with_RC import MI_Net_with_RC_WSOD
from MINNpy3.mi_Net import mi_Net_WSOD

from IMDB_study import getDictFeaturesPrecomputed,getTFRecordDataset
from IMDB import get_database
import numpy as np
import cv2
from tf_faster_rcnn.lib.model.test import get_blobs
import tensorflow as tf
from tf_faster_rcnn.lib.datasets.factory import get_imdb
#from Transform_Box import py_cpu_modif
import pickle
from tf_faster_rcnn.lib.model.nms_wrapper import nms
import os 
from LatexOuput import arrayToLatex
from sklearn.metrics import average_precision_score
import time
import os.path

MILmodel_tab = ['MI_Net','mi_Net','MI_Net_with_DS','MI_Net_with_RC','MI_Max_AddOneLayer_Keras']

def mainEval(dataset_nm='IconArt_v1',classe=0,k_per_bag = 300,metamodel = 'FasterRCNN',\
             demonet='res152_COCO',test=False,MILmodel='MI_Net',max_epoch=20,verbose=True):
    
#    dataset_nm='IconArt_v1'
#    classe=1
#    k_per_bag = 300
#    metamodel = 'FasterRCNN'
#    demonet='res152_COCO'
#    test=True
#    MILmodel='MI_Net_with_DS'
#    max_epoch = 1
    
    t0 = time.time()
                        
    if test:
        classe = 0
    
    if MILmodel=='MI_Net':
        MILmodel_fct = MI_Net_WSOD
    elif MILmodel=='MI_Max_AddOneLayer_Keras':
        MILmodel_fct = MI_Max_AddOneLayer_Keras
    elif MILmodel=='mi_Net':
        MILmodel_fct = mi_Net_WSOD
    elif MILmodel=='MI_Net_with_DS':
        MILmodel_fct = MI_Net_with_DS_WSOD
    elif MILmodel=='MI_Net_with_RC':
        MILmodel_fct = MI_Net_with_RC_WSOD
    else:
        print(MILmodel,'is unkwon')
        return(0)
    print('MILmodel',MILmodel,max_epoch)    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(dataset_nm)
    
    dataset,bags_full_label,mean_fea,std_fea = load_dataset(dataset_nm,classe=0,k_per_bag = k_per_bag,metamodel = metamodel,demonet=demonet)
    model_dict = {}
    
    for j in range(num_classes):
        if test and not(j==classe): 
            continue
        else:
            for k in range(len(dataset['train'])):
                a = list(dataset['train'][k])
                a[1] = [bags_full_label[k,j]]*k_per_bag
                a = tuple(a)
                dataset['train'][k] = a
                
        print('start training for class',j)
        model = MILmodel_fct(dataset,max_epoch=max_epoch,verbose=verbose)
        model_dict[j] = model
    
    t1 = time.time()
    print("--- Training duration :",str(t1-t0),' s')
    
    dict_name_file = getDictFeaturesPrecomputed(dataset_nm,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    
    name_file = dict_name_file['test']
    if metamodel=='EdgeBoxes':
        dim_rois = 4
    else:
        dim_rois = 5
    next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,dim_rois = dim_rois,
                                      num_classes = num_classes)
    
    dont_use_07_metric = False
    if dataset_nm=='VOC2007':
        imdb = get_imdb('voc_2007_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index)
    elif dataset_nm=='watercolor':
        imdb = get_imdb('watercolor_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index)
    elif dataset_nm=='PeopleArt':
        imdb = get_imdb('PeopleArt_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index)
    elif dataset_nm=='clipart':
        imdb = get_imdb('clipart_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index) 
    elif dataset_nm=='comic':
        imdb = get_imdb('comic_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index) 
    elif dataset_nm=='CASPApaintings':
        imdb = get_imdb('CASPApaintings_test',data_path=default_path_imdb)
        num_images = len(imdb.image_index) 
    elif dataset_nm=='IconArt_v1' or dataset_nm=='RMN':
        imdb = get_imdb('IconArt_v1_test',data_path=default_path_imdb)
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif 'IconArt_v1' in dataset_nm and not('IconArt_v1' ==dataset_nm):
        imdb = get_imdb('IconArt_v1_test',ext=dataset_nm.split('_')[-1],data_path=default_path_imdb)
#        num_images = len(imdb.image_index) 
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif dataset_nm in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        imdb = get_imdb('WikiTenLabels_test',data_path=default_path_imdb)
        #num_images = len(imdb.image_index) 
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    elif 'OIV5' in dataset_nm: # For OIV5 for instance !
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    else:
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    imdb.set_force_dont_use_07_metric(dont_use_07_metric)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    
    
    TEST_NMS = 0.3
    thresh = 0.0
    true_label_all_test = []
    predict_label_all_test = []
    name_all_test = []
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 16
    config.inter_op_parallelism_threads = 16
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    i = 0
    num_features = 2048
    while True:
        try:
            fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
            fc7s = np.divide(fc7s-mean_fea, std_fea).astype(np.float32)
            true_label_all_test += [labels]
            score_all = None
            for j in range(num_classes):
                if not(test):
                    model= model_dict[j]
                    predictions = model.predict(fc7s.reshape((-1,num_features)),batch_size=1)
                    if MILmodel=='MI_Net_with_DS':
                        predictions = predictions[3]
                    scores_all_j_k = predictions.reshape((fc7s.shape[0],1,fc7s.shape[1]))
                else:
                    if j==classe:
                        model= model_dict[j]
                        predictions = model.predict(fc7s.reshape((-1,num_features)),batch_size=1)
                        if MILmodel=='MI_Net_with_DS':
                            predictions = predictions[3]
                        scores_all_j_k = predictions.reshape((fc7s.shape[0],1,fc7s.shape[1]))
                if score_all is None:
                    score_all = scores_all_j_k
                else:
                    score_all = np.concatenate((score_all,scores_all_j_k),axis=1)
            predict_label_all_test +=  [np.max(score_all,axis=2)]
            
            for k in range(len(labels)):
                name_im = name_imgs[k].decode("utf-8")
                complet_name = path_to_img + str(name_im) + '.jpg'
                im = cv2.imread(complet_name)
                blobs, im_scales = get_blobs(im)
                roi = roiss[k,:]
                if metamodel=='EdgeBoxes':
                    roi_boxes =  roi / im_scales[0] 
                else:
                    roi_boxes =  roi[:,1:5] / im_scales[0]
                
                for j in range(num_classes):
                    scores = score_all[k,j,:]
                    #print(j,'scores',scores.shape)
                    inds = np.where(scores > thresh)[0]
                    cls_scores = scores[inds]
                    cls_boxes = roi_boxes[inds,:]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(cls_dets, TEST_NMS)
                    cls_dets = cls_dets[keep, :]
                    all_boxes[j][i] = cls_dets
                i += 1
            for l in range(len(name_imgs)): 
                if dataset_nm in ['IconArt_v1','VOC2007','watercolor','clipart',\
                                  'comic','CASPApaintings','WikiTenLabels','PeopleArt',\
                                  'MiniTrain_WikiTenLabels','WikiLabels1000training']:
                    name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                else:
                    name_all_test += [[name_imgs[l]]]
                    
        except tf.errors.OutOfRangeError:
            break

    sess.close()
    
    
    
    
    true_label_all_test = np.concatenate(true_label_all_test)
    predict_label_all_test = np.concatenate(predict_label_all_test,axis=0)
    name_all_test = np.concatenate(name_all_test)
    
    AP_per_class = []
    for j,classe in enumerate(classes):
            AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
            AP_per_class += [AP]
    print('Average Precision classification task :')
    print(arrayToLatex(AP_per_class,per=True))        
            
    max_per_image = 100
    num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels or IconArt_v1 case
    all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
    number_im = 0
    name_all_test = name_all_test.astype(str)
    for i in range(num_images_detect):
#        print(i)
        name_img = imdb.image_path_at(i)
        if dataset_nm=='PeopleArt':
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
#        print(name_img_ind[0])
        for j in range(1, imdb.num_classes):
            j_minus_1 = j-1
            if len(all_boxes[j_minus_1][name_img_ind[0]]) >0:
                all_boxes_order[j][i]  = all_boxes[j_minus_1][name_img_ind[0]]
        if max_per_image > 0 and len(all_boxes_order[j][i]) >0: 
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
    output_dir = path_data +'tmp/' + dataset_nm+'_mAP.txt'
    aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
    apsAt05 = aps
    print("Detection score (thres = 0.5): ",dataset_nm)
    print(arrayToLatex(aps,per=True))
    ovthresh_tab = [0.3,0.1,0.]
    for ovthresh in ovthresh_tab:
        aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
        if ovthresh == 0.1:
            apsAt01 = aps
        print("Detection score with thres at ",ovthresh,'with ',MILmodel)
        print(arrayToLatex(aps,per=True))
    
    t2 = time.time()
    print("--- Testing duration :",str(t2-t1),' s')
    
    return(apsAt05,apsAt01,AP_per_class)

def runSeveralMInet(dataset_nm='IconArt_v1',MILmodel='MI_Net',demonet = 'res152_COCO',\
                        k_per_bag=300,layer='fc7',num_rep = 10,metamodel = 'FasterRCNN',
                        printR=False,pm_only_on_mean=False,ReDo=False):
    """
    @param : printR if True, we print the results instead of compute them
    """
    verbose = False
    max_epoch=20    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    path_data_output = path_data +'VarStudy/'
    name_dict = path_data_output 
    if not(demonet== 'res152_COCO'):
        name_dict += demonet +'_'
    if not(layer== 'fc7'):
        name_dict += '_'+demonet
    name_dict +=  dataset_nm
    if not(num_rep==100):
        name_dict += '_numRep'+ str(num_rep)
    name_dict += '_'+MILmodel +'_'+str(max_epoch)
    name_dictAP = name_dict + '_APscore.pkl'
    multi = 100
    
    if not(printR):
        # Compute the performance
    
        if not(os.path.isfile(name_dictAP)) or ReDo:
            DictAP = {}
            ll = []
            l01 = []
            lclassif = []
            for r in range(num_rep):
                print('Reboot ',r,'on ',num_rep)
                apsAt05,apsAt01,AP_per_class = mainEval(dataset_nm=dataset_nm,\
                                     k_per_bag = k_per_bag,metamodel =metamodel,\
                                     demonet=demonet,test=False,\
                                     MILmodel=MILmodel,max_epoch=max_epoch,verbose=verbose)
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
        # Print the results
        onlyAP05 = False
        try:
            f= open(name_dictAP, 'rb')
            print(name_dictAP)
            DictAP = pickle.load(f)
            for Metric in DictAP.keys():
                string_to_print =  str(Metric) + ' & ' +MILmodel + ' '
                
                string_to_print += ' & '
                ll_all = DictAP[Metric] 
                if dataset_nm=='WikiTenLabels':
                    ll_all = np.delete(ll_all, [1,2,9], axis=1)         
                if not(dataset_nm=='PeopleArt'):
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
                    if not(pm_only_on_mean):
                        for mean_c,std_c in zip(mean_over_class,std_over_class):
                            s =  "{0:.1f} ".format(mean_c*multi) + ' $\pm$ ' +  "{0:.1f}".format(std_c*multi)
                            string_to_print += s + ' & '
                    else:
                        for mean_c,std_c in zip(mean_over_class,std_over_class):
                            s =  "{0:.1f} ".format(mean_c*multi)
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
            

if __name__ == '__main__':
#    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
#
#    for MILmodel in MILmodel_tab:
#        mainEval(MILmodel=MILmodel,max_epoch=20,test=False)
    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    for dataset_nm in database_tab:
        for MILmodel in MILmodel_tab:
            print(dataset_nm,MILmodel)
            runSeveralMInet(dataset_nm=dataset_nm,MILmodel=MILmodel,demonet = 'res152_COCO',\
                        k_per_bag=300,layer='fc7',num_rep = 10,metamodel = 'FasterRCNN',
                        printR=True,pm_only_on_mean=True,ReDo=False)
