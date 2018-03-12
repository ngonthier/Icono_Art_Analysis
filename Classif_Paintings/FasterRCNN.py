#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:42:41 2017

Based on the Tensorflow implementation of Faster-RCNNN 
https://github.com/endernewton/tf-faster-rcnn

Be careful it was a necessity to modify all the script of the library with stuff 
like ..lib etc
It is a convertion for Python 3

Faster RCNN re-scale  the  images  such  that  their  shorter  side  = 600 pixels  

@author: gonthier

You can find the weight here : https://partage.mines-telecom.fr/index.php/s/ep52PPAxSI932zY
You will have to modify the static path to the weights/models in each function : Sorry :( 
TODO : change that


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
from trouver_classes_parmi_K import MILSVM

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

def run_FasterRCNN_Perf_Paintings(TL = True,reDo=False):
    """
    Compute the performance on the Your Paintings subset ie Crowley on the output but also the best case on feature fc7 of the best proposal part
    TL : use the features maps of the best object score detection
    reDO : recompute the features maps
    """
    
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt = path +database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    sLength = len(df_test['name_img'])
    sLength_all = len(df_label['name_img'])
    name_img = df_test['name_img'][0]
    i = 0
    y_test = np.zeros((sLength,10))
    NETS_Pretrained = {'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt'
                   }
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}

    for demonet in NETS_Pretrained.keys():
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
          
        if not(TL):  
            net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales)
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
            
            scores_all_image = np.zeros((len(df_test),nbClasses))
            
            for i,name_img in  enumerate(df_test['name_img']):
                if i%1000==0:
                    print(i,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                scores, boxes = im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
                scores_max = np.max(scores,axis=0)
                scores_all_image[i,:] = scores_max
                for j in range(10):
                    if(classes_paitings[j] in list(df_test['classe'][df_test['name_img']==name_img])[0]):
                        y_test[i,j] = 1
                
            AP_per_class = []
            for k,classe in enumerate(classes_paitings):
                index_classe = np.where(np.array(CLASSES)==classe)[0][0]
                scores_per_class = scores_all_image[:,index_classe]
                #print(scores_per_class)
                #print(y_test[:,k],np.sum(y_test[:,k]))
                AP = average_precision_score(y_test[:,k],scores_per_class,average=None)
                AP_per_class += [AP]
                print("Average Precision for",classe," = ",AP)
            print(demonet," mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))
            
            sess.close()
        else:
            if database=='Paintings':
                item_name = 'name_img'
                path_to_img = '/media/HDD/data/Painting_Dataset/'
                classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
            path_data = path
            N = 1
            extL2 = ''
            nms_thresh = 0.0
            name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'.pkl'
            name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
            if not(os.path.isfile(name_pkl)) or reDo:
                if demonet == 'vgg16_COCO':
                    size_output = 4096
                elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                    size_output = 2048
                features_resnet = np.ones((sLength_all,size_output))
                classes_vectors = np.zeros((sLength_all,10))
                # Use the output of fc7 
                net.create_architecture("TEST", nbClasses,
                                      tag='default', anchor_scales=anchor_scales,
                                      modeTL= True,nms_thresh=nms_thresh) # default nms_thresh = 0.7
                saver = tf.train.Saver()
                saver.restore(sess, tfmodel)
                
                scores_all_image = np.zeros((len(df_test),nbClasses))
                features_resnet_dict= {}
                pkl = open(name_pkl_all_features, 'wb')
                
                for i,name_img in  enumerate(df_label[item_name]):
                    if i%1000==0:
                        print(i,name_img)
                        if not(i==0):
                            pickle.dump(features_resnet_dict,pkl)
                            features_resnet_dict= {}
                    complet_name = path_to_img + name_img + '.jpg'
                    im = cv2.imread(complet_name)
                    cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
                    features_resnet_dict[name_img] = fc7
                    
                    # Normally argmax roi_scores is 0
                    out = fc7[np.argmax(roi_scores),:]
                    features_resnet[i,:] = np.array(out)
                    if database=='VOC12' or database=='Paintings':
                        for j in range(10):
                            if( classes[j] in df_label['classe'][i]):
                                classes_vectors[i,j] = 1
                    
                X_train = features_resnet[df_label['set']=='train',:]
                y_train = classes_vectors[df_label['set']=='train',:]
                
                X_test= features_resnet[df_label['set']=='test',:]
                y_test = classes_vectors[df_label['set']=='test',:]
                
                X_val = features_resnet[df_label['set']=='validation',:]
                y_val = classes_vectors[df_label['set']=='validation',:]
                
                print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
                
                Data = [X_train,y_train,X_test,y_test,X_val,y_val]
                
                pickle.dump(features_resnet_dict,pkl)
                pkl.close()
                
                with open(name_pkl, 'wb') as pkl:
                    pickle.dump(Data,pkl)
                
                sess.close()
            
            # Compute the metric
            print("CV_Crowley = True") 
            Classification_evaluation(kind=demonet,kindnetwork='FasterRCNN',
                                      database='Paintings',L2=False,augmentation=False,
                                      classifier_name='LinearSVM',CV_Crowley=True)
            print("CV_Crowley = False") 
            Classification_evaluation(kind=demonet,kindnetwork='FasterRCNN',
                                      database='Paintings',L2=False,augmentation=False,
                                      classifier_name='LinearSVM',CV_Crowley=False)
 
def read_features_computePerfPaintings():
    """ Function to test if you can refind the same AP metric by reading the saved CNN features """
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
    name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
    features_resnet_dict = {}
    sLength_all = len(df_label['name_img'])
    if demonet == 'vgg16_COCO':
        size_output = 4096
    elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
        size_output = 2048
    features_resnet = np.ones((sLength_all,size_output))
    classes_vectors = np.zeros((sLength_all,10))
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
    print(len(features_resnet_dict))
    
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0 and not(i==0):
            print(i,name_img)
        fc7 = features_resnet_dict[name_img]
        out = fc7[0,:]
        features_resnet[i,:] = np.array(out)
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if(classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
    
    restarts = 0
    max_iters = 300
    #from trouver_classes_parmi_K import MILSVM
    X_train = features_resnet[df_label['set']=='train',:]
    y_train = classes_vectors[df_label['set']=='train',:]
    X_test= features_resnet[df_label['set']=='test',:]
    y_test = classes_vectors[df_label['set']=='test',:]
    X_val = features_resnet[df_label['set']=='validation',:]
    y_val = classes_vectors[df_label['set']=='validation',:]
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
    for j,classe in enumerate(classes):
        
#        y_test = label_classe[df_label['set']=='test',:]
#        X_test= features_resnet[df_label['set']=='test',:]   
        neg_ex = np.expand_dims(X_trainval[y_trainval[:,j]==0,:],axis=1)
        print(neg_ex.shape)
        pos_ex =  np.expand_dims(X_trainval[y_trainval[:,j]==1,:],axis=1)
        print(pos_ex.shape)
        
        classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                      max_iters=max_iters,symway=True,
                                      all_notpos_inNeg=False,gridSearch=True,
                                      verbose=True)     
        classifier = classifierMILSVM.fit(pos_ex, neg_ex)
        
        decision_function_output = classifier.decision_function(X_test)
        y_predict_confidence_score_classifier  = decision_function_output
        AP = average_precision_score(y_test[:,j],y_predict_confidence_score_classifier,average=None)
        print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
           
    
    #del(features_resnet_dict) 
    X_train = features_resnet[df_label['set']=='train',:]
    y_train = classes_vectors[df_label['set']=='train',:]
    
    X_test= features_resnet[df_label['set']=='test',:]
    y_test = classes_vectors[df_label['set']=='test',:]
    
    X_val = features_resnet[df_label['set']=='validation',:]
    y_val = classes_vectors[df_label['set']=='validation',:]
   
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)

    classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []
    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1,2]))
    param_grid = dict(C=cs)
    custom_cv = PredefinedSplit(np.hstack((-np.ones((1,X_train.shape[0])),np.zeros((1,X_val.shape[0])))).reshape(-1,1)) # For example, when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
    for i,classe in enumerate(classes):
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True),
                                param_grid=param_grid,n_jobs=-1,cv=custom_cv)
        grid.fit(X_trainval,y_trainval[:,i])  
        y_predict_confidence_score = grid.decision_function(X_test)
        y_predict_test = grid.predict(X_test) 
        AP = average_precision_score(y_test[:,i],y_predict_confidence_score,average=None)
        AP_per_class += [AP]
        print("Average Precision for",classe," = ",AP)
        test_precision = precision_score(y_test[:,i],y_predict_test)
        test_recall = recall_score(y_test[:,i],y_predict_test)
        print("Test precision = {0:.2f}, recall = {1:.2f}".format(test_precision,test_recall))
    print("mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))             
       
def run_FasterRCNN_demo():
    
    for demonet in NETS_Pretrained.keys():
        #demonet = 'res101_COCO'
        tf.reset_default_graph() # Needed to use different nets one after the other
        print(demonet)
        if 'VOC'in demonet:
            CLASSES = CLASSES_SET['VOC']
            anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
        elif 'COCO'in demonet:
            CLASSES = CLASSES_SET['COCO']
            anchor_scales = [4, 8, 16, 32]
        nbClasses = len(CLASSES)
        path_to_model = '/media/HDD/models/tf-faster-rcnn/'
        tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
        
        #tfmodel = os.path.join(path_to_model,DATASETS[dataset][0],NETS[demonet][0])
        print(tfmodel)
    #    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                              NETS[demonet][0])
        
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
                              tag='default', anchor_scales=anchor_scales)
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
    
        print('Loaded network {:s}'.format(tfmodel))
    
        im_names = ['loulou.jpg', 'cat.jpg', 'dog.jpg']
        DATA_DIR = '/media/HDD/data/Images/'
        #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
        #            '001763.jpg', '004545.jpg']
        for im_name in im_names:
            print('Demo for data/demo/{}'.format(im_name))
            imfile = os.path.join(DATA_DIR, im_name)
            im = cv2.imread(imfile)
            scores, boxes = im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            # Only single-image batch implemented !
            print(scores.shape)
            #print(scores)
    
            CONF_THRESH = 0.8
            NMS_THRESH = 0.3 # non max suppression
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if(len(inds)>0):
                    print(CLASSES[cls_ind])
        sess.close()
 
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
def vis_detections_list(im, class_name_list, dets_list, thresh=0.5):
    """Draw detected bounding boxes."""

    list_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8',	'#f58231','#911eb4','#46f0f0','#f032e6',	
                   '#d2f53c','#fabebe',	'#008080','#e6beff','#aa6e28','#fffac8','#800000',
                   '#aaffc3','#808000','#ffd8b1','#000080','#808080','#FFFFFF','#000000']	
    i_color = 0
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for class_name,dets in zip(class_name_list,dets_list):
        print(class_name,dets)
        inds = np.where(dets[:, -1] >= thresh)[0]
        if not(len(inds) == 0):
            color = list_colors[i_color]
            i_color += 1 % len(list_colors)
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
        
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=3.5)
                    )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
           
def FasterRCNN_bigImage():
    DATA_DIR =  '/media/HDD/data/Art Paintings from Web/'
    demonet = 'res152_COCO'
    tf.reset_default_graph() # Needed to use different nets one after the other
    print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32]
    nbClasses = len(CLASSES)
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    
    #tfmodel = os.path.join(path_to_model,DATASETS[dataset][0],NETS[demonet][0])
    print(tfmodel)
#    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
#                              NETS[demonet][0])
    
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
                          tag='default', anchor_scales=anchor_scales)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    im_name = 'L Adoration des mages - Jan Mabuse - 1515.jpg'
    print('Demo for data/demo/{}'.format(im_name))
    imfile = os.path.join(DATA_DIR, im_name)
    im = cv2.imread(imfile)
    scores, boxes = im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
   # Only single-image batch implemented !
    print('scores.shape',scores.shape)
    #print(scores)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3 # non max suppression
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if(len(inds)>0):
            print('CLASSES[cls_ind]',CLASSES[cls_ind])
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.show()
    sess.close()
    
def FasterRCNN_TransferLearning_outlier():
    """
    Compute the performance on the Your Paintings subset ie Crowley
    on the fc7 output but with an outlier detection version 
    """
    reDo = False
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    database = 'Paintings'
    databasetxt = database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    sLength = len(df_test['name_img'])
    sLength_all = len(df_label['name_img'])
    name_img = df_test['name_img'][0]
    i = 0
    y_test = np.zeros((sLength,10))
    NETS_Pretrained = {'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt'
                   }
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}

    for demonet in NETS_Pretrained.keys():
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
          
        if database=='Paintings':
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/Painting_Dataset/'
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        path_data = 'data/'
        N = 1
        extL2 = ''
        
        name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'.pkl'
        name_pkl = path_data + 'testTL.pkl'
        
        if not(os.path.isfile(name_pkl)) or reDo:
            print('Start computing image region proposal')
            if demonet == 'vgg16_COCO':
                size_output = 4096
            elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                size_output = 2048
            features_resnet_dict= {}
            features_resnet = np.ones((sLength_all,size_output))
            classes_vectors = np.zeros((sLength_all,10))
            # Use the output of fc7 
            net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,modeTL= True)
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
            
            scores_all_image = np.zeros((len(df_test),nbClasses))
            
    
            for i,name_img in  enumerate(df_label[item_name]):
                if i%1000==0:
                    print(i,name_img)
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
                features_resnet_dict[name_img] = fc7
    #            out = fc7[np.argmax(roi_scores),:]
    #            features_resnet[i,:] = np.array(out)
    #            if database=='VOC12' or database=='Paintings':
    #                for j in range(10):
    #                    if( classes[j] in df_label['classe'][i]):
    #                        classes_vectors[i,j] = 1
                 # We work on the class 0
#                if( classes[j] in df_label['classe'][i]):
#                    classes_vectors[i,j] = 1
        
            with open(name_pkl, 'wb') as pkl:
                pickle.dump(features_resnet_dict,pkl)
        
        print("Load data")
        features_resnet_dict = pickle.load(open(name_pkl, 'rb'))
        
        print("Learning of the normal distribution")
        j=0   
        normalDistrib =  [] # Each time you change the size of the array, it needs to be resized and every element needs to be copied. This is happening here too. 
        ElementsInClassTrain = []
        
        # Maximum size possible 
        
        normalDistrib = np.zeros((144508,2048))
        ElementsInClassTrain = np.zeros((3340, 2048))
        ElementsInClassTrainFirstElement = np.zeros((87, 2048))
        normalDistrib_i = 0
        ElementsInClassTrain_i = 0
        i = 0
        for index,row in df_label.iterrows():
            name_img = row[item_name]
            inClass = classes[j] in row['classe']
            inTest = row['set']=='test'
            if index%1000==0:
                print(index,name_img)
            if not(inTest):
                f = features_resnet_dict[name_img]
                if not(inClass):
                    normalDistrib[normalDistrib_i:normalDistrib_i+len(f),:] = f
                    normalDistrib_i += len(f)
                else:
                    ElementsInClassTrain[ElementsInClassTrain_i:ElementsInClassTrain_i+len(f),:] = f
                    ElementsInClassTrain_i += len(f)
                    ElementsInClassTrainFirstElement[i:i+1] = f[0,:]
                    i += 1
                    
        # I do not have anything to add that has not been said here. I just want to post a link the sklearn page about SVC which clarifies what is going on:
        # The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
        # Kernelized SVMs require the computation of a distance function between each point in the dataset, which is the dominating cost of O(nfeatures×n2observations).          
          
        # Outlier detection 
#        numberOfExemple = 1508
#        subsetNegativeExemple =  np.random.choice(len(normalDistrib),numberOfExemple)
#        subsetnormalDistrib = normalDistrib[subsetNegativeExemple,:]
#        #clf = LocalOutlierFactor(n_jobs=-1) # BCP + rapide que les autres méthodes
#        clf = svm.OneClassSVM(kernel="linear") # With numberOfExemple = 1500 we get AP for aeroplane  =  0.0265245509492
#        #clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=-1)
#        #clf = EllipticEnvelope()
#        print('Shape of normalDistrib :',normalDistrib.shape)
#        print('Shape of subsetnormalDistrib :',subsetnormalDistrib.shape)
#        print('Shape of ElementsInClassTrain :',ElementsInClassTrain.shape)
##        Shape of normalDistrib : (144508, 2048)
##        Shape of ElementsInClassTrain : (https://partage.mines-telecom.fr/index.php/s/ep52PPAxSI932zY3340, 2048)
#        clf.fit(subsetnormalDistrib)
#        print('End of training anomaly detector')
        
        # Detection of the outlier in the positive exemple images        
        
        # Classification version 
        numberOfExemple = 144508 #144508
#        subsetNegativeExemple =  np.random.choice(len(normalDistrib),numberOfExemple)
#        subsetnormalDistrib = normalDistrib[subsetNegativeExemple,:]
        subsetnormalDistrib = normalDistrib
        normalDistrib_class = np.zeros((numberOfExemple,1))
        ElementsInClassTrainFirstElement_class = np.ones((87,1))
        y_training = np.vstack((normalDistrib_class,ElementsInClassTrainFirstElement_class)).ravel()
        X_training = np.vstack((subsetnormalDistrib,ElementsInClassTrainFirstElement))
        classifier = svm.LinearSVC() # class_weight={1: 100000} doesn't improve at all
        classifier.fit(X_training,y_training)
        print("End training in a SVM one class versus one class manner")
        
        print("Test on image")
        
        numberOfTestEx = np.sum(df_label['set']=='test')
        y_predict_confidence_score = np.zeros((numberOfTestEx,1))
        y_predict_confidence_score_classifier= np.zeros((numberOfTestEx,1))
        y_test = np.zeros((numberOfTestEx,1)).ravel()
        numberImageTested = 0
        numberOfPositiveExemples = 0
        i = 0
        for index,row in df_label.iterrows():
            name_img = row[item_name]
            inClass = classes[j] in row['classe']
            inTest = row['set']=='test'
            if index%1000==0:
                print(index,name_img)
            if inTest:
#                y_pred_train_outlier = clf.decision_function(features_resnet_dict[name_img])
#                max_outlier_value = np.max(y_pred_train_outlier)
#                y_predict_confidence_score[i] = max_outlier_value
                
                #SVM version Average Precision for aeroplane  =  0.661635821571
#                y_pred_train = classifier.decision_function(features_resnet_dict[name_img]) 
#                max_value = np.max(y_pred_train)
                 # SVM version Average Precision for aeroplane  =  0.602695774327
#                data = features_resnet_dict[name_img][0,:].reshape(1, -1)
#                max_value = classifier.decision_function(data) 
                
                data = features_resnet_dict[name_img] # SVM version Average Precision for aeroplane  =  0.602695774327
                max_value = np.max(classifier.decision_function(data))
                
                y_predict_confidence_score_classifier[i] = max_value
                numberImageTested += 1
                if inClass:
                    numberOfPositiveExemples += 1
                    y_test[i] = 1
                i += 1
#        print(y_predict_confidence_score,y_test)
#        print(numberImageTested,"Images tested",numberOfPositiveExemples,"possible examples")   
#        print("Compute metric")
#        AP_outlier = average_precision_score(y_test,y_predict_confidence_score,average=None)
#        #AP_per_class += [AP]
#        print("Outlier version Average Precision for",classes[j]," = ",AP_outlier)
        AP_svm = average_precision_score(y_test,y_predict_confidence_score_classifier,average=None)
        print("SVM version Average Precision for",classes[j]," = ",AP_svm)  
               
        sess.close()
    
    
def FasterRCNN_TransferLearning_misvm():
    """
    Compute the performance on the Your Paintings subset ie Crowley
    on the fc7 output but with an Multi Instance SVM classifier for classifier the
    bag 
    """
    reDo = False
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    sLength = len(df_test['name_img'])
    name_img = df_test['name_img'][0]
    i = 0
    y_test = np.zeros((sLength,10))
    NETS_Pretrained = {'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt'
                   }
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}

    for demonet in NETS_Pretrained.keys():
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
          
        if database=='Paintings':
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/Painting_Dataset/'
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        path_data = path
        N = 1
        extL2 = ''
        
        nms_thresh = 0.5
        
        name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
        #name_pkl = path_data + 'testTL_withNMSthresholdProposal03.pkl'
        
        if not(os.path.isfile(name_pkl)) or reDo:
            print('Start computing image region proposal')
            if demonet == 'vgg16_COCO':
                size_output = 4096
            elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                size_output = 2048
            features_resnet_dict= {}
            # Use the output of fc7 
            # Parameter important 
            
            net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,
                                  modeTL= True,nms_thresh=nms_thresh)
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
            numberOfRegion = 0
            for i,name_img in  enumerate(df_label[item_name]):
                if i%1000==0:
                    print(i,name_img)
#                    with open(name_pkl, 'wb') as pkl:
#                        pickle.dump(features_resnet_dict,pkl)
#                    features_resnet_dict= {}
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
                #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
                features_resnet_dict[name_img] = fc7
                numberOfRegion += len(fc7)
                
            print("We have ",numberOfRegion,"regions proposol")
            # Avec un threshold a 0.1 dans le NMS de RPN on a 712523 regions
            
            sess.close()
            with open(name_pkl, 'wb') as pkl:
                pickle.dump(features_resnet_dict,pkl)
        
        print("Load data")
        features_resnet_dict = pickle.load(open(name_pkl, 'rb'))
        return(0)
        print("preparing data fro learning")
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        testMode = True
        jtest = 0
        for j,classe in enumerate(classes):
            if testMode and not(j==jtest):
                continue
            list_training_ex = []
            list_training_label = []
            list_test_ex = []
            y_test = []
            for index,row in df_label.iterrows():
                name_img = row[item_name]
                inClass = classes[j] in row['classe']
                inTest = row['set']=='test'
                f = features_resnet_dict[name_img]
                if index%1000==0:
                    print(index,name_img)
                if not(inTest):
                    #list_training_ex += [f]
                    if not(inClass):
                       list_training_ex += [f[0:5,:]]
                       list_training_label += [-1] # Label must be -1 or  1 
                    else:
                        list_training_ex += [f]
                        list_training_label += [1]
                else:
                    list_test_ex += [f]
                    if not(inClass):
                       y_test += [-1]
                    else:
                        y_test += [1]
                        
            print("Learning of the Multiple Instance Learning SVM")
            #classifier = misvm.SIL(kernel='linear', C=1.0) #SIL
#            cs = np.logspace(-5, -1, 5)
#            cs = np.hstack((cs,[0.2,1.,2.,10.]))
#            param_grid = dict(C=cs)
            # Construct classifiers
            classifiers = {}
            #classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0,scale_C=False)
            #classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)

            classifiers['MISVM'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=10,verbose=False,restarts=0)
#            from sklearn.svm import SVC
#            classifiermisvm = SVC(kernel='linear', max_iter=-1) 
#            classifiers['miSVM'] = misvm.miSVM(kernel='linear', C=1.0, max_iters=10)
            #classifiers['MISVM'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=2,verbose=False)
            #classifiermisvm = SklearnClassifier(misvm.MISVM(kernel='linear', C=1.0, max_iters=10))
#            classifiers['grid'] = GridSearchCV(classifiermisvm, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=-1)
            
            #classifier = misvm.miSVM(kernel='linear', C=1.0, max_iters=5)
            #classifier = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
            print("len list_training_ex",len(list_training_ex))
            APlist = {}
            for algorithm, classifier in classifiers.items():
                if (len(classifiers.items())> 1):
                    print(algorithm)
                classifier.fit(list_training_ex, list_training_label)
                y_predict_confidence_score_classifier = classifier.predict(list_test_ex) # Return value between -1 and 1 : score
                labels_test_predited = np.sign(y_predict_confidence_score_classifier) # numpy.sign(labels) to get -1/+1 class predictions
                y_predict_confidence_score_classifier = (y_predict_confidence_score_classifier + 1.)/2.
                print("number of test exemples",len(y_test),len(labels_test_predited))
                #print(y_test,labels_test_predited)
                test_precision = precision_score(y_test,labels_test_predited)
                test_recall = recall_score(y_test,labels_test_predited)
                F1 = f1_score(y_test,labels_test_predited)
                print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}".format(test_precision,test_recall,F1))
                AP = average_precision_score(y_test,y_predict_confidence_score_classifier,average=None)
                print("SVM version Average Precision for",classes[j]," = ",AP)
                precision_at_k = ranking_precision_score(np.array(y_test), y_predict_confidence_score_classifier,20)
                P20_per_class += [precision_at_k]
                AP_per_class += [AP]
                R_per_class += [test_recall]
                P_per_class += [test_precision]
                APlist[algorithm] = AP
            # For aeroplan we have with res152_COCO Average Precision for aeroplane  =  0.68
            # and Test precision = 0.97, recall = 0.55
            # Avec [f[0:5,:]] et MISVM C=1.0 on a AP de aeroplane de 0.71 
        print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
        print("mean Precision for all the data = {0:.3f}".format(np.mean(P_per_class)))  
        print("mean Recall for all the data = {0:.3f}".format(np.mean(R_per_class)))  
        print("mean Precision @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    
        print(AP_per_class)
        
def Compute_Faster_RCNN_features(demonet='res152_COCO',nms_thresh = 0.7,database='Paintings',
                                 augmentation=False,L2 =False,
                                 saved='all',verbose=True):
    """
    @param : demonet : teh kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : nms_thresh : the nms threshold on the Region Proposal Network
    
    """
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
    else:
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
    databasetxt = path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    if augmentation:
        raise NotImplementedError
        N = 50
    else: 
        N=1
    if L2:
        raise NotImplementedError
        extL2 = '_L2'
    else:
        extL2 = ''
    if saved=='all':
        savedstr = '_all'
    elif saved=='fc7':
        savedstr = ''
    elif saved=='pool5':
        savedstr = '_pool5'
    
    df_label = pd.read_csv(databasetxt,sep=",")
    
    tf.reset_default_graph() # Needed to use different nets one after the other
    if verbose: print(demonet)
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
      
    name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'
    +str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'.pkl'
    #name_pkl = path_data + 'testTL_withNMSthresholdProposal03.pkl'
    
    net.create_architecture("TEST", nbClasses,
                          tag='default', anchor_scales=anchor_scales,
                          modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    features_resnet_dict= {}
    pkl = open(name_pkl_all_features, 'wb')
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0:
            if verbose : print(i,name_img)
            if not(i==0):
                pickle.dump(features_resnet_dict,pkl) # Save the data
                features_resnet_dict= {}
        complet_name = path_to_img + name_img + '.jpg'
        im = cv2.imread(complet_name)
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
        #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
        if saved=='fc7':
            features_resnet_dict[name_img] = fc7
        elif saved=='pool5':
            features_resnet_dict[name_img] = pool5
        elif saved=='all':
            features_resnet_dict[name_img] = rois,roi_scores,fc7

    pickle.dump(features_resnet_dict,pkl)
    pkl.close()
        
def Illus_NMS_threshold_test():
    """
    The goal of this function is to test the modification of the NMS threshold 
    on the output provide by the algo 
    And plot the zone considered as the best by the Faster RCNN 
    """ 
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}
    path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/'
    demonet = 'res152_COCO'
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
      
    # List des images a test 
    path_to_img = '/media/HDD/output_exp/ClassifPaintings/Test_nms_threshold/'
    path_to_imgARTUK =  '/media/HDD/data/Painting_Dataset/'
    list_name_img = ['dog','acc_acc_ac_5289_624x544','not_ncmg_1941_23_624x544',
                     'Albertinelli Franciabigio Vièrge et saints','1979.18 01 p01',
                     'abd_aag_003796_624x544',
                     'Accademia - The Mystic Marriage of St. Catherine by Veronese'] # First come from Your Paintings and second from Wikidata
    list_dog = ['ny_yag_yorag_326_624x544', 'dur_dbm_770_624x544', 'ntii_skh_1196043_624x544', 'nti_ldk_884912_624x544', 'syo_bha_90009742_624x544', 'tate_tate_t00888_10_624x544', 'ntii_lyp_500458_624x544', 'ny_yag_yorag_37_b_624x544', 'ngs_ngs_ng_1193_f_624x544', 'dur_dbm_533_624x544']
    list_name_img += list_dog
    list_nms_thresh = [0.0,0.1,0.7]
    nms_thresh = list_nms_thresh[0]
    # First we test with a high threshold !!!
    plt.ion()
    for nms_thresh in list_nms_thresh:
        sess = tf.Session(config=tfconfig)
        print("nms_thresh",nms_thresh)
        net.create_architecture("TEST", nbClasses,
                                      tag='default', anchor_scales=anchor_scales,
                                      modeTL= True,nms_thresh=nms_thresh)
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        name_img = list_name_img[0]
        i=0
        for i,name_img in  enumerate(list_name_img):
            print(i,name_img)
            if name_img in list_dog:
                complet_name = path_to_imgARTUK + name_img + '.jpg'
            else:
                complet_name = path_to_img + name_img + '.jpg'
            im = cv2.imread(complet_name)
            #print("Image shape",im.shape)
            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im)  # This call net.TL_image 
            best_RPN_score_ROI = np.argmax(roi_scores)
            blobs, im_scales = get_blobs(im)
            print("best_RPN_score_ROI must be 0, and it is equal to ",best_RPN_score_ROI,"the score is",roi_scores[best_RPN_score_ROI]) # It must be 0
            if not(nms_thresh==0.0):
                best_roi = rois[best_RPN_score_ROI,:]
                #print(best_roi)
                #best_bbox_pred = bbox_pred[best_RPN_score_ROI,:]
                #print(bbox_pred.shape)
                #boxes = rois[:, 1:5] / im_scales[0]
                best_roi_boxes =  best_roi[1:5] / im_scales[0]
                best_roi_boxes_and_score = np.expand_dims(np.concatenate((best_roi_boxes,roi_scores[best_RPN_score_ROI])),axis=0)
                cls = ['best_object']
                #print(best_roi_boxes)
                vis_detections_list(im, cls, [best_roi_boxes_and_score], thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'.jpg'
                plt.savefig(name_output)
            else:
                roi_boxes =  rois[:,1:5] / im_scales[0]
                roi_boxes_and_score = np.concatenate((roi_boxes,roi_scores),axis=1)
                cls = ['object']*len(roi_boxes_and_score)
                #print(best_roi_boxes)
                vis_detections_list(im, cls, [roi_boxes_and_score], thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'.jpg'
                plt.savefig(name_output)
            if(nms_thresh==0.7):
                roi_boxes =  rois[:,1:5] / im_scales[0]
                roi_boxes_and_score = np.concatenate((roi_boxes,roi_scores),axis=1)
                cls = ['object']*len(roi_boxes_and_score)
                #print(best_roi_boxes)
                vis_detections_list(im, cls, [roi_boxes_and_score], thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'_allBoxes.jpg'
                plt.savefig(name_output)
                
            # Plot the k first score zone  
            k = 5
            roi_boxes =  rois[:,1:5] / im_scales[0]
            roi_boxes_and_score = np.concatenate((roi_boxes,roi_scores),axis=1)
            roi_boxes_and_score = roi_boxes_and_score[0:k,:]
            cls = ['object']*len(roi_boxes_and_score)
            #print(best_roi_boxes)
            vis_detections_list(im, cls, [roi_boxes_and_score], thresh=0.0)
            name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'_'+str(k)+'Boxes.jpg'
            plt.savefig(name_output)
            
        plt.close('all')
        tf.reset_default_graph()
        sess.close()
        
    
    
    return(0) # Not really necessary indead
        
def FasterRCNN_TL_MILSVM(reDo = False,normalisation=False):
    """
    Compute the performance on the Your Paintings subset ie Crowley
    on the fc7 output but with an Multi Instance SVM classifier for classifier the
    bag with the Said method
    @param reDo : recompute the feature even if it exists saved on the disk and erases the old one
    @param normalisation : normalisation of the date before doing the MILSVM from Said
    Attention cette fonction ne fonctionne pas et je n'ai pas trouver le bug, il ne 
    faut pas utiliser cette fonction mais plutot aller voir TL_MILSVM
    """
    print("Attention cette fonction ne fonctionne pas et je n'ai pas trouver le bug, il ne faut pas utiliser cette fonction mais plutot aller voir TL_MILSVM")
    TestMode_ComparisonWithBestObjectScoreKeep = True
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    #df_test = df_label[df_label['set']=='test']
    #sLength = len(df_test['name_img'])
    #sLength_all = len(df_label['name_img'])
    #name_img = df_test['name_img'][0]
    #i = 0
    #y_test = np.zeros((sLength,10))
    NETS_Pretrained = {'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt'
                   }
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}

    for demonet in NETS_Pretrained.keys():
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
          
        if database=='Paintings':
            item_name = 'name_img'
            path_to_img = '/media/HDD/data/Painting_Dataset/'
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        path_data = path
        N = 1
        extL2 = ''
        
        nms_thresh = 0.7
        
        name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
        #name_pkl = path_data + 'testTL_withNMSthresholdProposal03.pkl'
        
        if not(os.path.isfile(name_pkl)) or reDo:
            print('Start computing image region proposal')
            if demonet == 'vgg16_COCO':
                size_output = 4096
            elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                size_output = 2048
            features_resnet_dict= {}
            # Use the output of fc7 
            
            net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,
                                  modeTL= True,nms_thresh=nms_thresh)
            saver = tf.train.Saver()
            saver.restore(sess, tfmodel)
            numberOfRegion = 0
            with open(name_pkl, 'wb') as pkl:
                for i,name_img in  enumerate(df_label[item_name]):
                    if i%1000==0:
                        print(i,name_img)
                        if not(i==0):
                            pickle.dump(features_resnet_dict,pkl)
                            features_resnet_dict= {}
                    complet_name = path_to_img + name_img + '.jpg'
                    im = cv2.imread(complet_name)
                    cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
                    features_resnet_dict[name_img] = fc7
                    numberOfRegion += len(fc7)
                
                print("We have ",numberOfRegion,"regions proposol")
                # We have  292081 regions proposol avec un threshold a 0.0
                # Avec un threshold a 0.1 dans le NMS de RPN on a 712523 regions
                pickle.dump(features_resnet_dict,pkl)
            sess.close()
        
        print("Load data")
        features_resnet_dict = {}
        with open(name_pkl, 'rb') as pkl:
            for i,name_img in  enumerate(df_label[item_name]):
                if i%1000==0 and not(i==0):
                    features_resnet_dict_tmp = pickle.load(pkl)
                    if i==1000:
                        features_resnet_dict = features_resnet_dict_tmp
                    else:
                        features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
            features_resnet_dict_tmp = pickle.load(pkl)
            features_resnet_dict =  {**features_resnet_dict,**features_resnet_dict_tmp}
               
        print("preparing data fpr learning")
        print("Number of element in the base",len(features_resnet_dict))
        k_per_bag = 1
        AP_per_class = []
        P_per_class = []
        R_per_class = []
        P20_per_class = []
        testMode = True
        jtest = 0
        j = 0
        # TODO normaliser les donnees en moyenne variance, normaliser selon les features 
        for j,classe in enumerate(classes):
            if testMode and not(j==jtest):
                continue
            list_training_ex = []
            list_test_ex = []
            y_test = []
            pos_ex = None
            neg_ex = None
            print(j,classe)
            for index,row in df_label.iterrows():
                name_img = row[item_name]
#                print(classes[j],row['classe'])
                inClass = classes[j] in row['classe']
                inTest = row['set']=='test'
                f = features_resnet_dict[name_img]
                if index%1000==0:
                    print(index,name_img)
                if not(inTest):
                    if(len(f) >= k_per_bag):
                        bag = np.expand_dims(f[0:k_per_bag,:],axis=0)
                    else:
                        print("pourquoi t es la")
                        number_repeat = k_per_bag // len(f)  +1
                        #print(number_repeat)
                        f_repeat = np.repeat(f,number_repeat,axis=0)
                        #print(f_repeat.shape)
                        bag = np.expand_dims(f_repeat[0:k_per_bag,:],axis=0)
                    if not(inClass):
                        if neg_ex is None:
                            neg_ex = bag
                        else:
                            neg_ex = np.vstack((neg_ex,bag))
                    else:
                         if pos_ex is None:
                            pos_ex = bag
                         else:
                            pos_ex = np.vstack((pos_ex,bag))
                else:
                    list_test_ex += [f]
                    if not(inClass):
                        y_test += [0]
                    else:
                        y_test += [1]
            #del(features_resnet_dict) # Try to free the memory
            
            if normalisation == True:
                mean_training_ex = np.mean(np.vstack((pos_ex,neg_ex)),axis=(0,1))
                std_training_ex = np.std(np.vstack((pos_ex,neg_ex)),axis=(0,1))
#                if std_training_ex==0.0: std_training_ex=1.0
                neg_ex_norm = (neg_ex - mean_training_ex)/std_training_ex
                pos_ex_norm = (pos_ex - mean_training_ex)/std_training_ex
                        
            print("Learning of the Multiple Instance Learning SVM")
            restarts = 0
            max_iters = 300
            #from trouver_classes_parmi_K import MILSVM
            classifierMILSVM = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=restarts,
                                      max_iters=max_iters,symway=True,
                                      all_notpos_inNeg=False,gridSearch=True,
                                      verbose=True)
            if normalisation == True:
                classifier = classifierMILSVM.fit(pos_ex_norm, neg_ex_norm)
            else:
                classifier = classifierMILSVM.fit(pos_ex, neg_ex)
            print("End training")
            y_predict_confidence_score_classifier = np.zeros_like(y_test)
            labels_test_predited  =  np.zeros_like(y_test)
            
            for i,elt in enumerate(list_test_ex):
                if normalisation == True:
                    elt = (elt - mean_training_ex)/std_training_ex # TODO check if it is the right way to do
                try:
                    if not(TestMode_ComparisonWithBestObjectScoreKeep):
                        decision_function_output = classifier.decision_function(elt)
                    else:
                        # We only keep the best score object box
                        decision_function_output = classifier.decision_function(elt[0,:].reshape(1,-1))
                    y_predict_confidence_score_classifier[i] = np.max(decision_function_output) # Confidence on the result
                    if np.max(decision_function_output) > 0:
                        labels_test_predited[i] = 1 
                    else: 
                        labels_test_predited[i] =  0 # Label of the class 0 or 1
                except ValueError:
                    print('ValueError',i,elt.shape)
            test_precision = precision_score(y_test,labels_test_predited)
            test_recall = recall_score(y_test,labels_test_predited)
            F1 = f1_score(y_test,labels_test_predited)
            print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}".format(test_precision,test_recall,F1))
            AP = average_precision_score(y_test,y_predict_confidence_score_classifier,average=None)
            print("MIL-SVM version Average Precision for",classes[j]," = ",AP)
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
    
def FasterRCNN_TransferLearning_Test_Bidouille():
    DATA_DIR =  '/media/HDD/data/Art Paintings from Web/'
    demonet = 'res152_COCO'
    tf.reset_default_graph() # Needed to use different nets one after the other
    print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32]
    nbClasses = len(CLASSES)
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    
    #tfmodel = os.path.join(path_to_model,DATASETS[dataset][0],NETS[demonet][0])
    print(tfmodel)
#    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
#                              NETS[demonet][0])
    
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
      
    net.create_architecture('TEST', nbClasses,
                          tag='default', anchor_scales=anchor_scales,modeTL = True)
#    {'bbox_pred': <tf.Tensor 'add:0' shape=(?, 324) dtype=float32>,
# 'cls_pred': <tf.Tensor 'resnet_v1_152_5/cls_pred:0' shape=(?,) dtype=int64>,
# 'cls_prob': <tf.Tensor 'resnet_v1_152_5/cls_prob:0' shape=(?, 81) dtype=float32>,
# 'cls_score': <tf.Tensor 'resnet_v1_152_5/cls_score/BiasAdd:0' shape=(?, 81) dtype=float32>,
# 'rois': <tf.Tensor 'resnet_v1_152_3/rois/proposal:0' shape=(?, 5) dtype=float32>,
# 'rpn_bbox_pred': <tf.Tensor 'resnet_v1_152_3/rpn_bbox_pred/BiasAdd:0' shape=(1, ?, ?, 48) dtype=float32>,
# 'rpn_cls_pred': <tf.Tensor 'resnet_v1_152_3/rpn_cls_pred:0' shape=(?,) dtype=int64>,
# 'rpn_cls_prob': <tf.Tensor 'resnet_v1_152_3/rpn_cls_prob/transpose_1:0' shape=(1, ?, ?, 24) dtype=float32>,
# 'rpn_cls_score': <tf.Tensor 'resnet_v1_152_3/rpn_cls_score/BiasAdd:0' shape=(1, ?, ?, 24) dtype=float32>,
# 'rpn_cls_score_reshape': <tf.Tensor 'resnet_v1_152_3/rpn_cls_score_reshape/transpose_1:0' shape=(1, ?, ?, 2) dtype=float32>}

    
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))
    im_name = 'L Adoration des mages - Jan Mabuse - 1515.jpg'
    print('Demo for data/demo/{}'.format(im_name))
    imfile = os.path.join(DATA_DIR, im_name)
    im = cv2.imread(imfile)
    
    
    
    # If we use the top detection we have the 300 first case
    
    cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
    #print(cls_score, cls_prob, bbox_pred, rois, fc7,pool5)
    print(cls_score.shape, cls_prob.shape, bbox_pred.shape, rois.shape,roi_scores.shape, fc7.shape,pool5.shape)
    #(300, 81) (300, 81) (300, 324) (300, 5) (300, 2048) (300, 7, 7, 1024)
    
    #cls_prob = cls_prob[np.argmax(roi_scores),:]
    #bbox_pred = bbox_pred[np.argmax(roi_scores),:]
    
    # Only single-image batch implemented !
    scores, boxes = TL_im_detect_end(cls_prob, bbox_pred, rois,im)
    CONF_THRESH = 0.1 # Plot if the score for this class is superior to CONF_THRESH
    NMS_THRESH = 0.7 # non max suppression
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if(len(inds)>0):
            print('CLASSES[cls_ind]',CLASSES[cls_ind])
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.show()
    sess.close()
    
def FasterRCNN_ImagesObject():
    DATA_DIR =  '/media/HDD/data/Art Paintings from Web/'
    demonet = 'res152_COCO'
    tf.reset_default_graph() # Needed to use different nets one after the other
    print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32]
    nbClasses = len(CLASSES)
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    
    #tfmodel = os.path.join(path_to_model,DATASETS[dataset][0],NETS[demonet][0])
    print(tfmodel)
#    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
#                              NETS[demonet][0])
    
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
                          tag='default', anchor_scales=anchor_scales)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    dirs = os.listdir(DATA_DIR)
    for im_name in dirs:
    
        #    im_name = 'Adoration bergers Antoon.jpg'
        #    im_name = 'Adoration bergers Lorenzo.jpg'
        im_name_wt_ext, _ = im_name.split('.')
        #im_name = 'L Adoration des mages - Jan Mabuse - 1515.jpg'
        print('Demo for data/demo/{}'.format(im_name))
        imfile = os.path.join(DATA_DIR, im_name)
        im = cv2.imread(imfile)
        scores, boxes = im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
           # Only single-image batch implemented !
        print(scores.shape)
        #print(scores)
        
        CONF_THRESH = 0.5
        NMS_THRESH = 0.3 # non max suppression
        cls_list = []
        dets_list = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            cls_list += [cls]
            dets_list += [dets]
            if(len(inds)>0):
                print(CLASSES[cls_ind])
        vis_detections_list(im, cls_list, dets_list, thresh=CONF_THRESH)
        name_output = 'output_FasterRCNN/' + im_name_wt_ext + '_FasterRCNN.jpg'
        plt.savefig(name_output)
    plt.show()
    sess.close()
        
        
if __name__ == '__main__':
    ## Faster RCNN re-scale  the  images  such  that  their  shorter  side  = 600 pixels  
#    Illus_NMS_threshold_test()
#    run_FasterRCNN_Perf_Paintings(TL = True,reDo=True)
#    FasterRCNN_TL_MILSVM(reDo = False,normalisation=False)
#    read_features_computePerfPaintings()
#    FasterRCNN_TransferLearning_misvm()
#    FasterRCNN_TL_MILSVM()
    #FasterRCNN_ImagesObject()
    #run_FasterRCNN_demo()
    #run_FasterRCNN_Perf_Paintings()
    # List des nets a tester : VGG16-VOC12
    #  VGG16-VOC07
    # RESNET152 sur COCO
    # VGG16 sur COCO
    # RES101 sur VOC12
    Compute_Faster_RCNN_features(demonet='res152_COCO',nms_thresh = 0.7,database='Paintings',
                                 augmentation=False,L2 =False,
                                 saved='all',verbose=True)    