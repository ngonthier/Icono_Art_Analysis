#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:42:41 2017

Based on the Tensorflow implementation of Faster-RCNNN but it has been modified a lot
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
from tf_faster_rcnn.lib.model.test import im_detect,TL_im_detect,TL_im_detect_end,get_blobs,apply_nms
from tf_faster_rcnn.lib.model.nms_wrapper import nms
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
from trouver_classes_parmi_K import MILSVM
import pathlib
from tool_on_Regions import reduce_to_k_regions
from tf_faster_rcnn.lib.datasets.factory import get_imdb
from LatexOuput import arrayToLatex

CLASSESVOC = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSESCOCO = ('__background__','person', 'bicycle','car','motorcycle', 'aeroplane','bus',
               'train','truck','boat',
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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_reshape(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def run_FasterRCNN_Perf_Paintings(TL = True,reDo=False,feature_selection = 'MaxObject',
                                  nms_thresh = 0.0,CV_Crowley=True,database='Paintings'):
    """
    Compute the performance on the Your Paintings subset ie Crowley on the output 
    but also the best case on feature fc7 of the best proposal part
    This function compute the classification score 
    @param : TL : use the features maps of the best object score detection
    @param : reDO : recompute the features maps
    @param : feature_selection : 'MaxObject' or 'meanObject' mean on all regions or only keep the max
    @param : nms_thresh : Threshold in the RPN 
    @param : CV_Crowley : boolean, use the same CV splitting that crowley or use a 3 fold otherwise for the gridsearch of the C param in the LinearSVM
    @param : database : the name of the database to use 
    """
    
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
    path = '/media/HDD/output_exp/ClassifPaintings/'
    databasetxt = path +database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    
    if database=='Paintings':
        df_test = df_label[df_label['set']=='test']
        sLength = len(df_test[item_name])
        sLength_all = len(df_label[item_name])
        name_img = df_test[item_name][0]
        i = 0
        y_test = np.zeros((sLength,10))
        classes_vectors = np.zeros((sLength_all,10))
    elif database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
#        5491 images avant
#        5473 images gardees
        random_state = 0
        sLength_all = len(df_label[item_name])
        index = np.arange(0,sLength_all)
        index_trainval, index_test = train_test_split(index, test_size=0.6, random_state=random_state)
        classes_vectors = df_label.as_matrix(columns=classes)
        
    
    NETS_Pretrained = {'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt'
                   }
    NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}

    for demonet in NETS_Pretrained.keys():
        #demonet = 'res101_COCO'
        tf.reset_default_graph() # Needed to use different nets one after the other
        print(database,demonet,feature_selection)
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
            if not(database=='Paintings'):
                print("That is impossible.")
                raise NotImplementedError
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
                    if(classes[j] in list(df_test['classe'][df_test['name_img']==name_img])[0]):
                        y_test[i,j] = 1
                
            AP_per_class = []
            for k,classe in enumerate(classes):
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
            path_data = path
            if feature_selection =='meanObject':
                N = 'mean'
            else:
                N = 1
            if feature_selection=='MaxObject':
                strnmsthreshold = ''
            elif feature_selection =='meanObject':
                strnmsthreshold = '_'+str(nms_thresh)
            extL2 = ''
            name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+strnmsthreshold+'.pkl'
            name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+'.pkl'
            if not(os.path.isfile(name_pkl)) or reDo or not(os.path.isfile(name_pkl_all_features)):
                if demonet == 'vgg16_COCO':
                    size_output = 4096
                elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                    size_output = 2048
                features_resnet = np.ones((sLength_all,size_output))
                # Use the output of fc7 
                net.create_architecture("TEST", nbClasses,
                                      tag='default', anchor_scales=anchor_scales,
                                      modeTL= True,nms_thresh=nms_thresh) # default nms_thresh = 0.7
                saver = tf.train.Saver()
                saver.restore(sess, tfmodel)
                
                features_resnet_dict= {}
                pkl = open(name_pkl_all_features, 'wb')
                
                for i,name_img in  enumerate(df_label[item_name]):
                    if i%1000==0:
                        print(i,name_img)
                        if not(i==0):
                            pickle.dump(features_resnet_dict,pkl)
                            features_resnet_dict= {}
                    if database=='VOC12' or database=='Paintings':
                        complet_name = path_to_img + name_img + '.jpg'
                        name_sans_ext = name_img
                    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                        name_sans_ext = os.path.splitext(name_img)[0]
                        complet_name = path_to_img +name_sans_ext + '.jpg'       
                    im = cv2.imread(complet_name)
                    
                    cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, 
                                                                                              net, im) # Arguments: im (ndarray): a color image in BGR order
                    features_resnet_dict[name_img] = fc7
                    
                    # Normally argmax roi_scores is 0
                    if feature_selection == 'MaxObject':
                        out = fc7[np.argmax(roi_scores),:]
                        features_resnet[i,:] = np.array(out)
                    elif feature_selection =='meanObject':
                        out = np.mean(fc7,axis=0)
                        features_resnet[i,:] = np.array(out)
                        
                    if database=='VOC12' or database=='Paintings':
                        for j in range(10):
                            if( classes[j] in df_label['classe'][i]):
                                classes_vectors[i,j] = 1
                 
                pickle.dump(features_resnet_dict,pkl)
                pkl.close()
                    
                if database=='VOC12' or database=='Paintings':
                    X_train = features_resnet[df_label['set']=='train',:]
                    y_train = classes_vectors[df_label['set']=='train',:]
                    X_test= features_resnet[df_label['set']=='test',:]
                    y_test = classes_vectors[df_label['set']=='test',:]
                    X_val = features_resnet[df_label['set']=='validation',:]
                    y_val = classes_vectors[df_label['set']=='validation',:]
                    #print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
                    Data = [X_train,y_train,X_test,y_test,X_val,y_val]
                elif database=='Wikidata_Paintings_miniset_verif':
                    X_test= features_resnet[index_test,:]
                    y_test = classes_vectors[index_test,:]
                    X_trainval =features_resnet[index_trainval,:]
                    y_trainval =  classes_vectors[index_trainval,:]
                    Data = [X_trainval,y_trainval,X_test,y_test]
                    
                with open(name_pkl, 'wb') as pkl:
                    pickle.dump(Data,pkl)

                sess.close()
            
            # Compute the metric
            if CV_Crowley :
                print("CV_Crowley = True") 
            else:
                print("CV_Crowley = False")
            if CV_Crowley and not(database == 'Paintings'):
                print("This is not possible !!!")
                CV_Crowley = False
            if feature_selection=='MaxObject':
                nms_thresh = None
                
            Classification_evaluation(kind=demonet,kindnetwork='FasterRCNN',
                                      database=database,L2=False,augmentation=False,
                                      classifier_name='LinearSVM',CV_Crowley=CV_Crowley,
                                      feature_selection =feature_selection,nms_thresh=nms_thresh)
             

def localisation_pred_met(all_boxes_pred):
    return(0)

def run_FRCNN_Detection_perf(database='VOC2007'):
    """
    15 mai 2018
    Le but de cette fonction est d'evaluer les performances de classification et
    de detection sur Pascal VOC2007 test set et autres 
    """
    print('Evaluation of the detections performance on ',database)
    max_per_image= 100
    TEST_NMS = 0.3
    thresh= 0.05
    output_dir=  '/media/HDD/output_exp/ClassifPaintings/tmp/'
    input_dir=  '/media/HDD/output_exp/ClassifPaintings/'
    if database=='VOC2007' or  database=='clipart':
        per =True
        ext = '.csv'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
        if database=='VOC2007' : imdb = get_imdb('voc_2007_test')
        if database=='clipart' : imdb = get_imdb('clipart_test')
        num_classes = imdb.num_classes -1
        corr_voc_coco = [0,5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
    elif database=='watercolor':
        per =True
        ext = '.csv'
        classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
        imdb = get_imdb('watercolor_test')
        num_classes = imdb.num_classes-1
        corr_watercolor_coco = [0,2,15,3,16,17,1] # From Cooc to watercolor
        corr_watercolor_voc = [0,2,3,7,8,12,15]
    elif database=='PeopleArt':
        per =True
        ext = '.csv'
        classes =  ["person"]
        imdb = get_imdb('PeopleArt_test')
        num_classes = imdb.num_classes-1
        corr_watercolor_coco = [0,1] # From Cooc to PeopleArt
        corr_watercolor_voc = [0,15]
    
    imdb.set_force_dont_use_07_metric(True)
    num_images = len(imdb.image_index)
    
    databasetxt = input_dir +database+ext
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    if database=='VOC2007'  or  database=='clipart':
        y_true = df_test.as_matrix(columns=CLASSESVOC[1:])
    elif database=='watercolor':
        y_true = df_test.as_matrix(columns=classes)
    elif database=='PeopleArt':
        y_true = df_test.as_matrix(columns=classes)
    y_predict = np.zeros((num_images,num_classes))
    assert(y_true.shape==y_predict.shape)
    
    demonets = ['res152_COCO','res101_COCO','res101_VOC07']
    
    just_Sans_Regression = False
    
    for demonet in demonets:
        if just_Sans_Regression:
            continue
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes+1)]
        print(demonet)
        tf.reset_default_graph()
        if 'VOC'in demonet:
            CLASSES = CLASSES_SET['VOC']
            anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
        elif 'COCO'in demonet:
            CLASSES = CLASSES_SET['COCO']
            anchor_scales = [4, 8, 16, 32]
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
        else:
          raise NotImplementedError
        nbClasses = len(CLASSES)
        net.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales)
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        
        for i in range(num_images):
            if i%1000==0: print('Images #',i)
            im = cv2.imread(imdb.image_path_at(i))
            scores, boxes = im_detect(sess, net, im) # For COCO scores.shape = #boxes,81, boxes.shape = #boxes,4*81


            if 'COCO' in demonet:
                if database=='VOC2007':
                    scores = scores[:,corr_voc_coco]
                elif database=='watercolor' or database=='PeopleArt':
                    scores = scores[:,corr_watercolor_coco]
                boxes_tmp = np.zeros((len(scores),21*4))
                for j in range(1, imdb.num_classes):
                    if database=='VOC2007': j_tmp = corr_voc_coco[j]
                    elif database=='watercolor'or database=='PeopleArt':j_tmp = corr_watercolor_coco[j]
                    boxes_tmp[:,j*4:(j+1)*4] = boxes[:,j_tmp*4:(j_tmp+1)*4]
                boxes = boxes_tmp
            elif  'VOC' in demonet and (database=='watercolor'or database=='PeopleArt'):
                scores = scores[:,corr_watercolor_voc]
                boxes_tmp = np.zeros((len(scores),21*4))
                for j in range(1, imdb.num_classes):
                    j_tmp = corr_watercolor_voc[j]
                    boxes_tmp[:,j*4:(j+1)*4] = boxes[:,j_tmp*4:(j_tmp+1)*4]
                boxes = boxes_tmp
                

            # skip j = 0, because it's the background class
            for j in range(1, imdb.num_classes):
              inds = np.where(scores[:, j] > thresh)[0]
              cls_scores = scores[inds, j]
              cls_boxes = boxes[inds, j*4:(j+1)*4]
              cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
              keep = nms(cls_dets, TEST_NMS)
              cls_dets = cls_dets[keep, :]
              all_boxes[j][i] = cls_dets
              
              # Part classification
              scores_max = np.max(scores,axis=0)
              y_predict[i,:] = scores_max[1:]
        
        # Score de classification
        AP_per_class = []
        for k,classe in enumerate(imdb.classes):
            if not(k==0):
                kk = k -1 
                AP = average_precision_score(y_true[:,kk],y_predict[:,kk],average=None)
                AP_per_class += [AP]
                print("Average Precision Classification for",classe," = ",AP)
        print(demonet," mean Average Precision Classification = {0:.3f}".format(np.mean(AP_per_class)))
        
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] \
                    for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
           
        det_file = os.path.join(output_dir, 'detections_perf.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        
        print('Evaluating detections')
        aps = imdb.evaluate_detections(all_boxes, output_dir)
        
        # Rappel des scores :
        print(demonet)
        print(arrayToLatex(CLASSESVOC[1:],dtype=str))
        print("Classification task")
        print(arrayToLatex(AP_per_class,per=per))
        print("Detection task")
        print(arrayToLatex(aps,per=per))

    # We will know see the impact of the loss of the regression of the bounding box
    print('Impact of the abscence of bounding boxes regressions at the end')
    for demonet in demonets:
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes+1)]
        print(demonet)
        tf.reset_default_graph()
        if 'VOC'in demonet:
            CLASSES = CLASSES_SET['VOC']
            anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
        elif 'COCO'in demonet:
            CLASSES = CLASSES_SET['COCO']
            anchor_scales = [4, 8, 16, 32]
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
        else:
          raise NotImplementedError
        nms_thresh = TEST_NMS
    
        # init session
        sess = tf.Session(config=tfconfig)
        nbClasses = len(CLASSES)
        net.create_architecture("TEST", nbClasses,
                                          tag='default', anchor_scales=anchor_scales,
                                          modeTL= True,nms_thresh=nms_thresh) # default nms_thresh = 0.7
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        
        plot = False
        if plot:
            path_to_output2  = '/media/HDD/output_exp/ClassifPaintings/Perf_FasterRCNN/VOC2007_Test/'
            pathlib.Path(path_to_output2).mkdir(parents=True, exist_ok=True) 
        
        for i in range(num_images):
            if i%1000==0: print('Images #',i)
            name_tab = imdb.image_path_at(i)
            name_tab2 = name_tab.split('/')[-1]
#            name_im = name_tab2.split('.')[0]
            im = cv2.imread(imdb.image_path_at(i))
            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess,  net, im) 
    #        
    #        scores_end, boxes_end = TL_im_detect_end(cls_prob, bbox_pred, rois,im)
    #        if 'COCO' in demonet:
    #            scores_end = scores_end[:,corr_voc_coco]
    #            boxes_tmp = np.zeros((len(scores_end),21*4))
    #            for j in range(1, imdb.num_classes):
    #                j_tmp = corr_voc_coco[j]
    #                boxes_tmp[:,j*4:(j+1)*4] = boxes_end[:,j_tmp*4:(j_tmp+1)*4]
    #            boxes_end = boxes_tmp
    #
    #            # skip j = 0, because it's the background class
    #        for j in range(1, imdb.num_classes):
    #            inds = np.where(scores_end[:, j] > thresh)[0]
    ##            print(j,len(inds))
    #            cls_scores_end = scores_end[inds, j]
    #            cls_boxes_end = boxes_end[inds, j*4:(j+1)*4]
    #            cls_dets_end = np.hstack((cls_boxes_end, cls_scores_end[:, np.newaxis])) \
    #                .astype(np.float32, copy=False)
    #            keep = nms(cls_dets_end, TEST_NMS)
    #            cls_dets_end = cls_dets_end[keep, :]
    ##            print(len(keep))
                    
            # C'est cls_prob apres np.reshape(scores, [scores.shape[0], -1]) qui correspond au scores de im_detect 
            blobs, im_scales = get_blobs(im)
            roi =  rois[:,1:5] / im_scales[0]
            scores = np.reshape(cls_prob, [cls_prob.shape[0], -1])
            boxes = np.tile(roi, (1, scores.shape[1]))
            # For COCO scores.shape = #boxes,81, boxes.shape = #boxes,4*81
            
            if 'COCO' in demonet:
                if database=='VOC2007':
                    scores = scores[:,corr_voc_coco]
                elif database=='watercolor'or database=='PeopleArt':
                    scores = scores[:,corr_watercolor_coco]
            elif 'VOC' in demonet and (database=='watercolor'or database=='PeopleArt'):
                scores = scores[:,corr_watercolor_voc]
                    
            # skip j = 0, because it's the background class
            local_cls = []
            roi_boxes_and_score = None
            for j in range(1, imdb.num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
    #            print(j,len(inds))
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                  .astype(np.float32, copy=False)
                keep = nms(cls_dets, TEST_NMS)
    #            print(len(keep))
                cls_dets = cls_dets[keep, :]
                all_boxes[j][i] = cls_dets
    
            if plot:
                if len(cls_dets) > 0:
                    local_cls += [imdb.classes[j]]*len(cls_dets)
                    roi_boxes_score = np.expand_dims(cls_dets,axis=1)
    #                print(roi_boxes_score.shape)
                    if roi_boxes_and_score is None:
                        roi_boxes_and_score = roi_boxes_score
                    else:
                        roi_boxes_and_score= \
                        np.vstack((roi_boxes_score,roi_boxes_and_score))
    #               print(roi_boxes_and_score)
    #                print(local_cls)
    #                vis_detections_list(im, local_cls, roi_boxes_and_score, thresh=0.5)
    #                name_output = path_to_output2 + name_im + '_Regions.jpg'
    #                plt.savefig(name_output)
    #                plt.close()
              # Part classification
            scores_max = np.max(scores,axis=0)
            y_predict[i,:] = scores_max[1:]
            
        # Score de classification
        AP_per_class = []
        for k,classe in enumerate(imdb.classes):
            if not(k==0):
                kk = k -1 
                AP = average_precision_score(y_true[:,kk],y_predict[:,kk],average=None)
                AP_per_class += [AP]
                print("Average Precision Classification for",classe," = ",AP)
        print(demonet," mean Average Precision Classification = {0:.3f}".format(np.mean(AP_per_class)))
        
        # Rappel des scores :
        
        print(demonet)
        print(arrayToLatex(CLASSESVOC[1:],dtype=str))
        print("Classification task")
        print(arrayToLatex(AP_per_class,per=per))
        
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] \
                    for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
           
        det_file = os.path.join(output_dir, 'detections_perf.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        
        aps = imdb.evaluate_detections(all_boxes, output_dir)
        print("Detection task with thresh = ",TEST_NMS)
        print(arrayToLatex(aps,per=per))
        
        print('Evaluating detections')
        with_thres_comp = False
        if with_thres_comp:
            for thresh in np.arange(TEST_NMS-0.1,0.0,-0.1):
                all_boxes_after_nms = apply_nms(all_boxes, float(thresh))
                aps = imdb.evaluate_detections(all_boxes_after_nms, output_dir)
                print("Detection task with thresh = ",thresh)
                print(arrayToLatex(aps,per=per))
        
            
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
 
def vis_detections(im, class_name, dets, thresh=0.5,with_title=True):
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

    if with_title:
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                      fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
def vis_detections_list(im, class_name_list, dets_list, thresh=0.5,list_class=None):
    """Draw detected bounding boxes."""

    list_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8',	'#f58231','#911eb4','#46f0f0','#f032e6',	
                   '#d2f53c','#fabebe',	'#008080','#e6beff','#aa6e28','#fffac8','#800000',
                   '#aaffc3','#808000','#ffd8b1','#000080','#808080','#FFFFFF','#000000']	
    i_color = 0
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
   
    for class_name,dets in zip(class_name_list,dets_list):
        #print(class_name,dets)
        inds = np.where(dets[:, -1] >= thresh)[0]
        if not(len(inds) == 0):
            if list_class is None:
                color = list_colors[i_color]
                i_color = ((i_color + 1) % len(list_colors))
            else:
                i_color = np.where(np.array(list_class)==class_name)[0][0] % len(list_colors)
                color = list_colors[i_color]
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
                                 saved='all',verbose=True,filesave='pkl',k_regions=300):
    """
    @param : demonet : teh kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : nms_thresh : the nms threshold on the Region Proposal Network
    
    """
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
        num_classes = 10
        ext = '.txt'
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        num_classes = 20
        ext = '.txt'
        raise(NotImplementedError)
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        ext = '.csv'
        item_name = 'item'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        classes = ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
        num_classes = 10
    elif database=='VOC2007':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
        num_classes = 20
        ext = '.csv'
    elif database=='PeopleArt':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/PeopleArt/JPEGImages/'
        num_classes = 1
        ext = '.csv'
    elif database=='watercolor':
        num_classes = 6
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
    elif database=='clipart':
        num_classes = 20
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/cross-domain-detection/datasets/clipart/JPEGImages/'
    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
        num_classes = 5
        ext = '.txt'
    else:
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
        ext = '.txt'
    
    databasetxt = path_data + database + ext
    if database=='VOC2007' or database=='watercolor' or database=='clipart':
        df_label = pd.read_csv(databasetxt,sep=",",dtype=str)
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                  'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                  'turban':int,'set':str,'Anno':int}
        df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
    else:
        df_label = pd.read_csv(databasetxt,sep=",")
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
    
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
    
    tf.reset_default_graph() # Needed to use different nets one after the other
    if verbose: print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32] # we  use  3  aspect  ratios  and  4  scales (adding 64**2)
    nbClassesDemoNet = len(CLASSES)
    path_to_model = '/media/HDD/models/tf-faster-rcnn/'
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
      size_output = 4096
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
      size_output = 2048
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
      size_output = 2048
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
      
   
    net.create_architecture("TEST", nbClassesDemoNet,
                          tag='default', anchor_scales=anchor_scales,
                          modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    features_resnet_dict= {}
    
    sets = ['train','val','trainval','test']
    
    
    
    if filesave == 'pkl':
        name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'.pkl'
        pkl = open(name_pkl_all_features, 'wb')
    elif filesave =='tfrecords':
        if k_regions==300:
            k_per_bag_str = ''
        else:
            k_per_bag_str = '_k'+str(k_regions)
        dict_writers = {}
        for set_str in sets:
            name_pkl_all_features = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
            dict_writers[set_str] = tf.python_io.TFRecordWriter(name_pkl_all_features)
        if database=='Paintings':
            classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        if database=='VOC2007' or database=='clipart':
            classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
        if database=='watercolor':
            classes = ["bicycle", "bird","car", "cat", "dog", "person"]
        if database=='PeopleArt':
            classes = ["person"]
    
    Itera = 1000
    for i,name_img in  enumerate(df_label[item_name]):
        if filesave=='pkl':
            if not(k_regions==300):
                raise(NotImplemented)
            if i%Itera==0:
                if verbose : print(i,name_img)
                if not(i==0):
                    pickle.dump(features_resnet_dict,pkl) # Save the data
                    features_resnet_dict= {}
            if database in ['VOC2007','clipart','Paintings','watercolor','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                complet_name = path_to_img + name_img + '.jpg'
            elif database=='PeopleArt':
                complet_name = path_to_img + name_img
                name_sans_ext = os.path.splitext(name_img)[0]
            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                name_sans_ext = os.path.splitext(name_img)[0]
                complet_name = path_to_img +name_sans_ext + '.jpg'
            im = cv2.imread(complet_name)
            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
            if saved=='fc7':
                features_resnet_dict[name_img] = fc7
            elif saved=='pool5':
                features_resnet_dict[name_img] = pool5
            elif saved=='all':
                features_resnet_dict[name_img] = rois,roi_scores,fc7
                
        elif filesave=='tfrecords':
            if i%Itera==0:
                if verbose : print(i,name_img)
            if database in ['VOC2007','clipart','Paintings','watercolor','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                complet_name = path_to_img + name_img + '.jpg'
                name_sans_ext = name_img
            elif database=='PeopleArt':
                complet_name = path_to_img + name_img
                name_sans_ext = os.path.splitext(name_img)[0]
            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                name_sans_ext = os.path.splitext(name_img)[0]
                complet_name = path_to_img +name_sans_ext + '.jpg'
            im = cv2.imread(complet_name)
            height = im.shape[0]
            width = im.shape[1]
            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            
            if k_regions==300:
                num_regions = fc7.shape[0]
                num_features = fc7.shape[1]
                dim1_rois = rois.shape[1]
                classes_vectors = np.zeros((num_classes,1))
                rois_tmp = np.zeros((k_regions,5))
                roi_scores_tmp = np.zeros((k_regions,1))
                fc7_tmp = np.zeros((k_regions,size_output))
                rois_tmp[0:rois.shape[0],0:rois.shape[1]] = rois
                roi_scores_tmp[0:roi_scores.shape[0],0:roi_scores.shape[1]] = roi_scores
                fc7_tmp[0:fc7.shape[0],0:fc7.shape[1]] = fc7           
                rois = rois_tmp
                roi_scores =roi_scores_tmp
                fc7 = fc7_tmp
            else:
                # We will select only k_regions 
                new_nms_thresh = 0.0
                score_threshold = 0.1
                minimal_surface = 36*36
                
                num_regions = k_regions
                num_features = fc7.shape[1]
                dim1_rois = rois.shape[1]
                classes_vectors = np.zeros((num_classes,1))
                rois_reduce,roi_scores_reduce,fc7_reduce =  reduce_to_k_regions(k_regions,rois, \
                                                       roi_scores, fc7,new_nms_thresh, \
                                                       score_threshold,minimal_surface)
                if(len(fc7_reduce) >= k_regions):
                    rois = rois_reduce[0:k_regions,:]
                    roi_scores =roi_scores_reduce[0:k_regions,]
                    fc7 = fc7_reduce[0:k_regions,:]
                else:
                    number_repeat = k_regions // len(fc7_reduce)  +1
                    f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
                    roi_scores_repeat = np.repeat(roi_scores_reduce,number_repeat,axis=0)
                    rois_reduce_repeat = np.repeat(rois_reduce,number_repeat,axis=0)
                    rois = rois_reduce_repeat[0:k_regions,:]
                    roi_scores =roi_scores_repeat[0:k_regions,]
                    fc7 = f_repeat[0:k_regions,:]
               
            
            if database=='Paintings':
                for j in range(num_classes):
                    if(classes[j] in df_label['classe'][i]):
                        classes_vectors[j] = 1
            if database in ['VOC2007','clipart','watercolor','PeopleArt']:
                for j in range(num_classes):
                    value = int((int(df_label[classes[j]][i])+1.)/2.)
                    #print(value)
                    classes_vectors[j] = value
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                for j in range(num_classes):
                    value = int(df_label[classes[j]][i])
                    classes_vectors[j] = value
            
            #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
            if saved=='fc7':
                features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'num_regions': _int64_feature(num_regions),
                    'num_features': _int64_feature(num_features),
                    'fc7': _bytes_feature(tf.compat.as_bytes(fc7.tostring())),
                    'label' : _bytes_feature(tf.compat.as_bytes(classes_vectors.tostring())),
                    'name_img' : _bytes_feature(str.encode(name_sans_ext))})
            elif saved=='pool5':
                raise(NotImplementedError)
            elif saved=='all':
                features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'num_regions': _int64_feature(num_regions),
                    'num_features': _int64_feature(num_features),
                    'dim1_rois': _int64_feature(dim1_rois),
                    'rois': _floats_feature(rois),
                    'roi_scores': _floats_feature(roi_scores),
                    'fc7': _floats_feature(fc7),
                    'label' : _floats_feature(classes_vectors),
                    'name_img' : _bytes_feature(str.encode(name_sans_ext))})
            example = tf.train.Example(features=features)    
            
            if database=='VOC2007' or database=='PeopleArt':
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='val').any():
                    dict_writers['val'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
            if (database=='Wikidata_Paintings_miniset') or database=='Paintings':
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='validation').any():
                    dict_writers['val'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
            if database in ['watercolor','clipart','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
                    
    if filesave=='pkl':
        pickle.dump(features_resnet_dict,pkl)
        pkl.close()
    elif filesave=='tfrecords':
        for set_str  in sets:
            dict_writers[set_str].close()
        
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
                     'Adam Elsheimer - Il Contento - WGA7492',
                     'Accademia - The Mystic Marriage of St. Catherine by Veronese'] # First come from Your Paintings and second from Wikidata
    list_dog = ['ny_yag_yorag_326_624x544', 'dur_dbm_770_624x544', 'ntii_skh_1196043_624x544', 'nti_ldk_884912_624x544', 'syo_bha_90009742_624x544', 'tate_tate_t00888_10_624x544', 'ntii_lyp_500458_624x544', 'ny_yag_yorag_37_b_624x544', 'ngs_ngs_ng_1193_f_624x544', 'dur_dbm_533_624x544']
    list_name_img += list_dog
    list_nms_thresh = [0.7,0.0,0.1]
    nms_thresh = list_nms_thresh[0]
    # First we test with a high threshold !!!
    plt.ion()
    for nms_thresh in list_nms_thresh:
        plt.close('all')
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
                print(roi_boxes.shape,roi_scores.shape)
                roi_boxes_and_score = np.concatenate((roi_boxes,roi_scores),axis=1)
                cls = ['object']*len(roi_boxes_and_score)
                #print(best_roi_boxes)
                vis_detections_list(im, cls, [roi_boxes_and_score], thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'_allBoxes.jpg'
                plt.savefig(name_output)
                k = 30
                new_nms_thresh = 0.0
                score_threshold = 0.1
                minimal_surface = 36*36
                fc7 = np.zeros_like(roi_scores)
                rois,roi_scores, _ = reduce_to_k_regions(k,rois,roi_scores, fc7,new_nms_thresh,score_threshold,minimal_surface)
                roi_boxes =  rois[:,1:5] / im_scales[0]
                roi_scores = np.expand_dims(roi_scores,axis=1)
                print(roi_boxes.shape,roi_scores.shape)
                roi_boxes_and_score = np.concatenate((roi_boxes,roi_scores),axis=1)
                cls = ['object']*len(roi_boxes_and_score)
                #print(best_roi_boxes)
                vis_detections_list(im, cls, [roi_boxes_and_score], thresh=0.0)
                name_output = path_to_output + name_img + '_threshold_'+str(nms_thresh)+'_reduceBoxes.jpg'
                plt.savefig(name_output)
                plt.show()

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
  
def Illus_ScoreObjectness():
    """
    The goal of this function is to test the modification of the NMS threshold 
    on the output provide by the algo 
    And plot the zone considered as the best by the Faster RCNN 
    """ 
    classes  = CLASSESVOC[1:21]
    corr_voc_coco = [0,5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
    TEST_NMS = 0.7
    num_classes = 21
    thresh = 0.0
    path_to_output = '/media/HDD/output_exp/ClassifPaintings/Test_ObjectScore/'
    pathlib.Path(path_to_output).mkdir(parents=True, exist_ok=True) 
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
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
      net_TL = vgg16()
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
      net_TL = resnetv1(num_layers=101)
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
      net_TL = resnetv1(num_layers=152)
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
      
    # List des images a test 
    path_to_img = '/media/HDD/output_exp/ClassifPaintings/im/'
    
    # creation of the images :
#    random_im = np.clip(np.random.normal(loc=125,size=(600,600,3),scale=25),0,255)
    random_im = np.random.randint(0,255,size=(600,600,3))
    list_images = []
    list_images_name = []
    list_images += [random_im]
    list_images_name += ['random_im']
    list_color = {'red' :  (0,0,255),'blue':(255,0,0),'green':(0,255,0),'white':(255,255,255),\
                  'grey':(128,128,128),'golden':(11,134,184)} # Color code in BGR
    for color in list_color.keys():
        im = np.ones(shape=(600,600,3))
        colors=list_color[color]
        for i in range(3):
            im[:,:,i] = colors[i]
        list_images += [im.astype(int)]
        list_images_name += [color]
    complet_name = path_to_img + 'watercolor-photoshop-brush-1200x580.jpg'
    im = cv2.imread(complet_name)
    list_images += [im]
    list_images_name += ['watercolor']
    
    nms_thresh = 0.7
    number_box_keep = 10
    plt.ion()
    plt.close('all')    
    sess = tf.Session(config=tfconfig)
    net_TL.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,
                                  modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)       
    for im,im_name in zip(list_images,list_images_name):
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = \
        TL_im_detect(sess, net_TL, im)
        blobs, im_scales = get_blobs(im)
        roi =  rois[0:number_box_keep,1:5] / im_scales[0]
        scores = roi_scores[0:number_box_keep]
        number_box_keep_p1 = number_box_keep +1
        roimin =  rois[-number_box_keep_p1:-1,1:5] / im_scales[0]
        scoresmin = roi_scores[-number_box_keep_p1:-1]
        Boexsmin = np.hstack((roimin,scoresmin))
        Boexs = np.hstack((roi,scores))
        vis_detections(im, 'object max', Boexs, thresh=0.,with_title=False)
        name_output = path_to_output + im_name + '_ObjectScoresMax.jpg'
        plt.savefig(name_output)     
        vis_detections(im, 'object min', Boexsmin, thresh=0.,with_title=False)
        name_output = path_to_output + im_name + '_ObjectScoresMin.jpg'
        plt.savefig(name_output)     
    plt.close('all')
    return(0) # Not really necessary indead
    
def Illus_box_ratio():
    """
    The goal of this function is to test the modification of the NMS threshold 
    on the output provide by the algo 
    And plot the zone considered as the best by the Faster RCNN 
    """ 
    classes  = CLASSESVOC[1:21]
    corr_voc_coco = [0,5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
    TEST_NMS = 0.7
    num_classes = 21
    thresh = 0.0
    #NETS_Pretrained = {'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'}
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
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
      net_TL = vgg16()
    elif demonet == 'res50':
      raise NotImplementedError
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
      net_TL = resnetv1(num_layers=101)
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
      net_TL = resnetv1(num_layers=152)
    elif demonet == 'mobile':
      raise NotImplementedError
    else:
      raise NotImplementedError
      
    # List des images a test 
    path_to_img = '/media/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
    path_to_img = '/media/HDD/output_exp/ClassifPaintings/im/'
    list_name_img = ['000001']
    list_name_img = ['medaille-charms-need-dog']
    nms_thresh = 0.7
    # First we test with a high threshold !!!
    plt.ion()
    plt.close('all')
    sess = tf.Session(config=tfconfig)
    net.create_architecture("TEST", nbClasses,
                          tag='default', anchor_scales=anchor_scales)

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    name_img = list_name_img[0]
    i=0
    name_img = list_name_img[i]
    complet_name = path_to_img + name_img + '.jpg'
    im = cv2.imread(complet_name)
    

    # Partie normale de detection :
    scores_classic, boxes_classic = im_detect(sess, net, im) # For COCO scores.shape = #boxes,81, boxes.shape = #boxes,4*81
    if 'COCO' in demonet:
        scores_classic = scores_classic[:,corr_voc_coco]
        boxes_tmp = np.zeros((len(scores_classic),21*4))
        for j in range(1, num_classes):
            j_tmp = corr_voc_coco[j]
            boxes_tmp[:,j*4:(j+1)*4] = boxes_classic[:,j_tmp*4:(j_tmp+1)*4]
        boxes_classic = boxes_tmp

    j_class = 15 # Person
    j_class = 12 # Dog
    CONF_THRESH = 0.3
    NMS_THRESH = 0.9
    argmax = [np.argmax(scores_classic[:,j_class])]
#    scores_classic = scores_classic[argmax,:]
    scores_class_classic = scores_classic[argmax,j_class]
    boxes_class_classic = boxes_classic[argmax,j_class*4:(j_class+1)*4] 
    print(scores_class_classic)
    cls = classes[j_class-1]
    cls_scores = scores_class_classic
    cls_boxes = boxes_class_classic
    print(cls_scores.shape)
    print(cls_boxes.shape)
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    print(dets.shape)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    vis_detections(im, cls, dets, thresh=0.5)
    plt.show()
    
    tf.reset_default_graph()
    sess.close()
    
    sess = tf.Session(config=tfconfig)
    net_TL.create_architecture("TEST", nbClasses,
                                  tag='default', anchor_scales=anchor_scales,
                                  modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    
    #print("Image shape",im.shape)
    cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = \
        TL_im_detect(sess, net_TL, im)  # This call net.TL_image 
        
    for dim in [2048,1024,800,600,512,248,224]:
        resized = cv2.resize(im, (dim,dim), interpolation = cv2.INTER_AREA)
        features_maps = net_TL.extract_head(sess, np.expand_dims(resized,axis=0))
        print(resized.shape)
        print(features_maps.shape)
        s = features_maps.shape[1]
        ratio = dim//s
        print('ratio',ratio)
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = \
        TL_im_detect(sess, net_TL, resized)
    # dans cette version du Faster RCNN toute les images sont redimensionnes a du 600
    # pour le plus petit cote et la feature map final a une dimension de (1, 38, 38, 1024)
    # Soit H/16 et W/16 pour une image de 600*600
    blobs, im_scales = get_blobs(im)
    roi =  rois[:,1:5] / im_scales[0]
    list_area = []
    for i in range(len(roi)):
        list_area +=[(roi[i,2] - roi[i,0])*(roi[i,3] - roi[i,1])]
    print('Min area of the rois : ',np.min(list_area),'list len',len(list_area))
    
    scores = np.reshape(cls_prob, [cls_prob.shape[0], -1])
    boxes = np.tile(roi,21)
    if 'COCO' in demonet:
        scores = scores[:,corr_voc_coco]
    # skip j = 0, because it's the background class
    scores_class_TL = scores[argmax,j_class]
    boxes_class_TL = boxes[argmax,j_class*4:(j_class+1)*4] 
    print(scores_class_TL)
    cls = classes[j_class-1]
    cls_scores = scores_class_TL
    cls_boxes = boxes_class_TL
    print(cls_scores.shape)
    print(cls_boxes.shape)
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    print(dets.shape)
    dets = dets[keep, :]
    #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    vis_detections(im, cls, dets, thresh=0.5)
    plt.show()
    #input('wait')        
    #plt.close('all')
    
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
    Cette fonction ne marche pas
    """
    print("Attention cette fonction ne fonctionne pas et je n'ai pas trouver le bug, il ne faut pas utiliser cette fonction mais plutot aller voir TL_MILSVM")
    raise NotImplemented # TODO remove this function !
    TestMode_ComparisonWithBestObjectScoreKeep = True
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    path = '/media/HDD/output_exp/ClassifPaintings/'
    database = 'Paintings'
    databasetxt =path + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
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
    DATA_DIR =  '/media/HDD/data/Fondazione_Zeri/Selection_Olivier/'
    output_DIR = '/media/HDD/output_exp/ClassifPaintings/Zeri/'
    pathlib.Path(output_DIR).mkdir(parents=True, exist_ok=True)
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
        
        CONF_THRESH = 0.75
        NMS_THRESH = 0.5 # non max suppression
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
        name_output = output_DIR + im_name_wt_ext + '_FasterRCNN.jpg'
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
#    Compute_Faster_RCNN_features(demonet='res152_COCO',nms_thresh = 0.7,database='VOC2007',
#                                 augmentation=False,L2 =False,
#                                 saved='all',verbose=True,filesave='tfrecords')   
#    Compute_Faster_RCNN_features(demonet='res152_COCO',nms_thresh = 0.7,database='watercolor',
#                                 augmentation=False,L2 =False,
#                                 saved='all',verbose=True,filesave='tfrecords')   
    
    # Test pour calculer les performances en prenant la moyenne des regions retournees par le réseau
#    run_FasterRCNN_Perf_Paintings(TL = True,reDo=False,feature_selection = 'meanObject',nms_thresh = 0.0)
#    run_FasterRCNN_Perf_Paintings(TL = True,reDo=False,feature_selection = 'MaxObject',
#                                  nms_thresh = 0.7,database='Paintings') # Pour calculer les performances sur les paintings de Crowley 
#    run_FasterRCNN_Perf_Paintings(TL = True,reDo=False,feature_selection = 'MaxObject',CV_Crowley=False,
#                                  nms_thresh = 0.7,database='Wikidata_Paintings_miniset_verif') # Pour calculer les performances sur les paintings de Crowley 
#    run_FasterRCNN_Perf_Paintings(TL = True,reDo=False,feature_selection = 'meanObject',CV_Crowley=False,
#                                  nms_thresh = 0.7,database='Wikidata_Paintings_miniset_verif') # Pour calculer les performances sur les paintings de Crowley 
#    
#    run_FRCNN_Detection_perf(database='VOC2007')
   run_FRCNN_Detection_perf(database='PeopleArt')
#    Illus_box_ratio()
#     Illus_box_ratio()