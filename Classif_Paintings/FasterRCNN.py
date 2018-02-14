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
import matplotlib
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import SGDClassifier
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
from tf_faster_rcnn.lib.model.test import im_detect,TL_im_detect,TL_im_detect_end
from tf_faster_rcnn.lib.model.nms_wrapper import nms
import matplotlib.pyplot as plt
#from tf_faster_rcnn.tools.demo import vis_detections
import numpy as np
import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score
from Classifier_Evaluation import Classification_evaluation
import os.path

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

def compute_FasterRCNN_Perf_Paintings(TL = True):
    """
    Compute the performance on the Your Paintings subset ie Crowley on the output but also the best case on feature fc7 of the best proposal part
    """
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
            path_data = 'data/'
            N = 1
            extL2 = ''
            name_pkl = path_data+'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'.pkl'
            if demonet == 'vgg16_COCO':
                size_output = 4096
            elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
                size_output = 2048
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
            
            with open(name_pkl, 'wb') as pkl:
                pickle.dump(Data,pkl)
            
            sess.close()
            
            # Compute the metric
            Classification_evaluation(kind=demonet,kindnetwork='FasterRCNN',database='Paintings',L2=False,augmentation=False,classifier_name='LinearSVM')       
        
def compute_FasterRCNN_demo():
    
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
    
def FasterRCNN_TransferLearning():
    """
    Compute the performance on the Your Paintings subset ie Crowley
    on the fc7 output but with an outlier detection vision
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
    
    #input('WAit for end')
    
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
    
    #compute_FasterRCNN_Perf_Paintings()
    FasterRCNN_TransferLearning()
    #FasterRCNN_ImagesObject()
    #compute_FasterRCNN_demo()
    #compute_FasterRCNN_Perf_Paintings()
    # List des nets a tester : VGG16-VOC12
    #  VGG16-VOC07
    # RESNET152 sur COCO
    # VGG16 sur COCO
    # RES101 sur VOC12