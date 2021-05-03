#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:42:55 2018

@author: gonthier
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
    ,'vgg16_coco': ('/media/gonthier/HDD/models/tf-faster-rcnn/vgg16/vgg16_faster_rcnn_iter_1190000.ckpt',)    
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

path_to_model = '/media/gonthier/HDD/models/tf-faster-rcnn/'

def FasterRCNN_demo():
    """
    Demo code
    """
    DATA_DIR =  'data/'
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
    im_name = 'loulou.jpg'
    print('Demo for data/demo/{}'.format(im_name))
    imfile = os.path.join(DATA_DIR, im_name)
    im = cv2.imread(imfile)
    scores, boxes = im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
   # Only single-image batch implemented !
    print('scores.shape',scores.shape)
    #print(scores)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.5 # non max suppression
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
      
def compute_FasterRCNN_feature_TransferLearning():
    """
    Compute the fc7 of the FasterRCNN
    """
    path_to_img = 'data/'
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
        path_to_model = '/media/gonthier/HDD/models/tf-faster-rcnn/'
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
        path_data = 'data/'
        listIm = ['loulou.jpg']
        print('Start computing image region proposal')
        if demonet == 'vgg16_COCO':
            size_output = 4096
        elif demonet == 'res101_COCO' or demonet == 'res152_COCO' :
            size_output = 2048

        # Use the output of fc7 
        net.create_architecture("TEST", nbClasses,
                              tag='default', anchor_scales=anchor_scales,modeTL= True)
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)

        for i,name_img in  enumerate(listIm):
            if i%1000==0:
                print(i,name_img)
            complet_name = path_to_img + name_img
            im = cv2.imread(complet_name)
            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            print(fc7.shape)
            # The fc7 features map of the region of interest are return by the function TL_im_detect
    
if __name__ == '__main__':
    #FasterRCNN_demo()
    compute_FasterRCNN_feature_TransferLearning()