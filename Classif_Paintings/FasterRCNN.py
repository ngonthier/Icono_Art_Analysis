#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:42:41 2017

Based on the Tensorflow implementation of Faster-RCNNN 
https://github.com/endernewton/tf-faster-rcnn

Be careful it was a necessity to modify all the script of the library with stuff 
like ..lib etc
It is a convertion for Python 3


@author: gonthier
"""

import tensorflow as tf
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
from tf_faster_rcnn.lib.model.test import im_detect
from tf_faster_rcnn.lib.model.nms_wrapper import nms
import numpy as np
import os,cv2
import pandas as pd
from sklearn.metrics import average_precision_score

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

def compute_FasterRCNN_Perf_Paintings():
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    database = 'Paintings'
    databasetxt = database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    sLength = len(df_test['name_img'])
    name_img = df_test['name_img'][0]
    i = 0
    y_test = np.zeros((sLength,10))

    NETS_Pretrained = {'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt',
                   'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'
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
            anchor_scales = [4, 8, 16, 32]
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
            

    
if __name__ == '__main__':
    #compute_FasterRCNN_demo()
    compute_FasterRCNN_Perf_Paintings()
    # List des nets a tester : VGG16-VOC12
    #  VGG16-VOC07
    # RESNET152 sur COCO
    # VGG16 sur COCO
    # RES101 sur VOC12