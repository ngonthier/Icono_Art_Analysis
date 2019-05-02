#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:57:59 2019

@author: gonthier
"""

import tensorflow as tf

def parser_w_rois_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            with_rois_scores=False):
        # Perform additional preprocessing on the parsed data.
        if not(with_rois_scores):
            keys_to_features={
                        'rois': tf.FixedLenFeature([num_rois*5],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}
        else:
            keys_to_features={
                        'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                        'rois': tf.FixedLenFeature([num_rois*5],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}

#        keys_to_features={
#                    'height': tf.FixedLenFeature([], tf.int64),
#                    'width': tf.FixedLenFeature([], tf.int64),
#                    'num_regions':  tf.FixedLenFeature([], tf.int64),
#                    'num_features':  tf.FixedLenFeature([], tf.int64),
#                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
#                    'rois': tf.FixedLenFeature([5*num_rois],tf.float32),
#                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
#                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
#                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
#                    'name_img' : tf.FixedLenFeature([],tf.string)}
        print('record',record)
        print('keys_to_features',keys_to_features)
        parsed = tf.parse_single_example(record, keys_to_features)
        print('parsed',parsed)
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [num_rois,num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [num_rois,5])    
        if not(with_rois_scores):
            return fc7,rois, label,name_img
        else:
            roi_scores = parsed['roi_scores']
            return fc7,rois,roi_scores,label,name_img
        
k_per_bag = 300
get_roisScore = False
num_classes = 7
num_features = 2048
path = '/media/gonthier/HDD/output_exp/ClassifPaintings'
name = 'EdgeBoxes_res152_IconArt_v1_N1_TLforMIL_nms_0.7_all_train.tfrecords'
#name = 'FasterRCNN_res152_COCO_IconArt_v1_N1_TLforMIL_nms_0.7_all_train.tfrecords'
record = path + '/' +name
train_dataset = tf.data.TFRecordDataset(record)
print('train_dataset',train_dataset)
train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
    num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,num_rois=k_per_bag))
mini_batch_size = 10
print('train_dataset',train_dataset)
dataset_batch = train_dataset.batch(mini_batch_size)
dataset_batch.cache()
print(dataset_batch)
iterator = dataset_batch.make_one_shot_iterator()
print(iterator)
next_element = iterator.get_next()
sess = tf.Session()
print(next_element)
nx = sess.run(next_element)
print(nx)

#train_dataset <TFRecordDataset shapes: (), types: tf.string>
#record Tensor("arg0:0", shape=(), dtype=string)
#keys_to_features {'rois': FixedLenFeature(shape=[1500], dtype=tf.float32, default_value=None), 'fc7': FixedLenFeature(shape=[614400], dtype=tf.float32, default_value=None), 'label': FixedLenFeature(shape=[7], dtype=tf.float32, default_value=None), 'name_img': FixedLenFeature(shape=[], dtype=tf.string, default_value=None)}
#parsed {'fc7': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=(614400,) dtype=float32>, 'label': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=(7,) dtype=float32>, 'name_img': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=() dtype=string>, 'rois': <tf.Tensor 'ParseSingleExample/ParseSingleExample:3' shape=(1500,) dtype=float32>}
#train_dataset <MapDataset shapes: ((300, 2048), (300, 5), (7,), ()), types: (tf.float32, tf.float32, tf.float32, tf.string)>
#<BatchDataset shapes: ((?, 300, 2048), (?, 300, 5), (?, 7), (?,)), types: (tf.float32, tf.float32, tf.float32, tf.string)>
#<tensorflow.python.data.ops.iterator_ops.Iterator object at 0x7f05683f9e48>
#(<tf.Tensor 'IteratorGetNext_4:0' shape=(?, 300, 2048) dtype=float32>, <tf.Tensor 'IteratorGetNext_4:1' shape=(?, 300, 5) dtype=float32>, <tf.Tensor 'IteratorGetNext_4:2' shape=(?, 7) dtype=float32>, <tf.Tensor 'IteratorGetNext_4:3' shape=(?,) dtype=string>)
