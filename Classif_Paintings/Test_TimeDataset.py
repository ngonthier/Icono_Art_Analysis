#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:03:46 2018

@author: gonthier
"""

import tensorflow as tf
import numpy as np
import os, time, multiprocessing
import matplotlib.pyplot as plt

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def parser(record):
    num_features = 2000
    size_group = 300
    num_classes= 10
    class_indice = 0
    keys_to_features={
                'X': tf.FixedLenFeature([size_group*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_classes],tf.float32)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    label = parsed['label']
    label = tf.slice(label,[class_indice],[1])
    label = tf.squeeze(label) # To get a vector one dimension
    X = parsed['X']
    X= tf.reshape(X, [size_group,num_features])
    return X, label


def test_suffle_parse():
    # Definition of the size 
    num_features = 2000
    num_ex = 2000
    size_group = 300
    num_classes = 10
    batch_size= 480
    max_iters = 2
    buffer_size = 10000
    
    # Creation of the Dataset 
    filename_tfrecords = 'tmp.tfrecords'
    if not(os.path.isfile(filename_tfrecords)): # If the file doesn't exist we will create it
        print("Start creating the Dataset")
        writer = tf.python_io.TFRecordWriter(filename_tfrecords)
        
        for i in range(num_ex):
            if i % 1000 == 0: print("Step :",i)
            X = np.random.normal(size=(size_group,num_features))
            vectors =  2*np.random.randint(0,2,(num_classes,1))-1
            features=tf.train.Features(feature={
                        'X': _floats_feature(X),
                        'label' : _floats_feature(vectors)})
            example = tf.train.Example(features=features)       
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print("The dataset tfrecords already exist")
     
    train_dataset = tf.data.TFRecordDataset(filename_tfrecords)
    num_proc = multiprocessing.cpu_count()
    train_dataset = train_dataset.map(parser,
                                        num_parallel_calls=num_proc)
    dataset_shuffle = train_dataset.shuffle(buffer_size=buffer_size, \
                                            reshuffle_each_iteration=True,seed=1) 
    dataset_shuffle = dataset_shuffle.batch(batch_size)
    dataset_shuffle = dataset_shuffle.cache()
    dataset_shuffle = dataset_shuffle.repeat() 
    dataset_shuffle = dataset_shuffle.prefetch(10*batch_size) 
    shuffle_iterator = dataset_shuffle.make_initializable_iterator()
    X_, y_ = shuffle_iterator.get_next()
    with tf.device('/gpu:0'):
        W=tf.Variable(tf.random_normal([num_features], stddev=1.),name="weights")
        W=tf.reshape(W,(1,1,num_features))
        Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)
        Max=tf.reduce_max(Prod,axis=1)
        Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),y_))
        loss= tf.add(Tan,tf.reduce_sum(tf.multiply(W,W)))
    
        LR = 0.01
        restarts = 0
        optimizer = tf.train.GradientDescentOptimizer(LR) 
        config = tf.ConfigProto()
    #    config.intra_op_parallelism_threads = 16
    #    config.inter_op_parallelism_threads = 16
        config.gpu_options.allow_growth = True
        train = optimizer.minimize(loss)  
        print("The graph is defined")
        sess = tf.Session(config=config)
            
        durationTab = []
        
        for essai in range(restarts+1):
            # To do need to reinitialiszed
            t0 = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(shuffle_iterator.initializer)
            t1 = time.time()
            duration = t1 - t0
            print('Duration of initialization : ',duration)
            for step in range(max_iters):
                t0 = time.time()
                print(sess.run(y_))
                t1 = time.time()
                duration = t1 - t0
                print("Step ",str(step),' duration : ',duration)
                durationTab += [duration]
                
        sess.close()
    
    buffer_size=1
    train_dataset = tf.data.TFRecordDataset(filename_tfrecords)
    num_proc = multiprocessing.cpu_count()
    train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
                    map_func=parser, batch_size=batch_size,
                    num_parallel_batches=num_proc,drop_remainder=False))
    dataset_shuffle = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size,seed=1))
    dataset_shuffle = dataset_shuffle.prefetch(10*batch_size) 
    shuffle_iterator = dataset_shuffle.make_initializable_iterator()
    X_, y_ = shuffle_iterator.get_next()
    with tf.device('/gpu:0'):
        W=tf.Variable(tf.random_normal([num_features], stddev=1.),name="weights")
        W=tf.reshape(W,(1,1,num_features))
        Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)
        Max=tf.reduce_max(Prod,axis=1)
        Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),y_))
        loss= tf.add(Tan,tf.reduce_sum(tf.multiply(W,W)))
    
        LR = 0.01
        restarts = 0
        optimizer = tf.train.GradientDescentOptimizer(LR) 
        config = tf.ConfigProto()
    #    config.intra_op_parallelism_threads = 16
    #    config.inter_op_parallelism_threads = 16
        config.gpu_options.allow_growth = True
        train = optimizer.minimize(loss)  
        print("The graph is defined")
        sess = tf.Session(config=config)
            
        durationTab = []
        
        for essai in range(restarts+1):
            # To do need to reinitialiszed
            t0 = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(shuffle_iterator.initializer)
            t1 = time.time()
            duration = t1 - t0
            print('Duration of initialization : ',duration)
            for step in range(max_iters):
                t0 = time.time()
                print(sess.run(y_))
                t1 = time.time()
                duration = t1 - t0
                print("Step ",str(step),' duration : ',duration)
                durationTab += [duration]
                
        sess.close()

def test_train_w_dataset():

    # Definition of the size 
    num_features = 2000
    num_ex = 2000
    size_group = 300
    num_classes = 10
    batch_size= 480
    max_iters = 300
    buffer_size = 10000
    
    # Creation of the Dataset 
    filename_tfrecords = 'tmp.tfrecords'
    if not(os.path.isfile(filename_tfrecords)): # If the file doesn't exist we will create it
        print("Start creating the Dataset")
        writer = tf.python_io.TFRecordWriter(filename_tfrecords)
        
        for i in range(num_ex):
            if i % 1000 == 0: print("Step :",i)
            X = np.random.normal(size=(size_group,num_features))
            vectors =  2*np.random.randint(0,2,(num_classes,1))-1
            features=tf.train.Features(feature={
                        'X': _floats_feature(X),
                        'label' : _floats_feature(vectors)})
            example = tf.train.Example(features=features)       
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print("The dataset tfrecords already exist")
     
    train_dataset = tf.data.TFRecordDataset(filename_tfrecords)
    num_proc = multiprocessing.cpu_count()
    train_dataset = train_dataset.map(parser,
                                        num_parallel_calls=num_proc)
    dataset_shuffle = train_dataset.shuffle(buffer_size=buffer_size,
                                                 reshuffle_each_iteration=True) 
    dataset_shuffle = dataset_shuffle.batch(batch_size)
    dataset_shuffle = dataset_shuffle.cache()
    dataset_shuffle = dataset_shuffle.repeat() 
    dataset_shuffle = dataset_shuffle.prefetch(10*batch_size) 
    shuffle_iterator = dataset_shuffle.make_initializable_iterator()
    X_, y_ = shuffle_iterator.get_next()
    with tf.device('/gpu:0'):
        W=tf.Variable(tf.random_normal([num_features], stddev=1.),name="weights")
        W=tf.reshape(W,(1,1,num_features))
        Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)
        Max=tf.reduce_max(Prod,axis=1)
        Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),y_))
        loss= tf.add(Tan,tf.reduce_sum(tf.multiply(W,W)))
    
        LR = 0.01
        restarts = 1
        optimizer = tf.train.GradientDescentOptimizer(LR) 
        config = tf.ConfigProto()
    #    config.intra_op_parallelism_threads = 16
    #    config.inter_op_parallelism_threads = 16
        config.gpu_options.allow_growth = True
        train = optimizer.minimize(loss)  
        print("The graph is defined")
        sess = tf.Session(config=config)
            
        durationTab = []
        
        for essai in range(restarts+1):
            # To do need to reinitialiszed
            t0 = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(shuffle_iterator.initializer)
            t1 = time.time()
            duration = t1 - t0
            print('Duration of initialization : ',duration)
            for step in range(max_iters):
                t0 = time.time()
                sess.run(train)
                t1 = time.time()
                duration = t1 - t0
                print("Step ",str(step),' duration : ',duration)
                durationTab += [duration]
                
        sess.close()
#        plt.plot(durationTab)
#        plt.ylabel('Duration')
#        plt.xlabel('Iteration')
#        plt.show()

if __name__ == '__main__':
#    test_train_w_dataset()
    test_suffle_parse()