# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:22:51 2019

The goal of this script is to evaluate the

@author: gonthier
"""

from MILbenchmark.utils import getDataset,normalizeDataSetFull,getMeanPref,\
    getTest_and_Train_Sets,normalizeDataSetTrain,getClassifierPerfomance
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import tensorflow as tf
import pathlib
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # 1 to remove info, 2 to remove warning and 3 for all
import warnings
from trouver_classes_parmi_K import tf_MI_max

def evalPerf(dataset='Birds',dataNormalizationWhen=None,dataNormalization=None):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : dataset = Newsgroups, Bird or SIVAL
    """

    warnings.filterwarnings("ignore")
    Dataset=getDataset(dataset)

    list_names,bags,labels_bags,labels_instance = Dataset
    if dataNormalizationWhen=='onAllSet':
        bags = normalizeDataSetFull(bags,dataNormalization)

    dict_results = {}
    for c_i,c in enumerate(list_names):
        c_i = 12
        # Loop on the different class, we will consider each group one after the other
        print("For class :",c)
        labels_bags_c = labels_bags[c_i]
        labels_instance_c = labels_instance[c_i]
        if dataset=='Newsgroups':
            bags_c = bags[c_i]
        else:
            bags_c = bags
        D = bags_c,labels_bags_c,labels_instance_c

        perf,perfB=performExperimentWithCrossVal(D,dataset,
                                dataNormalizationWhen,dataNormalization,
                                GridSearch=False)
        mPerf = perf[0]
        stdPerf = perf[1]
        mPerfB = perfB[0]
        stdPerfB = perfB[1]
        ## Results
        print('=============================================================')
        print('-------------------------------------------------------------')
        print('- instances') # f1Score,UAR,aucScore,accuracyScore
        print('AUC: ',mPerf[2],' +/- ',stdPerf[2])
        print('UAR: ',mPerf[1],' +/- ',stdPerf[1])
        print('F1: ',mPerf[0],' +/- ',stdPerf[0])
        print('Accuracy: ',mPerf[3],' +/- ',stdPerf[3])
        print('- bags')
        print('AUC: ',mPerfB[2],' +/- ',stdPerfB[2])
        print('UAR: ',mPerfB[1],' +/- ',stdPerfB[1])
        print('F1: ',mPerfB[0],' +/- ',stdPerfB[0])
        print('Accuracy: ',mPerfB[3],' +/- ',stdPerfB[3])
        print('-------------------------------------------------------------')


def performExperimentWithCrossVal(D,dataset,dataNormalizationWhen,
                                  dataNormalization,GridSearch=False):

    bags,labels_bags_c,labels_instance_c  = D

    StratifiedFold = True

    nRep=10
    nFolds=10
    numMetric = 4

    size_biggest_bag = 0
    for elt in bags:
        size_biggest_bag = max(size_biggest_bag,len(elt))
    num_features = bags[0].shape[1]
    mini_batch_size_max = 2000
    opts = dataset,mini_batch_size_max,num_features,size_biggest_bag
    if dataset=='SIVAL':
        num_sample = 5
        perfObj=np.empty((num_sample,nRep,nFolds,numMetric))
        perfObjB=np.empty((num_sample,nRep,nFolds,numMetric))
        for k in range(num_sample):
            labels_bags_c_k = labels_bags_c[k]
            labels_instance_c_k = labels_instance_c[k]
            bags_k = bags[k]
            perfObj_k,perfObjB_k = doCrossVal(nRep,nFolds,numMetric,bags_k,labels_bags_c_k,labels_instance_c_k,StratifiedFold,opts)
            perfObj[k,:,:,:] = perfObj_k
            perfObjB[k,:,:,:] = perfObjB_k
    else:
        perfObj,perfObjB = doCrossVal(nRep,nFolds,numMetric,bags,labels_bags_c,labels_instance_c,StratifiedFold,opts)

    perf=getMeanPref(perfObj,dataset)
    perfB=getMeanPref(perfObjB,dataset)
    return(perf,perfB)

def doCrossVal(nRep,nFolds,numMetric,bags,labels_bags_c,labels_instance_c,StratifiedFold,opts):
    """
    This function perform a cross validation evaluation or StratifiedFold cross 
    evaluation
    """
    dataset,mini_batch_size_max,num_features,size_biggest_bag = opts 
    perfObj=np.empty((nRep,nFolds,numMetric))
    perfObjB=np.empty((nRep,nFolds,numMetric))
    for r in range(nRep):
        # Creation of nFolds splits
        # 
        #Use a StratifiedKFold to get the same distribution in positive and negative class in the train and test set
        if StratifiedFold:
            skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=r)
            fold = 0
            for train_index, test_index in skf.split(bags,labels_bags_c):
                labels_bags_c_train, labels_bags_c_test = \
                    getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                bags_train, bags_test = \
                    getTest_and_Train_Sets(bags,train_index,test_index)
                _ , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
                gt_instances_labels_stack = np.hstack(labels_instance_c_test)
                mini_batch_size = min(mini_batch_size_max,len(bags_train))
            
                #Training
                data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                                   num_features,'train',dataset)
                export_dir=trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                                      size_biggest_bag,num_features,mini_batch_size)

                # Testing
                data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,size_biggest_bag,num_features,'test',dataset)
                pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,data_path_test,bags_test,\
                                                                   size_biggest_bag,num_features,mini_batch_size)

                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
                fold += 1
        else:
            kf = KFold(n_splits=nFolds, shuffle=True, random_state=r)
            fold = 0
            for train_index, test_index in kf.split(labels_bags_c):
                labels_bags_c_train, labels_bags_c_test = \
                    getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                bags_train, bags_test = \
                    getTest_and_Train_Sets(bags,train_index,test_index)
                _ , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
                gt_instances_labels_stack = np.hstack(labels_instance_c_test)
                mini_batch_size = min(mini_batch_size_max,len(bags_train))

                #Training
                data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                                   num_features,'train',dataset)
                export_dir=trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                                      size_biggest_bag,num_features,mini_batch_size)

                # Testing
                data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,size_biggest_bag,num_features,'test',dataset)
                pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,data_path_test,bags_test,\
                                                                   size_biggest_bag,num_features,mini_batch_size)
                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
                fold += 1
    return(perfObj,perfObjB)

def parser(record,num_rois=300,num_features=2048):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'num_features':  tf.FixedLenFeature([], tf.int64),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([1],tf.float32)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
#    label = tf.slice(label,[classe_index],[1])
#    label = tf.squeeze(label) # To get a vector one dimension
    fc7 = parsed['fc7']
    fc7 = tf.reshape(fc7, [num_rois,num_features])         
    return fc7,label

def predict_MIMAX(export_dir,data_path_test,bags_test,size_biggest_bag,num_features,\
               mini_batch_size=256,verbose=False):
    """
    This function as to goal to predict on the test set
    """
    
      # To avoid the INFO and WARNING message
    predict_label_all_test = []
    head,tail = os.path.split(export_dir)
    export_dir_path = os.path.join(head)
    name_model_meta = export_dir + '.meta'

    test_dataset = tf.data.TFRecordDataset(data_path_test)
    test_dataset = test_dataset.map(lambda r: parser(r,num_rois=size_biggest_bag,num_features=num_features))
    dataset_batch = test_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()

    # Parameters of the classifier
    with_tanh = True
    with_tanh_alreadyApplied = False
    with_softmax = False
    with_softmax_a_intraining = False
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(name_model_meta)
        new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
        graph= tf.get_default_graph()
        
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        if with_tanh_alreadyApplied:
            try:
                Prod_best = graph.get_tensor_by_name("Tanh:0")
            except KeyError:
                try:
                     Prod_best = graph.get_tensor_by_name("Tanh_2:0")
                except KeyError:
                     Prod_best = graph.get_tensor_by_name("Tanh_1:0")
        else:
            Prod_best = graph.get_tensor_by_name("Prod:0")
        if with_tanh:
            if verbose: print('use of tanh')
            Tanh = tf.tanh(Prod_best)
            mei = tf.argmax(Tanh,axis=2)
            score_mei = tf.reduce_max(Tanh,axis=2)
        elif with_softmax:
            Softmax = tf.nn.softmax(Prod_best,axis=-1)
            mei = tf.argmax(Softmax,axis=2)
            score_mei = tf.reduce_max(Softmax,axis=2)
        elif with_softmax_a_intraining:
            Softmax=tf.multiply(tf.nn.softmax(Prod_best,axis=-1),Prod_best)
            mei = tf.argmax(Softmax,axis=2)
            score_mei = tf.reduce_max(Softmax,axis=2)
        else:
            if verbose: print('Tanh in testing time',Prod_best)
            mei = tf.argmax(Prod_best,axis=-1)
            score_mei = tf.reduce_max(Prod_best,axis=-1)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Evaluation Test
        while True:
            try:
                fc7s,labels = sess.run(next_element)
#                print(sess.run(next_element))
                if with_tanh:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Tanh], feed_dict={X: fc7s, y: labels})
                elif with_softmax or with_softmax_a_intraining:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Softmax], feed_dict={X: fc7s, y: labels})
                else:
                    PositiveExScoreAll = \
                    sess.run(Prod_best, feed_dict={X: fc7s, y: labels})
                for elt_i in range(PositiveExScoreAll.shape[1]):
                    predict_label_all_test += [PositiveExScoreAll[0,elt_i,:]]
            except tf.errors.OutOfRangeError:
                break
    tf.reset_default_graph()
    # Post processing
    instances_labels_pred = []
    bags_labels_pred = []
    for elt,predictions in zip(bags_test,predict_label_all_test):
        length_bag = elt.shape[0]
        np_predictions = np.array(predictions)
        instances_labels_pred += [np_predictions[0:length_bag]]
        bags_labels_pred += [np.max(np_predictions[0:length_bag])]

    bags_labels_pred=(np.hstack(bags_labels_pred))
    instances_labels_pred=(np.hstack(instances_labels_pred))
    
    if verbose:
        print('The model folder will be removed',export_dir_path)

    shutil.rmtree(export_dir_path)
    
    return(bags_labels_pred,instances_labels_pred)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def Create_tfrecords(bags, labels_bags,size_biggest_bag,num_features,nameset,dataset):
    """
    This function create a tfrecords from a list of bags and a list of bag labels
    """
    name = 'MIL' +dataset + nameset+'.tfrecords'
    directory = 'tmp'
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
        
    path_name = os.path.join(directory,name)
    if os.path.isfile(path_name):
        os.remove(path_name) # Need to remove the folder !
    writer = tf.python_io.TFRecordWriter(path_name)
    for elt,elt_label in zip(bags,labels_bags):
        number_repeat = size_biggest_bag // len(elt)  +1
        f_repeat = np.repeat(elt,number_repeat,axis=0)
        fc7 = f_repeat[0:size_biggest_bag,:]
        features=tf.train.Features(feature={
                        'num_features': _int64_feature(num_features),
                        #'fc7': _bytes_feature(tf.compat.as_bytes(fc7.tostring())), 
                        'fc7': _floats_feature(fc7), 
                        'label' : _floats_feature(elt_label)})   
        example = tf.train.Example(features=features) #.tostring()
        writer.write(example.SerializeToString())
    writer.close()

    return(path_name)

def trainMIMAX(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
               num_features,mini_batch_size):
    """
    This function train a tidy MIMAX model
    """
    path_model = os.path.join('tmp','MI_max')
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True) 

    tf.reset_default_graph()
    classifierMI_max = tf_MI_max(restarts=11,is_betweenMinus1and1=True, \
                                 num_rois=size_biggest_bag,num_classes=1, \
                                 num_features=num_features,mini_batch_size=mini_batch_size, \
                                 verbose=False)
    export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=-1,shuffle=False,restarts_paral='paral', \
                       WR=True)
    tf.reset_default_graph()
    return(export_dir)

if __name__ == '__main__':
#    evalPerf(dataset='Birds')
#    evalPerf(dataset='Newsgroups')
    evalPerf(dataset='SIVAL')
