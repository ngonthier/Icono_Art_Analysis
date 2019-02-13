# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:22:51 2019

The goal of this script is to evaluate the

@author: gonthier
"""

import os
import warnings
warnings.filterwarnings("ignore")

from MILbenchmark.utils import getDataset,normalizeDataSetFull,getMeanPref,\
    getTest_and_Train_Sets,normalizeDataSetTrain,getClassifierPerfomance
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import pathlib
import shutil
import sys
import misvm
from MILbenchmark.mialgo import sisvm,MIbyOneClassSVM,sixgboost

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 1 to remove info, 2 to remove warning and 3 for all
import tensorflow as tf

from trouver_classes_parmi_K import tf_MI_max
from trouver_classes_parmi_K_mi import tf_mi_model
import pickle


list_of_ClassicalMI = ['miSVM','SIL','SISVM','LinearSISVM','MIbyOneClassSVM',\
                       'SIXGBoost']

def EvaluationOnALot_ofParameters(dataset):
    """
    List of the parameters that can be used to get better results maybe on the 
    MILbenchmark
    """
    
    # List of the parameter that can improve the resultats
    C_tab = np.logspace(start=-3,stop=3,num=7)
    C_Searching_tab = [True,False]
    CV_Mode_tab = ['','CVforCsearch']
    restarts_tab = [0,11,99]
    LR_tab = np.logspace(-5,1,7)
    dataNormalization_tab = ['std','var','0-1']
    
    # defaults case
    default_C = 1.
    default_LR= 0.01
    default_C_Searching = False
    default_CV_Mode = ''
    default_restarts = 11
    default_dataNormalization = None
    opts_MIMAX = default_C,default_C_Searching,default_CV_Mode,default_restarts,default_LR
    pref_name_case = '_defaultMode'
    evalPerf(method='MIMAX',dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
    
    for dataNormalization in dataNormalization_tab:
        pref_name_case = '_Nor'+dataNormalization
        evalPerf(method='MIMAX',dataset=dataset,dataNormalizationWhen='onTrainSet',dataNormalization=dataNormalization,
                 reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
    
    
#    for C in C_tab:
#        if not(C==default_C):
#            opts_MIMAX = C,default_C_Searching,default_CV_Mode,default_restarts,default_LR
#            pref_name_case = 'MIMAX_C'+str(C) 
#            evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#                 reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
#        
#    for restarts in restarts_tab:
#        opts_MIMAX = default_C,default_C_Searching,default_CV_Mode,restarts,default_LR
#        pref_name_case = 'MIMAX_r'+str(restarts) 
#        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
#        
#    for C_Searching in C_Searching_tab:
#        opts_MIMAX = default_C,C_Searching,default_CV_Mode,default_restarts,default_LR
#        pref_name_case = 'MIMAX_r'+str(restarts) 
#        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
#        
#    for CV_Mode in CV_Mode_tab:
#        opts_MIMAX = default_C,default_C_Searching,CV_Mode,default_restarts,default_LR
#        pref_name_case = 'MIMAX_r'+str(CV_Mode) 
#        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
#        
#    for LR in LR_tab:
#        opts_MIMAX = default_C,default_C_Searching,CV_Mode,default_restarts,LR
#        pref_name_case = 'MIMAX_LR'+str(LR) 
#        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
#        
        
    


def evalPerf(method='MIMAX',dataset='Birds',dataNormalizationWhen=None,dataNormalization=None,
             reDo=False,opts_MIMAX=None,pref_name_case='',verbose=False):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM 
    @param : dataset = Newsgroups, Bird or SIVAL
    @param : dataNormalizationWhen : moment of the normalization of the data, 
        None = no normalization, onAllSet doing on all the set, onTrainSet 
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    """

    if verbose: print('Start evaluation performance on ',dataset,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = '_'+pref_name_case
    filename = method + '_' + dataset + pref_name_case + '.pkl'
    path_file_results = os.path.join(script_dir,'MILbenchmark','Results')
    file_results = os.path.join(path_file_results,filename)
    pathlib.Path(path_file_results).mkdir(parents=True, exist_ok=True) # creation of the folder if needed
    if reDo:
        results = {}
    else:
        try:
            results = pickle.load(open(file_results,'br'))
        except FileNotFoundError:
            results = {}
            
    Dataset=getDataset(dataset)
    list_names,bags,labels_bags,labels_instance = Dataset
              
    for c_i,c in enumerate(list_names):
        if not(c in results.keys()):
            # Loop on the different class, we will consider each group one after the other
            if verbose: print("Start evaluation for class :",c)
            labels_bags_c = labels_bags[c_i]
            labels_instance_c = labels_instance[c_i]
            if dataset in ['Newsgroups','SIVAL']:
                bags_c = bags[c_i]
            else:
                bags_c = bags
                
            if dataNormalizationWhen=='onAllSet':
                bags_c = normalizeDataSetFull(bags_c,dataNormalization)
                
            D = bags_c,labels_bags_c,labels_instance_c
    
            perf,perfB=performExperimentWithCrossVal(method,D,dataset,
                                    dataNormalizationWhen,dataNormalization,
                                    GridSearch=False,opts_MIMAX=opts_MIMAX,
                                    verbose=verbose)
            mPerf = perf[0]
            stdPerf = perf[1]
            mPerfB = perfB[0]
            stdPerfB = perfB[1]
            ## Results
            print('=============================================================')
            print("For class :",c)
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
            results[c] = [perf,perfB]
        pickle.dump(results,open(file_results,'bw'))

def performExperimentWithCrossVal(method,D,dataset,dataNormalizationWhen,
                                  dataNormalization,GridSearch=False,opts_MIMAX=None,
                                  verbose=False):

    bags,labels_bags_c,labels_instance_c  = D

    StratifiedFold = True
    if verbose and StratifiedFold: print('Use of the StratifiedFold cross validation')
    nRep=10
    nFolds=10
    numMetric = 4

    size_biggest_bag = 0
    for elt in bags:
        size_biggest_bag = max(size_biggest_bag,len(elt))
    if dataset=='SIVAL':
        num_features = bags[0][0].shape[1]
    else:
        num_features = bags[0].shape[1]
        
    mini_batch_size_max = 2000 # Maybe this value can be update depending on your GPU memory size
    opts = dataset,mini_batch_size_max,num_features,size_biggest_bag
    if dataset=='SIVAL':
        num_sample = 5
        perfObj=np.empty((num_sample,nRep,nFolds,numMetric))
        perfObjB=np.empty((num_sample,nRep,nFolds,numMetric))
        for k in range(num_sample):
            labels_bags_c_k = labels_bags_c[k]
            labels_instance_c_k = labels_instance_c[k]
            bags_k = bags[k]
            perfObj_k,perfObjB_k = doCrossVal(method,nRep,nFolds,numMetric,bags_k,\
                                              labels_bags_c_k,labels_instance_c_k,\
                                              StratifiedFold,opts,dataNormalizationWhen,\
                                              dataNormalization,opts_MIMAX=opts_MIMAX,\
                                              verbose=verbose)
            perfObj[k,:,:,:] = perfObj_k
            perfObjB[k,:,:,:] = perfObjB_k
    else:
        perfObj,perfObjB = doCrossVal(method,nRep,nFolds,numMetric,bags,labels_bags_c,\
                                      labels_instance_c,StratifiedFold,opts\
                                      ,dataNormalizationWhen,dataNormalization,\
                                      opts_MIMAX=opts_MIMAX,verbose=verbose)

    perf=getMeanPref(perfObj,dataset)
    perfB=getMeanPref(perfObjB,dataset)
    return(perf,perfB)

def doCrossVal(method,nRep,nFolds,numMetric,bags,labels_bags_c,labels_instance_c,StratifiedFold,opts,
               dataNormalizationWhen,dataNormalization,opts_MIMAX=None,verbose=False):
    """
    This function perform a cross validation evaluation or StratifiedFold cross 
    evaluation
    """
     
    perfObj=np.empty((nRep,nFolds,numMetric))
    perfObjB=np.empty((nRep,nFolds,numMetric))
    for r in range(nRep):
        # Creation of nFolds splits
        #Use a StratifiedKFold to get the same distribution in positive and negative class in the train and test set
        if StratifiedFold:
            
            skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=r)
            fold = 0

            for train_index, test_index in skf.split(bags,labels_bags_c):
                if verbose:
                    sys.stdout.write('Rep number : {:d}/{:d}, Fold number : {:d}/{:d} \r' \
                                     .format(r, nRep, fold, nFolds))
                    sys.stdout.flush()
                labels_bags_c_train, labels_bags_c_test = \
                    getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                bags_train, bags_test = \
                    getTest_and_Train_Sets(bags,train_index,test_index)
                _ , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
                    
                if dataNormalizationWhen=='onTrainSet':
                    bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)              
                    
                gt_instances_labels_stack = np.hstack(labels_instance_c_test)
                
                
                pred_bag_labels, pred_instance_labels =train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=opts_MIMAX)
 
                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
                fold += 1
        else:
            kf = KFold(n_splits=nFolds, shuffle=True, random_state=r)
            fold = 0
            for train_index, test_index in kf.split(labels_bags_c):
                if verbose: print('Fold number : ',fold,'over ',nFolds)
                labels_bags_c_train, labels_bags_c_test = \
                    getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                bags_train, bags_test = \
                    getTest_and_Train_Sets(bags,train_index,test_index)
                _ , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
                
                if dataNormalizationWhen=='onTrainSet':
                    bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization) 
                    
                gt_instances_labels_stack = np.hstack(labels_instance_c_test)
                
                pred_bag_labels, pred_instance_labels = train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=opts_MIMAX)
                
                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
                fold += 1
    return(perfObj,perfObjB)

def get_classicalMILclassifier(method):
    if method=='miSVM':
        classifier = misvm.miSVM(verbose=False)
    elif method=='SIL':
        classifier = misvm.SIL(verbose=False)
    elif method=='SISVM':
        classifier = sisvm.SISVM(verbose=False)
    elif method=='LinearSISVM':
        classifier = sisvm.LinearSISVM(verbose=False)
    elif method=='MIbyOneClassSVM':
        classifier = MIbyOneClassSVM.MIbyOneClassSVM(verbose=False)
    elif method=='SIXGBoost':
        classifier = sixgboost.SIXGBoost(verbose=False)
    else:
        print('Method unknown: ',method)
        raise(NotImplementedError)
    return(classifier)

def train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=None):
    
    if method == 'MIMAX':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))
    
        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir=trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,mini_batch_size)
    
    elif method in list_of_ClassicalMI:
        
        classifier = get_classicalMILclassifier(method)
        classifier.fit(bags_train, labels_bags_c_train)
        pred_bag_labels, pred_instance_labels = classifier.predict(bags_test,instancePrediction=True)
        
    else:
        print('Method unknown: ',method)
        raise(NotImplementedError)
        
    return(pred_bag_labels, pred_instance_labels)
    
    


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
    test_dataset = test_dataset.map(lambda r: parser(r,num_rois=size_biggest_bag\
                                                     ,num_features=num_features))
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
#            mei = tf.argmax(Tanh,axis=2)
#            score_mei = tf.reduce_max(Tanh,axis=2)
        elif with_softmax:
            Softmax = tf.nn.softmax(Prod_best,axis=-1)
#            mei = tf.argmax(Softmax,axis=2)
#            score_mei = tf.reduce_max(Softmax,axis=2)
        elif with_softmax_a_intraining:
            Softmax=tf.multiply(tf.nn.softmax(Prod_best,axis=-1),Prod_best)
#            mei = tf.argmax(Softmax,axis=2)
#            score_mei = tf.reduce_max(Softmax,axis=2)
        else:
            if verbose: print('Tanh in testing time',Prod_best)
#            mei = tf.argmax(Prod_best,axis=-1)
#            score_mei = tf.reduce_max(Prod_best,axis=-1)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Evaluation Test
        while True:
            try:
                fc7s,labels = sess.run(next_element)
                if with_tanh:
                    PositiveExScoreAll =\
                    sess.run(Tanh, feed_dict={X: fc7s, y: labels})
                elif with_softmax or with_softmax_a_intraining:
                    PositiveExScoreAll =\
                    sess.run(Softmax, feed_dict={X: fc7s, y: labels})
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
               num_features,mini_batch_size,opts_MIMAX=None):
    """
    This function train a tidy MIMAX model
    """
    path_model = os.path.join('tmp','MI_max')
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True) 

    tf.reset_default_graph()
    
    if not(opts_MIMAX is None):
        C,C_Searching,CV_Mode,restarts,LR = opts_MIMAX
    else:
        C,C_Searching,CV_Mode,restarts,LR = 1.0,False,None,11,0.01

    classifierMI_max = tf_MI_max(LR=LR,restarts=restarts,is_betweenMinus1and1=True, \
                                 num_rois=size_biggest_bag,num_classes=1, \
                                 num_features=num_features,mini_batch_size=mini_batch_size, \
                                 verbose=False,C=C,CV_Mode=CV_Mode,max_iters=300)
    C_values =  np.logspace(-3,2,6,dtype=np.float32)
    classifierMI_max.set_C_values(C_values)
    export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=-1,shuffle=False,restarts_paral='paral', \
                       WR=True,C_Searching=C_Searching)
    tf.reset_default_graph()
    return(export_dir)

if __name__ == '__main__':
#    evalPerf(dataset='Birds',reDo=True,dataNormalizationWhen='onTrainSet',dataNormalization='std',verbose=True)
#    evalPerf(method='LinearSISVM',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=True,verbose=False)
    evalPerf(method='SIXGBoost',dataset='Newsgroups',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
    evalPerf(method='SIXGBoost',dataset='SIVAL',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
#    evalPerf(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=True,verbose=False)
#    evalPerf(dataset='Newsgroups',reDo=True,verbose=False)
#    evalPerf(dataset='Birds',reDo=True,verbose=False)
#    evalPerf(dataset='SIVAL',reDo=False,verbose=False)
#    evalPerf(dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    evalPerf(dataset='Newsgroups')
#    evalPerf(method='MIMAX',dataset='Birds',verbose=True)
#    EvaluationOnALot_ofParameters(dataset='Birds')
#    EvaluationOnALot_ofParameters(dataset='Newsgroups')
#    EvaluationOnALot_ofParameters(dataset='SIVAL')
# A tester : SVM with SGD and then loss hinge in our case