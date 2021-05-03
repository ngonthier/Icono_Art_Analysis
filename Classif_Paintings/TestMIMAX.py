# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:35:02 2019

Goal of this script is to do some test on the MIMAX model 

@author: gonthier
"""


import os
import warnings
warnings.filterwarnings("ignore")

from MILbenchmark.utils import getDataset,normalizeDataSetFull,getMeanPref,\
    getTest_and_Train_Sets,normalizeDataSetTrain,getClassifierPerfomance
from MILbenchmark.Dataset.GaussianToy import createGaussianToySets,createMILblob,createGaussianToySets_MClasses
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
import numpy as np
import pathlib
import shutil
import sys
import misvm
from MILbenchmark.mialgo import sisvm,MIbyOneClassSVM,sixgboost,siDLearlyStop,\
    miDLearlyStop,miDLstab,ensDLearlyStop,kerasMultiMIMAX

from sklearn.metrics import roc_curve,f1_score,roc_auc_score


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from scipy.stats.stats import pearsonr

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 1 to remove info, 2 to remove warning and 3 for all
import tensorflow as tf

from trouver_classes_parmi_K import tf_MI_max,ModelHyperplan
#from trouver_classes_parmi_K_MultiPlan import tf_MI_max,ModelHyperplan
from trouver_classes_parmi_K_mi import tf_mi_model
import pickle

from MIbenchmarkage import train_and_test_MIL

from sklearn.preprocessing import MultiLabelBinarizer

list_of_MIMAXbasedAlgo = ['MIMAX','MIMAXaddLayer','IA_mi_model','MAXMIMAX']

path_tmp = '/media/gonthier/HDD/output_exp/ClassifPaintings/tmp/'
if not(os.path.exists(path_tmp)):
    path_tmp = 'tmp'
# test if the folder need exist :
path_needed = os.path.join(path_tmp,'MI_max')
pathlib.Path(path_needed).mkdir(parents=True, exist_ok=True)
path_needed = os.path.join(path_tmp,'MI_max_StoredW')
pathlib.Path(path_needed).mkdir(parents=True, exist_ok=True)

def testMultiClassGaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,
                        dataNormalizationWhen='onTrainSet',
                        dataNormalization='std',
                        reDo=True,opts_MIMAX=None,pref_name_case='',
                        verbose=False,overlap = False,end_name='',
                        specificCase='',k=100):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM, miSVM  or IA_mi_model,
        MIMAXaddLayer, SIDLearlyStop etc
    @param : dataset = GaussianToy
    @param : dataNormalizationWhen : moment of the normalization of the data,
        None = no normalization, onAllSet doing on all the set, onTrainSet
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    @param : overlap = False overlapping between the 2 classes
    @param : specificCase : we proposed different case of toy points clouds
        - 2clouds : 2 clouds distincts points of clouds as positives examples
        - 2cloudsOpposite : 2 points clouds positive at the opposite from the negatives
    @param : OnePtTraining : Only one point is available in the training set
    @param : k : the number of element per bag
    """

    list_specificCase = ['',None,'2clouds','2cloudsOpposite']
    if not(specificCase in list_specificCase):
        print(specificCase,'is unknown')
        raise(NotImplementedError)

    dataset_WR = dataset+'_WR'+str(WR)

    if verbose: print('Start evaluation performance on ',dataset_WR,'with WR = ',WR,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None
    n = 2 # Number of dimension
#        k = 100 # number of element in a bag
    np_pos = 50 # Number of positive examples
    np_neg = 250# Number of negative examples
    M = 4
    
    if dataset=='GaussianToy':
        Dataset=createGaussianToySets_MClasses(M=M,WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,
                                      overlap=overlap,specificCase=specificCase)
        list_names,bags,labels_bags,labels_instance = Dataset
    elif dataset=='blobs':
        list_names,bags,labels_bags,labels_instance = createMILblob(WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,Between01=False)
    else:
        raise(NotImplementedError)
#        prefixName = 'N'+str(n)+'_k'+str(k)+'_WR'+str(WR)+'_pos'+str(np_pos)+\
#            '_neg'+str(np_neg)

    if verbose: print('Start evaluation performance on ',dataset,'with WR = ',WR,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    for c_i,c in enumerate(list_names):
        # Loop on the different class, we will consider each group one after the other
        if verbose: print("Start evaluation for class :",c)
        labels_bags_c = labels_bags[c_i]
        labels_instance_c = labels_instance[c_i]
        bags_c = bags

        if dataNormalizationWhen=='onAllSet':
            bags_c = normalizeDataSetFull(bags_c,dataNormalization)

#        D = bags_c,labels_bags_c,labels_instance_c

        size_biggest_bag = 0
        for elt in bags:
            size_biggest_bag = max(size_biggest_bag,len(elt))
        if dataset=='SIVAL':
            num_features = bags[0][0].shape[1]
        else:
            num_features = bags[0].shape[1]

        mini_batch_size_max = 2000 # Maybe this value can be update depending on your GPU memory size
        opts = dataset,mini_batch_size_max,num_features,size_biggest_bag
        nFolds = 2
        skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=0)
        
        labels_bags_c_multiclass = []
        for elt in labels_bags_c:
            elt = (elt+1.)/2.
            if np.sum(elt)==0.:
                labels_bags_c_multiclass += [-1]
            else:
                labels_bags_c_multiclass += [np.where(elt==1.)[0]]
#        from sklearn.utils.multiclass import type_of_target
        labels_bags_c_multiclass = np.array(np.hstack(labels_bags_c_multiclass))
        print(len(labels_bags_c))
#        print(labels_bags_c_multiclass)
#        
#        -np.ones(shape=(len(labels_bags_c),))
#        print(labels_bags_c)
#        for i in range(M):
#            index_i = np.where(labels_bags_c[:,i]==1)[0]
#            print(index_i)
#            labels_bags_c_multiclass[index_i] = i
        
        train_index, test_index = skf.split(bags,labels_bags_c_multiclass)
        train_index = train_index[0]
        test_index = test_index[0]
        labels_bags_c_train, labels_bags_c_test = \
            getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
        bags_train, bags_test = \
            getTest_and_Train_Sets(bags,train_index,test_index)
        labels_instance_c_train , labels_instance_c_test = \
            getTest_and_Train_Sets(labels_instance_c,train_index,test_index)

        labels_instance_c_train = np.hstack(labels_instance_c_train)

        if dataNormalizationWhen=='onTrainSet':
            bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)

        gt_instances_labels_stack = np.hstack(labels_instance_c_test)

        # Dans le predict_MIMAX : ajout le num_class
        pred_bag_labels, pred_instance_labels =train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
               method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose)
        
        f1score = f1_score(gt_instances_labels_stack,pred_instance_labels)

        print(f1score)