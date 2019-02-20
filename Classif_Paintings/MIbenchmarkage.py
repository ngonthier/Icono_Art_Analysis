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
from MILbenchmark.Dataset.GaussianToy import createGaussianToySets
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import pathlib
import shutil
import sys
import misvm
from MILbenchmark.mialgo import sisvm,MIbyOneClassSVM,sixgboost

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 1 to remove info, 2 to remove warning and 3 for all
import tensorflow as tf

from trouver_classes_parmi_K import tf_MI_max
from trouver_classes_parmi_K_mi import tf_mi_model
import pickle


list_of_ClassicalMI = ['miSVM','SIL','SISVM','LinearSISVM','MIbyOneClassSVM',\
                       'SIXGBoost','MISVM']

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
    LR_tab = np.logspace(-4,0,5)
    dataNormalization_tab = ['std','var','0-1']
    dataNormalization_tab = ['std']
    
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
    
    for LR in LR_tab:
        opts_MIMAX = default_C,default_C_Searching,default_CV_Mode,default_restarts,LR
        pref_name_case = '_LR'+str(LR) 
        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
    
#    for C in C_tab:
#        if not(C==default_C):
#            opts_MIMAX = C,default_C_Searching,default_CV_Mode,default_restarts,default_LR
#            pref_name_case = '_C'+str(C) 
#            evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#                 reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
##        
#    for restarts in restarts_tab:
#        opts_MIMAX = default_C,default_C_Searching,default_CV_Mode,restarts,default_LR
#        pref_name_case = '_r'+str(restarts) 
#        evalPerf(dataset=dataset,dataNormalizationWhen=None,dataNormalization=default_dataNormalization,
#             reDo=False,opts_MIMAX=opts_MIMAX,pref_name_case=pref_name_case,verbose=False)
##        
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
        pref_name_case = pref_name_case
    if dataNormalizationWhen=='onTrainSet':
        pref_name_case += '_' +str(dataNormalization)
    filename = method + '_' + dataset + pref_name_case + '.pkl'
    filename = filename.replace('MISVM','bigMISVM')
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
 
def fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,dataNormalizationWhen=None,dataNormalization=None,
             reDo=False,opts_MIMAX=None,pref_name_case='',verbose=False,
             overlap = False,end_name=''):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM 
    @param : dataset = GaussianToy
    @param : dataNormalizationWhen : moment of the normalization of the data, 
        None = no normalization, onAllSet doing on all the set, onTrainSet 
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    @param : overlap = False overlapping between the 2 classes
    """
    dataset = 'GaussianToy_WR'+str(WR)
    
    if verbose: print('Start evaluation performance on ',dataset,'with WR = ',WR,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None
       
    n = 2
    k = 100
    np_pos = 50
    np_neg = 250 
#    np_pos = 2 
#    np_neg = 4 
    
    Dataset=createGaussianToySets(WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,overlap=overlap)
    list_names,bags,labels_bags,labels_instance = Dataset
    prefixName = 'N'+str(n)+'_k'+str(k)+'_WR'+str(WR)+'_pos'+str(np_pos)+'_neg'+str(np_neg)
    if overlap:
        prefixName += '_OL'
        
    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = pref_name_case
    filename = method + '_' + dataset + prefixName +pref_name_case +end_name + '.pkl'
    filename = filename.replace('MISVM','bigMISVM')
    path_file_results = os.path.join(script_dir,'MILbenchmark','ResultsToy')
    file_results = os.path.join(path_file_results,filename)
    pathlib.Path(path_file_results).mkdir(parents=True, exist_ok=True) # creation of the folder if needed
    if reDo:
        results = {}
    else:
        try:
            results = pickle.load(open(file_results,'br'))
        except FileNotFoundError:
            results = {}
      
    for c_i,c in enumerate(list_names):
        if not(c in results.keys()):
            # Loop on the different class, we will consider each group one after the other
            if verbose: print("Start fitting and plot for class :",c)
            labels_bags_c = labels_bags[c_i]
            labels_instance_c = labels_instance[c_i]
            bags_c = bags
                
            if dataNormalizationWhen=='onAllSet':
                bags_c = normalizeDataSetFull(bags_c,dataNormalization)
                
#            D = bags_c,labels_bags_c,labels_instance_c
            numMetric = 4
            StratifiedFold= True
            size_biggest_bag = 0
            for elt in bags:
                size_biggest_bag = max(size_biggest_bag,len(elt))
            if dataset=='SIVAL':
                num_features = bags[0][0].shape[1]
            else:
                num_features = bags[0].shape[1]
            mini_batch_size_max = 2000 # Maybe this value can be update depending on your GPU memory size
            opts = dataset,mini_batch_size_max,num_features,size_biggest_bag
            perf,perfB=plot_Hyperplan(method,numMetric,bags_c,labels_bags_c,labels_instance_c,
                                      StratifiedFold,opts,dataNormalizationWhen,
                                      dataNormalization,opts_MIMAX=None,
                                      verbose=verbose,prefixName=prefixName,
                                      end_name=end_name)
            
            
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
       
def evalPerfGaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,dataNormalizationWhen=None,dataNormalization=None,
             reDo=False,opts_MIMAX=None,pref_name_case='',verbose=False):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM 
    @param : dataset = GaussianToy
    @param : dataNormalizationWhen : moment of the normalization of the data, 
        None = no normalization, onAllSet doing on all the set, onTrainSet 
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    """
    dataset = 'GaussianToy_WR'+str(WR)
    
    if verbose: print('Start evaluation performance on ',dataset,'with WR = ',WR,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = pref_name_case
    filename = method + '_' + dataset + pref_name_case + '.pkl'
    filename = filename.replace('MISVM','bigMISVM')
    path_file_results = os.path.join(script_dir,'MILbenchmark','ResultsToy')
    file_results = os.path.join(path_file_results,filename)
    pathlib.Path(path_file_results).mkdir(parents=True, exist_ok=True) # creation of the folder if needed
    if reDo:
        results = {}
    else:
        try:
            results = pickle.load(open(file_results,'br'))
        except FileNotFoundError:
            results = {}
            
    Dataset=createGaussianToySets(WR=WR,n=2,k=100,np1=50,np2=250)
    list_names,bags,labels_bags,labels_instance = Dataset
              
    for c_i,c in enumerate(list_names):
        if not(c in results.keys()):
            # Loop on the different class, we will consider each group one after the other
            if verbose: print("Start evaluation for class :",c)
            labels_bags_c = labels_bags[c_i]
            labels_instance_c = labels_instance[c_i]
            bags_c = bags
                
            if dataNormalizationWhen=='onAllSet':
                bags_c = normalizeDataSetFull(bags_c,dataNormalization)
                
            D = bags_c,labels_bags_c,labels_instance_c
    
            perf,perfB=performExperimentWithCrossVal(method,D,dataset,
                                    dataNormalizationWhen,dataNormalization,
                                    nRep=1,nFolds=10,
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
                                  dataNormalization,nRep=10,nFolds=10,GridSearch=False,opts_MIMAX=None,
                                  verbose=False):

    bags,labels_bags_c,labels_instance_c  = D

    StratifiedFold = True
    if verbose and StratifiedFold: print('Use of the StratifiedFold cross validation')

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
    
def plot_Hyperplan(method,numMetric,bags,labels_bags_c,labels_instance_c,StratifiedFold,opts,
               dataNormalizationWhen,dataNormalization,opts_MIMAX=None,verbose=False,prefixName='',end_name=''):
    
    nRep= 1
    nFolds = 1
    fold= 0 
    r = 0
    perfObj=np.empty((nRep,nFolds,numMetric))
    perfObjB=np.empty((nRep,nFolds,numMetric))
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=r)
    train_index, test_index = skf.split(bags,labels_bags_c)
    train_index =train_index[0]
    test_index =test_index[0]
    labels_bags_c_train, labels_bags_c_test = \
        getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
    bags_train, bags_test = \
        getTest_and_Train_Sets(bags,train_index,test_index)
    _ , labels_instance_c_test = \
        getTest_and_Train_Sets(labels_instance_c,train_index,test_index)

    if dataNormalizationWhen=='onTrainSet':
        bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)              
        
    gt_instances_labels_stack = np.hstack(labels_instance_c_test)
    
    bags_test_vstack = np.vstack(bags_test)
    X = bags_test_vstack
    
    grid_size= 100
    filename= prefixName +'_' + method 
    if dataNormalizationWhen=='onTrainSet':
        filename += '_' +str(dataNormalization)
    filename += end_name + ".png"
    filename = filename.replace('MISVM','bigMISVM')
    path_filename =os.path.join('MILbenchmark','ResultsToy',filename)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    points = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        points.append(point)

    if method=='MIMAX':
        opts_MIMAX = 1.0,False,None,11,0.01 # Attention tu utilise 100 vecteurs
        pred_bag_labels, pred_instance_labels,result,bestloss = train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
               method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,pointsPrediction=points,get_bestloss=True)
    else:
        pred_bag_labels, pred_instance_labels,result = train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
               method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,pointsPrediction=points)
    # result is the predited class for the points 
    
    perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
    perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
    
    y = np.sign(pred_instance_labels) # The class prediction +1 or -1
    Z = np.zeros_like(xx)
    index_r= 0
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        Z[i,j] = np.sign(result[index_r])
        index_r+=1
    
    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.5)

    # Plot with the predictive value
    y_unique = np.unique(y)
    if len(y_unique) == 1:
        if y_unique[0]==1:
            color = 'r'
        else:
            color = 'b'
    else:
        color = y
            
    plt.scatter(X[:, 0], X[:, 1],
                c=color, cmap=cm.Paired,alpha=0.5)
#    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
#                c=flatten(y), cmap=cm.Paired)
    
    X_pos = X[np.where(gt_instances_labels_stack==1)[0],:]
    X_neg = X[np.where(gt_instances_labels_stack==-1)[0],:]
    plt.scatter(flatten(X_pos[:, 0]), flatten(X_pos[:, 1]), s=80, facecolors='none', edgecolors='r')
    plt.scatter(flatten(X_neg[:, 0]), flatten(X_neg[:, 1]), s=80, facecolors='none', edgecolors='b')

    dataset = 'GaussianToy'
    perf=getMeanPref(perfObj,dataset)
    perfB=getMeanPref(perfObjB,dataset)
    mPerf = perf[0]
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    titlestr = 'Classification for ' +method+' AUC: {0:.2f}, UAR: {1:.2f}, F1 : {2:.2f}'.format(mPerf[2],mPerf[1],mPerf[0])
    if method=='MIMAX':
        titlestr += ' BL : {0:.2f}'.format(bestloss[0])
    plt.title(titlestr)
    plt.savefig(path_filename)
    plt.show()
    

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
                       method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose)

   
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
                       method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose)
                
                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels)
                fold += 1
    return(perfObj,perfObjB)

def get_classicalMILclassifier(method,verbose=False):
    if method=='miSVM':
        classifier = misvm.miSVM(kernel='linear',C=1.0,restarts=10,max_iters=50,verbose=verbose)
    elif method=='MISVM':
        classifier = misvm.MISVM(kernel='linear',C=1.0,restarts=10,max_iters=50,verbose=verbose)
    elif method=='SIL':
        classifier = misvm.SIL(verbose=verbose)
    elif method=='SISVM':
        classifier = sisvm.SISVM(verbose=verbose)
    elif method=='LinearSISVM':
        classifier = sisvm.LinearSISVM(verbose=verbose)
    elif method=='MIbyOneClassSVM':
        classifier = MIbyOneClassSVM.MIbyOneClassSVM(verbose=verbose)
    elif method=='SIXGBoost':
        classifier = sixgboost.SIXGBoost(verbose=verbose)
    else:
        print('Method unknown: ',method)
        raise(NotImplementedError)
    return(classifier)

def train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=None,verbose=False,pointsPrediction=None,
                       get_bestloss=False):
    """
    pointsPrediction=points : other prediction to do, points size
    """
    
    if method == 'MIMAX':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))
    
        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir=trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX,
                              verbose=verbose,get_bestloss=get_bestloss)
    
        if get_bestloss:
            export_dir,best_loss = export_dir

        if not(pointsPrediction is None):
            labels_pointsPrediction = [np.array(0.)]*len(pointsPrediction)
            data_path_points = Create_tfrecords(pointsPrediction, labels_pointsPrediction,\
                                              size_biggest_bag,num_features,'points',dataset)
            _, points_instance_labels = predict_MIMAX(export_dir,\
                    data_path_points,pointsPrediction,size_biggest_bag,num_features,mini_batch_size,removeModel=False)
        
        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,mini_batch_size,removeModel=True) 
           
    
    elif method == 'MIMAXaddLayer':
        if not(pointsPrediction is None):
            raise(NotImplementedError)
        
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))
    
        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir=trainMIMAXaddLayer(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,mini_batch_size,
                predict_parall=False)
    
    elif method in list_of_ClassicalMI:
        
        classifier = get_classicalMILclassifier(method,verbose=verbose)
        classifier.fit(bags_train, labels_bags_c_train)
        pred_bag_labels, pred_instance_labels = classifier.predict(bags_test,instancePrediction=True)
        if not(pointsPrediction is None):
            _, points_instance_labels = classifier.predict(pointsPrediction,instancePrediction=True)
    else:
        print('Method unknown: ',method)
        raise(NotImplementedError)
    
    if pointsPrediction is None:
        if get_bestloss:
            return(pred_bag_labels, pred_instance_labels,best_loss) 
        else:
            return(pred_bag_labels, pred_instance_labels) 
    else:
        if get_bestloss:
            return(pred_bag_labels, pred_instance_labels,points_instance_labels,best_loss)
        else:
            return(pred_bag_labels, pred_instance_labels,points_instance_labels) 


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
               mini_batch_size=256,verbose=False,predict_parall=True,removeModel=True):
    """
    This function as to goal to predict on the test set
    removeModel = True : erase the model created 
    """
    
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
                if not(predict_parall):
                    labels = labels.ravel()
                if with_tanh:
                    PositiveExScoreAll =\
                    sess.run(Tanh, feed_dict={X: fc7s, y: labels})
                elif with_softmax or with_softmax_a_intraining:
                    PositiveExScoreAll =\
                    sess.run(Softmax, feed_dict={X: fc7s, y: labels})
                else:
                    PositiveExScoreAll = \
                    sess.run(Prod_best, feed_dict={X: fc7s, y: labels})
                if predict_parall:
                    for elt_i in range(PositiveExScoreAll.shape[1]):
                        predict_label_all_test += [PositiveExScoreAll[0,elt_i,:]]
                else:
                    for elt_i in range(PositiveExScoreAll.shape[0]): # Tres certainement sous optimal, 
                        # TODO ameliorer cela
                        predict_label_all_test += [PositiveExScoreAll[elt_i,:]]
            except tf.errors.OutOfRangeError:
                break
    tf.reset_default_graph()
    # Post processing
    instances_labels_pred = []
    bags_labels_pred = []
    for elt,predictions in zip(bags_test,predict_label_all_test):
#        print(predictions)
        length_bag = elt.shape[0]
        np_predictions = np.array(predictions)
        instances_labels_pred += [np_predictions[0:length_bag]]
        bags_labels_pred += [np.max(np_predictions[0:length_bag])]
    bags_labels_pred=(np.hstack(bags_labels_pred))
    instances_labels_pred=(np.hstack(instances_labels_pred))
    
    if verbose:
        print('The model folder will be removed',export_dir_path)

    if removeModel:
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
        if len(elt) < size_biggest_bag:
            number_repeat = size_biggest_bag // len(elt)  +1
            f_repeat = np.concatenate([elt]*number_repeat,axis=0)
            fc7 = f_repeat[0:size_biggest_bag,:]
        else:
            fc7 = elt[0:size_biggest_bag,:]
        features=tf.train.Features(feature={
                        'num_features': _int64_feature(num_features),
                        'num_regions': _int64_feature(size_biggest_bag),
                        #'fc7': _bytes_feature(tf.compat.as_bytes(fc7.tostring())), 
                        'fc7': _floats_feature(fc7), 
                        'label' : _floats_feature(elt_label)})   
        example = tf.train.Example(features=features) #.tostring()
        writer.write(example.SerializeToString())
    writer.close()

    return(path_name)

def trainMIMAX(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
               num_features,mini_batch_size,opts_MIMAX=None,verbose=False,get_bestloss=False):
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
                                 verbose=verbose,C=C,CV_Mode=CV_Mode,max_iters=300,debug=False)
    C_values =  np.logspace(-3,2,6,dtype=np.float32)
    classifierMI_max.set_C_values(C_values)
    export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=-1,shuffle=False,restarts_paral='paral', \
                       WR=True,C_Searching=C_Searching)
    
    if get_bestloss:
        bestloss = classifierMI_max.get_bestloss()
        tf.reset_default_graph()
        return(export_dir,bestloss)
    else:
        tf.reset_default_graph()
        return(export_dir)
    
def trainMIMAXaddLayer(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
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
                                 verbose=False,C=C,CV_Mode=CV_Mode,max_iters=300,
                                 AddOneLayer=True,Optimizer='Momentum')

    export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=0,shuffle=False,restarts_paral=None, \
                       WR=True,C_Searching=C_Searching)
    tf.reset_default_graph()
    return(export_dir)

def ToyProblemRun():
    
    list_of_algo= ['LinearSISVM','miSVM','MISVM','SIXGBoost']
    overlap_tab = [False,True]
    reDo = True
    for overlap in overlap_tab:
        for i in range(10):
            end_name= '_Rep' + str(i) 
            fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',
                                       WR=0.01,verbose=True,reDo=reDo,
                                       dataNormalizationWhen='onTrainSet',dataNormalization='std',
                                       overlap = overlap,end_name=end_name)
            fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',
                                       WR=0.01,verbose=True,reDo=reDo,
                                       dataNormalizationWhen=None,dataNormalization='std',
                                       overlap = overlap,end_name=end_name)
            
    for overlap in overlap_tab:
        for method in list_of_algo:
            fit_train_plot_GaussianToy(method=method,dataset='GaussianToy',
                                       WR=0.01,verbose=True,reDo=reDo,
                                       dataNormalizationWhen='onTrainSet',dataNormalization='std',
                                       overlap = overlap)
            
def BenchmarkRun():
    
    datasets = ['Birds','Newsgroups','SIVAL']
    list_of_algo= ['MIMAX','LinearSISVM','SIXGBoost','miSVM','MISVM']
    
    for method in list_of_algo:
        for dataset in datasets:
            evalPerf(method=method,dataset=dataset,reDo=False,verbose=False,
                     dataNormalizationWhen='onTrainSet',dataNormalization='std')

if __name__ == '__main__':
#    evalPerf(dataset='Birds',reDo=True,dataNormalizationWhen='onTrainSet',dataNormalization='std',verbose=True)
#    evalPerf(method='LinearSISVM',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=True,verbose=False)
#    evalPerf(method='SIXGBoost',dataset='Newsgroups',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
#    evalPerf(method='SIXGBoost',dataset='SIVAL',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
#    evalPerf(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=True,
#             pref_name_case='test',verbose=False)
#    evalPerf(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=True,
#             pref_name_case='test')
#    evalPerf(method='miSVM',dataset='Newsgroups',reDo=False,verbose=False,dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    evalPerf(method='BigMISVM',dataset='Birds',reDo=False,verbose=False,dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    evalPerf(method='MISVM',dataset='SIVAL',reDo=False,verbose=False,dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    evalPerf(method='MISVM',dataset='Newsgroups',reDo=False,verbose=False)
#    evalPerf(method='MISVM',dataset='Birds',reDo=False,verbose=False)
#    evalPerf(method='MISVM',dataset='SIVAL',reDo=False,verbose=False)
#    evalPerf(dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    evalPerf(dataset='Newsgroups')
#    evalPerf(method='MIMAX',dataset='Birds',verbose=True)
#    EvaluationOnALot_ofParameters(dataset='Newsgroups')
#    EvaluationOnALot_ofParameters(dataset='SIVAL')
#    EvaluationOnALot_ofParameters(dataset='Birds')
#    evalPerf(method='SIXGBoost',dataset='Newsgroups',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
#    evalPerf(method='SIXGBoost',dataset='SIVAL',dataNormalizationWhen='onTrainSet',dataNormalization='std',reDo=False,verbose=False)
#    
#    EvaluationOnALot_ofParameters(dataset='Newsgroups')
#    EvaluationOnALot_ofParameters(dataset='SIVAL')
# A tester : SVM with SGD and then loss hinge in our case
    
    # Gaussian Toy test
#    fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,verbose=True,reDo=True)
#    fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=1.,verbose=True,reDo=True,dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.1,verbose=True,reDo=True,dataNormalizationWhen='onTrainSet',dataNormalization='std')
#    fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,verbose=True,reDo=True,dataNormalizationWhen=None,dataNormalization='std',overlap=True)
#    fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,verbose=True,reDo=True,dataNormalizationWhen='onTrainSet',dataNormalization='std')

#    evalPerfGaussianToy(method='LinearSISVM',dataset='GaussianToy',WR=0.01,verbose=True)
#    evalPerfGaussianToy(method='LinearSISVM',dataset='GaussianToy',WR=0.003,verbose=True)
    ToyProblemRun()
    BenchmarkRun()