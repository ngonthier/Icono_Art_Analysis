# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:22:51 2019

The goal of this script is to evaluate  the different model on the
classical MIL benchmark as Birds, SIVAL or NewsGroups and on some Toy
Problem

@author: gonthier
"""

import os
import warnings
warnings.filterwarnings("ignore")

from MILbenchmark.utils import getDataset,normalizeDataSetFull,getMeanPref,\
    getTest_and_Train_Sets,normalizeDataSetTrain,getClassifierPerfomance
from MILbenchmark.Dataset.GaussianToy import createGaussianToySets,createMILblob
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
import numpy as np
import pathlib
import shutil
import sys
import misvm
from MILbenchmark.mialgo import sisvm,MIbyOneClassSVM,sixgboost,siDLearlyStop,\
    miDLearlyStop,miDLstab,ensDLearlyStop#,kerasMultiMIMAX

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


list_of_ClassicalMI = ['miSVM','SIL','SISVM','LinearSISVM','MIbyOneClassSVM',\
                       'SIXGBoost','MISVM','SIDLearlyStop','miDLearlyStop',\
                       'miDLstab','ensDLearlyStop','kerasMultiMIMAX']
list_of_MIMAXbasedAlgo = ['MIMAX','MIMAXaddLayer','IA_mi_model','MAXMIMAX']

path_tmp = '/media/HDD/output_exp/ClassifPaintings/tmp/'
if not(os.path.exists(path_tmp)):
    path_tmp = 'tmp'
# test if the folder need exist :
path_needed = os.path.join(path_tmp,'MI_max')
pathlib.Path(path_needed).mkdir(parents=True, exist_ok=True)
path_needed = os.path.join(path_tmp,'MI_max_StoredW')
pathlib.Path(path_needed).mkdir(parents=True, exist_ok=True)

def EvaluationOnALot_ofParameters(dataset):
    """
    List of the parameters that can be used to get better results maybe on the
    MILbenchmark
    """

    # List of the parameter that can improve the resultats
#    C_tab = np.logspace(start=-3,stop=3,num=7)
#    C_Searching_tab = [True,False]
#    CV_Mode_tab = ['','CVforCsearch','CV']
#    restarts_tab = [0,11,99]
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

def evalPerf(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',
             dataNormalization='std',
             reDo=False,opts_MIMAX=None,pref_name_case='',verbose=False,
             epochsSIDLearlyStop=10,nRep=10,nFolds=10,numMetric=7):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM, SIDLearlyStop
    @param : dataset = Newsgroups, Bird or SIVAL
    @param : dataNormalizationWhen : moment of the normalization of the data,
        None = no normalization, onAllSet doing on all the set, onTrainSet
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    @param : nRep number of repetition, it have to be equal to 10 for the benchmark evaluation
    @param : nFolds number of folds, it have to be equal to 10 for the benchmark evaluation
    @param : numMetric number of different metrics computed
    """

    if verbose: print('Start evaluation performance on ',dataset,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = pref_name_case
    if dataNormalizationWhen=='onTrainSet':
        pref_name_case += '_' +str(dataNormalization)
    if method in ['SIDLearlyStop','miDLearlyStop']:
        pref_name_case += 'epochs'+str(epochsSIDLearlyStop)
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
                                    nRep=nRep,nFolds=nFolds,numMetric=numMetric,
                                    GridSearch=False,opts_MIMAX=opts_MIMAX,
                                    verbose=verbose,epochsSIDLearlyStop=epochsSIDLearlyStop)
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
            if numMetric >= 5:
                print('AP : ',mPerf[4],' +/- ',stdPerf[4])
            if numMetric >= 6:
                print('Balanced Accuracy : ',mPerf[5],' +/- ',stdPerf[5])
            if numMetric >= 7:
                print('Matthew Coeff : ',mPerf[6],' +/- ',stdPerf[6])
            print('- bags')
            print('AUC: ',mPerfB[2],' +/- ',stdPerfB[2])
            print('UAR: ',mPerfB[1],' +/- ',stdPerfB[1])
            print('F1: ',mPerfB[0],' +/- ',stdPerfB[0])
            print('Accuracy: ',mPerfB[3],' +/- ',stdPerfB[3])
            if numMetric >= 5:
                print('AP : ',mPerfB[4],' +/- ',stdPerfB[4])
            if numMetric >= 6:
                print('Balanced Accuracy : ',mPerfB[5],' +/- ',stdPerfB[5])
            if numMetric >= 7:
                print('Matthew Coeff : ',mPerfB[6],' +/- ',stdPerfB[6])
            print('-------------------------------------------------------------')
            results[c] = [perf,perfB]
        pickle.dump(results,open(file_results,'bw'))

def plotDistribScorePerd(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',dataNormalization='std',
             reDo=True,opts_MIMAX=None,pref_name_case='',verbose=True,
             numberofW_to_keep = 12,number_of_reboots = 120,
             corr='cov'):
    """
    The goal of this function is to draw the histogram of the value of the loss function
    and the performance on a specific split of the dataset
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM
    @param : dataset = Newsgroups, Bird or SIVAL
    @param : dataNormalizationWhen : moment of the normalization of the data,
        None = no normalization, onAllSet doing on all the set, onTrainSet
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    @param : numberofW_to_keep : number of W keeped
    @param : number_of_reboots : number of differents runs for plotting the histogram
    @param : corr type of correlation we compute cov compute the covariance whereas
        pearsonr compute the pearson correlation coefficient
    """

    if verbose: print('Start evaluation performance on ',dataset,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = pref_name_case
    if dataNormalizationWhen=='onTrainSet':
        pref_name_case += '_' +str(dataNormalization)
    filename = method + 'PlotDistrib_' + dataset + pref_name_case + '.pkl'
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

    plt.ion()
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

            numMetric = 4

            size_biggest_bag = 0
            for elt in bags_c:
                size_biggest_bag = max(size_biggest_bag,len(elt))
            if dataset=='SIVAL':
                num_features = bags_c[0][0].shape[1]
            else:
                num_features = bags_c[0].shape[1]

            mini_batch_size_max = 2000 # Maybe this value can be update depending on your GPU memory size
            opts = dataset,mini_batch_size_max,num_features,size_biggest_bag

            if method in['MIMAX','IA_mi_model','MIMAXaddLayer'] and not(opts_MIMAX is None):
                C,C_Searching,CV_Mode,restarts,LR  = opts_MIMAX

            if dataset=='SIVAL':
                num_sample = 5
                perfObj=np.empty((num_sample,number_of_reboots,numMetric))
                perfObjB=np.empty((num_sample,number_of_reboots,numMetric))
                loss_values=np.empty((num_sample,number_of_reboots,))
                for k in range(num_sample):
                    labels_bags_c_k = labels_bags_c[k]
                    labels_instance_c_k = labels_instance_c[k]
                    bags_k = bags_c[k]
                    perfObj_k,perfObjB_k,loss_values_k = computePerfMImaxAllW(method,numberofW_to_keep,number_of_reboots,
                                    numMetric,bags_k,labels_bags_c_k,labels_instance_c_k,opts,
                                    dataNormalizationWhen,dataNormalization,opts_MIMAX=opts_MIMAX,verbose=False,
                                    test_size=0.1)
                    perfObj[k,:,:] = perfObj_k
                    perfObjB[k,:,:] = perfObjB_k
                    loss_values[k,:,:] = loss_values_k
            else:
                perfObj,perfObjB,loss_values = computePerfMImaxAllW(method,numberofW_to_keep,number_of_reboots,numMetric,bags_c,labels_bags_c,labels_instance_c,opts,
                                                                    dataNormalizationWhen,dataNormalization,opts_MIMAX=opts_MIMAX,verbose=False,
                                                                    test_size=0.1)
            data = [loss_values]
            data +=[perfObj[:,0]]
            data +=[perfObj[:,1]]
            data +=[perfObj[:,2]]
            if corr=='pearsonr':
                corrLossF1 = pearsonr(-loss_values,perfObj[:,0])[0]
                corrLossAUC = pearsonr(-loss_values,perfObj[:,1])[0]
                corrLossUAR = pearsonr(-loss_values,perfObj[:,2])[0]
            elif corr=='cov':
                corrLossF1 = np.cov(-loss_values,perfObj[:,0])[0,1]
                corrLossAUC = np.cov(-loss_values,perfObj[:,1])[0,1]
                corrLossUAR = np.cov(-loss_values,perfObj[:,2])[0,1]

            yaxes = ['loss','F1','UAR','AUC']
            titles = []
            titles += ['Loss function']
            if corr=='pearsonr':
                titles += ['Corr with F1 : {0:.2f}'.format(corrLossF1)]
                titles += ['Corr with AUC : {0:.2f}'.format(corrLossAUC)]
                titles += ['Corr with UAR : {0:.2f}'.format(corrLossUAR)]
            elif corr=='cov':
                titles += ['Cov with F1 : {0:.2f}'.format(corrLossF1)]
                titles += ['Cov with AUC : {0:.2f}'.format(corrLossAUC)]
                titles += ['Cov with UAR : {0:.2f}'.format(corrLossUAR)]

            f,a = plt.subplots(2,2,figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
            a = a.ravel()
            for idx,ax in enumerate(a):
                ax.hist(data[idx])
                ax.set_title(titles[idx])
                ax.set_ylabel(yaxes[idx])

            add_to_name = ''
            if method in['MIMAX','IA_mi_model','MIMAXaddLayer'] and not(opts_MIMAX is None):
                if not(C==1.0):
                    add_to_name += '_C'+str(C)
                if not(LR==0.01):
                    add_to_name += '_LR'+str(LR)
                if C_Searching:
                    add_to_name += '_C_Searching'
                if not(CV_Mode==''):
                    add_to_name += '_' + CV_Mode

            titlestr = 'Distribution of Loss Function for ' +c+' in '+dataset+\
            ' with best over '+str(numberofW_to_keep) +' W'
            plt.suptitle(titlestr,fontsize=12)
            script_dir = os.path.dirname(__file__)
            filename = method + '_' + dataset +'_' +c +pref_name_case +'_'+corr

            if dataNormalizationWhen=='onTrainSet':
                filename += '_' +str(dataNormalization)
            filename += '_W'+ str(numberofW_to_keep)+".png"
            filename = filename.replace('MISVM','bigMISVM')
            path_filename =os.path.join(script_dir,'MILbenchmark','Results','Hist',filename)
            head,tail = os.path.split(path_filename)
            pathlib.Path(head).mkdir(parents=True, exist_ok=True)
#            plt.tight_layout()
            plt.savefig(path_filename)
            plt.show()
            plt.close()

def plotROCcurve(method='MIMAX',dataset='Birds',dataNormalizationWhen='onTrainSet',
             dataNormalization='std',
             reDo=True,opts_MIMAX=None,pref_name_case='ROCCurve',verbose=False,
             epochsSIDLearlyStop=1,nRep=1,nFolds=10,numMetric=7):
    """
    This function evaluate the performance of our MIMAX algorithm
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM or miSVM, SIDLearlyStop
    @param : dataset = Newsgroups, Bird or SIVAL
    @param : dataNormalizationWhen : moment of the normalization of the data,
        None = no normalization, onAllSet doing on all the set, onTrainSet
    @param : dataNormalization : kind of normalization possible : std, var or 0-1
    @param : reDo : it will erase the results file
    @param : opts_MIMAX optimion for the MIMAX (i.e.  C,C_Searching,CV_Mode,restarts,LR)
    @param : pref_name_case prefixe of the results file name
    @param : verbose : print some information
    @param : nRep number of repetition, it have to be equal to 10 for the benchmark evaluation
    @param : nFolds number of folds, it have to be equal to 10 for the benchmark evaluation
    @param : numMetric number of different metrics computed
    """

    if verbose: print('Start evaluation performance on ',dataset,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    script_dir = os.path.dirname(__file__)
    if not(pref_name_case==''):
        pref_name_case = pref_name_case
    if dataNormalizationWhen=='onTrainSet':
        pref_name_case += '_' +str(dataNormalization)
    filename = method + 'PlotROC_' + dataset + pref_name_case + '.pkl'
    filename = filename.replace('MISVM','bigMISVM')
    path_file_results = os.path.join(script_dir,'MILbenchmark','Results','ROCcurve')
    file_results = os.path.join(path_file_results,filename)
    pathlib.Path(path_file_results).mkdir(parents=True, exist_ok=True) # creation of the folder if needed
    if reDo:
        results = {}
    else:
        try:
            results = pickle.load(open(file_results,'br'))
        except FileNotFoundError:
            results = {}

    plt.ion()
    Dataset=getDataset(dataset)
    list_names,bags,labels_bags,labels_instance = Dataset
    print(list_names)
    for c_i,c in enumerate(list_names):
        if not(c in results.keys()):
            # Loop on the different class, we will consider each group one after the other
#            if verbose:
            print("Start evaluation for class :",c)
            labels_bags_c = labels_bags[c_i]
            labels_instance_c = labels_instance[c_i]
            if dataset in ['Newsgroups','SIVAL']:
                bags_c = bags[c_i]
            else:
                bags_c = bags

            if dataNormalizationWhen=='onAllSet':
                bags_c = normalizeDataSetFull(bags_c,dataNormalization)

            size_biggest_bag = 0
            for elt in bags_c:
                size_biggest_bag = max(size_biggest_bag,len(elt))
            if dataset=='SIVAL':
                num_features = bags_c[0][0].shape[1]
            else:
                num_features = bags_c[0].shape[1]

            mini_batch_size_max = 2000 # Maybe this value can be update depending on your GPU memory size
            opts = dataset,mini_batch_size_max,num_features,size_biggest_bag

            if method in['MIMAX','IA_mi_model','MIMAXaddLayer'] and not(opts_MIMAX is None):
                C,C_Searching,CV_Mode,restarts,LR  = opts_MIMAX

            list_pred = []
            list_gt = []

            perfObj=np.empty((nFolds,numMetric))
            perfObjB=np.empty((nFolds,numMetric))

            if dataset=='SIVAL':
                raise(NotImplementedError)
            else:

                # Creation of nFolds splits
                #Use a StratifiedKFold to get the same distribution in positive and negative class in the train and test set
                r = 0
                skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=r)
                fold = 0

                for train_index, test_index in skf.split(bags_c,labels_bags_c):
                    if verbose:
                        sys.stdout.write('Rep number : {:d}/{:d}, Fold number : {:d}/{:d} \r' \
                                         .format(r, nRep, fold, nFolds))
                        sys.stdout.flush()
                    labels_bags_c_train, labels_bags_c_test = \
                        getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                    bags_train, bags_test = \
                        getTest_and_Train_Sets(bags_c,train_index,test_index)
                    labels_instance_c_train , labels_instance_c_test = \
                        getTest_and_Train_Sets(labels_instance_c,train_index,test_index)

                    if dataNormalizationWhen=='onTrainSet':
                        bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)

                    gt_instances_labels_stack = np.hstack(labels_instance_c_test).ravel()


                    pred_bag_labels, pred_instance_labels =train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                           method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,epochsSIDLearlyStop=epochsSIDLearlyStop)

                    labels_instance_c_train = np.hstack(labels_instance_c_train)
                    print("Number of positive instances train",np.sum((labels_instance_c_train+1)/2.))
                    print("Number of positive bags train",np.sum((np.hstack(labels_bags_c_train)+1)/2.))
                    print("Number of neagative instances train",-np.sum((labels_instance_c_train-1)/2.))
                    print("Number of positive instances test",np.sum((np.hstack(labels_instance_c_test)+1)/2.))
                    print("Number of positive predicted instances test",np.sum((np.sign(pred_instance_labels)+1)/2.))
                    print("F1 score on test :",f1_score(np.hstack(labels_instance_c_test), np.sign(pred_instance_labels),labels=[-1,1]))


                    list_gt += [gt_instances_labels_stack]
                    list_pred += [pred_instance_labels]
                    perfObj[fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels,numMetric=numMetric)
                    perfObjB[fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels,numMetric=numMetric)
                    fold += 1
#                    perf = getMeanPref(perfObj,dataset)
#                    perfB = getMeanPref(perfObjB,dataset)

            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')
            for gt_instances_labels_stack,pred_instance_labels in zip(list_gt,list_pred):
                gt_instances_labels_stack = gt_instances_labels_stack.ravel()
                pred_instance_labels = pred_instance_labels.ravel()
#                print(gt_instances_labels_stack)
#                print(pred_instance_labels.shape)
                fpr, tpr, _ = roc_curve(gt_instances_labels_stack, pred_instance_labels,pos_label=1)
                f1 = f1_score(gt_instances_labels_stack, np.sign(pred_instance_labels),labels=[-1,1])
                aucScore = roc_auc_score(gt_instances_labels_stack, pred_instance_labels)

                label_i = "F1 : {0:.2f}, AUC : {1:.2f}".format(f1,aucScore)
                plt.plot(fpr, tpr, label=label_i)

            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()
            add_to_name = ''
            if method in['MIMAX','IA_mi_model','MIMAXaddLayer'] and not(opts_MIMAX is None):
                if not(C==1.0):
                    add_to_name += '_C'+str(C)
                if not(LR==0.01):
                    add_to_name += '_LR'+str(LR)
                if C_Searching:
                    add_to_name += '_C_Searching'
                if not(CV_Mode==''):
                    add_to_name += '_' + CV_Mode

            titlestr = 'ROC curve for ' +c+' in '+dataset
            plt.suptitle(titlestr,fontsize=12)
            script_dir = os.path.dirname(__file__)
            filename = method + '_' + dataset +'_' +c +'_'+pref_name_case +add_to_name
            if dataNormalizationWhen=='onTrainSet':
                filename += '_' +str(dataNormalization)
            filename = filename.replace('MISVM','bigMISVM')
            path_filename =os.path.join(script_dir,'MILbenchmark','Results','ROCcurve',filename)
            head,tail = os.path.split(path_filename)
            pathlib.Path(head).mkdir(parents=True, exist_ok=True)
#            plt.tight_layout()
            plt.savefig(path_filename)
        plt.show()
        plt.close()




def fit_train_plot_GaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,\
                               dataNormalizationWhen='onTrainSet',dataNormalization='std',\
                               reDo=True,opts_MIMAX=None,\
                               pref_name_case='',verbose=False,\
                               overlap = False,end_name='',specificCase='',\
                               OnePtTraining=False):
    """
    This function evaluate the performance of our MIMAX algorithm and plot the
    Hyperplan for the case 2D n=2
    @param : method = MIMAX, SIL, siSVM, MIbyOneClassSVM, miSVM  or IA_mi_model,
        MIMAXaddLayer etc
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
    """

    list_specificCase = ['',None,'2clouds','2cloudsOpposite']
    if not(specificCase in list_specificCase):
        print(specificCase,'is unknown')
        raise(NotImplementedError)

    dataset = 'GaussianToy_WR'+str(WR)

    if verbose: print('Start evaluation performance on ',dataset,'with WR = ',WR,'method :',method)

    if dataNormalization==None: dataNormalizationWhen=None

    n = 2
    k = 100
    np_pos = 50
    np_neg = 250
#    np_pos = 4
#    np_neg = 6

    Dataset=createGaussianToySets(WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,
                                  overlap=overlap,specificCase=specificCase)
    list_names,bags,labels_bags,labels_instance = Dataset
    prefixName = 'N'+str(n)+'_k'+str(k)+'_WR'+str(WR)+'_pos'+str(np_pos)+\
        '_neg'+str(np_neg)

    if overlap:
        prefixName += '_OL'
    if not(specificCase is None):
        prefixName += specificCase
    if OnePtTraining:
        prefixName += '_OnePt'

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
            numMetric = 7
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
                                      end_name=end_name,OnePtTraining=OnePtTraining)


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
            if numMetric >= 5:
                print('AP : ',mPerf[4],' +/- ',stdPerf[4])
            if numMetric >= 6:
                print('Balanced Accuracy : ',mPerf[5],' +/- ',stdPerf[5])
            if numMetric >= 7:
                print('Matthew Coeff : ',mPerf[6],' +/- ',stdPerf[6])
            print('- bags')
            print('AUC: ',mPerfB[2],' +/- ',stdPerfB[2])
            print('UAR: ',mPerfB[1],' +/- ',stdPerfB[1])
            print('F1: ',mPerfB[0],' +/- ',stdPerfB[0])
            print('Accuracy: ',mPerfB[3],' +/- ',stdPerfB[3])
            if numMetric >= 5:
                print('AP : ',mPerfB[4],' +/- ',stdPerfB[4])
            if numMetric >= 6:
                print('Balanced Accuracy : ',mPerfB[5],' +/- ',stdPerfB[5])
            if numMetric >= 7:
                print('Matthew Coeff : ',mPerfB[6],' +/- ',stdPerfB[6])
            print('-------------------------------------------------------------')
            results[c] = [perf,perfB]
        pickle.dump(results,open(file_results,'bw'))

def evalPerfGaussianToy(method='MIMAX',dataset='GaussianToy',WR=0.01,
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

    # Number of dimension
    n_list = [2,3,10,50,100,250,300,400,500,750,1000,2048]
    dict_perf = {}
    for n in n_list[::-1]:
        np_pos = 50 # Number of positive examples
        np_neg = 250# Number of negative examples

        if dataset=='GaussianToy':
            Dataset=createGaussianToySets(WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,
                                          overlap=overlap,specificCase=specificCase)
            list_names,bags,labels_bags,labels_instance = Dataset
        elif dataset=='blobs':
            list_names,bags,labels_bags,labels_instance = createMILblob(WR=WR,n=n,k=k,np1=np_pos,np2=np_neg,Between01=False)
        else:
            raise(NotImplementedError)

        if verbose: print('Start evaluation performance on ',dataset,'with WR = ',WR,'method :',method)

        if dataNormalization==None: dataNormalizationWhen=None

        numMetric = 7

        for c_i,c in enumerate(list_names):
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
                                    nRep=1,nFolds=2,
                                    GridSearch=False,opts_MIMAX=opts_MIMAX,
                                    verbose=verbose,numMetric=numMetric)
            mPerf = perf[0]
            stdPerf = perf[1]
            dict_perf[n] = [mPerf,stdPerf]
#            mPerfB = perfB[0]
#            stdPerfB = perfB[1]
            ## Results
            print('=============================================================')
            print("For class :",c)
            print('-------------------------------------------------------------')
            print('- instances') # f1Score,UAR,aucScore,accuracyScore
            print('AUC: ',mPerf[2],' +/- ',stdPerf[2])
            print('UAR: ',mPerf[1],' +/- ',stdPerf[1])
            print('F1: ',mPerf[0],' +/- ',stdPerf[0])
            print('Accuracy: ',mPerf[3],' +/- ',stdPerf[3])
            if numMetric >= 5:
                print('AP : ',mPerf[4],' +/- ',stdPerf[4])
            if numMetric >= 6:
                print('Balanced Accuracy : ',mPerf[5],' +/- ',stdPerf[5])
            if numMetric >= 7:
                print('Matthew Coeff : ',mPerf[6],' +/- ',stdPerf[6])
#            print('- bags')
#            print('AUC: ',mPerfB[2],' +/- ',stdPerfB[2])
#            print('UAR: ',mPerfB[1],' +/- ',stdPerfB[1])
#            print('F1: ',mPerfB[0],' +/- ',stdPerfB[0])
#            print('Accuracy: ',mPerfB[3],' +/- ',stdPerfB[3])
#            if numMetric >= 5:
#                print('AP : ',mPerfB[4],' +/- ',stdPerfB[4])
#            if numMetric >= 6:
#                print('Balanced Accuracy : ',mPerfB[5],' +/- ',stdPerfB[5])
#            if numMetric >= 7:
#                print('Matthew Coeff : ',mPerfB[6],' +/- ',stdPerfB[6])
            print('-------------------------------------------------------------')
    list_metrics = ['F1','UAR','AUC','Accuracy']    
    for i,metric in enumerate(list_metrics):
        mPerf = []
        stdPerf = []
        for n in n_list:
            mPerf += [dict_perf[n][0][i]]
            stdPerf += [dict_perf[n][1][i]]
        plt.semilogx(n_list,mPerf)
#        plt.errorbar(n_list,mPerf,stdPerf)
        plt.xlabel('d dimension')
        plt.ylabel(metric)
        titre = method + ' '+metric +' fct to d'
        if overlap:
            titre += ' with overlap'
        plt.title(titre)
        plt.tight_layout()
        plt.show()



def performExperimentWithCrossVal(method,D,dataset,dataNormalizationWhen,
                                  dataNormalization,nRep=10,nFolds=10,numMetric=5,
                                  GridSearch=False,opts_MIMAX=None,
                                  verbose=False,epochsSIDLearlyStop=10):

    bags,labels_bags_c,labels_instance_c  = D

    StratifiedFold = True
    if verbose and StratifiedFold: print('Use of the StratifiedFold cross validation')

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
                                              verbose=verbose,
                                              epochsSIDLearlyStop=epochsSIDLearlyStop)
            perfObj[k,:,:,:] = perfObj_k
            perfObjB[k,:,:,:] = perfObjB_k
    else:
        perfObj,perfObjB = doCrossVal(method,nRep,nFolds,numMetric,bags,labels_bags_c,\
                                      labels_instance_c,StratifiedFold,opts\
                                      ,dataNormalizationWhen,dataNormalization,\
                                      opts_MIMAX=opts_MIMAX,verbose=verbose,
                                      epochsSIDLearlyStop=epochsSIDLearlyStop)

    perf=getMeanPref(perfObj,dataset)
    perfB=getMeanPref(perfObjB,dataset)
    return(perf,perfB)

def plot_Hyperplan(method,numMetric,bags,labels_bags_c,labels_instance_c,
                   StratifiedFold,opts,
               dataNormalizationWhen,dataNormalization,opts_MIMAX=None,
               verbose=False,prefixName='',end_name='',OnePtTraining=False):
    """
    This fucntion will plot the plan in 2D
    """
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
    if OnePtTraining:
        index_pos_pt = np.random.choice(np.where(np.vstack(labels_bags_c_train)==1.)[0],1)
        index_neg_pts = np.where(np.vstack(labels_bags_c_train)==-1.)[0]
        indextotal = np.concatenate((index_pos_pt,index_neg_pts))
        local_index = 0
        labels_bags_c_train_tmp = []
        bags_train_tmp = []
        for local_index in range(len(labels_bags_c_train)):
            if local_index  in indextotal:
                labels_bags_c_train_tmp += [labels_bags_c_train[local_index]]
                bags_train_tmp += [bags_train[local_index]]
        bags_train = bags_train_tmp
        labels_bags_c_train = labels_bags_c_train_tmp
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

    if method in['MIMAX','IA_mi_model','MIMAXaddLayer']:
        pred_bag_labels, pred_instance_labels,result,bestloss = train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
               method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,pointsPrediction=points,get_bestloss=True)
    else:
        pred_bag_labels, pred_instance_labels,result = train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
               method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,pointsPrediction=points)
    # result is the predited class for the points

    perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels,numMetric=numMetric)
    perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels,numMetric=numMetric)

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

#    plt.scatter(X[:, 0], X[:, 1],
#                c=color, cmap=cm.Paired,alpha=0.5)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(color), cmap=cm.Paired)

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



    add_to_name = ''
    if method in['MIMAX','IA_mi_model','MIMAXaddLayer'] and not(opts_MIMAX is None):
        C,C_Searching,CV_Mode,restarts,LR = opts_MIMAX
        if not(restarts==49):
            add_to_name += '_r'+str(restarts)
        if not(C==1.0):
            add_to_name += '_C'+str(C)
        if not(LR==0.01):
            add_to_name += '_LR'+str(LR)
        if C_Searching:
            add_to_name += '_C_Searching'
        if not(CV_Mode=='') or not(CV_Mode is None):
            add_to_name += '_' + CV_Mode

    titlestr = 'Classification for ' +method+' AUC: {0:.2f}, UAR: {1:.2f}, F1 : {2:.2f}'.format(mPerf[2],mPerf[1],mPerf[0])
    if method in['MIMAX','IA_mi_model','MIMAXaddLayer']:
        try:
            bestloss_length = len(bestloss)
            if bestloss_length==1:
                titlestr += ' BL : {0:.2f}'.format(bestloss[0])
            else:
                titlestr += ' BL : {0:.2f}'.format(bestloss)
        except TypeError:
            titlestr += ' BL : {0:.2f}'.format(bestloss)
    plt.title(titlestr)
    plt.savefig(path_filename)
    plt.show()
    plt.close()

    return(perf,perfB)

def doCrossVal(method,nRep,nFolds,numMetric,bags,labels_bags_c,labels_instance_c,StratifiedFold,opts,
               dataNormalizationWhen,dataNormalization,opts_MIMAX=None,verbose=False,
               epochsSIDLearlyStop=10):
    """
    This function perform a cross validation evaluation or StratifiedFold cross
    evaluation
    """

    debug = False

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
                labels_instance_c_train , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)

                labels_instance_c_train = np.hstack(labels_instance_c_train)
                if debug:
                    print("Number of positive instances train",np.sum((labels_instance_c_train+1)/2.))
                    print("Number of positive bags train",np.sum((np.hstack(labels_bags_c_train)+1)/2.))
                    print("Number of neagative instances train",-np.sum((labels_instance_c_train-1)/2.))

                if dataNormalizationWhen=='onTrainSet':
                    bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)

                gt_instances_labels_stack = np.hstack(labels_instance_c_test)


                pred_bag_labels, pred_instance_labels =train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,epochsSIDLearlyStop=epochsSIDLearlyStop)
                if debug:
                    print("Number of positive instances test",np.sum((np.hstack(labels_instance_c_test)+1)/2.))
                    print("Number of positive predicted instances test",np.sum((np.sign(pred_instance_labels)+1)/2.))
                    print("F1 score :",f1_score(np.hstack(labels_instance_c_test), np.sign(pred_instance_labels),labels=[-1,1]))
                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels,numMetric=numMetric)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels,numMetric=numMetric)
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
                       method,opts,opts_MIMAX=opts_MIMAX,verbose=verbose,epochsSIDLearlyStop=epochsSIDLearlyStop)

                perfObj[r,fold,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels,numMetric=numMetric)
                perfObjB[r,fold,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels,numMetric=numMetric)
                fold += 1
    return(perfObj,perfObjB)

def computePerfMImaxAllW(method,numberofW_to_keep,number_of_reboots ,numMetric,bags,labels_bags_c,labels_instance_c,opts,
               dataNormalizationWhen,dataNormalization,opts_MIMAX=None,verbose=False,
               test_size=0.1):
    """
    This function perform an evaluation on all the vector computed by Mi_max method
    """
    dataset,mini_batch_size_max,num_features,size_biggest_bag = opts



    perfObj=np.empty((number_of_reboots,numMetric))
    loss_values=np.empty((number_of_reboots,))
    perfObjB=np.empty((number_of_reboots,numMetric))


    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

    for train_index, test_index in sss.split(bags,labels_bags_c):
        labels_bags_c_train, labels_bags_c_test = \
            getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
        bags_train, bags_test = \
            getTest_and_Train_Sets(bags,train_index,test_index)
        _ , labels_instance_c_test = \
            getTest_and_Train_Sets(labels_instance_c,train_index,test_index)

        if dataNormalizationWhen=='onTrainSet':
            bags_train,bags_test = normalizeDataSetTrain(bags_train,bags_test,dataNormalization)

        gt_instances_labels_stack = np.hstack(labels_instance_c_test)

        mini_batch_size = min(mini_batch_size_max,len(bags_train))

        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        path_model = os.path.join(path_tmp)
        pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)

        tf.reset_default_graph()

        if opts_MIMAX is None:
            C,C_Searching,CV_Mode,restarts,LR = 1.0,False,None,120,0.01
        else:
            C,C_Searching,CV_Mode,restarts,LR = opts_MIMAX

        restarts = numberofW_to_keep*number_of_reboots

        classifierMI_max = tf_MI_max(LR=LR,restarts=restarts,is_betweenMinus1and1=True, \
                                     num_rois=size_biggest_bag,num_classes=1, \
                                     num_features=num_features,mini_batch_size=mini_batch_size, \
                                     verbose=verbose,C=C,CV_Mode=CV_Mode,max_iters=300,debug=False)

        export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                           class_indice=-1,shuffle=False,restarts_paral='paral', \
                           WR=True,C_Searching=C_Searching,storeVectors=True)

        name_dictW = export_dir
        with open(name_dictW, 'rb') as f:
            Dict = pickle.load(f)
        Wstored = Dict['Wstored']
        Bstored =  Dict['Bstored']
        Lossstored = Dict['Lossstored']
#        np_pos_value =  Dict['np_pos_value']
#        np_neg_value =  Dict['np_neg_value']

        modelcreator = ModelHyperplan(norm='',epsilon=0.01,mini_batch_size=mini_batch_size,
                                      num_features=num_features,num_rois=size_biggest_bag,num_classes=1,
                                      with_scores=False,seuillage_by_score=False,
                                      restarts=numberofW_to_keep-1)
        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)

        class_indice = -1
        ## Compute the best vectors
        for l in range(number_of_reboots):
#            print('reboot :',l)
            Wstored_extract = Wstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
            W_tmp = np.reshape(Wstored_extract,(-1,num_features),order='F')
            b_tmp =np.reshape( Bstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep],(-1,1,1),order='F')
            Lossstoredextract = Lossstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep]
            loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
            ## Creation of the model
            export_dir_local =  modelcreator.createIt(path_model,class_indice,W_tmp,b_tmp,loss_value)
#            number_zone = 300
#            Number_of_positif_elt = 1
#            dict_class_weight = {0:np_neg_value*number_zone ,1:np_pos_value* Number_of_positif_elt}
#            parameters=False,False,False,False
            pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir_local,\
                data_path_test,bags_test,size_biggest_bag,num_features,mini_batch_size,removeModel=True)

            perfObj[l,:]=getClassifierPerfomance(y_true=gt_instances_labels_stack,y_pred=pred_instance_labels,numMetric=numMetric)
            perfObjB[l,:]=getClassifierPerfomance(y_true=labels_bags_c_test,y_pred=pred_bag_labels,numMetric=numMetric)
            loss_values[l] = np.min(loss_value)

    return(perfObj,perfObjB,loss_values)

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
    elif method=='SIDLearlyStop':
        classifier = siDLearlyStop.SIDLearlyStop(verbose=verbose)
    elif method=='miDLearlyStop':
        classifier = miDLearlyStop.miDLearlyStop(verbose=verbose,max_iter=10)
    elif method=='miDLstab':
        classifier = miDLstab.miDLstab(verbose=verbose,max_iter=10,epochs_final=20)
    elif method=='kerasMultiMIMAX':
        classifier = kerasMultiMIMAX.kerasMultiMIMAX(verbose=verbose,epochs=2)
    elif method=='ensDLearlyStop':
        classifier = ensDLearlyStop.ensDLearlyStop(verbose=verbose,n_models=5)
    else:
        print('Method unknown: ',method)
        raise(NotImplementedError)
    return(classifier)

def train_and_test_MIL(bags_train,labels_bags_c_train,bags_test,labels_bags_c_test,\
                       method,opts,opts_MIMAX=None,verbose=False,
                       pointsPrediction=None,
                       get_bestloss=False,epochsSIDLearlyStop=10):
    """
    pointsPrediction=points : other prediction to do, points size
    
    /!\ For the moment we are only doing the binary case
    """

    # Number of class : 
    num_class = 1

    if method == 'MIMAX':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))

        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir= trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX,
                              verbose=verbose,get_bestloss=get_bestloss)

        if get_bestloss:
            export_dir,best_loss = export_dir

        if not(pointsPrediction is None):
            labels_pointsPrediction = [np.array(0.)]*len(pointsPrediction)
            data_path_points = Create_tfrecords(pointsPrediction, labels_pointsPrediction,\
                                              size_biggest_bag,num_features,'points',dataset)
            _, points_instance_labels = predict_MIMAX(export_dir,\
                    data_path_points,pointsPrediction,size_biggest_bag,
                    num_features,num_class,mini_batch_size,removeModel=False)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,num_class,
                mini_batch_size,removeModel=True)

    elif method == 'MAXMIMAX':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))
        AggregW = 'maxOfTanh'
        proportionToKeep = 1/12
        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir = trainMIMAX(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX,
                              verbose=verbose,get_bestloss=get_bestloss,
                              AggregW=AggregW,proportionToKeep=proportionToKeep)

        if get_bestloss:
            export_dir,best_loss = export_dir

        if not(pointsPrediction is None):
            labels_pointsPrediction = [np.array(0.)]*len(pointsPrediction)
            data_path_points = Create_tfrecords(pointsPrediction, labels_pointsPrediction,\
                                              size_biggest_bag,num_features,'points',dataset)
            print(data_path_points)
            _, points_instance_labels = predict_MIMAX(export_dir,\
                    data_path_points,pointsPrediction,size_biggest_bag,
                    num_features,num_class,mini_batch_size,AggregW=AggregW,removeModel=False)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        print(data_path_test)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,num_class,mini_batch_size
                ,AggregW=AggregW,removeModel=True)

    elif method == 'IA_mi_model':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))

        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir = trainIA_mi_model(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,opts_MIMAX=opts_MIMAX,
                              verbose=verbose,get_bestloss=get_bestloss)

        if get_bestloss:
            export_dir,best_loss = export_dir

        if not(pointsPrediction is None):
            labels_pointsPrediction = [np.array(0.)]*len(pointsPrediction)
            data_path_points = Create_tfrecords(pointsPrediction, labels_pointsPrediction,\
                                              size_biggest_bag,num_features,'points',dataset)
            _, points_instance_labels = predict_MIMAX(export_dir,\
                    data_path_points,pointsPrediction,size_biggest_bag,num_features,
                    num_class,mini_batch_size,removeModel=False)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,
                num_class,mini_batch_size,removeModel=True)


    elif method == 'MIMAXaddLayer':
        dataset,mini_batch_size_max,num_features,size_biggest_bag = opts
        mini_batch_size = min(mini_batch_size_max,len(bags_train))

        #Training
        data_path_train = Create_tfrecords(bags_train, labels_bags_c_train,size_biggest_bag,\
                                           num_features,'train',dataset)
        export_dir=trainMIMAXaddLayer(bags_train, labels_bags_c_train,data_path_train,\
                              size_biggest_bag,num_features,mini_batch_size,
                              opts_MIMAX=opts_MIMAX,get_bestloss=get_bestloss)

        if get_bestloss:
            export_dir,best_loss = export_dir

        if not(pointsPrediction is None):
            labels_pointsPrediction = [np.array(0.)]*len(pointsPrediction)
            data_path_points = Create_tfrecords(pointsPrediction, labels_pointsPrediction,\
                                              size_biggest_bag,num_features,'points',dataset)
            _, points_instance_labels = predict_MIMAX(export_dir,\
                    data_path_points,pointsPrediction,size_biggest_bag,num_features,
                    mini_batch_size,removeModel=False,predict_parall=False)

        # Testing
        data_path_test = Create_tfrecords(bags_test, labels_bags_c_test,\
                                          size_biggest_bag,num_features,'test',dataset)
        pred_bag_labels, pred_instance_labels = predict_MIMAX(export_dir,\
                data_path_test,bags_test,size_biggest_bag,num_features,mini_batch_size,
                predict_parall=False)

    elif method in list_of_ClassicalMI:

        classifier = get_classicalMILclassifier(method,verbose=verbose)
        if method=='SIDLearlyStop':
            classifier.fit(bags_train, labels_bags_c_train,epochs=epochsSIDLearlyStop)
        else:
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


def parser(record,num_rois=300,num_features=2048,num_class=1):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'num_features':  tf.FixedLenFeature([], tf.int64),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_class],tf.float32)}
    parsed = tf.parse_single_example(record, keys_to_features)

    # Cast label data into int32
    label = parsed['label']
#    label = tf.slice(label,[classe_index],[1])
#    label = tf.squeeze(label) # To get a vector one dimension
    fc7 = parsed['fc7']
    fc7 = tf.reshape(fc7, [num_rois,num_features])
    return fc7,label

def predict_MIMAX(export_dir,data_path_test,bags_test,size_biggest_bag,num_features,\
               num_class=1,mini_batch_size=256,verbose=False,predict_parall=True,AggregW=None,
               removeModel=True):
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
                                                     ,num_features=num_features,\
                                                     num_class=num_class))
    dataset_batch = test_dataset.batch(mini_batch_size)
    dataset_batch.cache()
    iterator = dataset_batch.make_one_shot_iterator()
    next_element = iterator.get_next()

    # Parameters of the classifier
    with_tanh = True
    with_tanh_alreadyApplied = False
    with_softmax = False
    with_softmax_a_intraining = False
    if not(AggregW is None): 
        if'Tanh' in AggregW or 'Sign' in AggregW: 
            # The tanh transformation have already be done
            with_tanh = False
            with_tanh_alreadyApplied = True

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
                Prod_best = graph.get_tensor_by_name("Tanh_2:0")
            except KeyError:
                try:
                    Prod_best = graph.get_tensor_by_name("Tanh_1:0")
                except KeyError:
                     Prod_best = graph.get_tensor_by_name("Tanh:0")
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
    directory = path_tmp
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    path_name = os.path.join(directory,name)
    if os.path.isfile(path_name):
        os.remove(path_name) # Need to remove the folder !
    writer = tf.python_io.TFRecordWriter(path_name)
    i = 0
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
                        'label' : _floats_feature(elt_label),
                        'name_img' : _bytes_feature(str.encode(str(i)))})
        i += 1
        example = tf.train.Example(features=features) #.tostring()
        writer.write(example.SerializeToString())
    writer.close()

    return(path_name)

def trainMIMAX(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
               num_features,mini_batch_size,opts_MIMAX=None,verbose=False,
               get_bestloss=False,AggregW=None,proportionToKeep=0.):
    """
    This function train a tidy MIMAX model
    """
    path_model = os.path.join(path_tmp,'MI_max')
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)
    
    try:
        M= len(labels_bags_c_train[0])
    except TypeError:
        M = 1

    tf.reset_default_graph()

    if not(opts_MIMAX is None):
        C,C_Searching,CV_Mode,restarts,LR = opts_MIMAX
    else:
        C,C_Searching,CV_Mode,restarts,LR = 1.0,False,None,11,0.01

    if not(AggregW is None):
        numberOfFinalVector = 8
        restarts = (restarts+1)*numberOfFinalVector-1

    classifierMI_max = tf_MI_max(LR=LR,restarts=restarts,is_betweenMinus1and1=True, \
                                 num_rois=size_biggest_bag,num_classes=M, \
                                 num_features=num_features,mini_batch_size=mini_batch_size, \
                                 verbose=verbose,C=C,CV_Mode=CV_Mode,max_iters=300,
                                 debug=False,AggregW=AggregW,proportionToKeep=proportionToKeep)
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

def trainIA_mi_model(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
               num_features,mini_batch_size,opts_MIMAX=None,verbose=False,get_bestloss=False):
    """
    This function train a tidy mi_model,with IA = instance label assignation
    """
    path_model = os.path.join(path_tmp,'MI_max')
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)

    tf.reset_default_graph()

    if not(opts_MIMAX is None):
        C,C_Searching,CV_Mode,restarts,LR = opts_MIMAX
    else:
        C,C_Searching,CV_Mode,restarts,LR = 1.0,False,None,11,0.01
    classifierIA_mi = tf_mi_model(LR=LR,C=C,restarts=restarts,num_rois=size_biggest_bag,
                       max_iters=300,verbose=verbose,mini_batch_size=mini_batch_size,
                       num_features=num_features,debug=False,num_classes=1,CV_Mode=CV_Mode,
                       is_betweenMinus1and1=True)

    C_values =  np.logspace(-3,2,6,dtype=np.float32)
    classifierIA_mi.set_C_values(C_values)

    export_dir = classifierIA_mi.fit_mi_model_tfrecords(data_path=data_path_train, \
                   class_indice=-1,shuffle=False,WR=True,restarts_paral='paral',
                   C_Searching=C_Searching)

    if get_bestloss:
        bestloss = classifierIA_mi.get_bestloss()
        tf.reset_default_graph()
        return(export_dir,bestloss)
    else:
        tf.reset_default_graph()
        return(export_dir)

def trainMIMAXaddLayer(bags_train, labels_bags_c_train,data_path_train,size_biggest_bag,
               num_features,mini_batch_size,opts_MIMAX=None,get_bestloss=False):
    """
    This function train a tidy MIMAX model with one layer more
    """
    path_model = os.path.join(path_tmp,'MI_max')
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
                                 AddOneLayer=True,Optimizer='lbfgs')

    export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
                       class_indice=0,shuffle=False,restarts_paral='', \
                       WR=True,C_Searching=C_Searching)
    tf.reset_default_graph()
    if get_bestloss:
        bestloss = classifierMI_max.get_bestloss()
        tf.reset_default_graph()
        return(export_dir,bestloss)
    else:
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



def BenchmarkRunSIDLearlyStop(firstset=0):

    epochs_list = [1,3,10]
    datasets = ['Newsgroups','Birds','SIVAL']
    if firstset==1:
        datasets = datasets[::-1]
    list_of_algo= ['SIDLearlyStop']
    np.random.shuffle(epochs_list)
    np.random.shuffle(datasets)
    # Warning the IA_mi_model repeat the element to get bag of equal size that can lead to bad results !

    for epochs in epochs_list:
        for method in list_of_algo:
            for dataWhen,dataNorm in zip(['onTrainSet',None],['std',None]):
                for dataset in datasets:
                    print('==== ',method,dataset,epochs,' ====')
                    evalPerf(method=method,dataset=dataset,reDo=False,verbose=False,
                             dataNormalizationWhen=dataWhen,dataNormalization=dataNorm,
                             epochsSIDLearlyStop=epochs)

def BenchmarkRunmiDLearlyStop(firstset=0):

    epochs_list =[1,3,10]
    datasets = ['Newsgroups','Birds','SIVAL']
    if firstset==1:
        datasets = datasets[::-1]
    list_of_algo= ['miDLearlyStop']
    np.random.shuffle(epochs_list)
    np.random.shuffle(datasets)
    # Warning the IA_mi_model repeat the element to get bag of equal size that can lead to bad results !

    for epochs in epochs_list:
        for method in list_of_algo:
            for dataWhen,dataNorm in zip([None,'onTrainSet'],[None,'std']):
                for dataset in datasets:
                    print('==== ',method,dataset,epochs,' ====')
                    evalPerf(method=method,dataset=dataset,reDo=False,verbose=False,
                             dataNormalizationWhen=dataWhen,dataNormalization=dataNorm,
                             epochsSIDLearlyStop=epochs)

def BenchmarkRun():

    datasets = ['SIVAL','Birds','Newsgroups']
    list_of_algo= ['MIMAX','IA_mi_model','MIMAXaddLayer']
    # Warning the IA_mi_model repeat the element to get bag of equal size that can lead to bad results !

    for method in list_of_algo:
        for dataset in datasets:
            print('==== ',method,dataset,' ====')
            for dataWhen,dataNorm in zip(['onTrainSet',None],['std',None]):
#            for dataWhen,dataNorm in zip(['onTrainSet'],['std']):
                evalPerf(method=method,dataset=dataset,reDo=False,verbose=False,
                         dataNormalizationWhen=dataWhen,dataNormalization=dataNorm)

def BenchmarkRunClassical():

    datasets = ['Birds','Newsgroups']
    list_of_algo= ['LinearSISVM','miSVM','MISVM']
    np.random.shuffle(list_of_algo)
    np.random.shuffle(datasets)

    for method in list_of_algo:
        for dataset in datasets:
            print('==== ',method,dataset,' ====')
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
#    ToyProblemRun()
    BenchmarkRun()
