# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:49:39 2018

@author: gonthier
"""
    
import numpy as np
from .Dataset.ExtractBirds import ExtractBirds
from .Dataset.ExtractSIVAL import ExtractSIVAL,ExtractSubsampledSIVAL
from .Dataset.ExtractNewsgroups import ExtractNewsgroups
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,recall_score
#from getMethodConfig import getMethodConfig
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer,SCORERS
#import pickle
#import os

#def mainTestFunction(allMethods,dataset,dataNormalizationWhen=None,
#                     dataNormalization=None,reDo=False,n_jobs=-1):
#    """
#    @param : allMethods list of the methods to test
#    @param : dataset string describing the dataset
#    @param : dataNormalizationWhen = 'onAllSet' or 'OnTrainSet' or None
#    @param : dataNormalization = 'std' or 'var', '0-1' or None
#    @param : reDo : erase the results file
#    @param : n_jobs : number of jobs (processors) for the GridSearch case
#    """
#    ## PARAMETERS
#    if isinstance(allMethods, str):
#        allMethods=[allMethods]
#
#    Dataset=getDataset(dataset)
#    ## LOAD DATASET
#
#    print('=============================================================')
#    print('= DATA SET ACQUIRED: ')
#    print('-------------------------------------------------------------')
#    ## TEST METHODS ON THE DATA SET
#    script_dir = os.path.dirname(__file__)
#    filename = dataset + '.pkl'
#    file_results = os.path.join(script_dir,'Results',filename)
#    if reDo:
#        results = {}
#    else:
#        try:
#            results = pickle.load(open(file_results))
#        except FileNotFoundError:
#            results = {}
#        
#    
#    for i in range(len(allMethods)): # Boucle sur les methodes
#        method=allMethods[i]
#        if not(method in results.keys()):
#            print('= Method: ',method)
#            # get config
#            opt = getMethodConfig(method,dataset)
#            list_names,bags,labels_bags,labels_instance = Dataset
#            if dataNormalizationWhen=='onAllSet':
#                bags = normalizeDataSetFull(bags,dataNormalization)
#            
#            for c_i,c in enumerate(list_names):
#                # Loop on the different class, we will consider each group one after the other
#                print("For class :",c)
#                labels_bags_c = labels_bags[c_i]
#                labels_instance_c = labels_instance[c_i]
#                D = bags,labels_bags_c,labels_instance_c
#                perf,perfB=performExperimentWithCrossVal(D,method,opt,dataset,
#                                        dataNormalizationWhen,dataNormalization,
#                                        n_jobs=n_jobs)
#                    
#            ## Results
#            print('=============================================================')
#            print('= ',method)
#            print('-------------------------------------------------------------')
#            print('- instances') # f1Score,UAR,aucScore,accuracyScore
#            print('AUC: ',perf[2,:])
#            print('UAR: ',perf[1,:])
#            print('F1: ',perf[0,:])
#            print('Accuracy: ',perf[3,:])
#            print('- bags')
#            print('AUC: ',perfB[2,:])
#            print('UAR: ',perfB[1,:])
#            print('F1: ',perfB[0,:])
#            print('Accuracy: ',perfB[3,:])
#            print('-------------------------------------------------------------')
#            results[method] = [perf,perfB]
#    pickle.dump(results,open(file_results,'w'))
#        


def normalizeDataSetFull(D,dataNormalization):
    """
    In those case we renormalize the dataset in taking into account the test set
    that's maybe not so rigourus
    """
    
    eps = 10**(-16)
    if dataNormalization=='std':
        Dfull = np.concatenate(D)
        mean_D = np.mean(Dfull,axis=0)
        std_D = np.std(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-mean_D)/std_D
    elif dataNormalization=='0-1':
        Dfull = np.concatenate(D)
        min_D = np.min(Dfull,axis=0)
        max_D = np.max(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-min_D)/(max_D-min_D)
    elif dataNormalization in ['var','variance']:
        Dfull = np.concatenate(D)
        mean_D = np.mean(Dfull,axis=0)
        var_D = np.var(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-mean_D)/var_D
    return(D)
    
def normalizeDataSetTrain(D,DT,dataNormalization):
    """
    In those case we renormalize but compting the case on the training set and 
    apply it to the test set also
    """
    
    eps = 10**(-16)
    if dataNormalization=='std':
        Dfull = np.concatenate(D)
        mean_D = np.mean(Dfull,axis=0)
        std_D = np.std(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-mean_D)/std_D
        for i,elt in enumerate(DT):
            DT[i] =  (elt-mean_D)/std_D
    elif dataNormalization=='0-1':
        Dfull = np.concatenate(D)
        min_D = np.min(Dfull,axis=0)
        max_D = np.max(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-min_D)/(max_D-min_D)
        for i,elt in enumerate(DT):
            DT[i] =  (elt-min_D)/(max_D-min_D)
    elif dataNormalization in ['var','variance']:
        Dfull = np.concatenate(D)
        mean_D = np.mean(Dfull,axis=0)
        var_D = np.var(Dfull,axis=0) +eps
        for i,elt in enumerate(D):
            D[i] =  (elt-mean_D)/var_D
        for i,elt in enumerate(DT):
            DT[i] =  (elt-mean_D)/var_D
    return(D,DT)

def getTest_and_Train_Sets(Data,indextrain,indextest):
    """
    Split the list of data in train and test set according to the index lists
    provided
    """
    DataTrain = [ Data[i] for i in indextrain]
    DataTest = [ Data[i] for i in indextest]
    return(DataTrain,DataTest)

def getClassifierPerfomance(y_true, y_pred):
    """
    This function compute 4 differents metrics :
    metrics = [f1Score,UAR,aucScore,accuracyScore]
    """
    f1Score = f1_score(y_true, np.sign(y_pred),labels=[-1,1])
    # F1-score is the harmonic mean between precision and recall.
    aucScore = roc_auc_score(y_true, y_pred) 
    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    UAR = recall_score(y_true, np.sign(y_pred),average='macro',labels=[-1,1]) 
    # Calculate metrics for each label, and find their unweighted mean. 
    # This does not take label imbalance into account. :
    # ie UAR = unweighted average recall (of each class)
    accuracyScore = accuracy_score(y_true, np.sign(y_pred)) # Accuracy classification score
    metrics = [f1Score,UAR,aucScore,accuracyScore] 
    return(metrics)
    
def getMeanPref(perfO=None,dataset=None):
    if not(dataset is None):
        if  len(perfO.shape)==4:
            mean = np.mean(perfO,axis=(0,1,2))
            std = np.std(perfO,axis=(0,1,2))
        if  len(perfO.shape)==3:
            mean = np.mean(perfO,axis=(0,1))
            std = np.std(perfO,axis=(0,1))
    elif dataset=='SIVAL':
        if  len(perfO.shape)==5:
            mean = np.mean(perfO,axis=(0,1,2,3))
            std = np.std(perfO,axis=(0,1,2,3))
        if  len(perfO.shape)==4:
            mean = np.mean(perfO,axis=(0,1,2))
            std = np.std(perfO,axis=(0,1,2))
    return([mean,std])
    

def getDataset(dataset=None):

    if 'SIVALfull'==dataset:
        Dataset = ExtractSIVAL()
    if 'SIVAL'==dataset:
        Dataset = ExtractSubsampledSIVAL() # Subsampled version
    if 'Birds'==dataset:
        Dataset = ExtractBirds()
    if 'Newsgroups'==dataset:
        Dataset = ExtractNewsgroups()

    return Dataset 
    

    