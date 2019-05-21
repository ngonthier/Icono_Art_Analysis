# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:09:15 2019

@author: gonthier
"""

def getMethodConfig(method,dataset,expType=None):
    """
    method is a string containing the name of the MIL methods to use
    dataset is a string containing the name of the data set
    expType is a string containing the ype of experiment to conduct
    OUTPUT:
    opt is an object containing the field used to configure the method
    specified for the dataset and experiment type.
    """
#    opt = None
    
    # Option selon la methode switch method
    if method=='miSVM':
        opt_method = cfgmiSVM(dataset)
    elif method=='MISVM':
        opt_method = cfgMISVM(dataset)
    elif method in ['SIL']:
        opt_method = cfgSIL(dataset)
    elif method in ['SISVM','SI-SVM']:
        opt_method = cfgSISVM(dataset)
    elif method=='MIdummy':
        opt_method = None,None
    else:
        print('Method unknwon')
        raise(NotImplementedError)

    return(opt_method)
    
def cfgSIL(dataset): 
   
    if dataset in ['Newsgroups','Spam','Birds']:
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100],
                      'p':[2],'gamma':[0.001,0.01,0.1,1,10]}
        scoring  = 'roc_auc'
    else:
        # 'Dont know how to configure for dataset
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100], 
              'p':[2],'gamma':[0.001,0.01,0.1,1,10]}
        scoring  = 'roc_auc'
        
    return(parameters,scoring)
    
def cfgSISVM(dataset): 
   
    if dataset in ['Newsgroups','Spam','Birds']:
        parameters = [{'C': [1, 10, 100], 'kernel': ['poly'],'degree':[2]},
        {'C':[1, 10, 100], 'gamma': [0.001,0.01,0.1,1,10], 'kernel': ['rbf']},] 
    # specifies that two grids should be explored: one with a linear kernel and 
    #C values in [1, 10, 100, 1000], and the second one with an RBF kernel, 
    #and the cross-product of C values ranging in [1, 10, 100, 1000] and gamma values in 
        scoring  = 'roc_auc'
    else:
        # 'Dont know how to configure for dataset
        parameters = {'kernel':('poly','rbf'), 'C':[1,10,100], 
              'degree':[2],'gamma':[0.001,0.01,0.1,1,10]}
        scoring  = 'roc_auc'
        
    return(parameters,scoring)
    
    
def cfgmiSVM(dataset): 
   
    if dataset in ['Newsgroups','Spam','Birds']:
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100],
                      'p':[2],'gamma':[0.001,0.01,0.1,1,10],'max_iters':[30],
                      'restarts':[0]}
        scoring  = 'roc_auc'
    else:
        # 'Dont know how to configure for dataset
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100], 
              'p':[2],'gamma':[0.001,0.01,0.1,1,10],'max_iters':[30],
              'restarts':[0]}
        scoring  = 'roc_auc'
        
    return(parameters,scoring)
    
def cfgMISVM(dataset): 
   
    if dataset in ['Newsgroups','Spam','Birds']:
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100],
                      'p':[2],'gamma':[0.001,0.01,0.1,1,10],'max_iters':[100],
                      'restarts':[0]}
        scoring  = 'roc_auc'
    else:
        # 'Dont know how to configure for dataset
        parameters = {'kernel':('polynomial','rbf'), 'C':[1,10,100], 
              'p':[2],'gamma':[0.001,0.01,0.1,1,10],'max_iters':[100],
              'restarts':[0]}
        scoring  = 'roc_auc'
        
    return(parameters,scoring)
