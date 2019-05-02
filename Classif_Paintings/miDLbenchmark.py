#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:30:10 2019

@author: gonthier
"""

from MIbenchmarkage import evalPerf
import numpy as np

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
                    
if __name__ == '__main__':
    BenchmarkRunmiDLearlyStop()