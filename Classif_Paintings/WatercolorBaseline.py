#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:44:03 2019

@author: gonthier
"""

from Baseline_script import Baseline_FRCNN_TL_Detect

def BaselineRunAll():
    """ Run severals baseline model on two datasets
    """
    
    datasets = ['watercolor']
    list_methods =['MISVM','miSVM','SISVM','MAX1','MAXA']
    normalisation = False
    restarts = 0
    max_iter = 50
    variance_thres = 0.9
    for database in datasets:
        for method in list_methods: 
            if method in ['MAXA','MAX1','SISVM']:
                GS_tab = [False,True]
                PCA_tab = [True,False]
            else:
                GS_tab = [False]
                PCA_tab = [True]
            for GS in GS_tab:
                for PCAuse in PCA_tab:
                    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                            normalisation= normalisation,baseline_kind=method,verbose=False,
                            gridSearch=GS,k_per_bag=300,n_jobs=4,PCAuse=PCAuse,variance_thres= variance_thres,
                            restarts=restarts,max_iter=max_iter,reDo=False)
    restarts = 10
    for database in datasets:
        for method in ['MISVM','miSVM']:
            Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                    normalisation= normalisation,baseline_kind=method,verbose=False,
                    gridSearch=False,k_per_bag=300,n_jobs=4,PCAuse=PCAuse,variance_thres= variance_thres,
                    restarts=restarts,max_iter=max_iter,reDo=False)
            
if __name__ == '__main__':
    BaselineRunAll()