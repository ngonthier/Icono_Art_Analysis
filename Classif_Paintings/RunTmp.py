#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:10:46 2019

@author: gonthier
"""

from Exp_runs import mainDatabase,mainDatabase_HL,mainDatabase_mi_model
from Baseline_script import Baseline_FRCNN_TL_Detect
from CNNfeatures import Compute_EdgeBoxesAndCNN_features

def precomputeEdgeBoxes():
    datasets = ['clipart']
    for database in datasets:
        try:
            Compute_EdgeBoxesAndCNN_features(database=database,k_regions=300)
        except Exception as e:
            print(e)
            pass 

def OtherBaselines():
    datasets = ['comic','CASPApaintings']
    list_methods =['MAX1','MAXA','MISVM','miSVM']
    normalisation = False
    restarts = 0
    max_iter = 50
    variance_thres = 0.9
    for database in datasets:
        for method in list_methods: 
            if method in ['MAXA','MAX1','SISVM']:
                GS_tab = [False,True]
            else:
                GS_tab = [False]
            for GS in GS_tab:
                try:
                    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                            normalisation= normalisation,baseline_kind=method,verbose=False,
                            gridSearch=GS,k_per_bag=300,n_jobs=1,PCAuse=False,
                            restarts=restarts,max_iter=max_iter,reDo=False)
                except Exception as e:
                    print(e)
                    pass 
                
def comicMaxA():
    datasets = ['comic']
    list_methods =['MAX1','MAXA','MISVM','miSVM']
    list_methods =['MISVM','miSVM']
    normalisation = False
    restarts = 0
    max_iter = 50
    variance_thres = 0.9
    for database in datasets:
        for method in list_methods: 
            if method in ['MAXA','MAX1','SISVM']:
                GS_tab = [True]
            else:
                GS_tab = [False]
            for GS in GS_tab:
                try:
                    Baseline_FRCNN_TL_Detect(demonet = 'res152_COCO',database =database,Test_on_k_bag=False,
                            normalisation= normalisation,baseline_kind=method,verbose=False,
                            gridSearch=GS,k_per_bag=300,n_jobs=1,PCAuse=False,
                            restarts=restarts,max_iter=max_iter,reDo=True)
                except Exception as e:
                    print(e)
                    pass 


if __name__ == '__main__':
    # Precompute EdgeBoxes data
    #precomputeEdgeBoxes() # Cela n a pas marché
#    mainDatabase(database_tab=['clipart','comic','CASPApaintings']) # To compute the EdgeBoxes scores
#    mainDatabase_HL(database_tab=['IconArt_v1','watercolor','PeopleArt','CASPApaintings','comic','clipart'])
    comicMaxA()
#    mainDatabase_mi_model(database_tab=['IconArt_v1','watercolor','PeopleArt','CASPApaintings','comic','clipart'])
    

