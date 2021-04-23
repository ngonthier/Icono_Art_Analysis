#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:49:51 2019

@author: gonthier
"""

from TL_MIL import tfR_FRCNN
from RunVarStudyAll import Study_eval_perf_onSplit_of_IconArt
from RunVarStudyAll import VariationStudyPart1,VariationStudyPart2,VariationStudyPart3,\
    unefficient_way_MaxOfMax_evaluation
from IMDB_study import modify_EdgeBoxesWrongBoxes

from CNNfeatures import Compute_EdgeBoxesAndCNN_features

from MILnet_eval import mainEval

def main():
    
    
    for database in ['watercolor','PeopleArt']:
        for loss_type in ['','hinge']:
            # IconArt MaxOfMax hinge loss
            for with_scores in [True,False]:
                tfR_FRCNN(demonet = 'res152_COCO',database=database, ReDo=False,
                              verbose = True,testMode = False,jtest = 'cow',
                              PlotRegions = False,saved_clf=False,RPN=False,
                              CompBest=False,Stocha=True,k_per_bag=300,
                              parallel_op=True,CV_Mode='',num_split=2,
                              WR=True,init_by_mean =None,seuil_estimation='',
                              restarts=11,max_iters_all_base=3000,LR=0.01,with_tanh=True,
                              C=1.0,Optimizer='GradientDescent',norm='',
                              transform_output='tanh',with_rois_scores_atEnd=False,
                              with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                              Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
                              k_intopk=1,C_Searching=False,predict_with='',
                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
                              loss_type=loss_type,storeVectors=False,storeLossValues=False,
                              MaxOfMax=True)
            
            # MaxOfMMean hinge loss
            for with_scores in [True,False]:
                tfR_FRCNN(demonet = 'res152_COCO',database=database, ReDo=False,
                              verbose = True,testMode = False,jtest = 'cow',
                              PlotRegions = False,saved_clf=False,RPN=False,
                              CompBest=False,Stocha=True,k_per_bag=300,
                              parallel_op=True,CV_Mode='',num_split=2,
                              WR=True,init_by_mean =None,seuil_estimation='',
                              restarts=11,max_iters_all_base=3000,LR=0.01,with_tanh=True,
                              C=1.0,Optimizer='GradientDescent',norm='',
                              transform_output='tanh',with_rois_scores_atEnd=False,
                              with_scores=with_scores,epsilon=0.01,restarts_paral='paral',
                              Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
                              k_intopk=1,C_Searching=False,predict_with='',
                              gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                              thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
                              loss_type=loss_type,storeVectors=False,storeLossValues=False,
                              MaxMMeanOfMax=True)
        

    tfR_FRCNN(demonet = 'res152_COCO',database = 'VOC2007', ReDo=True,
                  verbose = True,testMode = False,jtest = 'cow',
                  PlotRegions = False,saved_clf=False,RPN=False,
                  CompBest=False,Stocha=True,k_per_bag=300,
                  parallel_op=True,CV_Mode='',num_split=2,
                  WR=True,init_by_mean =None,seuil_estimation='',
                  restarts=11,max_iters_all_base=3000,LR=0.01,with_tanh=True,
                  C=1.0,Optimizer='GradientDescent',norm='',
                  transform_output='tanh',with_rois_scores_atEnd=False,
                  with_scores=True,epsilon=0.01,restarts_paral='paral',
                  Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
                  k_intopk=1,C_Searching=False,predict_with='',
                  gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
                  thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
                  loss_type='hinge',storeVectors=False,storeLossValues=False)
    
    
    
    # MaxOfMax !!! 
   # unefficient_way_MaxOfMax_evaluation()
    
    # EdgeBoxes MiModel k 300
#    tfR_FRCNN(demonet = 'res152',database = 'VOC2007', ReDo=True,
#                                          verbose = True,testMode = False,jtest = 'cow',
#                                          PlotRegions = False,saved_clf=False,RPN=False,
#                                          CompBest=False,Stocha=True,k_per_bag=300,
#                                          parallel_op=True,CV_Mode='',num_split=2,
#                                          WR=True,init_by_mean =None,seuil_estimation='',
#                                          restarts=11,max_iters_all_base=300,LR=0.01,with_tanh=True,
#                                          C=1.0,Optimizer='GradientDescent',norm='',
#                                          transform_output='tanh',with_rois_scores_atEnd=False,
#                                          with_scores=False,epsilon=0.01,restarts_paral='paral',
#                                          Max_version='',w_exp=10.0,seuillage_by_score=False,seuil=0.9,
#                                          k_intopk=1,C_Searching=False,predict_with='',
#                                          gridSearch=False,thres_FinalClassifier=0.5,n_jobs=1,
#                                          thresh_evaluation=0.05,TEST_NMS=0.3,AggregW='',proportionToKeep=0.25,
#                                          loss_type='',storeVectors=False,storeLossValues=False,
#                                          metamodel='EdgeBoxes',model='mi_model')
    
    # Hinge loss sur les autres bases d'images
#    for dataset in ['watercolor','PeopleArt','VOC2007']:
#    for dataset in ['VOC2007']:
#        VariationStudyPart1(database=dataset,scenarioSubset=[3,22])
#        VariationStudyPart2(database=dataset,scenarioSubset=[3,22],withoutAggregW=True)
#        VariationStudyPart3(database=dataset,scenarioSubset=[3,22],withoutAggregW=True)
#    unefficient_way_MaxOfMax_evaluation()
#    Compute_EdgeBoxesAndCNN_features(database='watercolor',k_regions=300)
#    Compute_EdgeBoxesAndCNN_features(database='VOC2007',k_regions=300)
#    Compute_EdgeBoxesAndCNN_features(k_regions=2000)
#    Compute_EdgeBoxesAndCNN_features(database='watercolor',k_regions=2000)
#    Compute_EdgeBoxesAndCNN_features(database='VOC2007',k_regions=2000)
    
    # IconArt v1 avec vgg16_COCO as features extractor
    # without score and with score on fc7 and fc6
#    VariationStudyPart1(database='IconArt_v1',scenarioSubset=[0,5],demonet = 'vgg16_COCO')
#    VariationStudyPart2(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO')
#    VariationStudyPart1(database='IconArt_v1',scenarioSubset=[0,5],demonet = 'vgg16_COCO',layer='fc6')
#    VariationStudyPart2(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO',layer='fc6')
#    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO')
#    VariationStudyPart3(database='IconArt_v1',scenarioSubset=[0,5],withoutAggregW=True,demonet = 'vgg16_COCO',layer='fc6')
##    tfR_FRCNN(demonet = 'res152',database = 'VOC2007', ReDo=False,
#          verbose = True,testMode = False,jtest = 'cow',
#          PlotRegions = False,saved_clf=False,RPN=False,
#          CompBest=False,Stocha=True,k_per_bag=300,
#          parallel_op=True,CV_Mode='',num_split=2,
#          WR=True,init_by_mean =None,seuil_estimation='',
#          restarts=11,max_iters_all_base=300,LR=0.01,
#          C=1.0,Optimizer='GradientDescent',norm='',
#          transform_output='tanh',with_rois_scores_atEnd=False,
#          with_scores=False,epsilon=0.01,restarts_paral='paral',
#          predict_with='MI_max',
#          AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False,
#          metamodel='EdgeBoxes')

#    tfR_FRCNN(demonet = 'res152',database = 'IconArt_v1', ReDo=False,
#          verbose = True,testMode = False,jtest = 'cow',
#          PlotRegions = False,saved_clf=False,RPN=False,
#          CompBest=False,Stocha=True,k_per_bag=300,
#          parallel_op=True,CV_Mode='',num_split=2,
#          WR=True,init_by_mean =None,seuil_estimation='',
#          restarts=11,max_iters_all_base=300,LR=0.01,
#          C=1.0,Optimizer='GradientDescent',norm='',
#          transform_output='tanh',with_rois_scores_atEnd=False,
#          with_scores=False,epsilon=0.01,restarts_paral='paral',
#          predict_with='MI_max',
#          AggregW =None ,proportionToKeep=1.0,model='mi_model',debug=False,
#          metamodel='EdgeBoxes')
#    tfR_FRCNN(demonet = 'res152',database = 'VOC2007', ReDo=False,
#          verbose = True,testMode = False,jtest = 'cow',
#          PlotRegions = False,saved_clf=False,RPN=False,
#          CompBest=False,Stocha=True,k_per_bag=300,
#          parallel_op=True,CV_Mode='',num_split=2,
#          WR=True,init_by_mean =None,seuil_estimation='',
#          restarts=11,max_iters_all_base=300,LR=0.01,
#          C=1.0,Optimizer='GradientDescent',norm='',
#          transform_output='tanh',with_rois_scores_atEnd=False,
#          with_scores=False,epsilon=0.01,restarts_paral='paral',
#          predict_with='MI_max',
#          AggregW =None ,proportionToKeep=1.0,model='mi_model',debug=False,
#          metamodel='EdgeBoxes')
#    modify_EdgeBoxesWrongBoxes(database='watercolor',k_per_bag = 300,\
#                               metamodel = 'EdgeBoxes',demonet='res152')
    
    
    # EdgeBoxes on watercolor ! plus rien ne marche !!!
#    try:
#        tfR_FRCNN(demonet = 'res152',database = 'watercolor', ReDo=True,
#              verbose = True,testMode = False,jtest = 'cow',
#              PlotRegions = False,saved_clf=False,RPN=False,
#              CompBest=False,Stocha=True,k_per_bag=300,
#              parallel_op=True,CV_Mode='',num_split=2,
#              WR=True,init_by_mean =None,seuil_estimation='',
#              restarts=11,max_iters_all_base=300,LR=0.01,
#              C=1.0,Optimizer='GradientDescent',norm='',
#              transform_output='tanh',with_rois_scores_atEnd=False,
#              with_scores=False,epsilon=0.01,restarts_paral='paral',
#              predict_with='MI_max',
#              AggregW =None ,proportionToKeep=1.0,model='MI_max',debug=False,
#              metamodel='EdgeBoxes')
#        tfR_FRCNN(demonet = 'res152',database = 'watercolor', ReDo=True,
#              verbose = True,testMode = False,jtest = 'cow',
#              PlotRegions = False,saved_clf=False,RPN=False,
#              CompBest=False,Stocha=True,k_per_bag=300,
#              parallel_op=True,CV_Mode='',num_split=2,
#              WR=True,init_by_mean =None,seuil_estimation='',
#              restarts=11,max_iters_all_base=300,LR=0.01,
#              C=1.0,Optimizer='GradientDescent',norm='',
#              transform_output='tanh',with_rois_scores_atEnd=False,
#              with_scores=False,epsilon=0.01,restarts_paral='paral',
#              predict_with='MI_max',
#              AggregW =None ,proportionToKeep=1.0,model='mi_model',debug=False,
#              metamodel='EdgeBoxes')
#    except:
#        pass
    
        

if __name__ == '__main__':
    main()