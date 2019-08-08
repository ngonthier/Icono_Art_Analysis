#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:49:51 2019

@author: gonthier
"""

from TL_MIL import tfR_FRCNN
from RunVarStudyAll import Study_eval_perf_onSplit_of_IconArt
from RunVarStudyAll import VariationStudyPart1,VariationStudyPart2,VariationStudyPart3,\
    unefficient_way_MaxOfMax_evaluation,unefficient_way_OneHiddenLayer_evaluation,\
    unefficient_evaluation_PrintResults
from IMDB_study import modify_EdgeBoxesWrongBoxes

from CNNfeatures import Compute_EdgeBoxesAndCNN_features

from MILnet_eval import runSeveralMInet

def ExperienceRuns():
    
    database_tab =  ['IconArt_v1','watercolor','PeopleArt']
    for database in database_tab:
        # Study of the impact of the number of restarts
        r_tab = [0,99] # 11 by default
        for r in r_tab:
            VariationStudyPart1(database,[0],num_rep = 10,r=r)
            VariationStudyPart2(database,[0],num_rep = 10,r=r)
#                VariationStudyPart2(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer)
        # Batch size
        bs_tab = [32,126,500] # 1000 by default
        for bs in bs_tab:
            VariationStudyPart1(database,[0],num_rep = 10,bs=bs)
            VariationStudyPart2(database,[0],num_rep = 10,bs=bs)
        
        # C value
        C_tab = [0.01,0.1,0.5,1.5] # 1.0 by default
        for C in C_tab:
            VariationStudyPart1(database,[0],num_rep = 10,C=C)
            VariationStudyPart2(database,[0],num_rep = 10,C=C)
    

def main():
    
    # Liste des choses que tu as a faire tourner :
    
    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
    database_tab =  ['IconArt_v1','watercolor','PeopleArt']
    database_tab =  ['watercolor','PeopleArt']
    for database in database_tab:
        for MILmodel in MILmodel_tab:
            try:
                runSeveralMInet(dataset_nm=database,MILmodel=MILmodel)
            except Exception as e:
                print(e)
                pass   
    
    # MIMAX - MIMAXS avec et sans score avec et sans hinge loss Sur les trois datasets
#    print('!!!!!!!!! MIMAX - MIMAXS avec et sans score avec et sans hinge loss Sur les trois datasets')
#    # With GradientDescent optimizer
#    optim_list =['GradientDescent','lbfgs']
#    optim_list =['GradientDescent']
#    for Optimizer in optim_list:
#        for database in ['IconArt_v1','watercolor','PeopleArt']:
#            try: 
#    #            Number 0 : MIMAX-score
#    #            Number 3 : hinge loss with score
#    #            Number 5 : MIMAX without score
#    #            Number 22 : hinge loss without score
#                VariationStudyPart1(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer)
#                VariationStudyPart2(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer)
#            except Exception:
#                pass    
#
#            # MaxOfMax 
#            for MaxOfMax,MaxMMeanOfMax in [[True,False],[False,True]]:
#                try: 
#                    if Optimizer=='GradientDescent':
#                        max_iters_all_base = 3000
#                    elif Optimizer=='lbfgs':
#                        max_iters_all_base = 300
#                    unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
#                                                Optimizer=Optimizer,MaxOfMax=MaxOfMax,\
#                                                MaxMMeanOfMax=MaxMMeanOfMax,max_iters_all_base = max_iters_all_base)
#                except Exception:
#                    pass 
#            try: 
#                if Optimizer=='GradientDescent':
#                    max_iters_all_base = 300
#                elif Optimizer=='lbfgs':
#                    max_iters_all_base = 300
#                unefficient_way_OneHiddenLayer_evaluation(database=database,num_rep = 10,Optimizer=Optimizer,
#                                                      max_iters_all_base=max_iters_all_base) 
#            except Exception as e:
#                print(e)
#                pass 

               
   
            


def PrintResults():
    #for Optimizer in ['GradientDescent','lbfgs']:
    for Optimizer in ['GradientDescent']:
        for database in ['IconArt_v1','watercolor','PeopleArt']:
            try: 
    #            Number 0 : MIMAX-score
    #            Number 3 : hinge loss with score
    #            Number 5 : MIMAX without score
    #            Number 22 : hinge loss without score
                VariationStudyPart3(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer)
            except Exception:
                pass    

            # MaxOfMax 
            for MaxOfMax,MaxMMeanOfMax in [[True,False],[False,True]]:
                try: 
                    if Optimizer=='GradientDescent':
                        max_iters_all_base = 3000
                    elif Optimizer=='lbfgs':
                        max_iters_all_base = 300
                    unefficient_evaluation_PrintResults(database=database,num_rep = 10,
                                                Optimizer=Optimizer,MaxOfMax=MaxOfMax,\
                                                MaxMMeanOfMax=MaxMMeanOfMax,max_iters_all_base = max_iters_all_base)
                except Exception:
                    pass 
            try: 
                if Optimizer=='GradientDescent':
                    max_iters_all_base = 3000
                elif Optimizer=='lbfgs':
                    max_iters_all_base = 300
                unefficient_evaluation_PrintResults(database=database,num_rep = 10,Optimizer=Optimizer,
                                                      max_iters_all_base=max_iters_all_base,AddOneLayer=True,
                                                      MaxOfMax=False,MaxMMeanOfMax=False)
            except Exception:
                pass 

               
    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
    for database in ['IconArt_v1','watercolor','PeopleArt']:
        for MILmodel in MILmodel_tab:
            try:
                runSeveralMInet(dataset_nm=database,MILmodel=MILmodel,printR=True)
            except Exception:
                pass         
if __name__ == '__main__':
    ExperienceRuns()
#    main()
    #PrintResults()