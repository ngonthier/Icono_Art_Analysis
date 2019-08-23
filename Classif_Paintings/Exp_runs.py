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
    unefficient_evaluation_PrintResults,unefficient_way_mi_model_evaluation
from IMDB_study import modify_EdgeBoxesWrongBoxes
import matplotlib.pyplot as plt
import numpy as np
from CNNfeatures import Compute_EdgeBoxesAndCNN_features

from MILnet_eval import runSeveralMInet

def ExperienceRuns(database_tab =  ['IconArt_v1','watercolor','PeopleArt']):
        
    for database in database_tab:
        # Study of the impact of the number of restarts
        #r_tab = [0,99] # 11 by default
        r_tab = [0,1,4,11,49,99,149] # 11 by default
        for r in r_tab:
            VariationStudyPart1(database,[0],num_rep = 10,r=r)
            VariationStudyPart2(database,[0],num_rep = 10,r=r)
#                VariationStudyPart2(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer)
        # Batch size
        bs_tab = [8,16,32,126,256,500,1000] # 1000 by default
        if database=='clipart':
            bs_tab = [8,16,32,126,256,500]
        #bs_tab = [8] # 1000 by default
        for bs in bs_tab:
            VariationStudyPart1(database,[0],num_rep = 10,bs=bs)
            VariationStudyPart2(database,[0],num_rep = 10,bs=bs)
        
        # C value
        #C_tab = [0.01,0.1,0.5,1.5] # 1.0 by default
        C_tab = [0.01,0.1,0.5,1.0,1.5,2.,10.] # 1.0 by default
        for C in C_tab:
            VariationStudyPart1(database,[0],num_rep = 10,C=C)
            VariationStudyPart2(database,[0],num_rep = 10,C=C)
        
def print_run_studyParam():
    """
    In this function we plot the evolution of the different parameters evaluation
    """
    database_tab =  ['IconArt_v1','watercolor','PeopleArt','clipart','comic']
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    makers = ['P','X','o','v','D','h','^']
    
    list_param = ['r','bs','C']
    for param in list_param:
        if param=='r':
            p_tab = [0,1,4,11,49,99,149] # 11 by default
            tt =' the number of reinitialization'
            tx = r'Number of reinitialization'
            name_figure ='Reinit'
        if param=='bs':
            p_tab = [8,16,32,126,256,500,1000]
            tt = 'the batch size'
            tx = r'Batch size'
            name_figure ='bs'
        if param=='C':
            p_tab =  [0.01,0.1,0.5,1.0,1.5,2.,10.] 
            tt = 'the regularization term'
            tx = r'Regularization term'
            name_figure ='C'
    
        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        multi = 100 # Multiplicative term to get something between 0 and 100 % 
        for database,c,m in zip(database_tab,colors,makers):
            x = []
            y = []
            std = []
            i = 0
            if param=='bs' and database=='clipart':
                p_tab_local = [8,16,32,126,256,500]
            else:
                p_tab_local = p_tab
            for p in p_tab_local:
                if param=='r':
                    ll_all = VariationStudyPart3(database,[0],num_rep = 10,r=p,withoutAggregW=True)
                if param=='bs':
                    ll_all = VariationStudyPart3(database,[0],num_rep = 10,bs=p,withoutAggregW=True)
                if param=='C':
                    ll_all = VariationStudyPart3(database,[0],num_rep = 10,C=p,withoutAggregW=True)
                if not(database=='PeopleArt'):
                    mean_over_reboot = np.mean(ll_all,axis=1) # Moyenne par ligne / reboot 
        #                            print(mean_over_reboot.shape)
                    std_of_mean_over_reboot = np.std(mean_over_reboot)
                    mean_of_mean_over_reboot = np.mean(mean_over_reboot)
                else:
                    std_of_mean_over_reboot = np.std(ll_all)
                    mean_of_mean_over_reboot = np.mean(ll_all)
                x += [i]
                y += [mean_of_mean_over_reboot*multi]
                std += [std_of_mean_over_reboot*multi]
                i+= 1
            #plt.plot(x,y,c,label=database)
            label = database.replace('_v1','')
            plt.errorbar(x, y,linestyle=':', yerr=std,marker=m,c=c,label=label,solid_capstyle='projecting', capsize=5)
        plt.xticks(x, p_tab)
        plt.legend(loc='best')
        plt.xlabel(tx,fontsize=12)
        plt.ylabel(r'mAP ( \% )',fontsize=12)
        fname = 'fig/'+name_figure +'.png'
        plt.savefig(fname)
#        str_t = r"Evolution of the impact of "+ tt
#        plt.title(str_t,
#              fontsize=16)
        plt.show()
            
def mainDatabase(database_tab=['clipart']):
   optim_list =['GradientDescent']
   for Optimizer in optim_list:
        for database in database_tab:
            # If Faster RCNN [0,5,3,22] for scenario_tab
            # MI_max
            scenario_tab = [0,5,3,22]
            try: 
            #            Number 0 : MIMAX-score
            #            Number 3 : hinge loss with score
            #            Number 5 : MIMAX without score
            #            Number 22 : hinge loss without score
                VariationStudyPart1(database,scenario_tab,num_rep = 10,Optimizer=Optimizer)
                VariationStudyPart2(database,scenario_tab,num_rep = 10,Optimizer=Optimizer)
            except Exception as e:
                 print(e)
                 pass 
            
            # MaxOfMax 
            for MaxOfMax in [True]:
                try: 
                    if Optimizer=='GradientDescent':
                        max_iters_all_base = 3000
                    elif Optimizer=='lbfgs':
                        max_iters_all_base = 300
                    unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                                                Optimizer=Optimizer,MaxOfMax=MaxOfMax,\
                                                MaxMMeanOfMax=False,
                                                max_iters_all_base = max_iters_all_base)
                except Exception:
                    pass   
            
            for metamodel,demonet,scenario_tab in zip(['EdgeBoxes'],['res152'],[[5,22]]):
                try: 
        #            Number 0 : MIMAX-score
        #            Number 3 : hinge loss with score
        #            Number 5 : MIMAX without score
        #            Number 22 : hinge loss without score
                    VariationStudyPart1(database,scenario_tab,num_rep = 10,Optimizer=Optimizer,
                                        metamodel=metamodel,demonet=demonet)
                    VariationStudyPart2(database,scenario_tab,num_rep = 10,Optimizer=Optimizer,
                                        metamodel=metamodel,demonet=demonet)
                except Exception as e:
                    print(e)
                    pass  
                
   MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']

   for dataset_nm in database_tab:
       for MILmodel in MILmodel_tab:
           runSeveralMInet(dataset_nm=dataset_nm,MILmodel=MILmodel)
           
def mainDatabase_HL(database_tab=['clipart'],scores_tab = [True],loss_tab = ['']):
   optim_list =['GradientDescent']
   for Optimizer in optim_list:
        for database in database_tab:
            try: 
                unefficient_way_OneHiddenLayer_evaluation(database=database,num_rep = 10,
                                Optimizer='GradientDescent',
                                max_iters_all_base = 300,num_features_hidden=256,
                                number_restarts = 11,scores_tab =scores_tab,loss_tab = loss_tab)
            except Exception as e:
                 print(e)
                 pass
             
def mainDatabase_mi_model(database_tab=['clipart'],scores_tab = [True],loss_tab = ['']):
   optim_list =['GradientDescent']
   for Optimizer in optim_list:
        for database in database_tab:
            try:  #  Il semblerait qu'il soit nécessaire de faire 3000 iterations dans le cas 
                 # de mi_model
                unefficient_way_mi_model_evaluation(database=database,num_rep = 10,
                                Optimizer='GradientDescent',
                                max_iters_all_base = 3000,
                                number_restarts = 11,scores_tab =scores_tab,loss_tab = loss_tab)
            except Exception as e:
                 print(e)
                 pass

def main():
    
    # Liste des choses que tu as a faire tourner :
    
#    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
#    database_tab =  ['IconArt_v1','watercolor','PeopleArt']
#    database_tab =  ['watercolor','PeopleArt']
#    for database in database_tab:
#        for MILmodel in MILmodel_tab:
#            try:
#                runSeveralMInet(dataset_nm=database,MILmodel=MILmodel)
#            except Exception as e:
#                print(e)
#                pass   
    
    # MIMAX - MIMAXS avec et sans score avec et sans hinge loss Sur les trois datasets
#    print('!!!!!!!!! MIMAX - MIMAXS avec et sans score avec et sans hinge loss Sur les trois datasets')
#    # With GradientDescent optimizer
#    optim_list =['GradientDescent','lbfgs']
#    optim_list =['GradientDescent']
#    for Optimizer in optim_list:
#        for database in ['IconArt_v1','watercolor','PeopleArt']:
#            # If Faster RCNN [0,5,3,22] for scenario_tab
#            for metamodel,demonet,scenario_tab in zip(['EdgeBoxes'],['res152'],[[5,22]]):
#                try: 
#        #            Number 0 : MIMAX-score
#        #            Number 3 : hinge loss with score
#        #            Number 5 : MIMAX without score
#        #            Number 22 : hinge loss without score
#                    VariationStudyPart1(database,scenario_tab,num_rep = 10,Optimizer=Optimizer,
#                                        metamodel=metamodel,demonet=demonet)
#                    VariationStudyPart2(database,scenario_tab,num_rep = 10,Optimizer=Optimizer,
#                                        metamodel=metamodel,demonet=demonet)
#                except Exception as e:
#                    print(e)
#                    pass    
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
            
    # Instance based model mi_model
    for database in ['IconArt_v1','watercolor','PeopleArt']:
        try:
            unefficient_way_mi_model_evaluation(database,num_rep =3,
                                        Optimizer='GradientDescent',
                                        max_iters_all_base=3000)
#        # If Faster RCNN [0,5,3,22] for scenario_tab
#        scenario_tab = [0,5,3,22]
#        max_iters_all_base = 3000
#        try: 
##            Number 0 : MIMAX-score
##            Number 3 : hinge loss with score
##            Number 5 : MIMAX without score
##            Number 22 : hinge loss without score
#            VariationStudyPart1(database,scenario_tab,num_rep = 10,
#                                Optimizer='GradientDescent',model='mi_model',
#                                max_iters_all_base=max_iters_all_base)
#            VariationStudyPart2(database,scenario_tab,num_rep = 10,
#                                Optimizer='GradientDescent',model='mi_model',
#                                max_iters_all_base=max_iters_all_base)
        except Exception as e:
            print(e)
            pass
            
    # One hidden layer model !
    optim_list = ['GradientDescent']
    for Optimizer in optim_list:
        for database in ['IconArt_v1','watercolor','PeopleArt']:
            try: 
                if Optimizer=='GradientDescent':
                    max_iters_all_base = 3000
                elif Optimizer=='lbfgs':
                    max_iters_all_base = 300
                # Pour 3 epochs 0 restarts cela prend : 50s soit 7 jours par cas :( 
                # avec hidden layer = 25 : il faudrait 10 milles secondes par run....
                unefficient_way_OneHiddenLayer_evaluation(database=database,num_rep =3,
                                                          Optimizer=Optimizer,
                                                          max_iters_all_base=max_iters_all_base,
                                                          num_features_hidden=2048,
                                                          number_restarts=1) 
            except Exception as e:
                print(e)
                pass 


def PrintResults(database_tab=['IconArt_v1','watercolor','PeopleArt','clipart'],pm_only_on_mean=True):
    """
    @param : pm_only_on_mean : only print the pm std for the mean over the classes
    """
    #for Optimizer in ['GradientDescent','lbfgs']:
    
    for Optimizer in ['GradientDescent']:
        for database in database_tab:
            print('=== MImax with and without score ===')
            try: 
    #            Number 0 : MIMAX-score
    #            Number 3 : hinge loss with score
    #            Number 5 : MIMAX without score
    #            Number 22 : hinge loss without score
                VariationStudyPart3(database,[0,5,3,22],num_rep = 10,Optimizer=Optimizer,
                                    pm_only_on_mean=pm_only_on_mean)
            except Exception:
                pass    
            
            print('=== EdgeBoxes MImax ===')
            for metamodel,demonet,scenario_tab in zip(['EdgeBoxes'],['res152'],[[5,22]]):
                try: 
                    VariationStudyPart3(database,scenario_tab,num_rep = 10,
                                        Optimizer=Optimizer,metamodel=metamodel,
                                        demonet=demonet,pm_only_on_mean=pm_only_on_mean)
                except Exception:
                    pass 
            # MaxOfMax 
            print('=== MaxOfMax ===')
            for MaxOfMax,MaxMMeanOfMax in [[True,False],[False,True]]:
                try: 
                    if Optimizer=='GradientDescent':
                        max_iters_all_base = 3000
                    elif Optimizer=='lbfgs':
                        max_iters_all_base = 300
                    unefficient_evaluation_PrintResults(database=database,num_rep = 10,
                                                Optimizer=Optimizer,MaxOfMax=MaxOfMax,\
                                                MaxMMeanOfMax=MaxMMeanOfMax,
                                                max_iters_all_base = max_iters_all_base)
                except Exception:
                    pass 
                
            print('=== MiModel ===')
            scenario_tab = [0,5,3,22]
            max_iters_all_base = 3000
            try: 
    #            Number 0 : MIMAX-score
    #            Number 3 : hinge loss with score
    #            Number 5 : MIMAX without score
    #            Number 22 : hinge loss without score
                VariationStudyPart3(database,scenario_tab,num_rep = 10,
                                    Optimizer=Optimizer,model='mi_model',
                                    max_iters_all_base=max_iters_all_base)
            except Exception as e:
                print(e)
                pass 
                
            print('=== Add One Layer ===')
            try: 
                if Optimizer=='GradientDescent':
                    max_iters_all_base = 3000
                elif Optimizer=='lbfgs':
                    max_iters_all_base = 300
                unefficient_evaluation_PrintResults(database=database,num_rep = 10,Optimizer=Optimizer,
                                                      max_iters_all_base=max_iters_all_base,
                                                      AddOneLayer=True,
                                                      MaxOfMax=False,MaxMMeanOfMax=False,
                                                      num_features_hidden=2048)
            except Exception:
                pass 

    print('=== Revisiting MIL Nets ===')               
    MILmodel_tab = ['MI_Net','MI_Net_with_DS','MI_Net_with_RC','mi_Net']
    for database in database_tab:
        for MILmodel in MILmodel_tab:
            try:
                runSeveralMInet(dataset_nm=database,MILmodel=MILmodel,printR=True,
                                pm_only_on_mean=pm_only_on_mean)
            except Exception:
                pass         
if __name__ == '__main__':
    ExperienceRuns(database_tab=['CASPApaintings'])
    mainDatabase(database_tab=['CASPApaintings','comic','clipart'])
    mainDatabase_HL(database_tab=['IconArt_v1','watercolor','PeopleArt','CASPApaintings','comic','clipart'])
#    PrintResults()