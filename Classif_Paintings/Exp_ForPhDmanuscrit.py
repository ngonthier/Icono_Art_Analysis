# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:20:44 2020


This file contains the differents experiments I need to run for my PhD manuscript

Pour faire tourner sur le cluster il va d abord falloir faire le precomputing du 
model Faster RCNN 
Les resultats seront la : /media/gonthier/HDD/output_exp/ClassifPaintings
et ensuite il faudra les copier dans le fichier : 
    /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/data/ClassifPaintings

@author: gonthier
"""

# from MIbenchmarkage import fit_train_plot_GaussianToy

from RunVarStudyAll import VariationStudyPart1,VariationStudyPart2,VariationStudyPart3,\
    unefficient_way_MaxOfMax_evaluation,unefficient_way_OneHiddenLayer_evaluation,\
    unefficient_evaluation_PrintResults,unefficient_way_mi_model_evaluation

### 1 / Toy Model pour MaxOfMax et MIMAX-HL

# def run_Missing_ToyExample():

#     overlap_tab = [False,True]
#     specificCase_tab = [None,'2cloudsOpposite']
#     specificCase_tab = ['2cloudsOpposite']
#     reDo = True
#     number_rep = 1
    
#     list_method = ['MIMAX','MIMAXaddLayer','MaxOfMax','IA_mi_model']    
    
#     # Cas AddOneLayer True : MIMAX HL
#     for method in list_method:
#         for overlap in overlap_tab:
#             for specificCase in specificCase_tab:
#                 end_name= '_PhDmanuscript'
#                 fit_train_plot_GaussianToy(method=method,dataset='GaussianToy',
#                                            WR=0.01,verbose=True,reDo=reDo,specificCase=specificCase,
#                                            dataNormalizationWhen='onTrainSet',dataNormalization='std',
#                                            overlap = overlap,end_name=end_name)
#                 fit_train_plot_GaussianToy(method=method,dataset='GaussianToy',
#                                            WR=0.01,verbose=True,reDo=reDo,specificCase=specificCase,
#                                            dataNormalizationWhen=None,dataNormalization='std',
#                                            overlap = overlap,end_name=end_name)
### miperceptron pour les 6 datasets artistiques !
def miperceptron_for_artistist_dataset():
    # Instance based model mi_model
    for database in ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']:
        try:
            unefficient_way_mi_model_evaluation(database,num_rep =3,
                                        Optimizer='GradientDescent',
                                        max_iters_all_base=3000,scores_tab = [True])
        except Exception as e:
            print(e)
            pass        

### Pascal VOC2007  : 

#- Pascal VOC2007 : il va falloir faire tourner :
#	Feature RESNET 152 COCO
#	Feature RESNET 101 VOC 
#	Pour mi-perceptron+ MI-MAX HL + polyhedral MIMAX (avec et sans score !) au moins 10 fois je dirai
#	
        
def PascalVOC_sanity_check(): 
    
    
    
    # Max Of Max : 
    database = 'VOC2007'
    for demonet in ['res152_COCO','res101_VOC07']:
        try: 
            unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,
                            number_restarts = 11,scores_tab =[True,False],
                            demonet=demonet)
        except Exception as e:
             print(e)
             pass
    

    # MIMAX HL
    for demonet in ['res152_COCO','res101_VOC07']:
        try: 
            unefficient_way_OneHiddenLayer_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,num_features_hidden=256,
                            number_restarts = 11,scores_tab =[True,False],
                            demonet=demonet)
        except Exception as e:
             print(e)
             pass
         
#    # Instance based model mi_model : doesn t converge and we don t know why ...
#    for demonet in ['res152_COCO','res101_VOC07']:
#        try:
#            unefficient_way_mi_model_evaluation(database,num_rep =3,
#                                        Optimizer='GradientDescent',
#                                        max_iters_all_base=300,scores_tab = [True,False],
#                                        demonet = demonet)
#        except Exception as e:
#            print(e)
#            pass 





#### Score use
#	Faire avec score multiply et score additive pour les 6 datasets !
            
def CVmode_MIMAX():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    for database in database_tab:
        # MI_max
        scenario_tab = [25]
        try: 
#            Number 25 : CV mode
            VariationStudyPart1(database,scenario_tab,num_rep = 10)
            VariationStudyPart2(database,scenario_tab,num_rep = 10)
        except Exception as e:
             print(e)
             pass 
def Other_way_to_use_score_MIMAX():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    for database in database_tab:
        # MI_max
        scenario_tab = [23,24]
        try: 
#            Number 23 : multiplication Tanh and score
#            Number 24 : Addition Tanh and score
            VariationStudyPart1(database,scenario_tab,num_rep = 10)
            VariationStudyPart2(database,scenario_tab,num_rep = 10)
        except Exception as e:
             print(e)
             pass 
         
def TwoThousandsboxes_MIMAX():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    for database in database_tab:
        # MI_max score
        scenario_tab = [0]
        try: 

            VariationStudyPart1(database,scenario_tab,num_rep = 10,
                                k_per_bag=2000)
            VariationStudyPart2(database,scenario_tab,num_rep = 10,
                                k_per_bag=2000)
        except Exception as e:
             print(e)
             pass 
         
def Other_way_to_use_score_MaxOfMax():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
        
    for database in database_tab :
        ## obj_score_add_tanh
        print('- obj_score_add_tanh -')
        try: 
            unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,
                            number_restarts = 11,scores_tab =[False],
                            obj_score_add_tanh=True)
        except Exception as e:
             print(e)
             pass
        ## obj_score_mul_tanh
        print('- obj_score_mul_tanh -')
        try: 
            unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,
                            number_restarts = 11,scores_tab =[False],
                            obj_score_mul_tanh=True)
        except Exception as e:
             print(e)
             pass
         
def MaxOfMax():
    """
    Le but de ce script va etre de verifier les performances de MaxOfMax sur ces 
    datasets la car on a un std a 0 ce qui est vraiment bizarre !! 
    """
    database_tab = ['clipart','comic','CASPApaintings']
        
    for database in database_tab :
        try: 
            unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,
                            number_restarts = 11,scores_tab =[True,False],
                            loss_tab=['','hinge'])
        except Exception as e:
             print(e)
             pass
         
def IcoOther_way_to_use_score_MaxOfMax():
    database_tab = ['IconArt_v1']
        
    for database in database_tab :
        ## obj_score_mul_tanh
        print('- obj_score_mul_tanh -')
        try: 
            unefficient_way_MaxOfMax_evaluation(database=database,num_rep = 10,
                            Optimizer='GradientDescent',
                            max_iters_all_base = 3000,
                            number_restarts = 11,scores_tab =[False],
                            obj_score_mul_tanh=True)
        except Exception as e:
             print(e)
             pass
         
def MaxOfMax_other_way_score_print_results():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
        
    for database in database_tab :
        print(database)
        print('- obj_score_add_tanh -') 
        unefficient_evaluation_PrintResults(database=database,num_rep = 10,
                                        Optimizer='GradientDescent',
                                        MaxOfMax=True,MaxMMeanOfMax=False,
                                        max_iters_all_base=3000,AddOneLayer=False,
                                        num_features_hidden=256,pm_only_on_mean=False,
                                        mi_model=False,
                                        seuillage_by_score=False,
                                        obj_score_add_tanh=True,
                                        obj_score_mul_tanh = False,
                                        with_scores_tab=[False],
                                        loss_type_tab=[''])
        print('- obj_score_mul_tanh -') 
        unefficient_evaluation_PrintResults(database=database,num_rep = 10,
                                        Optimizer='GradientDescent',
                                        MaxOfMax=True,MaxMMeanOfMax=False,
                                        max_iters_all_base=3000,AddOneLayer=False,
                                        num_features_hidden=256,pm_only_on_mean=False,
                                        mi_model=False,
                                        seuillage_by_score=False,
                                        obj_score_add_tanh=False,
                                        obj_score_mul_tanh = True,
                                        with_scores_tab=[False],
                                        loss_type_tab=[''])
        
def print_old_results_MaxOfMax_artistic_datasets():
    database_tab = ['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
        
    print('=== Polyhedral MIMAX model de base ===')
    print('== avec et sans score avec et sans la loss hinge ==')
    for database in database_tab :
        print(database)
        unefficient_evaluation_PrintResults(database=database,num_rep = 10,
                                        Optimizer='GradientDescent',
                                        MaxOfMax=True,MaxMMeanOfMax=False,
                                        max_iters_all_base=3000,AddOneLayer=False,
                                        num_features_hidden=256,pm_only_on_mean=True,
                                        mi_model=False,
                                        seuillage_by_score=False,
                                        obj_score_add_tanh=False,
                                        obj_score_mul_tanh = False,
                                        with_scores_tab=[True,False],
                                        loss_type_tab=['','hinge'])
        
def print_old_results_MIMAX_artistic_datasets():
        
    print('=== MIMAX model de base ===')
    print('== avec et sans score avec et sans la loss hinge ==')
    VariationStudyPart3(['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
                        ,scenarioSubset=[0,5,3,22],withoutAggregW=True,num_rep=10,
                        pm_only_on_mean=True)
        

if __name__ == '__main__':                                       
    
#    print('TwoThousandsboxes_MIMAX')
#    TwoThousandsboxes_MIMAX()

    print('CVmode_MIMAX')
#    CVmode_MIMAX()
    # To print performance 
    # VariationStudyPart3(['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    #                     ,scenarioSubset=[25],withoutAggregW=True,num_rep=10)
    
    # print('Other_way_to_use_score_MIMAX')
    # #Other_way_to_use_score_MIMAX()
    # # To print performance 
    # VariationStudyPart3(['IconArt_v1','watercolor','PeopleArt','clipart','comic','CASPApaintings']
    #                     ,scenarioSubset=[23,24],withoutAggregW=True,num_rep=10)
    
    
    print('Other_way_to_use_score_MaxOfMax')
#    Other_way_to_use_score_MaxOfMax()
    MaxOfMax_other_way_score_print_results()
#    
#    print('miperceptron_for_artistist_dataset')
#    miperceptron_for_artistist_dataset()
#    
#    print('PascalVOC_sanity_check')
#    PascalVOC_sanity_check()

    