# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:15:51 2020

List des figures a faire pour le papier DHNord 

@author: gonthier
"""

import os
import pathlib
#import matplotlib
#import tensorflow as tf

from CompNet_FT_lucidIm import do_lucid_vizu_for_list_model,print_performance_FineTuned_network
from Activation_for_model import plot_images_Pos_Images

#import lucid_utils



def do_TopK_figures(list_models_name,list_layer_index_to_print,suffix_tab=[''],
                    dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path='',
                    alreadyAtInit=False,
                    ReDo=False,
                    FTmodel=True):
    
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    for model_name in list_models_name:
        print('#### ',model_name)
              
        for layer_name,num_feature in list_layer_index_to_print:
        
            if model_name=='pretrained':
                #assert(alreadyAtInit==False)
                output_path_for_img = os.path.join(output_path,model_name)
                plot_images_Pos_Images(dataset,model_name,constrNet,
                                layer_name=layer_name,
                                num_feature=num_feature,
                                numberIm=numberIm,stats_on_layer=stats_on_layer,
                                suffix='',
                                FTmodel=True,
                                output_path_for_img=output_path_for_img,
                                ReDo=ReDo)
            elif not(model_name=='random'):
                
                for suffix in suffix_tab:
                    output_path_for_img = os.path.join(output_path,model_name+suffix)
                    plot_images_Pos_Images(dataset,model_name,constrNet,
                                layer_name=layer_name,
                                num_feature=num_feature,
                                numberIm=numberIm,stats_on_layer=stats_on_layer,
                                suffix=suffix,
                                FTmodel=FTmodel,
                                output_path_for_img=output_path_for_img,
                                alreadyAtInit=alreadyAtInit,ReDo=ReDo)
                    
            else:
                raise(NotImplementedError('random model not implemented'))

if __name__ == '__main__': 
    
    # To get the performance results
    
    print_performance_FineTuned_network(constrNet='InceptionV1',
                                        list_models_name=['RASTA_small01_modif'],
                                        suffix_tab=[''],latexOutput=True,print_all=True)
    
    # Liste figure pour AdaptationFiltersRASTA de DHNord 2020 paper
    suffix_tab = ['','1']
    list_features = [['mixed4d_pool_reduce_pre_relu',63],['mixed4b_3x3_bottleneck_pre_relu',35],['mixed4d_3x3_pre_relu',52]]
#    list_features = [['mixed4d_3x3_pre_relu',52]]
    list_models = ['pretrained','RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision',
                        'RASTA_big001_modif_deepSupervision']
#    list_models = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
#                        'RASTA_small001_modif_deepSupervision',
#                        'RASTA_big001_modif_deepSupervision']
    # Il y a un pb non resolu avec le pretrained model !
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 suffix_tab = suffix_tab,
                                 output_path=output_path,constrNet='InceptionV1')
    
    # Afficher les 100 images qui repondent le plus pour ces filtres là :
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=suffix_tab,dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    
    # Plot Top100 images with the image already present at the initialisation 
    # surrounded by green 
    list_features = [['mixed4d_pool_reduce_pre_relu',63],['mixed4b_3x3_bottleneck_pre_relu',35],['mixed4d_3x3_pre_relu',52]]
    list_models = ['RASTA_small01_modif']
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=True)
    
    # Pour la figure Mid-level detectors can be learned from scratch
    list_models = ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200']
    list_features = [['mixed5a_3x3_bottleneck_pre_relu',1],['mixed4d_5x5_pre_relu',50]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1')
    
    # Pour la figure low level feature 
    list_models = ['pretrained','RASTA_small01_modif']
    list_features = [['conv2d1_pre_relu',30],['mixed3a_3x3_pre_relu',12],['mixed3a_5x5_bottleneck_pre_relu',8]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1')
    
    # Pour la figure High-level filters seem poly-semantic; feat vizu + top images associées
    list_models = ['pretrained','RASTA_small01_modif']
    list_features = [['mixed5b_pool_reduce_pre_relu',92],['mixed5b_3x3_pre_relu',33],['mixed5b_5x5_pre_relu',82]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1')
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=False)
    
    # Model from scratch completement
    list_models = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    list_features = [['mixed4d',8],['mixed4d',16],['mixed4d',66]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','V3DHNORD','im_old')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1')
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=False)
    
    # Same model but random initialisation
    from CompNet_FT_lucidIm import do_lucid_vizu_for_list_model,print_performance_FineTuned_network
    from Figures_For_DHNord import do_TopK_figures
    import os
    list_models = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    list_features = [['mixed4d',8],['mixed4d',16],['mixed4d',66]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','V3DHNORD','im_old')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1',init_model_use=True)
    
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    FTmodel=False)
    
    
    # Print fine-tuned models :
    list_models_name = ['RASTA_small01_modif',
                        'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                        'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    print_performance_FineTuned_network(constrNet='InceptionV1',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=True)
    
    # Figures pour Model trained on RASTA and then on IconArt 
    list_models_name = ['IconArt_v1_big01_modif_XXRASTA_small01_modifXX','RASTA_small01_modif',
                        'pretrained']
    list_models_name = ['IconArt_v1_small01_modif']
    list_features = [['mixed4d_3x3_bottleneck_pre_relu',64],
                     ['mixed4c_5x5_bottleneck_pre_relu',1],
                     ['mixed4e_3x3_pre_relu',92],
                     ['mixed4c_3x3_bottleneck_pre_relu',78],
                     ['mixed4c_pool_reduce_pre_relu',2],
                     ['mixed4d_5x5_pre_relu',33],
                     ['mixed4d_5x5_pre_relu',49]
                     ]
    list_features = [['mixed4c_3x3_bottleneck_pre_relu',78],
                     ['mixed4c_pool_reduce_pre_relu',2],
                     ['mixed4d_5x5_pre_relu',49]
                     ]
    do_TopK_figures(list_models_name=list_models_name,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='IconArt_v1',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    do_TopK_figures(list_models_name=list_models_name,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    
    do_lucid_vizu_for_list_model(list_models_name=list_models_name,list_layer_index_to_print=list_features,
                                 output_path=output_path,constrNet='InceptionV1')
    
    # Suite aux remarques de Yann : plus de feature visu sur les couches intermediaires :
    list_features = [['mixed4b',361],['mixed4d',56],['mixed4d',89],['mixed4d',106]]
#    list_features = [['mixed4d_3x3_pre_relu',52]]
    list_models = ['RASTA_small01_modif']
    #list_models = ['pretrained']
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 suffix_tab = [''],
                                 output_path=output_path,constrNet='InceptionV1')
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=True)
    
    # Faire l'image des draperies !
    list_features = [['mixed4c_3x3_bottleneck_pre_relu',78]]
    list_models = ['pretrained','RASTA_small01_modif']
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=True)
    
    # Figures pour la presentation : Top 10
    suffix_tab = ['','1']
    list_features = [['mixed4d_pool_reduce_pre_relu',63],['mixed4b_3x3_bottleneck_pre_relu',35],['mixed4d_3x3_pre_relu',52]]
    list_models = ['pretrained','RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision',
                        'RASTA_big001_modif_deepSupervision']
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    # Afficher les 100 images qui repondent le plus pour ces filtres là :
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=suffix_tab,dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    list_models = ['pretrained','RASTA_small01_modif']
    list_features = [['mixed5b_pool_reduce_pre_relu',92],['mixed5b_3x3_pre_relu',33],['mixed5b_5x5_pre_relu',82]]
    output_path = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=False)
    
    list_models = ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    list_features = [['mixed4d',8],['mixed4d',16],['mixed4d',66]]
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=False)
    list_models_name = ['IconArt_v1_big01_modif_XXRASTA_small01_modifXX','RASTA_small01_modif','pretrained','IconArt_v1_small01_modif']
    list_features = [['mixed4c_3x3_bottleneck_pre_relu',78],
                     ['mixed4c_pool_reduce_pre_relu',2],
                     ['mixed4d_5x5_pre_relu',49]
                     ]
    do_TopK_figures(list_models_name=list_models_name,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='IconArt_v1',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    do_TopK_figures(list_models_name=list_models_name,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path)
    
    list_features = [['mixed4b',361],['mixed4d',56],['mixed4d',89],['mixed4d',106]]
    list_models = ['RASTA_small01_modif','pretrained']
    do_TopK_figures(list_models_name=list_models,
                    list_layer_index_to_print=list_features,
                    suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 9,
                    stats_on_layer = 'meanAfterRelu',
                    output_path=output_path,
                    alreadyAtInit=False)

    