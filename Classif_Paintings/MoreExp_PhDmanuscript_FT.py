# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:57:11 2020

Experiences en plus a faire tourner pour la these : 
    fine tuning des modeles avec parfois de la visualization des features ! 

@author: gonthier
"""

from CompNet_FT_lucidIm import Comparaison_of_FineTunedModel,print_performance_FineTuned_network

from StatsConstr_ClassifwithTL import learn_and_eval
from keras_resnet_utils import getBNlayersResNet50

from Gram_For_Better_Feat_Vizu import get_confusion_matrices_for_list_models

### Cas de Paintings IconArt en mode Off-the-shelf feature extraction InceptionV1
def perf_OffTheshelf_InceptionV1_IconArt_ArtUK_baseline():
        
    print('IconArt - InceptionV1')
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='avgpool',\
                constrNet='InceptionV1',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer=None,
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='avgpool',\
                constrNet='InceptionV1',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer=None,
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
        
    print('Paintings - InceptionV1')
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='avgpool',\
                constrNet='InceptionV1',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer=None,
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='avgpool',\
                constrNet='InceptionV1',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer=None,
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4) 

### Cas de Paintings IconArt et RASTA en mode Off-the-shelf feature extraction
def perf_IconArt_ArtUK_RASTA_baseline_TL():
    
    print('RASTA')
    learn_and_eval('RASTA',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('RASTA',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    
    print('IconArt - VGG')
    # Diverge
    # learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
    #             constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
    #             pretrainingModif=False,return_best_model=True,weights='imagenet',\
    #             optimizer='SGD',opt_option=[0.01],
    #             transformOnFinalLayer='GlobalAveragePooling2D',
    #             epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.1,0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_small01_modif_GAP',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    
    
    print('IconArt - ResNet50')
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_small01_modif_GAP',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    
    print('Paintings - VGG')
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.1,0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_small01_modif_GAP',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
                constrNet='VGG',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)  
    
    
    print('Paintings - ResNet50')
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='imagenet',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_small01_modif_GAP',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='MLP1',features='conv5_block3_out',\
                constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
                pretrainingModif=False,return_best_model=True,weights='RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',\
                optimizer='SGD',opt_option=[0.01],
                transformOnFinalLayer='GlobalAveragePooling2D',
                epochs=20,cropCenter=True,verbose=True,SGDmomentum=0.9,decay=1e-4)
    
    
    


### Cas de Paintings et IconArt pour fine-tune les modeles et voir les performances
def print_IconArtv1_performance(latexOutput=False):
    # For Classification performance different setup
    # 'IconArt_v1_big01_modif_GAP', diverge
    list_models_name_VGG = ['IconArt_v1_small01_modif_GAP',
                            'IconArt_v1_big001_modif_GAP',
                            'IconArt_v1_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                            'IconArt_v1_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name_VGG,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
    list_models_name_ResNet = ['IconArt_v1_small01_modif_GAP',
                            'IconArt_v1_big01_modif_GAP',
                            'IconArt_v1_big001_modif_GAP',
                            'IconArt_v1_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                            'IconArt_v1_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name_ResNet,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
    # list_models_name=['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
    #                   'IconArt_v1_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
    # print_performance_FineTuned_network(constrNet='InceptionV1_slim',
    #                                     list_models_name=list_models_name,
    #                                     suffix_tab=[''],latexOutput=latexOutput)
    
    
def print_Paintings_performance(latexOutput=False):
    # For Classification performance different setup
    # 'Paintings_big01_modif_GAP',  diverge
    list_models_name_VGG = ['Paintings_small01_modif_GAP',
                            'Paintings_big001_modif_GAP',
                            'Paintings_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                            'Paintings_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name_VGG,
                                        suffix_tab=[''],latexOutput=latexOutput)
    list_models_name_ResNet = ['Paintings_small01_modif_GAP',
                            'Paintings_big01_modif_GAP',
                            'Paintings_big001_modif_GAP',
                            'Paintings_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                            'Paintings_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name_ResNet,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
#    list_models_name=['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                      'IconArt_v1_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
#    print_performance_FineTuned_network(constrNet='InceptionV1_slim',
#                                        list_models_name=list_models_name,
#                                        suffix_tab=[''],latexOutput=latexOutput)
    
    
### RASTA performance : 

def print_RASTA_performance(latexOutput=False):
    list_models_name = ['RASTA_small01_modif_GAP',
                        #'RASTA_small01_modif_GAP_adam_unfreeze20_SmallDataAug',
                        'RASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                        'RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG'
                        ]
                        
    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)
    # 'RASTA_small01_modif_GAP'
    # Top-1 accuracy : 55.99%
    # Top-3 accuracy : 82.71%
    # Top-5 accuracy : 91.70%
    # ResNet50 & RASTA\_small01\_modif\_GAP & 55.99 & 82.71 & 91.70\\

    #'RASTA_small01_modif_GAP_adam_unfreeze20_SmallDataAug',
    # Top-1 accuracy : 46.81%
    # Top-3 accuracy : 77.02%
    # Top-5 accuracy : 87.36%
    # ResNet50 & RASTA\_small01\_modif\_GAP\_adam\_unfreeze20\_SmallDataAug & 46.81 & 77.02 & 87.36\\


    list_models_name = ['RASTA_small01_modif_GAP',
                        'RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                        'RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG'] 
    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)

### Cas de RASTA fine-tuning et Visualisation

def RASTA_ResNet_VGG_feat_vizu():
#list_model_name_5 = ['RASTA_small01_modif',
    #                       'RASTA_big001_modif_adam_unfreeze50_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze50_SmallDataAug_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze20_ep200',
    #                      'RASTA_big001_modif_adam_unfreeze20_SmallDataAug_ep200',
    #                     ]
    # 'RASTA_small01_modif_GAP',
    #                       'RASTA_big001_modif_GAP_adam_unfreeze50',
    #                       'RASTA_big001_modif_GAP_adam_unfreeze50_SmallDataAug',
    #                       'RASTA_big001_modif_GAP_adam_unfreeze50_randomCrop',
#                            'RASTA_big001_modif_GAP_adam_unfreeze50_RandForUnfreezed_randomCrop',
#                          'RASTA_big001_modif_GAP_adam_unfreeze20',
#                          'RASTA_big001_modif_GAP_adam_unfreeze20_SmallDataAug',
#                          'RASTA_big001_modif_GAP_adam_unfreeze20_randomCrop',

###%  A faire tourner plus tard !
     list_model_name_5 = ['RASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200'] 
## Provide 60% on Top1 
     Comparaison_of_FineTunedModel(list_model_name_5,constrNet='ResNet50') 

    
#    list_model_name_5 = ['RASTA_big001_modif_GAP_adam_unfreeze20_RandForUnfreezed_randomCrop'] 
## Provide 60% on Top1 
#     Comparaison_of_FineTunedModel(list_model_name_5,constrNet='ResNet50') 
#    # InceptionV1 and ResNet50 models have been trained => need to look at the results ! 
#    #Test avec RMSprop non fait !
# #'RASTA_big001_modif_adam_unfreeze8_SmallDataAug_ep200','RASTA_big001_modif_adam_unfreeze8_SmallDataAug_ep200',
#     list_model_name_4 = ['RASTA_big001_modif_GAP_adam_unfreeze8',
#                          'RASTA_big001_modif_GAP_adam_unfreeze8_SmallDataAug',
#                         'RASTA_big0001_modif_GAP_adam_unfreeze8',
#                         'RASTA_big0001_modif_GAP_adam_unfreeze8_SmallDataAug',
#                         'RASTA_big0001_modif_GAP_adam_unfreeze8',
#                         'RASTA_big001_modif_GAP_RMSprop_unfreeze8_SmallDataAug',
#                         'RASTA_big0001_modif_GAP_RMSprop_unfreeze8_SmallDataAug',
#                        ]
     list_model_name_4 = ['RASTA_small01_modif_GAP',
                           # 'RASTA_big01_modif_GAP',
                           # 'RASTA_big001_modif_GAP',
                            'RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                            'RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG',
                            ]
     Comparaison_of_FineTunedModel(list_model_name_4,constrNet='VGG')

### Cas ou l on regroupe les modeles ensemble


def combine_ens_models():
    
    list_model_name= ['RASTA_small01_modif_GAP',
                      'RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                      'RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG']
    print('VGG ensembles : ',list_model_name)
    get_confusion_matrices_for_list_models(list_model_name=list_model_name,
                                           list_constrNet=['VGG'],
                                           bagging_method = 'mean')
    list_model_name= ['RASTA_small01_modif_GAP',
                      'RASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                      'RASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedG']
    print('ResNet50 ensembles : ',list_model_name)
    get_confusion_matrices_for_list_models(list_model_name=list_model_name,
                                           list_constrNet=['ResNet50'],
                                           bagging_method = 'mean')

### Cas de Paintins et IconArt avec un passage intermediaire par RASTA

def print_perform_Paintings_IconArt_RASTA_intermediaire(latexOutput=False):
    
    # VGG IconArt
    list_models_name = ['IconArt_v1_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
    
    # ResNet IconArt
    list_models_name = ['IconArt_v1_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)
    # VGG Paintings
    list_models_name = ['Paintings_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
    
    # ResNet Paintings
    list_models_name = ['Paintings_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)


def exp_BN_only():
    """In this exp we only fine-tuned the batch normalization of the models"""
    
    target_dataset_tab = ['Paintings','IconArt_v1']
    
    ## VGG VGGAdaIn train only the batch normalisation
    constrNet = 'VGGAdaIn'
    style_layers = ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
    'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
    'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
    'block5_conv1','block5_conv2','block5_conv3','block5_conv4']
    opt_option_tab=[[0.1,0.01],[0.01]]
    optimizer='SGD'
    SGDmomentum=0.9
    decay=1e-4
    features = 'block5_pool'
    final_clf = 'MLP1'
    transformOnFinalLayer='GlobalAveragePooling2D'
    return_best_model = True
    epochs=20
    cropCenter = True
    getBeforeReLU = True
    
    regulOnNewLayer = None
    nesterov = False
    dropout = None
    onlyPlot = False
    freezingType = 'FromBottom'

    pretrainingModif = False
    
    for opt_option in opt_option_tab:
        print('opt_option',opt_option)
        for target_dataset in target_dataset_tab:
        #            print(constrNet,style_layers)
            metrics = learn_and_eval(target_dataset,constrNet=constrNet,kind_method='FT',\
                                    epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                    forLatex=True,optimizer=optimizer,\
                                    pretrainingModif=pretrainingModif,freezingType=freezingType,\
                                    opt_option=opt_option,cropCenter=cropCenter,\
                                    style_layers=style_layers,getBeforeReLU=getBeforeReLU,\
                                    final_clf=final_clf,features=features,\
                                    return_best_model=return_best_model,\
                                    onlyReturnResult=onlyPlot,verbose=True,\
                                    dropout=dropout,regulOnNewLayer=regulOnNewLayer,\
                                    nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
            print(target_dataset,metrics)
                
    ## ResNet50 :
        
    constrNet = 'ResNet50AdaIn'
    getBeforeReLU = True
    #batch_size = 16 
    features = 'conv5_block3_out' 
    #style_layers = getBNlayersResNet50() # En fait cela ne marche pas il y a un soucis 
    # getBNlayersResNet50 ne correspond pas aux couches dans le modele et je en sais pas pourquoi 
    style_layers = ['conv1_bn',
                    'conv2_block1_1_bn', 
                    'conv2_block1_2_bn', 
                    'conv2_block1_0_bn', 
                    'conv2_block1_3_bn', 
                    'conv2_block2_1_bn', 
                    'conv2_block2_2_bn', 
                    'conv2_block2_3_bn', 
                    'conv2_block3_1_bn', 
                    'conv2_block3_2_bn', 
                    'conv2_block3_3_bn', 
                    'conv3_block1_1_bn', 
                    'conv3_block1_2_bn', 
                    'conv3_block1_0_bn', 
                    'conv3_block1_3_bn', 
                    'conv3_block2_1_bn', 
                    'conv3_block2_2_bn', 
                    'conv3_block2_3_bn', 
                    'conv3_block3_1_bn', 
                    'conv3_block3_2_bn', 
                    'conv3_block3_3_bn', 
                    'conv3_block4_1_bn', 
                    'conv3_block4_2_bn', 
                    'conv3_block4_3_bn', 
                    'conv4_block1_1_bn', 
                    'conv4_block1_2_bn', 
                    'conv4_block1_0_bn', 
                    'conv4_block1_3_bn', 
                    'conv4_block2_1_bn', 
                    'conv4_block2_2_bn',  
                    'conv4_block2_3_bn',  
                    'conv4_block3_1_bn',  
                    'conv4_block3_2_bn',  
                    'conv4_block3_3_bn', 
                    'conv4_block4_1_bn', 
                    'conv4_block4_2_bn', 
                    'conv4_block4_3_bn', 
                    'conv4_block5_1_bn', 
                    'conv4_block5_2_bn', 
                    'conv4_block5_3_bn', 
                    'conv4_block6_1_bn', 
                    'conv4_block6_2_bn', 
                    'conv4_block6_3_bn',  
                    'conv5_block1_1_bn', 
                    'conv5_block1_2_bn', 
                    'conv5_block1_0_bn', 
                    'conv5_block1_3_bn',  
                    'conv5_block2_1_bn', 
                    'conv5_block2_2_bn',  
                    'conv5_block2_3_bn',
                    'conv5_block3_1_bn', 
                    'conv5_block3_2_bn',
                    'conv5_block3_3_bn']    
    for opt_option in opt_option_tab:
        print('opt_option',opt_option)
        for target_dataset in target_dataset_tab:
    #            print(constrNet,style_layers)
            metrics = learn_and_eval(target_dataset=target_dataset,constrNet=constrNet,\
                                     forLatex=True,verbose=True,\
                                     kind_method='FT',epochs=epochs,transformOnFinalLayer=transformOnFinalLayer,\
                                     pretrainingModif=pretrainingModif,freezingType=freezingType,\
                                     optimizer=optimizer,opt_option=opt_option, #batch_size=batch_size,\
                                     final_clf=final_clf,features=features,return_best_model=return_best_model,\
                                     onlyReturnResult=onlyPlot,style_layers=style_layers,
                                     cropCenter=cropCenter,dropout=dropout,regulOnNewLayer=regulOnNewLayer,\
                                     nesterov=nesterov,SGDmomentum=SGDmomentum,decay=decay)
            print(target_dataset,metrics)
            
    # InceptionV1 adaIn a faire !
            


if __name__ == '__main__': 
    
    # Baseline perfo 
    #perf_IconArt_ArtUK_RASTA_baseline_TL()
    #perf_OffTheshelf_InceptionV1_IconArt_ArtUK_baseline()
    
    # Exp avec BatchNorm model
    #exp_BN_only()
       
    # Combine 3 nets together
    #combine_ens_models()
    
    # # Classif performance
    # print_IconArtv1_performance()
    # print_Paintings_performance()
    # print_RASTA_performance()
    # print_perform_Paintings_IconArt_RASTA_intermediaire()
    
    # Feat vizu autre reseaux pour RASTA
    RASTA_ResNet_VGG_feat_vizu()