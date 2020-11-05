# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:57:11 2020

Experiences en plus a faire tourner pour la these : 
    fine tuning des modeles avec parfois de la visualization des features ! 

@author: gonthier
"""

from CompNet_FT_lucidIm import Comparaison_of_FineTunedModel,print_performance_FineTuned_network

### Cas de Paintings et IconArt pour fine-tune les modeles et voir les performances

def print_IconArtv1_performance(latexOutput=True):
    # For Classification performance different setup
    list_models_name_VGG = ['IconArt_v1_small01_modif_GAP',
                            'IconArt_v1_big01_modif_GAP',
                            'IconArt_v1_big001_modif_GAP',
                            'IconArt_v1_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                            'IconArt_v1_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name_VGG,
                                        suffix_tab=[''],latexOutput=latexOutput)
    list_models_name_ResNet = ['IconArt_v1_small01_modif_GAP',
                            'IconArt_v1_big01_modif_GAP',
                            'IconArt_v1_big001_modif_GAP',
                            'IconArt_v1_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                            'IconArt_v1_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name_ResNet,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
#    list_models_name=['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                      'IconArt_v1_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
#    print_performance_FineTuned_network(constrNet='InceptionV1_slim',
#                                        list_models_name=list_models_name,
#                                        suffix_tab=[''],latexOutput=latexOutput)
def print_Paintings_performance(latexOutput=True):
    # For Classification performance different setup
    list_models_name_VGG = ['Paintings_small01_modif_GAP',
                            'Paintings_big01_modif_GAP',
                            'Paintings_big001_modif_GAP',
                            'Paintings_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                            'Paintings_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='VGG',
                                        list_models_name=list_models_name_VGG,
                                        suffix_tab=[''],latexOutput=latexOutput)
    list_models_name_ResNet = ['Paintings_small01_modif_GAP',
                            'Paintings_big01_modif_GAP',
                            'Paintings_big001_modif_GAP',
                            'Paintings_big0001_modif_GAP_adam_unfreeze20_RandForUnfreezed_SmallDataAug_ep200',
                            'Paintings_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            ]

    print_performance_FineTuned_network(constrNet='ResNet50',
                                        list_models_name=list_models_name_ResNet,
                                        suffix_tab=[''],latexOutput=latexOutput)
    
#    list_models_name=['IconArt_v1_big001_modif_adam_unfreeze84_SmallDataAug_ep200',
#                      'IconArt_v1_big001_modif_Adadelta_unfreeze84_MediumDataAug_ep200']
#    print_performance_FineTuned_network(constrNet='InceptionV1_slim',
#                                        list_models_name=list_models_name,
#                                        suffix_tab=[''],latexOutput=latexOutput)
    
    


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
                            'RASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
                            ]
     Comparaison_of_FineTunedModel(list_model_name_4,constrNet='VGG')



### Cas de Paintins et IconArt avec un passage intermediaire par RASTA

def print_perform_Paintings_IconArt_RASTA_intermediaire(latexOutput=True):
    
    # VGG IconArt
    list_models_name = ['IconArt_v1_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX'
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
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='ResNet',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)
    # VGG Paintings
    list_models_name = ['Paintings_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big001_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX'
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
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX'
                        ]

    print_performance_FineTuned_network(constrNet='ResNet',
                                        list_models_name=list_models_name,
                                        suffix_tab=[''],latexOutput=latexOutput)


if __name__ == '__main__': 
    
    print_IconArtv1_performance()
    print_Paintings_performance()
    RASTA_ResNet_VGG_feat_vizu()
    print_perform_Paintings_IconArt_RASTA_intermediaire()