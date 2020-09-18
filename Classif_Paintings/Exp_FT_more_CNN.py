#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:32:59 2020

The goal of this script is just to run more architecture !

@author: gonthier
"""

from CompNet_FT_lucidIm import Comparaison_of_FineTunedModel

if __name__ == '__main__': 
    
    
    # For ResNet model
    list_model_RASTA = ['RASTA_small01_modif_GAP',
                        'RASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
                        'RASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
    
    Comparaison_of_FineTunedModel(list_model_RASTA,constrNet='ResNet50')
    
    list_models_name_P = ['Paintings_small01_modif_GAP',
                        'Paintings_big01_modif_GAP',
                        'Paintings_big001_modif_GAP',
                        'Paintings_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big001_modif_GAP_XXRASTA_small01_modif_GAPXX']
    
    Comparaison_of_FineTunedModel(list_models_name_P,constrNet='ResNet50')

    list_models_name_I = ['IconArt_v1_small01_modif_GAP',
                        'IconArt_v1_big01_modif_GAP',
                        'IconArt_v1_big001_modif_GAP',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_small01_modif_GAPXX']
    
    Comparaison_of_FineTunedModel(list_models_name_I,constrNet='ResNet50')
    
    ## For VGG
    list_model_RASTAVGG = ['RASTA_small01_modif_GAP',
                        'RASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200',
                        'RASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
     
    Comparaison_of_FineTunedModel(list_model_RASTAVGG,constrNet='VGG') 
    
    list_models_name_P = ['Paintings_small01_modif_GAP',
                        'Paintings_big01_modif_GAP',
                        'Paintings_big001_modif_GAP',
                        'Paintings_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'Paintings_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'Paintings_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'Paintings_big001_modif_GAP_XXRASTA_small01_modif_GAPXX']
    
    Comparaison_of_FineTunedModel(list_models_name_P,constrNet='VGG')

    list_models_name_I = ['IconArt_v1_small01_modif_GAP',
                        'IconArt_v1_big01_modif_GAP',
                        'IconArt_v1_big001_modif_GAP',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big0001_modif_GAP_adam_unfreeze8_RandForUnfreezed_SmallDataAug_ep200XX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_big001_modif_GAP_RandInit_randomCrop_deepSupervision_ep200_LRschedGXX',
                        'IconArt_v1_small01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big01_modif_GAP_XXRASTA_small01_modif_GAPXX',
                        'IconArt_v1_big001_modif_GAP_XXRASTA_small01_modif_GAPXX']
    
    Comparaison_of_FineTunedModel(list_models_name_I,constrNet='VGG')