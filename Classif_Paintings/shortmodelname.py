#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:06:10 2020

@author: gonthier
"""

def get_list_shortcut_name_model():

    possible_datasets = ['IconArt_v1','RMN','RASTA']
    possible_lr = ['big0001_modif','small001_modif','big001_modif','small01_modif','big01_modif']
    possible_opt = ['','_adam','_Adadelta','_RMSprop']
    possible_freeze=  ['','_unfreeze50','_unfreeze84','_unfreeze44','_unfreeze20','_unfreeze8']
    # _unfreeze50 for InceptionV1 to train starting at mixed4a_3x3_bottleneck_pre_relu
    # but _unfreeze84 for InceptionV1_slim to train at 
    #  Mixed_4b_Branch_1_a_1x1_conv : because the name of the layer are not the same !
    possible_loss= ['','_cosineloss']
    possibleInit = ['','_RandInit','_RandForUnfreezed']
    possible_crop = ['','_randomCrop']
    possible_Sup = ['','_deepSupervision']
    possible_Aug = ['','_dataAug','_SmallDataAug','_MediumDataAug']
    possible_epochs = ['','_ep120','_ep200','_ep1']
    possible_clipnorm = ['','_cn1','_cn10']
    possible_LRSched = ['','_LRschedG','_RedLROnPlat'] # For LR scheduler
    possible_dropout = ['','_dropout04','_dropout070704'] # For LR scheduler
    # For the parameters based on : https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    # Use the learning rate at 0.01 and the list dropout
    possible_lastEpochs = ['','_LastEpoch']
    
    list_finetuned_models_name = []
    for dataset in possible_datasets:
        for lr in possible_lr:
            for opt in possible_opt:
                for f in possible_freeze:
                    for loss in possible_loss:
                        for init in  possibleInit:
                            for crop in possible_crop:
                                for sup in possible_Sup:
                                    for aug in possible_Aug:
                                        for ep in possible_epochs:
                                            for le in possible_lastEpochs:
                                                for c in possible_clipnorm:
                                                    for ls in possible_LRSched:
                                                        for dp in possible_dropout:
                                                            list_finetuned_models_name +=[dataset+'_'+lr+opt+f+loss+init+crop+sup+aug+ep+c+ls+dp+le]
                                
    return(list_finetuned_models_name)