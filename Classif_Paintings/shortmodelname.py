#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:06:10 2020

@author: gonthier
"""

def get_list_shortcut_name_model():
    """
    Without the twice transferred !
    """

    # Dans l etat actuel cela donne 29859840 possibilite je en vous laissse pas 
    # imaginer pour cela a exploser avec la fct get_list_shortcut_name_model_wTwiceTrained
    

    possible_datasets = ['IconArt_v1','RMN','RASTA','Paintings']
    possible_lr = ['big0001_modif','small001_modif','big001_modif','small01_modif','big01_modif']
    possible_llt = ['','_GAP','_GMP']
    # For the last layer transformation !!! GlobalAveragePooling2D
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
            for llt in possible_llt:
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
                                                                list_finetuned_models_name +=[dataset+'_'+lr+llt+opt+f+loss+init+crop+sup+aug+ep+c+ls+dp+le]
                                    
    return(list_finetuned_models_name)
    
def test_if_the_name_is_correct(model_name):
    """
    The goal of this fct is to test if the name_model is possible or not
    """
    
    possible_datasets = ['IconArt_v1','RMN','RASTA','Paintings']
    possible_lr = ['_big0001_modif','_small001_modif','_big001_modif','_small01_modif','_big01_modif']
    possible_llt = ['_GAP','_GMP']
    # For the last layer transformation !!! GlobalAveragePooling2D
    possible_opt = ['_adam','_Adadelta','_RMSprop']
    possible_freeze=  ['_unfreeze50','_unfreeze84','_unfreeze44','_unfreeze20','_unfreeze8']
    # _unfreeze50 for InceptionV1 to train starting at mixed4a_3x3_bottleneck_pre_relu
    # but _unfreeze84 for InceptionV1_slim to train at 
    #  Mixed_4b_Branch_1_a_1x1_conv : because the name of the layer are not the same !
    possible_loss= ['_cosineloss']
    possibleInit = ['_RandInit','_RandForUnfreezed']
    possible_crop = ['_randomCrop']
    possible_Sup = ['_deepSupervision']
    possible_Aug = ['_dataAug','_SmallDataAug','_MediumDataAug']
    possible_epochs = ['_ep120','_ep200','_ep1']
    possible_clipnorm = ['_cn1','_cn10']
    possible_LRSched = ['_LRschedG','_RedLROnPlat'] # For LR scheduler
    possible_dropout = ['_dropout04','_dropout070704'] # For LR scheduler
    # For the parameters based on : https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    # Use the learning rate at 0.01 and the list dropout
    possible_lastEpochs = ['_LastEpoch']
    correct_name = False
    for dataset in possible_datasets:
        #print(dataset,model_name)
        if dataset in model_name:
            correct_name = True
            model_name_new = model_name.replace(dataset,'')
            break
    if not(correct_name):
        if not(model_name in [None,'RandForUnfreezed','imagenet']):
            print('Dataset is missing in',model_name)
        return(False)
        #raise(ValueError('Dataset is missing'))
    correct_name = False
    for lr in possible_lr:
        if lr in model_name:
            correct_name = True
            model_name_new = model_name_new.replace(lr,'')
            break
    if not(correct_name):
        print('lr is missing in ',model_name)
        return(False)
        #raise(ValueError('lr is missing'))

    for llt in possible_llt:
        model_name_new = model_name_new.replace(llt,'')
        
    for opt in possible_opt:
        model_name_new = model_name_new.replace(llt,'')
    for f in possible_freeze:
        model_name_new = model_name_new.replace(llt,'')
    for loss in possible_loss:
        model_name_new = model_name_new.replace(llt,'')
    for init in  possibleInit:
        model_name_new = model_name_new.replace(llt,'')
    for crop in possible_crop:
        model_name_new = model_name_new.replace(llt,'')
    for sup in possible_Sup:
        model_name_new = model_name_new.replace(llt,'')
    for aug in possible_Aug:
        model_name_new = model_name_new.replace(llt,'')
    for ep in possible_epochs:
        model_name_new = model_name_new.replace(llt,'')
    for le in possible_lastEpochs:
        model_name_new = model_name_new.replace(llt,'')
    for c in possible_clipnorm:
        model_name_new = model_name_new.replace(llt,'')
    for ls in possible_LRSched:
        model_name_new = model_name_new.replace(llt,'')
    for dp in possible_dropout:
        model_name_new = model_name_new.replace(llt,'')
    
    if not(model_name_new==''):
        print(model_name_new + ' parameter is unknonw sorry.')
        return(False)
        #raise(ValueError(model_name_new + 'is unknonw sorry.'))
        
    return(True)

def test_if_the_name_is_correct_wTwiceTrained(model_name):    

    if 'XX' in model_name:
        splittedXX = model_name.split('XX')
        first_model = splittedXX[1]
        # Test if first model is correct 
        is_correct = test_if_the_name_is_correct(first_model)
        if not(is_correct):
            print('The first trained model is incorrect :',first_model)
            return(False)
        name_wo_XX = splittedXX[0][:-1] + splittedXX[2]
        is_correct = test_if_the_name_is_correct(name_wo_XX)
        return(is_correct)
    else:
        return(test_if_the_name_is_correct(model_name))
        
def get_list_shortcut_name_model_wTwiceTrained():
    list_finetuned_models_name = get_list_shortcut_name_model()
    # We will replace the possibleInit by the name of the other model to use ! 
    possibleInit = ['','_RandInit','_RandForUnfreezed']
    for model_name in list_finetuned_models_name:
        possibleInit += ['_XX'+model_name+'XX']
     
    possible_datasets = ['IconArt_v1','RMN','RASTA','Paintings']
    possible_lr = ['big0001_modif','small001_modif','big001_modif','small01_modif','big01_modif']
    possible_llt = ['','_GAP','_GMP']
    # For the last layer transformation !!! GlobalAveragePooling2D
    possible_opt = ['','_adam','_Adadelta','_RMSprop']
    possible_freeze=  ['','_unfreeze50','_unfreeze84','_unfreeze44','_unfreeze20','_unfreeze8']
    # _unfreeze50 for InceptionV1 to train starting at mixed4a_3x3_bottleneck_pre_relu
    # but _unfreeze84 for InceptionV1_slim to train at 
    #  Mixed_4b_Branch_1_a_1x1_conv : because the name of the layer are not the same !
    possible_loss= ['','_cosineloss']
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
    
    list_finetuned_models_name_twiceTrained = []
    for dataset in possible_datasets:
        for lr in possible_lr:
            for llt in possible_llt:
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
                                                                list_finetuned_models_name_twiceTrained +=[dataset+'_'+lr+llt+opt+f+loss+init+crop+sup+aug+ep+c+ls+dp+le]
                                    
    return(list_finetuned_models_name_twiceTrained)