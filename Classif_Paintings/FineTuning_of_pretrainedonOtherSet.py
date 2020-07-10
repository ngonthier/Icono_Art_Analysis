#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:34:48 2020

Le but de ce script est d'evaluer la pertinence d'utiliser un modèle fine-tuné 
sur RASTA pour l'appliquer a IconArt ou bien Paintings 

@author: gonthier
"""

from StatsConstr_ClassifwithTL import FineTuneModel,learn_and_eval
from CompNet_FT_lucidIm import get_fine_tuned_model
from IMDB import get_database

def FT_other_model(model_name,constrNet,target_dataset='IconArt_v1'):
    """
    @param : model_name : name of the model fine-tuned on other dataset before
    """
    
    ft_model = get_fine_tuned_model(model_name,constrNet)
    
    # TODO finir ici !!! 
    # Il faudra peut etre faire passer les modeles fine-tuned dans l'argument weights au lieu de 
    # imagenet ou autre et qui va charger les autres models deja !!! pour tester 

    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
        path_data,Not_on_NicolasPC = get_database(target_dataset)
    
    model_trained_on_target_set = FineTuneModel(ft_model,dataset=target_dataset,df=df_label,\
                        x_col=item_name,y_col=classes,path_im=path_to_img,\
                        str_val=str_val,num_classes=len(classes),epochs=epochs,\
                        Net=constrNet,plotConv=plotConv,batch_size=batch_size,\
                        cropCenter=cropCenter,return_best_model=return_best_model,\
                        NoValidationSetUsed=NoValidationSetUsed,\
                        RandomValdiationSet=RandomValdiationSet,\
                        deepSupervision=deepSupervision,dataAug=dataAug,\
                        last_epochs_model_path=last_epochs_model_path,\
                        history_path=local_history_path,randomCrop=randomCrop,\
                        LR_scheduling=LR_scheduling,\
                        imSize=imSize)

def trainedonRASTA_finetuned_IconArt_v1():
    FT_other_model(model_name='RASTA_small01_modif',constrNet='InceptionV1',target_dataset='IconArt_v1')