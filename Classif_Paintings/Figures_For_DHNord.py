# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:15:51 2020

List des figures a faire pour le papier DHNord 

@author: gonthier
"""

import os
import pathlib
import matplotlib
import tensorflow as tf

from CompNet_FT_lucidIm import do_lucid_vizu_for_list_model
from Activation_for_model import plot_images_Pos_Images

import lucid_utils



def do_TopK_figures(list_models_name,list_layer_index_to_print,suffix_tab=[''],dataset='RASTA',
                    constrNet='InceptionV1',
                    numberIm = 100,
                    stats_on_layer = 'meanAfterRelu'):
    
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    for model_name in list_models_name:
        print('#### ',model_name)
              
        for layer_name,num_feature in list_layer_index_to_print:
        
            if model_name=='pretrained':
                plot_images_Pos_Images(dataset,model_name,constrNet,
                                layer_name=layer_name,
                                num_feature=num_feature,
                                numberIm=numberIm,stats_on_layer=stats_on_layer,suffix='',
                                FTmodel=True)
            elif not(model_name=='random'):
                
                for suffix in suffix_tab:
                    plot_images_Pos_Images(dataset,model_name,constrNet,
                                layer_name=layer_name,
                                num_feature=num_feature,
                                numberIm=numberIm,stats_on_layer=stats_on_layer,suffix=suffix,
                                FTmodel=True)

if __name__ == '__main__': 
    
    # Liste figure pour AdaptationFiltersRASTA de DHNord 2020 paper
    suffix_tab = ['','1']
    list_features = [['mixed4d_pool_reduce_pre_relu',64],['mixed4b_3x3_bottleneck_pre_relu',35],['mixed4d_3x3_pre_relu',52]]
    list_models = ['RASTA_small01_modif','RASTA_small001_modif','RASTA_big001_modif',
                        'RASTA_small001_modif_deepSupervision','RASTA_big001_modif_deepSupervision']
    # Il y a un pb non resolu avec le pretrained model !
    output_path = path_lucid_model = os.path.join(os.sep,'Users','gonthier','Travail','DHNordPaper','im')
    do_lucid_vizu_for_list_model(list_models_name=list_models,list_layer_index_to_print=list_features,
                                 suffix_tab = suffix_tab,
                                 output_path=output_path,constrNet='InceptionV1')
    
    # 
    
    
    