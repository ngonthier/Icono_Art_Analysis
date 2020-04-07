# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:39:41 2020

The goal of this script is to compute the mean value of each features maps of 
the whole image from a given training dataset for a given network

@author: gonthier
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras import Model

import numpy as np
import platform
import pathlib
import os
import pickle
import matplotlib

from StatsConstr_ClassifwithTL import predictionFT_net
from googlenet import inception_v1_oldTF as Inception_V1
from IMDB import get_database
from plots_utils import plt_multiple_imgs
from CompNet_FT_lucidIm import get_fine_tuned_model

def get_Network(Net):
    weights = 'imagenet'
    
    if Net=='VGG':
        imagenet_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=weights)
    elif Net == 'InceptionV1':
        imagenet_model = Inception_V1(include_top=False, weights=weights)
    else:
        raise(NotImplementedError)
        
    return(imagenet_model)

def get_Model_that_output_meanActivation(model):
    
    list_outputs = []
    list_outputs_name = []
    
    for layer in model.layers:
        if  isinstance(layer, Conv2D) :
            layer_output = layer.output
            mean_each_feature = tf.keras.backend.mean(layer_output, axis=[1,2], keepdims=False)
            list_outputs += [mean_each_feature]
            list_outputs_name += [layer.name]
            
    new_model = Model(model.input,list_outputs)
    
    return(new_model,list_outputs_name)
    
    
def compute_meanValue_Feature(dataset,model_name,constrNet,suffix=''):
    """
    This function will compute the mean activation of each features maps for all
    the convolutionnal layers 
    """
    K.set_learning_phase(0) #IE no training
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    
    cropCenter = True
    
    if model_name=='pretrained':
        base_model = get_Network(constrNet)
    else:
        # Pour ton windows il va falloir copier les model .h5 finetuné dans ce dossier la 
        # C:\media\gonthier\HDD2\output_exp\Covdata\RASTA\model
        base_model = get_fine_tuned_model(model_name,constrNet=constrNet,suffix=suffix)
        
    model,list_outputs_name = get_Model_that_output_meanActivation(base_model)
    #print(model.summary())
    activations = predictionFT_net(model,df_train,x_col=item_name,y_col=classes,path_im=path_to_img,
                     Net=constrNet,cropCenter=cropCenter)
    print('activations len and shape',len(activations),activations[0].shape)
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    act_plus_layer = [list_outputs_name,activations]
    save_file = os.path.join(output_path,'activations_per_img.pkl')
    with open(save_file, 'wb') as handle:
        pickle.dump(act_plus_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return(list_outputs_name,activations)
  
def plot_images_Pos_Images(dataset,model_name,constrNet,
                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
                                                num_feature=64,
                                                numberIm=9):
    """
    This function will plot k image a given layer with a given features number
    """
    cropCenter = True
    printNearZero = False
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(dataset)
    df_train = df_label[df_label['set']=='train']
    name_images = df_train[item_name].values
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    # For images
    output_path_for_img = os.path.join(output_path,'ActivationsImages')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(output_path_for_img).mkdir(parents=True, exist_ok=True) 
        
    save_file = os.path.join(output_path,'activations_per_img.pkl')
    
    if os.path.exists(save_file):
        # The file exist
        with open(save_file, 'rb') as handle:
            act_plus_layer = pickle.load(handle)
            [list_outputs_name,activations] = act_plus_layer
    else:
        list_outputs_name,activations = compute_meanValue_Feature(dataset,model_name,constrNet)
    
    for layer_name_inlist,activations_l in zip(list_outputs_name,activations):
        if layer_name==layer_name_inlist:
            print(layer_name,num_feature)
            activations_l_f = activations_l[:,num_feature]
            where_activations_l_f_pos = np.where(activations_l_f>0)[0]
            activations_l_f_pos = activations_l_f[where_activations_l_f_pos]
            name_images_l_f_pos = name_images[where_activations_l_f_pos]
            argsort = np.argsort(activations_l_f_pos)[::-1]
            # Most positive images
            list_most_pos_images = name_images_l_f_pos[argsort[0:numberIm]]
            act_most_pos_images = activations_l_f_pos[argsort[0:numberIm]]
            title_imgs = []
            for act in act_most_pos_images:
                str_act = '{:.02f}'.format(act)
                title_imgs += [str_act]
            name_fig = dataset+'_'+layer_name+'_'+str(num_feature)+'_Most_Pos_Images_NumberIm'+str(numberIm)
            plt_multiple_imgs(list_images=list_most_pos_images,path_output=output_path_for_img,\
                              path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
                              Net=None,title_imgs=title_imgs)
            
#            # Slightly positive images : TODO
#            list_slightly_pos_images = name_images_l_f_pos[argsort[-numberIm:]]
#            act_slightly_pos_images = activations_l_f_pos[argsort[-numberIm:]]
#            title_imgs = []
#            for act in act_slightly_pos_images:
#                str_act = '{:.02f}'.format(act)
#                title_imgs += [str_act]
#            name_fig = dataset+'_'+layer_name+'_'+str(num_feature) +'_Slightly_Pos_Images_NumberIm'+str(numberIm)
#            plt_multiple_imgs(list_images=list_slightly_pos_images,path_output=output_path_for_img,\
#                              path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
#                              Net=None,title_imgs=title_imgs)
            
            # Positive near zero images
            if printNearZero:
                list_nearZero_pos_images = name_images_l_f_pos[argsort[-numberIm:]]
                act_nearZero_pos_images = activations_l_f_pos[argsort[-numberIm:]]
                title_imgs = []
                for act in act_nearZero_pos_images:
                    str_act = '{:.02f}'.format(act)
                    title_imgs += [str_act]
                name_fig = dataset+'_'+layer_name+'_'+str(num_feature) +'_Near_Zero_Pos_Images_NumberIm'+str(numberIm)
                plt_multiple_imgs(list_images=list_nearZero_pos_images,path_output=output_path_for_img,\
                                  path_img=path_to_img,name_fig=name_fig,cropCenter=cropCenter,
                                  Net=None,title_imgs=title_imgs)
    
if __name__ == '__main__': 
    # Petit test 
    #compute_meanValue_Feature(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1')
    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
                                                num_feature=64,
                                                numberIm=9)
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4d_3x3_bottleneck_pre_relu',
#                                                num_feature=64,
#                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
                                                layer_name='mixed4d_3x3_pre_relu',
                                                num_feature=52,
                                                numberIm=81)
#    # mixed4d_pool_reduce_pre_reluConv2D_63_RASTA_small01_modif.png	
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4d_pool_reduce_pre_relu',
#                                                num_feature=63,
#                                                numberIm=81)
#    #Nom de fichier	mixed4b_3x3_bottleneck_pre_reluConv2D_35_RASTA_small01_modif.png	
#    plot_images_Pos_Images(dataset='RASTA',model_name='RASTA_small01_modif',constrNet='InceptionV1',
#                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
#                                                num_feature=35,
#                                                numberIm=81)
#    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
#                                                layer_name='mixed4b_3x3_bottleneck_pre_relu',
#                                                num_feature=35,
#                                                numberIm=81)
    plot_images_Pos_Images(dataset='RASTA',model_name='pretrained',constrNet='InceptionV1',
                                                layer_name='mixed4d_pool_reduce_pre_relu',
                                                num_feature=63,
                                                numberIm=81)
    # Nom de fichier	mixed3a_5x5_bottleneck_pre_reluConv2D_8_RASTA_small01_modif.png	


