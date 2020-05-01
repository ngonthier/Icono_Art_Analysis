# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:36:04 2020

The goal of this script is to look at the gram matrices of a fine-tuned network
To see the eigen values of its to see if we can have a better features visualisation

Pour le moment, on calcule les covariances après sous-traction de la moyenne spatiale
d'une image donnée et non pas après soustraction de la moyenne globale sur tout le 
dataset

@author: gonthier
"""

import numpy as np
from numpy import linalg as LA
import os
import platform
import pathlib
import time
import matplotlib.pyplot as plt

from tensorflow.python.keras import backend as K

from Study_Var_FeaturesMaps import get_dict_stats

from CompNet_FT_lucidIm import get_fine_tuned_model,convert_finetuned_modelToFrozenGraph

from lucid_utils import print_PCA_images

def compute_mean_var_of_GramMatrix(model_name = 'RASTA_small01_modif',classe = None,\
                                   layer='mixed4d_pre_relu'):
    
    constrNet = 'InceptionV1'
    cropCenter = True
    set_ = 'train'
    
    if 'RASTA' in model_name:
        source_dataset = 'RASTA'
    elif 'IconArt_v1' in model_name:
        source_dataset = 'IconArt_v1'
    number_im_considered = None
    style_layers = [layer]
    whatToload = 'all'
    if not(classe is None):
        classe_str = classe
    else:
        classe_str = ''
    
    #classe = 'Color_Field_Painting'
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet,model_name)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet,model_name)
    path_output_lucid_im = os.path.join(output_path,'PCAlucid')

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if classe is None:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_std_cov_matrix.npy')
    else:
        path_data_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_mean_cov_matrix.npy')
        path_data_std_cov_matrix = os.path.join(output_path,style_layers[0]+'_'+str(classe)+'_std_cov_matrix.npy')

    K.set_learning_phase(0)
    suffix = ''
    suffix_str = suffix
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    
    if not (os.path.isfile(path_data_cov_matrix) and os.path.isfile(path_data_std_cov_matrix)):
    
        fine_tuned_model, _ = get_fine_tuned_model(model_name,constrNet=constrNet,suffix='',get_Metrics=False)

        dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',set=set_,getBeforeReLU=False,\
                       Net=constrNet,style_layers_imposed=[],\
                       list_mean_and_std_source=[],list_mean_and_std_target=[],\
                       cropCenter=cropCenter,BV=True,sizeIm=224,model_alreadyLoaded=fine_tuned_model,\
                       name_model=model_name,\
                       randomCropValid=False,classe=classe)
        # Dans le cas de RASTA ce fichier fait plus de 60Go
        
        
        stats_layer = dict_stats[layer]
        print('len(stats_layer)',len(stats_layer))
        del dict_stats
        number_img = len(stats_layer)
        [cov,mean] = stats_layer[0]
        features_size,_ = cov.shape
        
        mean_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        mean_squared_value_cov_matrix = np.zeros((features_size,features_size),dtype=np.float64)
        for i in range(number_img):
            [cov,mean] = stats_layer[i]
            mean_cov_matrix += cov/number_img
            cov_squared = cov**2
            mean_squared_value_cov_matrix += (cov_squared)/number_img
            
            isnan_cov_squared = np.isnan(cov_squared)
            where_is_nan = np.where(isnan_cov_squared)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,cov[ci,cj],cov_squared[ci,cj])
            isnan_mean_cov = np.isnan(mean_cov_matrix)
            where_is_nan = np.where(isnan_mean_cov)
            for ci,cj in zip(where_is_nan[0],where_is_nan[1]):
                print('nan in cov_squared',i,ci,cj,isnan_mean_cov[ci,cj])
            
        squared_mean_cov_matrix = mean_cov_matrix**2
        var_cov_matrix = mean_squared_value_cov_matrix - squared_mean_cov_matrix
        var_cov_matrix = np.clip(var_cov_matrix,0.0,np.inf)
        std_cov_matrix = np.sqrt(var_cov_matrix)
        
        if classe is None:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
        else:
            np.save(path_data_cov_matrix, mean_cov_matrix)
            np.save(path_data_std_cov_matrix, std_cov_matrix)
    else:
        mean_cov_matrix = np.load(path_data_cov_matrix)
        std_cov_matrix = np.load(path_data_std_cov_matrix)
        
    features_size,_ = mean_cov_matrix.shape
    print(style_layers[0],str(classe))
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of Mean of the cov matrices'.format(np.mean(mean_cov_matrix),np.median(mean_cov_matrix),np.min(mean_cov_matrix),np.max(mean_cov_matrix)))
    #Mean 9.48e+01, median 8.80e+01, min -1.40e+03 and max 4.97e+03 of Mean of the cov matrices
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of abs Mean of the cov matrices'.format(np.mean(np.abs(mean_cov_matrix)),np.median(np.abs(mean_cov_matrix)),np.min(np.abs(mean_cov_matrix)),np.max(np.abs(mean_cov_matrix))))
    #Mean 1.47e+02, median 1.16e+02, min 6.99e-04 and max 4.97e+03 of abs Mean of the cov matrices
    print('Mean {0:.2e}, median {1:.2e}, min {2:.2e} and max {3:.2e} of std of the cov matrices'.format(np.mean(std_cov_matrix),np.median(std_cov_matrix),np.min(std_cov_matrix),np.max(std_cov_matrix)))
    #Mean 3.62e+02, median 3.35e+02, min 8.02e+01 and max 3.00e+03 of std of the cov matrices
    
    where_std_sup_mean = np.where(std_cov_matrix > mean_cov_matrix)
    print('Number of time std > mean in the 528*528 = 278784 matrice :',len(where_std_sup_mean[0]))
    # 271913
    where_std_sup_abs_mean = np.where(std_cov_matrix > np.abs(mean_cov_matrix))
    print('Number of time std > mean in the 528*528 = 278784 matrice :',len(where_std_sup_abs_mean[0]))
    # 271679   

#    mixed4d_pre_relu None
#    Mean 9.48e+01, median 8.80e+01, min -1.40e+03 and max 4.97e+03 of Mean of the cov matrices
#    Mean 1.47e+02, median 1.16e+02, min 6.99e-04 and max 4.97e+03 of abs Mean of the cov matrices
#    Mean 3.62e+02, median 3.35e+02, min 8.02e+01 and max 3.00e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 271913
#    Number of time std > mean in the 528*528 = 278784 matrice : 271679
#    
#    mixed4d_pre_relu Abstract_Art
#    Mean 1.23e+02, median 1.11e+02, min -1.29e+03 and max 3.57e+03 of Mean of the cov matrices
#    Mean 1.74e+02, median 1.38e+02, min 3.37e-03 and max 3.57e+03 of abs Mean of the cov matrices
#    Mean 3.47e+02, median 3.21e+02, min 6.84e+01 and max 2.71e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 266607
#    Number of time std > mean in the 528*528 = 278784 matrice : 266325
#    
#    mixed4d_pre_relu Color_Field_Painting
#    Mean 9.37e+01, median 8.63e+01, min -9.74e+02 and max 2.47e+03 of Mean of the cov matrices
#    Mean 1.21e+02, median 9.77e+01, min 2.04e-05 and max 2.47e+03 of abs Mean of the cov matrices
#    Mean 2.44e+02, median 2.27e+02, min 4.11e+01 and max 2.10e+03 of std of the cov matrices
#    Number of time std > mean in the 528*528 = 278784 matrice : 263827
#    Number of time std > mean in the 528*528 = 278784 matrice : 263111
    
    # Compute the eigen values 
    #del std_cov_matrix,where_std_sup_abs_mean,where_std_sup_mean
    eigen_values, eigen_vectors = LA.eig(mean_cov_matrix)
    
    pathlib.Path(path_output_lucid_im).mkdir(parents=True, exist_ok=True) 
    plt.figure()
    plt.scatter(np.arange(0,len(eigen_values)),eigen_values,s=4)
    plt.ylabel('Eigen Value')
    plt.savefig(os.path.join(path_output_lucid_im,'EigenValues_'+layer+'_'+classe_str+'.png'),\
                dpi=300)
    plt.close()
    
    print('Eigen values 10 first value',eigen_values[0:10])
    #print('First eigen vector :',eigen_vectors[:,0])
    print('Max imag part first vector :',np.max(np.imag(eigen_vectors[:,0])))
    eigen_vectors = np.real(eigen_vectors)

    num_components_draw = 10
    
    name_pb = 'tf_graph_'+constrNet+model_name+suffix_str+'.pb'
    if not(os.path.isfile(os.path.join(path_lucid_model,name_pb))):
        name_pb = convert_finetuned_modelToFrozenGraph(model_name,
                                   constrNet=constrNet,path=path_lucid_model,suffix=suffix)
    
    if constrNet=='VGG':
        input_name_lucid ='block1_conv1_input'
    elif constrNet=='InceptionV1':
        input_name_lucid ='input_1'
    for comp_number in range(num_components_draw):
        weights = eigen_vectors[:,comp_number]
        #weights = weights[0:1]
        #print('weights',weights)
        #time.sleep(.300)
        prexif_name = '_PCA'+str(comp_number)
        if not(classe is None):
           prexif_name += '_'+classe 
        index_features_withinLayer_all = np.arange(0,features_size)
        print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights,\
                         index_features_withinLayer=index_features_withinLayer_all,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_pos = prexif_name + '_PosContrib'
        where_pos = np.where(weights>0.)[0]
        weights_pos = list(weights[where_pos])
        print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights_pos,\
                         index_features_withinLayer=where_pos,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_pos,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        prexif_name_neg = prexif_name + '_NegContrib'
        where_neg = np.where(weights>0.)[0]
        weights_neg = list(-weights[where_neg])
        print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=weights_neg,\
                         index_features_withinLayer=where_neg,\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_neg,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        where_max = np.argmax(weights)
        prexif_name_max = prexif_name+  '_Max'+str(where_max)
        print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=[1.],\
                         index_features_withinLayer=[where_max],\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        where_min = np.argmin(weights)
        prexif_name_max = prexif_name+  '_Min'+str(where_min)
        print_PCA_images(model_path=os.path.join(path_lucid_model,name_pb),
                         layer_to_print=layer,weights=[1.],\
                         index_features_withinLayer=[where_min],\
                         path_output=path_output_lucid_im,prexif_name=prexif_name_max,\
                         input_name=input_name_lucid,Net=constrNet,sizeIm=256)
        
        # Faire que les positives après
        
#        print_images(model_path=name_pb,list_layer_index_to_print,path_output='',prexif_name='',\
#                 input_name='block1_conv1_input',Net='VGG')
#    
#    
#    matrix_of_cov_matrix = np.empty((number_img,features_size,features_size),dtype=np.float32)
#    matrix_of_mean_matrix = np.empty((number_img,features_size),dtype=np.float32)
#    print('matrix_of_cov_matrix',matrix_of_cov_matrix.shape)
#    list_cov = []
#    list_mean = []
    
#    matrix_of_cov_matrix = np.memmap('cov_matrix.dat', dtype=np.float32,
#              mode='w+', shape=(number_img,features_size,features_size))
#    matrix_of_mean_matrix = np.memmap('mean_matrix.dat', dtype=np.float32,
#              mode='w+', shape=(number_img,features_size))
#    for i in range(number_img):
#        [cov,mean] = stats_layer[i]
#        matrix_of_cov_matrix[i,:,:] = cov
#        matrix_of_mean_matrix[i,:] = mean
    
#    del stats_layer
    
if __name__ == '__main__':
    compute_mean_var_of_GramMatrix(model_name = 'RASTA_small01_modif',classe = None,\
                                   layer='mixed4d_pre_relu')
    compute_mean_var_of_GramMatrix(model_name = 'RASTA_small01_modif',classe ='Color_Field_Painting',\
                                   layer='mixed4d_pre_relu')
    compute_mean_var_of_GramMatrix(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe = None,\
                                   layer='mixed4d_pre_relu')
    compute_mean_var_of_GramMatrix(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe ='Mary',\
                                   layer='mixed4d_pre_relu')
    compute_mean_var_of_GramMatrix(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe = None,\
                                   layer='mixed4b_pre_relu')
    compute_mean_var_of_GramMatrix(model_name = 'IconArt_v1_big001_modif_adam_SmallDataAug_ep200',classe ='Mary',\
                                   layer='mixed4b_pre_relu')
    
    
    
    

