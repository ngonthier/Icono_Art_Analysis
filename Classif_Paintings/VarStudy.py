#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 18:13:30 2018

PCA des vecteurs W aider par 
eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_ACP_Python.pdf

@author: gonthier
"""
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import pathlib

from IMDB_study import getDictFeaturesFasterRCNN,getTFRecordDataset
from IMDB import get_database
from MINNpy3.mil_nets.WSOD_datasets import load_dataset

import itertools
import tensorflow as tf
from tf_faster_rcnn.lib.model.nms_wrapper import nms
import os
import cv2

def drawCircles(X,n,p,pca):
    print('Plot vectors in the cercle')
    eigval = (n-1)/n*pca.explained_variance_
    plt.figure()
    plt.plot(np.arange(1,p+1),eigval)
    plt.title("Scree plot")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.show()
    #racine carrée des valeurs propres 
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p,p))
    for k in range(p):
        corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
    #cercle des corrélations 
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    #affichage des étiquettes (noms des variables) 
    for j in range(p):
        plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1])) 
    #ajouter les axes 
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)  
    plt.title('Qualité de représentation des variables (COS²)')
    plt.show()
    
    
    
    cos2var = corvar**2
    print(pd.DataFrame({'id':X.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))
    #contributions 
    ctrvar = cos2var
    for k in range(p):
        ctrvar[:,k] = ctrvar[:,k]/eigval[k]
    #on n'affiche que pour les deux premiers axes
    print(pd.DataFrame({'id':X.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))
    
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    #affichage des étiquettes (noms des variables) 
    for j in range(p):
        plt.annotate(X.columns[j],(ctrvar[j,0],ctrvar[j,1])) 
    #ajouter les axes 
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)  
    plt.title('Contribution des variables aux axes (CTR)')
    #  La contribution est également basée sur le carré de la corrélation, mais relativisée par l’importance de l’axe
    plt.show()
    


def PCAplots(Wstored,Lossstored):
     # PCA 
    n_components = Wstored.shape[2]
    num_samples = Wstored.shape[1]
    d =pd.DataFrame(Wstored[0,:,:])
    print('PCA without reduction')
    pca = PCA(n_components=n_components,svd_solver='full')
    principalComponents = pca.fit_transform(Wstored[0,:,:]) # On the first class
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    singular_values = pca.singular_values_
    print('Max  of exxplained variance ratio',np.max(explained_variance_ratio))
    plt.figure()
    plt.semilogy(explained_variance_ratio)
    plt.title('Semilog of Variance ratio PCA explained')
    plt.figure()
    plt.plot(explained_variance_ratio_cumsum)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Cumsum of Variance ratio PCA explained')
    print('Cumsum over 0.9 for component number',np.where(explained_variance_ratio_cumsum>0.9)[0][0])
    print('Cumsum over 0.95 for component number',np.where(explained_variance_ratio_cumsum>0.95)[0][0])
    print('explained_variance_ratio of the three first components',explained_variance_ratio[0:3])
    input('wait to close an continue')
    plt.close('all')
    target1 = Lossstored[0,:]
    target2 = np.zeros_like(target1)
    for i in range(9):
        target2[np.where(target1<np.percentile(target1, 10*(i+1))) and np.where(target1>np.percentile(target1, 10*(i)))] = i
    target2[np.where(target1>np.percentile(target1, 90))] = 9
    list_target = [target1,target2]
    
    for target in list_target:
        plt.figure()
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
                c=target, edgecolor='none', alpha=0.5)
    #            cmap=plt.cm.get_cmap('spectral', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.title('First 2 components of ACP')
        # Plot 3D
    #    fig = plt.figure(1, figsize=(4, 3))
        fig = plt.figure()
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=target,
               edgecolor='k')
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.show()
    input('wait to close an continue')
    plt.close('all')
    #scree plot
    X = d
    n = num_samples
    p = n_components
    if n > p:
        drawCircles(X,n,p,pca)
    
    input('wait to close an continue')
    plt.close('all') 
    # PCA with reduction 
    print('PCA with reduction')
    Wstore_scaled = preprocessing.scale(Wstored[0,:,:]) # Remove mean and divide by std 
    dreduc =pd.DataFrame(Wstore_scaled)
    pca = PCA(n_components=n_components,svd_solver='full')
    principalComponents = pca.fit_transform(Wstore_scaled) # On the first class
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    singular_values = pca.singular_values_
    print('Max  of exxplained variance ratio',np.max(explained_variance_ratio))
    plt.figure()
    plt.semilogy(explained_variance_ratio)
    plt.title('Semilog of Variance ratio PCA explained')
    plt.figure()
    plt.plot(explained_variance_ratio_cumsum)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Cumsum of Variance ratio PCA explained')
    print('Cumsum over 0.9 for component number',np.where(explained_variance_ratio_cumsum>0.9)[0][0])
    print('Cumsum over 0.95 for component number',np.where(explained_variance_ratio_cumsum>0.95)[0][0])
    print('explained_variance_ratio of the three first components',explained_variance_ratio[0:3])
    input('wait to close an continue')
    plt.close('all')
#    target = np.sign(Lossstored[0,:])
    for target in list_target:
        plt.figure()
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
                c=target, edgecolor='none', alpha=0.5)
    #            cmap=plt.cm.get_cmap('spectral', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.title('First 2 components of ACP')
        # Plot 3D
    #    fig = plt.figure(1, figsize=(4, 3))
        fig = plt.figure()
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=target,
               edgecolor='k')
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.title('Version 3D')
        plt.show()
    input('wait to close an continue')
    plt.close('all')
    if n > p:
        drawCircles(dreduc,n,p,pca)

def Etude_Wvectors():
    
    plt.ion()
    
    name_dict = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/1533740133.283923.pkl'
    name_dict = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/MImax_PeopletArt_10800W_withScore_classicLoss.pkl'
    with open(name_dict, 'rb') as f:
        Dict = pickle.load(f)
    
    Wstored = Dict['Wstored']
    num_classes,num_samples,num_features = Wstored.shape
    Bstored = Dict['Bstored']
    Wstored_w_bias = np.expand_dims(np.append(Wstored[0,:,:],np.reshape(Bstored,(num_samples,1)),axis=1),axis=0)
    Lossstored = Dict['Lossstored']
    np_pos_value = Dict['np_pos_value']
    np_neg_value = Dict['np_neg_value']
    print('Dimension of Wstored',Wstored.shape)
  
    
    # STD of the features
    std_Wstored = np.std(Wstored[0,:,:],axis=0)
    mean_ofStd = np.mean(std_Wstored)
    print('mean_ofStd',mean_ofStd)
    print('Element with the strongest std',np.argmax(std_Wstored),'value :',np.max(std_Wstored))
    std_Wstored_sorted = np.sort(std_Wstored)
    plt.figure()
    plt.plot(std_Wstored_sorted)
    plt.title('STD of features (sorted)')
    Wstored_w_bias_std = np.std(Wstored_w_bias[0,:,:],axis=0)
    print('Element with the strongest std',np.argmax(Wstored_w_bias_std),'value :',np.max(Wstored_w_bias_std))
    std_Wstored_sorted = np.sort(Wstored_w_bias_std)
    plt.figure()
    plt.plot(std_Wstored_sorted)
    plt.title('STD of features and bias (sorted)')
    
    # Sign of the features
    Wstored_sign = np.sign(Wstored[0,:,:])
    Wpos = np.sum((1 +Wstored_sign)/2.,axis=0)
    Wneg = np.sum((1 -Wstored_sign)/2.,axis=0)
    plt.figure()
    plt.hist(Wpos,num_samples+2,normed=True)
    plt.title('Histogram of the features always positive')
    Wstored_sign_sum = (num_samples - Wpos )/num_samples
    plt.figure()
    plt.hist(Wstored_sign_sum,1000)
    plt.title('Visualisation of the sign changing features')
    input('wait to close an continue')
    plt.close('all')
#    # Matrice de correlation
#    d =pd.DataFrame(Wstored[0,:,:])
#    corr = d.corr()
#    mask = np.zeros_like(corr, dtype=np.bool)
#    mask[np.triu_indices_from(mask)] = True
#    
#    # Set up the matplotlib figure
#    f, ax = plt.subplots(figsize=(11, 9))
#    
#    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(220, 10, as_cmap=True)
#    
#    # Draw the heatmap with the mask and correct aspect ratio
#    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})    
#    
#    index = np.triu_indices_from(corr, k=1)
#    list_corr = []
#    corr_np = np.array(corr)
#    for x,y in zip(index[0],index[1]):
#        list_corr += [np.abs(corr_np[x,y])]
#    plt.figure()
#    plt.hist(list_corr,100)
#    plt.title('Histogram of the absolute value of the correlation between features')
    
    PCAplots(Wstored,Lossstored)
    input('wait to close an continue')
    plt.close('all')
    
    ## with bias
    print('PCA with bias without reduction')
    PCAplots(Wstored_w_bias,Lossstored)
    input('wait to close an continue')
    plt.close('all')
#    
    # Valeur de la fonction de cout 
    plt.figure()
    plt.hist(Lossstored[0,:],100)
    plt.title('Histogram of the loss values')
    print('extrema of the loss',np.min(Lossstored),np.max(Lossstored))

def plotDecroissanceFct():
    """
    Le but de cette fonction est d afficher la decroissance de la fonction de cout 
    au cours du temps
    """
    export_dir = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withScore_ValuesLoss.pkl'
    export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withoutScore_ValuesLoss.pkl'


    list_export_dir = [export_dir,export_dir_withoutscore]
    list_plots_elt = ['with score','without score']
    iterations = np.arange(1200)
    for elt,export_dir in zip(list_plots_elt,list_export_dir):
        with open(export_dir, 'rb') as f:
            Dict = pickle.load(f)
        all_loss_value = Dict['all_loss_value']
        print(all_loss_value.shape)
        
        plt.figure()
        for i in range(12):
            plt.plot(iterations,all_loss_value[0,:,i])
        title = 'Loss value '+ elt +' in the loss function for the PeopleArt Dataset'
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.show()
        
    export_dir = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withScore_ValuesLoss_CsearchCVmodeTrue.pkl'
    export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withoutScore_ValuesLoss_CsearchCVmodeTrue.pkl'

    list_export_dir = [export_dir,export_dir_withoutscore]
    list_plots_elt = ['with score','without score']
    
    list_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8',	'#f58231','#911eb4','#46f0f0','#f032e6',	
               '#d2f53c','#fabebe',	'#008080','#e6beff','#aa6e28','#fffac8','#800000',
               '#aaffc3','#808000','#ffd8b1','#000080','#808080','#FFFFFF','#000000']	
    i_color = 0
    iterations = np.arange(600)
    for elt,export_dir in zip(list_plots_elt,list_export_dir):
        with open(export_dir, 'rb') as f:
            Dict = pickle.load(f)
        all_loss_value = Dict['all_loss_value']
        print(all_loss_value.shape)
        plt.figure()
        for i in range(all_loss_value.shape[2]):
            plt.plot(iterations,all_loss_value[0,:,i],color=list_colors[i_color])
            i_color = (i_color + 1) % 12
        title = 'Loss value '+ elt +' in the loss function for the PeopleArt Dataset with Csearch'
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.show()   
 
def plot_AddValues(matrix,titre,loss_value=None):
    
    s1,s2 = matrix.shape
    fig, ax = plt.subplots(figsize=(7,7))
    
    if not(loss_value is None):
        # We want to show all ticks...
        ax.set_xticks(np.arange(s1))
        # ... and label them with the respective list entries
        values = []
        for l in loss_value:
            values += ['{0:.2f}'.format(l)]
        ax.set_xticklabels(values)

    im = ax.imshow(matrix)

    plt.title(titre)

    for i in range(s1):
        for j in range(s2):
            textij = ' {0:.1f}'.format(matrix[i, j])
            text = ax.text(j, i, textij,
                       ha="center", va="center", color="w")
    
    ax.set_title(titre)
    fig.tight_layout()
    plt.show()

def Estimate_Contrib_Vectors(MaxOfMax=False,MaxMMeanOfMax=False,withscore=False):
    if MaxOfMax:
        export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse____MaxOfMax.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore_MaxOfMax.pkl'
    elif MaxMMeanOfMax:
        export_dir_withoutscore ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse____MaxMMeanOfMax.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore_MaxMMeanOfMax.pkl'
    else:
        export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse__.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore.pkl'
    
    numberofW_to_keep = 12
    num_features = 2048
    num_classes = 7
    
    if withscore:
        name_dictW = export_dir
    else:
        name_dictW = export_dir_withoutscore
    
    all_possible_cases = itertools.product([-1,1],repeat=numberofW_to_keep)
    
    # TODO faire en sorte que l'on puisse afficher les contributions par vecteurs.... 
    
    with open(name_dictW, 'rb') as f:
        Dict = pickle.load(f)
        Wstored = Dict['Wstored']
        Bstored =  Dict['Bstored']
        Lossstored = Dict['Lossstored']
        np_pos_value =  Dict['np_pos_value'] 
        np_neg_value =  Dict['np_neg_value']
        
        dataset_nm = 'IconArt_v1'
        item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC =\
            get_database(dataset_nm)
        
        k_per_bag = 300
        metamodel = 'FasterRCNN'
        demonet='res152_COCO'
        dict_name_file = getDictFeaturesFasterRCNN(dataset_nm,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    
        
        # Evaluation contribution on train set
        dict_vectors = {}
        dict_bias = {}
        dict_results = {}
        right_classif = {}
        for j in range(num_classes):
            dict_vectors[j] = Wstored[j::num_classes,:,:]
            dict_bias[j] = Bstored[j::num_classes]
            dict_results[j] = np.zeros(shape=(len(all_possible_cases),))
            right_classif[j] = np.zeros(shape=(numberofW_to_keep+1,))
        
        name_file = dict_name_file['trainval']
        if metamodel=='EdgeBoxes':
            dim_rois = 4
        else:
            dim_rois = 5
        next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,dim_rois = dim_rois)
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 16
        config.inter_op_parallelism_threads = 16
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        i = 0
        num_features = 2048
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                # fc7s size batchsize,300,2048
                for j in range(num_classes):
                    # Avec normalement W de taille [12,1,2048]
                    product = np.einsum('ijk,bck->ibc',fc7s,dict_vectors[j])[:,:,0] +dict_bias[j]
                    max_product_along_boxes = np.max(product,axis=-1)
                    max_product_along_im = np.int((np.sign(np.max(max_product_along_boxes,axis=-1)) + 1)/2)
                    sgn = np.sign(max_product_along_boxes,dtype=np.int,casting='unsafe')
                    label_predicted =np.int((sgn + 1)/2)
                    for k in range(len(sgn)):
                        index_in_poss_case = np.where(all_possible_cases==sgn[k])[0]
                        dict_results[j][index_in_poss_case] += 1
                    for wi in range(numberofW_to_keep):
                        right_classif[j][wi] += len(np.where(label_predicted[:,wi]==labels[j,:])[0])
                    # Dans le cas collaboratif 
                    right_classif[j][numberofW_to_keep] += len(np.where(max_product_along_im==labels[j,:])[0])
                    
            except tf.errors.OutOfRangeError:
                break
    
        sess.close()
                
        # Evaluation test
        name_file = dict_name_file['test']
        next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,dim_rois = dim_rois)
        
        dont_use_07_metric = False
        if dataset_nm=='VOC2007':
            imdb = get_imdb('voc_2007_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            num_images = len(imdb.image_index)
        elif dataset_nm=='watercolor':
            imdb = get_imdb('watercolor_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            num_images = len(imdb.image_index)
        elif dataset_nm=='PeopleArt':
            imdb = get_imdb('PeopleArt_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            num_images = len(imdb.image_index)
        elif dataset_nm=='clipart':
            imdb = get_imdb('clipart_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            num_images = len(imdb.image_index) 
        elif dataset_nm=='IconArt_v1' or dataset_nm=='RMN':
            imdb = get_imdb('IconArt_v1_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        elif 'IconArt_v1' in dataset_nm and not('IconArt_v1' ==dataset_nm):
            imdb = get_imdb('IconArt_v1_test',ext=dataset_nm.split('_')[-1])
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
    #        num_images = len(imdb.image_index) 
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        elif dataset_nm in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
            imdb = get_imdb('WikiTenLabels_test')
            imdb.set_force_dont_use_07_metric(dont_use_07_metric)
            #num_images = len(imdb.image_index) 
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        else:
            num_images =  len(df_label[df_label['set']=='test'][item_name])
        
        dict_all_boxes = {}
        for wi in range(numberofW_to_keep+1):
            dict_all_boxes[wi] = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        
        
        TEST_NMS = 0.3
        thresh = 0.0
        true_label_all_test = []
        predict_label_all_test = []
        name_all_test = []
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 16
        config.inter_op_parallelism_threads = 16
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        i = 0
        num_features = 2048
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                for j in range(num_classes):
                    # Avec normalement W de taille [12,1,2048]
                    product = np.einsum('ijk,bck->ibc',fc7s,dict_vectors[j])[:,:,0] +dict_bias[j]
                    max_product_along_vectors = np.max(product,axis=1)
                dict_scores = {}
                for wi in range(numberofW_to_keep):
                    dict_scores[wi] = product[:,wi,:]
                dict_scores[numberofW_to_keep] = max_product_along_vectors
                
                for wi in range(numberofW_to_keep):
                    score_all = dict_scores[wi]
                    for k in range(len(labels)):
                        name_im = name_imgs[k].decode("utf-8")
                        complet_name = path_to_img + str(name_im) + '.jpg'
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        roi = roiss[k,:]
                        if metamodel=='EdgeBoxes':
                            roi_boxes =  roi / im_scales[0] 
                        else:
                            roi_boxes =  roi[:,1:5] / im_scales[0]
                        
                        for j in range(num_classes):
                            scores = score_all[k,j,:]
                            #print(j,'scores',scores.shape)
                            inds = np.where(scores > thresh)[0]
                            cls_scores = scores[inds]
                            cls_boxes = roi_boxes[inds,:]
                            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                            keep = nms(cls_dets, TEST_NMS)
                            cls_dets = cls_dets[keep, :]
                            dict_all_boxes[wi][j][i] = cls_dets
#                    i += 1
                wi =   numberofW_to_keep
                score_all = dict_scores[wi]
                for k in range(len(labels)):
                    name_im = name_imgs[k].decode("utf-8")
                    complet_name = path_to_img + str(name_im) + '.jpg'
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    roi = roiss[k,:]
                    if metamodel=='EdgeBoxes':
                        roi_boxes =  roi / im_scales[0] 
                    else:
                        roi_boxes =  roi[:,1:5] / im_scales[0]
                    
                    for j in range(num_classes):
                        scores = score_all[k,j,:]
                        #print(j,'scores',scores.shape)
                        inds = np.where(scores > thresh)[0]
                        cls_scores = scores[inds]
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        dict_all_boxes[wi][j][i] = cls_dets
                    i += 1
                for l in range(len(name_imgs)): 
                    if dataset_nm in ['IconArt_v1','VOC2007','watercolor','clipart','WikiTenLabels','PeopleArt','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                        name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                    else:
                        name_all_test += [[name_imgs[l]]]
                        
            except tf.errors.OutOfRangeError:
                break
    
        sess.close()
        
        true_label_all_test = np.concatenate(true_label_all_test)
        predict_label_all_test = np.concatenate(predict_label_all_test,axis=0)
        name_all_test = np.concatenate(name_all_test)
        
#            name_all_test = np.concatenate(name_all_test)
#    
#    AP_per_class = []
#    for j,classe in enumerate(classes):
#            AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
#            AP_per_class += [AP]
#    print('Average Precision classification task :')
#    print(arrayToLatex(AP_per_class,per=True))        
    for wi in range(numberofW_to_keep+1):
        all_boxes =  dict_all_boxes[wi]
        max_per_image = 100
        num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels or IconArt_v1 case
        all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
        number_im = 0
        name_all_test = name_all_test.astype(str)
        for i in range(num_images_detect):
    #        print(i)
            name_img = imdb.image_path_at(i)
            if dataset_nm=='PeopleArt':
                name_img_wt_ext = name_img.split('/')[-2] +'/' +name_img.split('/')[-1]
                name_img_wt_ext_tab =name_img_wt_ext.split('.')
                name_img_wt_ext = '.'.join(name_img_wt_ext_tab[0:-1])
            else:
                name_img_wt_ext = name_img.split('/')[-1]
                name_img_wt_ext =name_img_wt_ext.split('.')[0]
            name_img_ind = np.where(np.array(name_all_test)==name_img_wt_ext)[0]
            #print(name_img_ind)
            if len(name_img_ind)==0:
                print('len(name_img_ind), images not found in the all_boxes')
                print(name_img_wt_ext)
                raise(Exception)
            else:
                number_im += 1 
    #        print(name_img_ind[0])
            for j in range(1, imdb.num_classes):
                j_minus_1 = j-1
                if len(all_boxes[j_minus_1][name_img_ind[0]]) >0:
                    all_boxes_order[j][i]  = all_boxes[j_minus_1][name_img_ind[0]]
            if max_per_image > 0 and len(all_boxes_order[j][i]) >0: 
                image_scores = np.hstack([all_boxes_order[j][i][:, -1]
                            for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes_order[j][i][:, -1] >= image_thresh)[0]
                        all_boxes_order[j][i] = all_boxes_order[j][i][keep, :]
        assert (number_im==num_images_detect) # To check that we have the all the images in the detection prediction
        det_file = os.path.join(path_data, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
        output_dir = path_data +'tmp/' + dataset_nm+'_mAP.txt'
        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
        apsAt05 = aps
        if wi < numberofW_to_keep:
            print('For vector number :',wi)
        else:
            print('For the max Of max')
        print("Detection score (thres = 0.5): ",dataset_nm)
        print(arrayToLatex(aps,per=True))
        ovthresh_tab = [0.3,0.1,0.]
        for ovthresh in ovthresh_tab:
            aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
            if ovthresh == 0.1:
                apsAt01 = aps
            print("Detection score with thres at ",ovthresh,'with ',model)
            print(arrayToLatex(aps,per=True))
        
def CovarianceOfTheVectors(MaxOfMax=False,MaxMMeanOfMax=False,withscore=False):
    
    # Need to create those files if they don't exist look at RunVarStudyAll
    path_to_fig = 'fig/Cov/'
    if MaxOfMax:
        path_to_fig += 'MaxOfMax'
        export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse____MaxOfMax.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore_MaxOfMax.pkl'
    elif MaxMMeanOfMax:
        path_to_fig += 'MaxMMeanOfMax'
        export_dir_withoutscore ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse____MaxMMeanOfMax.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore_MaxMMeanOfMax.pkl'
    else:
        path_to_fig += 'MI_max'
        export_dir_withoutscore = '/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse__.pkl'
        export_dir ='/media/gonthier/HDD/output_exp/ClassifPaintings/VarStudy/IconArt_v1_Wvectors_C_SearchingFalse___WithScore.pkl'
    
    numberofW_to_keep = 12
#    number_of_reboots = 100
    num_features = 2048
    num_classes = 7
    
    if withscore:
        path_to_fig += 'S/'
        name_dictW = export_dir
    else:
        name_dictW = export_dir_withoutscore
        path_to_fig += '/'

    pathlib.Path(path_to_fig).mkdir(parents=True, exist_ok=True) 
        
    with open(name_dictW, 'rb') as f:
        Dict = pickle.load(f)
        Wstored = Dict['Wstored']
        Bstored =  Dict['Bstored']
        Lossstored = Dict['Lossstored']
        np_pos_value =  Dict['np_pos_value'] 
        np_neg_value =  Dict['np_neg_value']
        
        # First we will compute the covariance matrices of 12 of the W vectors without selection
        l = 0
        
        for j in range(num_classes):
            Wstored_extract = Wstored[j::num_classes,:,:]
            Wtmp = Wstored_extract[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
            Wtmp = Wtmp.reshape((numberofW_to_keep,num_features))
            cov_matrix = np.cov(Wtmp)
            corrcoef_matrix = np.corrcoef(Wtmp)
            Lossstoredextract = Lossstored[j,:]
            Lossstoredextract = Lossstoredextract[l*numberofW_to_keep:(l+1)*numberofW_to_keep]
            loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
            
            titre= 'Cov without selection classe : '+str(j)
            plot_AddValues(cov_matrix,titre,loss_value)
            namefig =  path_to_fig +'Cov'+str(j)+'.png'
            plt.savefig(namefig)
            titre= 'CorrCoeff without selection classe : '+str(j)
            plot_AddValues(corrcoef_matrix,titre,loss_value)
            namefig =  path_to_fig +'Corr'+str(j)+'.png'
            plt.savefig(namefig)
            
            
##            print(cov_matrix)
##            plt.imshow(cov_matrix)
##            
##            plt.title(titre)
##            plt.show()
##            plt.imshow(corrcoef_matrix)
##            titre= 'CorrCoeff without selection classe : '+str(j)
##            plt.title(titre)
##            plt.show()
#        
#        # We will add the biases to the vectors
#        
#        for l in range(number_of_reboots):
#            Wstored_extract = Wstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep,:]
#            W_tmp = np.reshape(Wstored_extract,(-1,num_features),order='F')
#            b_tmp =np.reshape( Bstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep],(-1,1,1),order='F')
#            Lossstoredextract = Lossstored[:,l*numberofW_to_keep:(l+1)*numberofW_to_keep]
#            loss_value = np.reshape(Lossstoredextract,(-1,),order='F')
        
        
    


if __name__ == '__main__':
    Etude_Wvectors()
