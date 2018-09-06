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
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

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
    
    name_dict = '/media/HDD/output_exp/ClassifPaintings/MI_max_StoredW/1533740133.283923.pkl'
    name_dict = '/media/HDD/output_exp/ClassifPaintings/MI_max_StoredW/MImax_PeopletArt_10800W_withScore_classicLoss.pkl'
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
    export_dir = '/media/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withScore_ValuesLoss.pkl'
#    export_dir = '/media/HDD/output_exp/ClassifPaintings/MI_max_StoredW/PeopleArt_withoutScore_ValuesLoss.pkl'
    with open(export_dir, 'rb') as f:
        Dict = pickle.load(f)
    all_loss_value = Dict['all_loss_value']
    print(all_loss_value.shape)
    loss_value = np.reshape(all_loss_value,(-1,),order='F')
    print(loss_value.shape)

if __name__ == '__main__':
    Etude_Wvectors()
