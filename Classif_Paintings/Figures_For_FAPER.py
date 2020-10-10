# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:37:37 2020

Do the figure for the FAPER paper : CKA and l2 norm distance paper

@author: gonthier
"""

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Dist_CNNs import get_linearCKA_bw_nets,comp_cka_for_paper

CB_color_cycle = ['#377eb8', '#ff7f00','#984ea3', '#4daf4a','#A2C8EC','#e41a1c',
                  '#f781bf', '#a65628', '#dede00','#FFBC79','#999999']

title_corr = {'pretrained': 'pretrained on ImageNet',
              'RASTA_small01_modif' : 'FT on RASTA (Mode A training 1)',
              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200' : 'The end from scratch',
              'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG' : 'From scratch',
              'RASTA_small01_modif1' : 'FT on RASTA (Mode A training 2)',
              'RASTA_small001_modif' : 'FT on RASTA (Mode B training 1)',
              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init': 'Random Init',
              'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init': 'Random Init'
                }

def cka_fct_layers_plot():
    
    matplotlib.use('Agg')
    plt.switch_backend('agg')
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")
    
    dataset = 'RASTA'
    
    l_pairs,l_dico = comp_cka_for_paper(dataset=dataset) # All the data 
    # Now need to find the wanted pairs
    
    all_pairs = [['pretrained','RASTA_small01_modif'],
                ['pretrained','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200'],
                ['pretrained','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG'],
                ['RASTA_small01_modif','RASTA_small01_modif1'],
                ['RASTA_small01_modif','RASTA_small001_modif'],
                ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init'],
                ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init']]
    
    
    
    list_layers=['conv2d0','conv2d1',
                 'conv2d2','mixed3a',
                 'mixed3b','mixed4a',
                 'mixed4b','mixed4c',
                 'mixed4d','mixed4e',
                 'mixed5a','mixed5b']
    
    #set tick marks for grid
    labels_ = []
    for l in list_layers:
        labels_+= [l.replace('_','\_')]
    
    
    l_rasta_dico = []
#    l_rasta_pairs = []
#    
#    # On RASTA first 
#    list_models = ['pretrained',
#                   'RASTA_small01_modif',
#                   'RASTA_small001_modif',
#                   'RASTA_big001_modif',
#                   'RASTA_small001_modif_deepSupervision',
#                   'RASTA_big001_modif_deepSupervision',
#                   'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200',
#                   'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG',
#                   'RASTA_big0001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
#    
#    list_with_suffix = ['RASTA_small01_modif',
#                   'RASTA_small001_modif',
#                   'RASTA_big001_modif',
#                   'RASTA_small001_modif_deepSupervision',
#                   'RASTA_big001_modif_deepSupervision']
#    
#    all_pairs = itertools.combinations(list_models, r=2)
         
    list_modified_in_unfreeze50 = ['mixed4a',
              'mixed4b','mixed4c',
              'mixed4d','mixed4e',
              'mixed5a','mixed5b']
    
    list_net = []
    plt.figure()
    ax = plt.subplot(111)
    for p,pair in enumerate(all_pairs):
        netA,netB = pair
        # TODO here !!!
        dico = get_linearCKA_bw_nets(dataset=dataset,netA=netA,netB=netB,
                                                     list_layers=list_layers)

        list_cka = []
        list_index_cka = []
        for i,layer in enumerate(list_layers):
            cka_l = dico[layer]
            
            #if dataset == 'RASTA' and ('RandForUnfreezed' in  netA or 'RandForUnfreezed' in  netB):
            # cas du randinit
            if (('RandForUnfreezed' in netA) and (dataset == 'RASTA' or netB=='pretrained')) or (('RandForUnfreezed' in netB) and (dataset == 'RASTA' or netA=='pretrained')):
                if not('unfreeze50' in  netA or 'unfreeze50' in  netB):
                   raise(NotImplementedError)
                if layer in list_modified_in_unfreeze50:
                   list_cka += [cka_l]
                   list_index_cka += [i]
            else:
                list_cka += [cka_l]
                list_index_cka += [i]
        
        label_p = title_corr[netA] +' vs '+title_corr[netB]
        plt.plot(list_index_cka, list_cka, marker='o',color=CB_color_cycle[p], label=label_p)
        
    plt.grid(True)
    plt.ylim((0.,1.))
        
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=4)
    
    ax.set_xticks(np.arange(len(list_layers)))
    ax.set_xticklabels(labels_, rotation=45)
    
    plt.show()

if __name__=='__main__':
    cka_fct_layers_plot()    