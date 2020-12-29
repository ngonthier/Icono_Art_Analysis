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
import platform
import os
import tikzplotlib

from Dist_CNNs import get_linearCKA_bw_nets,comp_cka_for_paper,get_l2norm_bw_nets,\
                        comp_l2_for_paper

CB_color_cycle = ['#377eb8', '#4daf4a','#A2C8EC', '#ff7f00','#984ea3','#e41a1c',
                  '#f781bf', '#a65628', '#dede00','#FFBC79','#999999']
list_markers = ['o','s','d','*','v','^','<','>','h','H','X','1','2','3','4','8','p','d','$f$','P']


title_corr = {'pretrained': 'pretrained on ImageNet',
              'RASTA_small01_modif' : 'FT on RASTA (Mode A training 1)',
              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200' : 'The end from scratch',
              'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG' : 'From scratch',
              'RASTA_small01_modif1' : 'FT on RASTA (Mode A training 2)',
              'RASTA_small001_modif' : 'FT on RASTA (Mode B training 1)',
              'RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init': 'Random Init',
              'RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init': 'Random Init'
                }

def cka_fct_layers_plot(forPhDmanuscript=False,side_legend=True,output_img='png'):
    
    #matplotlib.use('Agg')
    #plt.switch_backend('agg')
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")
    
    dataset = 'RASTA'
    
    l_pairs,l_dico = comp_cka_for_paper(dataset=dataset) # All the data 
    # Now need to find the wanted pairs
    
    all_pairs = [['pretrained','RASTA_small01_modif'],
                ['RASTA_small01_modif','RASTA_small01_modif1'],
                ['RASTA_small01_modif','RASTA_small001_modif'],
                ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init'],
                ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init'],
                ['pretrained','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200'],
                ['pretrained','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG']
                ]
    case_str = "ForPaper_"
    if forPhDmanuscript:
        case_str = "ForPhDmodels_"
        all_pairs = [['pretrained','RASTA_small01_modif'],
                ['RASTA_small01_modif','RASTA_small01_modif1'],
                ['RASTA_small01_modif','RASTA_small001_modif'],
                ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init'],
                ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init'],
                ['pretrained','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200'],
                ['pretrained','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG'],
                ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_small01_modif'],
                ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG','RASTA_small01_modif']
                ]
    
    
    
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
    
    if output_img=='png':
        markersize=10
        linewidth=3
    elif output_img=='tikz':
       markersize=5
       linewidth=2
        
    list_net = []
    plt.figure()
    ax = plt.subplot(111)
    for p,pair in enumerate(all_pairs):
        netA,netB = pair
        dico = None
        
        for local_pair, local_dico in zip(l_pairs,l_dico):
            netC,netD = local_pair
            if (netA==netC and netB==netD) or (netA==netD and netB==netC):
                dico = local_dico
                continue
        if dico is None:
            initA = False
            initB = False
            suffixA =''
            suffixB = ''
            netA_l = netA
            netB_l = netB
            if '_init' in netA: 
                initA=True
                netA_l = netA.replace('_init','')
            if '_init' in netB: 
                initB=True
                netB_l = netB.replace('_init','')
            if netA[-1] == '1': 
                suffixA='1'
                netA_l = netA[0:-1]
            if netB[-1] == '1': 
                suffixB='1'
                netB_l = netB[0:-1]
            dico = get_linearCKA_bw_nets(dataset=dataset,netA=netA_l,netB=netB_l,
                                         initB=initB,initA=initA,
                                         suffixA=suffixA,suffixB=suffixB,
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
        
        if forPhDmanuscript:
            if (netA=='RASTA_small01_modif' and netB=='RASTA_small01_modif1') or (netA=='RASTA_small01_modif'and netB=='RASTA_small001_modif'):
                label_p = title_corr[netA] +r' vs \\ '+title_corr[netB]
            else:
                label_p = title_corr[netA] +' vs '+title_corr[netB]
        else:
            label_p = title_corr[netA] +' vs '+title_corr[netB]
        plt.plot(list_index_cka, list_cka,linestyle='--', marker=list_markers[p],
                 color=CB_color_cycle[p], label=label_p, markersize=markersize,
                 linewidth=linewidth)
        print(netA,netB,list_cka)
        
    plt.grid(True)
    plt.ylim((0.,1.01))
    plt.yticks(fontsize=18)
        
    ax.set_xlabel("Layer",fontsize=20)
    ax.set_ylabel("CKA (linear)",fontsize=20)
    
    ncol= 2
    loc='upper center'
    bbox_to_anchor=(0.5, 1.2)
    if forPhDmanuscript:
#        ncol = 3
#        bbox_to_anchor=(0.5, 1.15)
        ncol= 1
        bbox_to_anchor=(1.01, 0.5)
        loc='center left'
    if side_legend:
        ncol= 1
        bbox_to_anchor=(1.01, 0.5)
        loc='center left'
    ax.legend(loc=loc,  bbox_to_anchor=bbox_to_anchor, # under  bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=ncol,fontsize=16)
#    ax.legend(loc='upper center',  bbox_to_anchor=bbox_to_anchor, # under  bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=False, ncol=ncol,fontsize=18)
    
    
    ax.set_xticks(np.arange(len(list_layers)))
    ax.set_xticklabels(labels_, rotation=45,fontsize=18)
        
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel')
    # For images
    
    if output_img=='png':
        if side_legend:
            plt.tight_layout()
        plt.tight_layout()
        path_fig = os.path.join(output_path,case_str+'_CKA_per_layer.png')
        plt.savefig(path_fig,bbox_inches='tight')
        plt.show()
    if output_img=='tikz':
        path_fig = os.path.join(output_path,case_str+'_CKA_per_layer.tex')
        tikzplotlib.save(path_fig)
    
    
    
    
    
def l2norm_fct_layers_plot(side_legend=False,output_img='png'):
    
    #matplotlib.use('Agg')
    #plt.switch_backend('agg')
    matplotlib.rcParams['text.usetex'] = True
    sns.set()
    sns.set_style("whitegrid")
    
    dataset = 'RASTA'
    
    l_pairs,l_dico = comp_l2_for_paper(dataset=dataset) # All the data 
    # Now need to find the wanted pairs
    
    list_layers,dict_layers_diff,l2_norm_total = l_dico[0]
    
    try:
        list_layers.remove('head0_bottleneck_pre_relu')
    except ValueError:
        pass
    try:
        list_layers.remove('head1_bottleneck_pre_relu')
    except ValueError:
        pass
    
    # Ces 2 cas la n'ont pas de raison d'etre car les filtres peuvent etre n'importe ou :
    ['pretrained','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200'],
    ['pretrained','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG'],
    
    all_pairs = [['pretrained','RASTA_small01_modif'],
                ['RASTA_small01_modif','RASTA_small01_modif1'],
                ['RASTA_small01_modif','RASTA_small001_modif'],
                ['RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200_init'],
                ['RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG','RASTA_big001_modif_RandInit_randomCrop_deepSupervision_ep200_LRschedG_init']]

    case_str = "ForPaper_"
    
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
         
    list_modified_in_unfreeze50 = ['mixed4a_3x3_bottleneck_pre_relu',
                                 'mixed4a_5x5_bottleneck_pre_relu',
                                 'mixed4a_1x1_pre_relu',
                                 'mixed4a_3x3_pre_relu',
                                 'mixed4a_5x5_pre_relu',
                                 'mixed4a_pool_reduce_pre_relu',
                                 'mixed4b_3x3_bottleneck_pre_relu',
                                 'mixed4b_5x5_bottleneck_pre_relu',
                                 'mixed4b_1x1_pre_relu',
                                 'mixed4b_3x3_pre_relu',
                                 'mixed4b_5x5_pre_relu',
                                 'mixed4b_pool_reduce_pre_relu',
                                 'mixed4c_3x3_bottleneck_pre_relu',
                                 'mixed4c_5x5_bottleneck_pre_relu',
                                 'mixed4c_1x1_pre_relu',
                                 'mixed4c_3x3_pre_relu',
                                 'mixed4c_5x5_pre_relu',
                                 'mixed4c_pool_reduce_pre_relu',
                                 'mixed4d_3x3_bottleneck_pre_relu',
                                 'mixed4d_5x5_bottleneck_pre_relu',
                                 'mixed4d_1x1_pre_relu',
                                 'mixed4d_3x3_pre_relu',
                                 'mixed4d_5x5_pre_relu',
                                 'mixed4d_pool_reduce_pre_relu',
                                 'mixed4e_3x3_bottleneck_pre_relu',
                                 'mixed4e_5x5_bottleneck_pre_relu',
                                 'mixed4e_1x1_pre_relu',
                                 'mixed4e_3x3_pre_relu',
                                 'mixed4e_5x5_pre_relu',
                                 'mixed4e_pool_reduce_pre_relu',
                                 'mixed5a_3x3_bottleneck_pre_relu',
                                 'mixed5a_5x5_bottleneck_pre_relu',
                                 'mixed5a_1x1_pre_relu',
                                 'mixed5a_3x3_pre_relu',
                                 'mixed5a_5x5_pre_relu',
                                 'mixed5a_pool_reduce_pre_relu',
                                 'mixed5b_3x3_bottleneck_pre_relu',
                                 'mixed5b_5x5_bottleneck_pre_relu',
                                 'mixed5b_1x1_pre_relu',
                                 'mixed5b_3x3_pre_relu',
                                 'mixed5b_5x5_pre_relu',
                                 'mixed5b_pool_reduce_pre_relu']
    
    list_net = []
    plt.figure()
    ax = plt.subplot(111)
    all_list_l2norm = []
    for p,pair in enumerate(all_pairs):
        netA,netB = pair
        dico = None
        
        for local_pair, local_dico in zip(l_pairs,l_dico):
            netC,netD = local_pair
            if (netA==netC and netB==netD) or (netA==netD and netB==netC):
                _,dico,_ = local_dico
                continue
        if dico is None:
            initA = False
            initB = False
            suffixA =''
            suffixB = ''
            netA_l = netA
            netB_l = netB
            if '_init' in netA: 
                initA=True
                netA_l = netA.replace('_init','')
            if '_init' in netB: 
                initB=True
                netB_l = netB.replace('_init','')
            if netA[-1] == '1': 
                suffixA='1'
                netA_l = netA[0:-1]
            if netB[-1] == '1': 
                suffixB='1'
                netB_l = netB[0:-1]
            list_name_layers_A,dico,l2_norm_total = get_l2norm_bw_nets(dataset=dataset,netA=netA_l,netB=netB_l,
                                         initB=initB,initA=initA,
                                         suffixA=suffixA,suffixB=suffixB,
                                         list_layers=list_layers)

        list_l2norm = []
        list_index_l2 = []
        for i,layer in enumerate(list_layers):
            l2_l = dico[layer]
            
            #if dataset == 'RASTA' and ('RandForUnfreezed' in  netA or 'RandForUnfreezed' in  netB):
            # cas du randinit
            if (('RandForUnfreezed' in netA) and (dataset == 'RASTA' or netB=='pretrained')) or (('RandForUnfreezed' in netB) and (dataset == 'RASTA' or netA=='pretrained')):
                if not('unfreeze50' in  netA or 'unfreeze50' in  netB):
                   raise(NotImplementedError)
                if layer in list_modified_in_unfreeze50:
                   list_l2norm += [l2_l]
                   list_index_l2 += [i]
            else:
                list_l2norm += [l2_l]
                list_index_l2 += [i]
        
        label_p = title_corr[netA] +' vs '+title_corr[netB]
        all_list_l2norm += list_l2norm
        plt.plot(list_index_l2, list_l2norm,linestyle='--', marker=list_markers[p],color=CB_color_cycle[p], label=label_p)
        #marker='o' pour un rond
        print(netA,netB,list_l2norm)
        
    plt.grid(True)
    plt.ylim((0.,np.max(all_list_l2norm)+0.1))
        
    ax.set_xlabel("Convolutional layer")
    ax.set_ylabel("$\ell_{2}$ norm")
    
    ax.legend(loc='upper center',  bbox_to_anchor=(0.5, 1.1), # under  bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=3)
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel')
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel')

    if output_img=='png':
        if side_legend:
            plt.tight_layout()
        plt.tight_layout()
        path_fig = os.path.join(output_path,case_str+'_l2norm_per_layer.png')
        plt.savefig(path_fig,bbox_inches='tight')
        plt.show()
    if output_img=='tikz':
        path_fig = os.path.join(output_path,case_str+'_l2norm_per_layer.tex')
        tikzplotlib.save(path_fig)
    
    #ax.set_xticks(np.arange(len(list_layers)))
    #ax.set_xticklabels(labels_, rotation=45)
    
    #plt.show()
    #input("wait")

if __name__=='__main__':
    #cka_fct_layers_plot(forPhDmanuscript=False,side_legend=True,output_img='tikz')
    cka_fct_layers_plot(forPhDmanuscript=True,side_legend=True,output_img='tikz')
    #l2norm_fct_layers_plot()