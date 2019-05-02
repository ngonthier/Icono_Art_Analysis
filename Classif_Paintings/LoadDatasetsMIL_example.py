# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:43:42 2019

@author: gonthier
"""

from MILbenchmark.utils import getDataset,normalizeDataSetFull

dataset = 'Birds' # 'Newgroups or SIVAL


Dataset=getDataset(dataset)
list_names,bags,labels_bags,labels_instance = Dataset
          
for c_i,c in enumerate(list_names):
    # Loop on the different class, we will consider each group one after the other
    print("Start evaluation for class :",c)
    labels_bags_c = labels_bags[c_i]
    labels_instance_c = labels_instance[c_i]
    if dataset in ['Newsgroups','SIVAL']:
        bags_c = bags[c_i]
    else:
        bags_c = bags

    # Pour normaliser les sacs
    # bags_c = normalizeDataSetFull(bags_c,'std')
        
    D = bags_c,labels_bags_c,labels_instance_c