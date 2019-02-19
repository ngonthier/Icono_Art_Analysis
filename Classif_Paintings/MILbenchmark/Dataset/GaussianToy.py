# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:04:36 2019

@author: gonthier
"""

import numpy as np

npt=np.float32

def createGaussianToySets(WR=0.01,n=20,k=300,np1=20,np2=200,overlap=True):
    """
    
    La premiere feature est décalé pour obtenir une classe différente
    
    WR : Witness rate = proportion of positive examples in the positive bags    
    # Variables communes 
    n=20  # Number of featues
    k=300 # Number of element in the bag 
    # la classe p1
    np1=20 # Number of positive bag
    np2=200 # Number of negative bag
    overlap=True 
    """
    
    np.random.seed(19680801)
    
    number_of_positive = int(np.floor(k*WR))
    if number_of_positive < 1:
        number_of_positive = 1
        print('Must have at least one element per bag !')
    
    def gen_vect_p1():
        return npt(np.random.randn(n))
    
    def gen_vect_p2():
        g0=np.random.randn(n)
        if overlap:
            g0[0:1]+=4
        else:
            g0[0:1]+=8
        return npt(g0)
    
    def gen_paquet_p2():
        tab=np.zeros((k,n),dtype=npt)
        for i in range(k):
            tab[i]=gen_vect_p2()
        return tab
    
    def gen_paquet_p1(N=np1):
        tab=np.zeros((k,n),dtype=npt)
        instances_labels = -np.ones(shape=(k,),dtype=npt)
        positive_instance_index = np.random.choice(k, number_of_positive)
        for i in range(k):
            if (i in positive_instance_index):
                tab[i]=gen_vect_p1()
                instances_labels[i] = 1.
            else:
                tab[i]=gen_vect_p2()
        return tab,instances_labels
    
    def gen_data_p1(N=np1):
        tab=np.zeros((N,k,n),dtype=npt)
        choisi=np.zeros((N,),dtype=np.int32)
        for i in range(N):
            tab[i],choisi[i]=gen_paquet_p1()
        return tab,choisi
        
    def gen_data_p2(N=np2):
        tab=np.zeros((N,k,n),dtype=npt)
        for i in range(N):
            tab[i]=gen_paquet_p2()
        return tab

#    data_p1,choisi=gen_data_p1()
#    data_p2=gen_data_p2()
    
    bags = []
    labels_bags = []
    labels_instance = []
    
    # Generation of the negative bags
    for i in range(np2):
        bags += [gen_paquet_p2()]
        labels_bags += [np.array(-1.)]
        labels_instance += [-1.*np.ones(shape=(k,))]
    
    # Generation of the positive bags
    for j in range(np1):
        tab,instances_labels = gen_paquet_p1()
        bags += [tab]
        labels_bags += [np.array(1.)]
        labels_instance += [instances_labels]
    
#    print(data_p1.shape) #(200, 30, 2048)
#    print(data_p2.shape) #(4000, 30, 2048)
    
    list_names = ['OneGaussianToy']
    
    Dataset = list_names,bags,[labels_bags],[labels_instance]
            
    return(Dataset)