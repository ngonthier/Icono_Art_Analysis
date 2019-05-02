# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:04:36 2019

@author: gonthier
"""

import numpy as np
from sklearn.datasets.samples_generator import make_blobs

npt=np.float32

def createMILblob(WR=0.01,n=20,k=300,np1=20,np2=200,Between01=False,
                  ):
    """
    @param WR : Witness rate = proportion of positive examples in the positive bags    
    @param n=20  # Number of featues
    @param k=300 # Number of element in the bag 
    @param np1=20 # Number of positive bag
    @param np2=200 # Number of negative bag
    @param : Between01 : if True the label of the class are 0-1 and not -1 and +1
    """
    np_pos = np1
    np_neg = np2
    number_of_positive = int(np.floor(k*WR))
    if number_of_positive < 1:
        number_of_positive = 1
        print('Must have at least one element per bag !')
        
    n_samples = (np_pos + np_neg)*k*2
    X,y=make_blobs(n_samples=n_samples,centers=2,n_features=n)
    Xneg = X[np.where(y==0)[0],:]
    Xpos = X[np.where(y==1)[0],:]
    Xlist = []
    labels_bags = []
    labels_instance = []
    labels_bags = []
    for i in range(np_neg): # Dot the negative bags
        Xtmp = Xneg[i*k:(i+1)*k,:]
        Xlist += [Xtmp]
        if Between01:
            labels_bags += [np.array(0.,dtype=npt)]
            labels_instance += [np.zeros(shape=(k,),dtype=npt)]
        else:
            labels_bags += [np.array(-1.,dtype=npt)]
            labels_instance += [-1.*np.ones(shape=(k,),dtype=npt)]

    indexneg = np_neg*k+1
    indexpos = 0
    for j in range(np_pos):
        # For one bag
        tab=np.zeros((k,n),dtype=npt)
        labels_bags += [np.array(1.,dtype=npt)]
        if Between01:
            instances_labelslocal = -np.zeros(shape=(k,),dtype=npt)
        else:
            instances_labelslocal = -np.ones(shape=(k,),dtype=npt)
        positive_instance_index = np.random.choice(k, number_of_positive)
        for i in range(k):
            if (i in positive_instance_index):
                tab[i]=Xpos[indexpos:indexpos+1,:]
                indexpos+=1
                instances_labelslocal[i] = 1.
            else:
                tab[i]=Xneg[indexneg:indexneg+1,:]
                indexneg += 1
        labels_instance += [instances_labelslocal]
        Xlist += [tab]

    Dataset = ['blobs'],Xlist,[labels_bags],[labels_instance]
    return(Dataset)

def createGaussianToySets(WR=0.01,n=20,k=300,np1=20,np2=200,overlap=False,
                          Between01=False,specificCase='',scaleWell=True):
    """
    
    La premiere feature est décalé pour obtenir une classe différente
    
    @param WR : Witness rate = proportion of positive examples in the positive bags    
    # Variables communes 
    @param n=20  # Number of featues
    @param k=300 # Number of element in the bag 
    # la classe p1
    @param np1=20 # Number of positive bag
    @param np2=200 # Number of negative bag
    @param overlap=False  : overlapping between the point clouds
    @param : Between01 : if True the label of the class are 0-1 and not -1 and +1
    @param : specificCase : we proposed different case of toy points clouds
        - 2clouds : 2 clouds distincts points of clouds as positives examples
        - 2cloudsOpposite : 2 points clouds positive at the opposite from the negatives
        - 
    """
    
    list_specificCase = ['',None,'2clouds','2cloudsOpposite']
    if not(specificCase in list_specificCase):
        print(specificCase,'is unknown')
        raise(NotImplementedError)
    
    np.random.seed(19680801)
    
    number_of_positive = int(np.floor(k*WR))
    if number_of_positive < 1:
        number_of_positive = 1
        print('Must have at least one element per bag !')
    
    if overlap:
        shift_negative_instances = 4
    else:
        shift_negative_instances = 8
        if scaleWell:
            shift_negative_instances *= np.sqrt(n-1)
    
    def gen_vect_p1():
        """
        Function that generate the features of the positives elements
        """
        features = npt(np.random.randn(n))
#        if specificCase=='' or specificCase is None:
#             # Standard case
        if specificCase=='2clouds':
            features[1:2] += np.random.choice([-4,4], 1)[0]
        if specificCase=='2cloudsOpposite':
            features[0:1] += np.random.choice([0,shift_negative_instances*2], 1)[0]
            
        return features
    
    def gen_vect_p2():
        g0=np.random.randn(n)
        g0[0:1]+=shift_negative_instances
        return npt(g0)
    
    def gen_paquet_p2():
        tab=np.zeros((k,n),dtype=npt)
        for i in range(k):
            tab[i]=gen_vect_p2()
        return tab
    
    def gen_paquet_p1(N=np1):
        """
        Fct that return the positives bags !
        """
        tab=np.zeros((k,n),dtype=npt)
        if Between01:
            instances_labels = -np.zeros(shape=(k,),dtype=npt)
        else:
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
        
        if Between01:
            labels_bags += [np.array(0.,dtype=npt)]
            labels_instance += [np.zeros(shape=(k,),dtype=npt)]
        else:
            labels_bags += [np.array(-1.,dtype=npt)]
            labels_instance += [-1.*np.ones(shape=(k,),dtype=npt)]
    
    # Generation of the positive bags
    for j in range(np1):
        tab,instances_labels = gen_paquet_p1()
        bags += [tab]
        labels_bags += [np.array(1.,dtype=npt)]
        labels_instance += [instances_labels]
    
#    print(data_p1.shape) #(200, 30, 2048)
#    print(data_p2.shape) #(4000, 30, 2048)
    
    list_names = ['OneGaussianToy']
    
    Dataset = list_names,bags,[labels_bags],[labels_instance]
            
    return(Dataset)
    
def createGaussianToySets_MClasses(M=4,WR=0.01,n=20,k=300,np1=20,np2=200,overlap=False,
                          Between01=False,specificCase='',scaleWell=True):
    """
    
    La premiere feature est décalé pour obtenir une classe différente
    
    @param M : number of classes 
    @param WR : Witness rate = proportion of positive examples in the positive bags    
    # Variables communes 
    @param n=20  # Number of featues
    @param k=300 # Number of element in the bag 
    # la classe p1
    @param np1=20 # Number of positive bag
    @param np2=200 # Number of negative bag
    @param overlap=False  : overlapping between the point clouds
    @param : Between01 : if True the label of the class are 0-1 and not -1 and +1
    @param : specificCase : we proposed different case of toy points clouds
        - 2clouds : 2 clouds distincts points of clouds as positives examples
        - 2cloudsOpposite : 2 points clouds positive at the opposite from the negatives
        - 
    """
    assert(n>=2)
#    list_specificCase = ['',None,'2clouds','2cloudsOpposite']
    list_specificCase = ['',None]
    if not(specificCase in list_specificCase):
        print(specificCase,'is unknown')
        raise(NotImplementedError)
    
    np.random.seed(19680801)
    
    number_of_positive = int(np.floor(k*WR))
    if number_of_positive < 1:
        number_of_positive = 1
        print('Must have at least one element per bag !')
    
    if overlap:
        shift_negative_instances = 4
    else:
        shift_negative_instances = 8
        if scaleWell:
            shift_negative_instances *= np.sqrt(n-1)
    
    def gen_vect_p1(classe=0):
        """
        Function that generate the features of the positives elements
        """
        features = npt(np.random.randn(n))
#        if specificCase=='' or specificCase is None:
#             # Standard case
#        if specificCase=='2clouds':
#            features[1:2] += np.random.choice([-4,4], 1)[0]
#        if specificCase=='2cloudsOpposite':
#            features[0:1] += np.random.choice([0,shift_negative_instances*2], 1)[0]
        if classe % 4==0:
            features[0:1] += shift_negative_instances*(classe//4)
        if classe % 4==1:
            features[0:1] -= shift_negative_instances*(classe//4)
        if classe % 4==2:
            features[1:2] += shift_negative_instances*(classe//4)
        if classe % 4==3:
            features[1:2] -= shift_negative_instances*(classe//4)
        return features
    
    def gen_vect_p2():
        g0=np.random.randn(n)
        
        return npt(g0)
    
    def gen_paquet_p2():
        tab=np.zeros((k,n),dtype=npt)
        for i in range(k):
            tab[i]=gen_vect_p2()
        return tab
    
    def gen_paquet_p1(classe=0,N=np1):
        """
        Fct that return the positives bags !
        """
        tab=np.zeros((k,n),dtype=npt)
        if Between01:
            instances_labels = -np.zeros(shape=(M,k),dtype=npt)
        else:
            instances_labels = -np.ones(shape=(M,k),dtype=npt)
        positive_instance_index = np.random.choice(k, number_of_positive)
        for i in range(k):
            if (i in positive_instance_index):
                tab[i]=gen_vect_p1(classe=classe)
                instances_labels[classe,i] = 1.
            else:
                tab[i]=gen_vect_p2()
        return tab,instances_labels
    
    def gen_data_p1(classe=0,N=np1):
        tab=np.zeros((N,k,n),dtype=npt)
        choisi=np.zeros((N,),dtype=np.int32)
        for i in range(N):
            tab[i],choisi[i]=gen_paquet_p1(classe=classe)
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
        
        if Between01:
            labels_bags += [np.zeros(shape=(M,),dtype=npt)]
            labels_instance += [np.zeros(shape=(M,k),dtype=npt)]
        else:
            labels_bags += [-1.*np.ones(shape=(M,),dtype=npt)]
            labels_instance += [-1.*np.ones(shape=(M,k),dtype=npt)]
    
    # Generation of the positive bags
    for mc in range(M):
        for j in range(np1):
            tab,instances_labels = gen_paquet_p1(classe=mc)
            bags += [tab]
            if Between01:
                label_bag = np.zeros(shape=(M,),dtype=npt)
            else:
                label_bag = -1.*np.ones(shape=(M,),dtype=npt)
            label_bag[mc] = 1
            labels_bags += [label_bag]
            labels_instance += [instances_labels]
    
#    print(data_p1.shape) #(200, 30, 2048)
#    print(data_p2.shape) #(4000, 30, 2048)
    
    assert(len(bags)==len(labels_bags))
    
    list_names = ['OneGaussianToyMClasses']
    
    Dataset = list_names,bags,[labels_bags],[labels_instance]
            
    return(Dataset)
