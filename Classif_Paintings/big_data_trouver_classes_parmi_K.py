#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:47:54 2018

@author: said
"""

import tensorflow as tf
import numpy as np
npt=np.float32
tft=tf.float32
import time, os
import matplotlib.pyplot as plt
import multiprocessing
#%% On genere des vecteurs selon deux distributions p1 et p2 dans R^n
# On les regroupe par paquets de k vecteurs
# Un paquet est positif si il contient un vecteur p1
# sinon, il est négatif
# Le problème: Ne voyant que des paquets, peut-on séparer p1 de p2 par un 
# produit scalaire 

# La variable a chercher est w un vecteur de R^n et un biais b
# La loss peut etre de deux formes
# FORME 1: somme (tanh max_i (w|x_i+b)) *signe(i) 
#### La somme sur i porte sur 1..k et signe(i) =1 si p1
# FORME 2: la même sauf que pour un exemple négatif on ne fait pas le max car
#   on sait que tous les vecteurs d'un paquet negatif sont négatifs 
# individuellement

#%% Variables communes 
n=2048
k=300
# la classe p1
np1=177
np2=1777

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def parser(record):
    num_features = n
    size_group = k
    num_classes= 1
    keys_to_features={
                'X': tf.FixedLenFeature([size_group*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_classes],tf.float32)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    label = parsed['label']
    label = tf.squeeze(label) # To get a vector one dimension
    X = parsed['X']
    X= tf.reshape(X, [size_group,num_features])
    return X, label

def gen_vect_p1():
    return npt(np.random.randn(n))

def gen_vect_p2():
    g0=np.random.randn(n)
    g0[0:2]+=4
    return npt(g0)
    #return npt(np.random.randn(n)+2)

def gen_paquet_p2():
    tab=np.zeros((k,n),dtype=npt)
    for i in range(k):
        tab[i]=gen_vect_p2()
    return tab
    
    
def gen_paquet_p1(N=np1):
    tab=np.zeros((k,n),dtype=npt)
    choisi=np.random.randint(k)
    for i in range(k):
        if (i==choisi):
            tab[i]=gen_vect_p1()
        else:
            tab[i]=gen_vect_p2()
    return tab,choisi

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
        #print(i/np2*100)
    return tab

data_p1,choisi=gen_data_p1()
data_p2=gen_data_p2()

with tf.Session() as sess:
    filename_tfrecords = 'tmp2.tfrecords'
    if not(os.path.isfile(filename_tfrecords)): # If the file doesn't exist we will create it
        print("Start creating the Dataset")
        writer = tf.python_io.TFRecordWriter(filename_tfrecords)
        
        for i in range(np1+np2):
            if i % 1000 == 0: print("Step :",i)
            X = np.random.normal(size=(k,n))
            vectors =  2*np.random.randint(0,2,(1,1))-1
            features=tf.train.Features(feature={
                        'X': _floats_feature(X),
                        'label' : _floats_feature(vectors)})
            example = tf.train.Example(features=features)       
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print("The dataset tfrecords already exist")

#data_p1=np.float32(np.random.randn(np1,k,n))
#data_p2=np.float32(np.random.randn(np2,k,n))

#%%

idcs2=[k for k in range(np2)]
idcs1=[k for k in range(np1)]

idcs2shuff=idcs2.copy()
idcs1shuff=idcs1.copy()

##### VARIABLES IMPORTANTES
EPOCHS=10 #NOMBRE D'EPOCHS A FAIRE COMPTEE SUR LES EXEMPLES NEGATIFS
PAS1=100 #TAILLE DU BATCH POUR LES EXEMPLES POSITIFS
PAS2=1000 # TAILLE DU BATCH POUR LES EXEMPLES NEGATIFS

#%%   Le graphe de calcul pour le cas où les exemples positifs et negatifs
# sont traites de maniere symetrique

X1=tf.placeholder(tft,(PAS1,k,n))
X2=tf.placeholder(tft,(PAS2,k,n))
W=tf.placeholder(tft,[n])
b=tf.placeholder(tft,[1])

W1=tf.reshape(W,(1,1,n))


Prod1=tf.reduce_sum(tf.multiply(W1,X1),axis=2)+b
Max1=tf.reduce_max(Prod1,axis=1)
Tan1=tf.reduce_sum(tf.tanh(Max1))/np1

Prod2=tf.reduce_sum(tf.multiply(W1,X2),axis=2)+b
Max2=tf.reduce_max(Prod2,axis=1)
Tan2=tf.reduce_sum(tf.tanh(Max2))/np2


loss=Tan1-Tan2-tf.reduce_sum(W*W)

gr=tf.gradients(loss,[W,b])

sess=tf.InteractiveSession()


bestloss=-1
for essai in range(1): #on fait 5 essais et on garde la meilleur loss
    sess.run(tf.global_variables_initializer())
    
    W_init=npt(np.random.randn(n))
    W_init=W_init/np.linalg.norm(W_init)
    b_init=npt(np.random.randn(1))
    dico={W:W_init,b:b_init}
    
    W_x=W_init.copy()
    b_x=b_init.copy()
    LR=0.01
    
    ## premier maniere de faire 
    
#    
    nbpas=int(np.ceil(EPOCHS*np2/PAS2))
    nbidcs2=int(np.ceil(nbpas*PAS2/np2)) #on veut un nombre entier de batchs
    nbidcs1=int(np.ceil(nbpas*PAS1/np1))
    
    np.random.shuffle(idcs2shuff)
    np.random.shuffle(idcs1shuff)
    listidc1=idcs1shuff.copy()
    listidc2=idcs2shuff.copy()
    
    for i in range(nbidcs1-1):
        np.random.shuffle(idcs1shuff)
        listidc1=np.append(listidc1,idcs1shuff.copy())
    
    for i in range(nbidcs2-1):
        np.random.shuffle(idcs2shuff)
        listidc2=np.append(listidc2,idcs2shuff.copy())
    
    compte_elt = False
    number_let = 0
    t0=time.time()
    for i in range(nbpas): #boucle avec batchs: a la fin on aura fait EPOCHS
                            #visites des exemples negatifs
        i1=i*PAS1
        i2=i*PAS2
        #print('avancement ',i/nbpas*100)
        if compte_elt:
            number_let += len(data_p1[listidc1[i1:i1+PAS1],:,:]) + len(data_p2[listidc2[i2:i2+PAS2],:,:])
        dico={W:W_x,b:b_x,X1:data_p1[listidc1[i1:i1+PAS1],:,:],
              X2:data_p2[listidc2[i2:i2+PAS2]]}
#        sor=sess.run([gr],feed_dict=dico)
        gr_value=sess.run(gr,feed_dict=dico)
        b_x=b_x+LR*gr_value[1]
        W_x=W_x+LR*gr_value[0]
        #print('etape ',i,'loss=', sor[2],'Tan1=',sor[0],\
         #     'Tan2=',sor[1],'norme de W=', np.linalg.norm(W_x))
#        b_x=b_x+LR*sor[3][1]
#        W_x=W_x+LR*sor[3][0]
#    if (essai==0) | (sor[2]>bestloss):
#        W_best=W_x.copy()
#        b_best=b_x.copy()
#        bestloss=sor[2]
    print ('essai =', essai,'bestloss=',bestloss)
    t1=time.time()
    print ('TEMPS ECOULE=',t1-t0)
    if compte_elt: print('Number of element used :',number_let)
    ## Deuxieme maniere de faire 
    #HACK POUR TESTER VITESSE CONTIGUE
    number_let = 0
    t0=time.time()
    for i in range(nbpas): #boucle avec batchs: a la fin on aura fait EPOCHS
                            #visites des exemples negatifs
        i1=np.random.randint(0,np1-PAS1)
        i2=np.random.randint(0,np2-PAS2)
        if compte_elt: number_let += len(data_p1[i1:i1+PAS1,:,:]) + len(data_p2[i2:i2+PAS2,:,:])
        dico={W:W_x,b:b_x,X1:data_p1[i1:i1+PAS1,:,:],
              X2:data_p2[i2:i2+PAS2,:,:]}
        gr_value=sess.run(gr,feed_dict=dico)
        b_x=b_x+LR*gr_value[1]
        W_x=W_x+LR*gr_value[0]
        #print('etape ',i,'loss=', sor[2],'Tan1=',sor[0],\
         #     'Tan2=',sor[1],'norme de W=', np.linalg.norm(W_x))
#        b_x=b_x+LR*sor[3][1]
#        W_x=W_x+LR*sor[3][0]
#    if (essai==0) | (sor[2]>bestloss):
#        W_best=W_x.copy()
#        b_best=b_x.copy()
#        bestloss=sor[2]
    print ('essai =', essai,'bestloss=',bestloss)
    t1=time.time()
    print ('TEMPS ECOULE avec donnees contigue =',t1-t0)
    if compte_elt: print('Number of element used :',number_let)
#        dico={W:W_x,b:b_x,X1:data_p1[listidc1[i1:i1+PAS1],:,:],
#              X2:data_p2[listidc2[i2:i2+PAS2]]}
#         FIN HACK (DECOMMENTER deux lignes precedentes)    

    ## Use of Dataset from TF
    batch_size = PAS1 + PAS2
    print(batch_size,nbpas)
    buffer_size= 10000
    train_dataset = tf.data.TFRecordDataset(filename_tfrecords)
    num_proc = multiprocessing.cpu_count()
    train_dataset = train_dataset.map(parser,
                                        num_parallel_calls=num_proc)
    dataset_shuffle = train_dataset.shuffle(buffer_size=buffer_size,
                                                 reshuffle_each_iteration=True) 
    dataset_shuffle = dataset_shuffle.batch(batch_size)
    dataset_shuffle = dataset_shuffle.cache()
    dataset_shuffle = dataset_shuffle.repeat() 
    dataset_shuffle = dataset_shuffle.prefetch(10*batch_size) 
    shuffle_iterator = dataset_shuffle.make_initializable_iterator()
    X_, y_ = shuffle_iterator.get_next()
    W=tf.Variable(tf.random_normal([n], stddev=1.),name="weights")
    W=tf.reshape(W,(1,1,n))
    Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)
    Max=tf.reduce_max(Prod,axis=1)
    Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),y_))
    loss= tf.add(Tan,tf.reduce_sum(tf.multiply(W,W)))
    sess.close()

    LR = 0.01
    restarts = 0
    optimizer = tf.train.GradientDescentOptimizer(LR) 
    config = tf.ConfigProto()
#    config.intra_op_parallelism_threads = 16
#    config.inter_op_parallelism_threads = 16
    config.gpu_options.allow_growth = True
    train = optimizer.minimize(loss)  
    print("The graph is defined")
    sess = tf.Session(config=config)
        
    durationTab = []
    number_let = 0
    for essai in range(restarts+1):
        # To do need to reinitialiszed
        t0 = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(shuffle_iterator.initializer)
        t1 = time.time()
        duration = t1 - t0
        print('Duration of initialization : ',duration)
        t0 = time.time()
        for step in range(nbpas):
            if compte_elt:
                X_temp = sess.run(X_)
                number_let += len(X_temp)
            sess.run(train)
        t1 = time.time()
        duration = t1 - t0
        print("Duration with Dataset from TF ",str(step),' duration : ',duration)

    if compte_elt: print('Number of element used :',number_let)

        
    
    
    
#    #%  Validation du vecteur choisi: Trouve-t-il le bon vecteur parmi les k
#    #  de la calsse 1 ? En fait, il est pertinent de le faire sur la base
#    # d'apprentissage car ce que l'on veut c'est singulariser les vecteurs
#    # qui sont bien dans la classe qui nous interesse.
#    # apres on peut sortir ces vecteurs et refaire une SVM dessus
#    dico={W:W_best,b:b_best} 
#    sor=sess.run([Prod1],feed_dict=dico)
#    
#    pr1=sor[0]
#    mei=pr1.argmax(axis=1)
#    
#    scor=0
#    for i in range(np1):
#        if mei[i]==choisi[i]:
#            scor+=1
#    print('m=',m,'score = ',scor/np1*100, '%')
#    W_best[0:10]   
#    sess.close()
#    tf.reset_default_graph()
#
#plt.plot(abs(W_x))
#plt.show()  
##print ('Tan1=',sor[0],'Tan2=',sor[1])
#
##for i in range(np1):
##    print(sor[0][i]-(sor[1].reshape((1,n))*data_p1[i]).sum(axis=1).max())
#    
##%%   FORME 2 NON ENCORE IMPLEMENTEE
#
#
##%% test interface de classif 
#t0=time.time()
#for k in range(10):
#    im=npt(np.random.randn(256,256))
#    plt.imshow(im,cmap='gray')
#    plt.show(block=False)
#    reponse=input('classe?  ')
#    plt.close()
#    print('la reponse etait ',reponse)
#
#t1=time.time()
#print('temp total= ',t1-t0)
##%%
