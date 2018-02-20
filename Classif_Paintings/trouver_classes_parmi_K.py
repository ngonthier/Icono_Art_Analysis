#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:47:54 2018

@author: said
"""

import tensorflow as tf
import numpy as np
npt=np.float32
tft=np.float32
import time
import matplotlib.pyplot as plt

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
k=30
# la classe p1
np1=200
np2=2000

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
    return tab

data_p1,choisi=gen_data_p1()
data_p2=gen_data_p2()

#%%   Le graphe de calcul pour le cas où les exemples positifs et negatifs
# sont traites de maniere symetrique
for m in range(1): #juste pour refaire m fois l'experience
    X1=tf.constant(data_p1,dtype=tft)
    X2=tf.constant(data_p2,dtype=tft)
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
    for essai in range(5): #on fait 5 essais et on garde la meilleur loss
        sess.run(tf.global_variables_initializer())
        
        W_init=npt(np.random.randn(n))
        W_init=W_init/np.linalg.norm(W_init)
        b_init=npt(np.random.randn(1))
        dico={W:W_init,b:b_init}
        
        W_x=W_init.copy()
        b_x=b_init.copy()
        LR=0.01
        t0=time.time()
        for i in range(300): 
            dico={W:W_x,b:b_x}
            sor=sess.run([Tan1,Tan2,loss,gr],feed_dict=dico)
            #print('etape ',i,'loss=', sor[2],'Tan1=',sor[0],\
             #     'Tan2=',sor[1],'norme de W=', np.linalg.norm(W_x))
            b_x=b_x+LR*sor[3][1]
            W_x=W_x+LR*sor[3][0]
        if (essai==0) | (sor[2]>bestloss):
            W_best=W_x.copy()
            b_best=b_x.copy()
            bestloss=sor[2]
        print ('essai =', essai,'bestloss=',bestloss)
        t1=time.time()
        print ('TEMPS ECOULE=',t1-t0)
    
    
    #%%  Validation du vecteur choisi: Trouve-t-il le bon vecteur parmi les k
    #  de la calsse 1 ? En fait, il est pertinent de le faire sur la base
    # d'apprentissage car ce que l'on veut c'est singulariser les vecteurs
    # qui sont bien dans la classe qui nous interesse.
    # apres on peut sortir ces vecteurs et refaire une SVM dessus
    dico={W:W_best,b:b_best} 
    sor=sess.run([Prod1],feed_dict=dico)
    
    pr1=sor[0]
    mei=pr1.argmax(axis=1)
    
    scor=0
    for i in range(np1):
        if mei[i]==choisi[i]:
            scor+=1
    print('m=',m,'score = ',scor/np1*100, '%')
    W_best[0:10]   
    sess.close()
    tf.reset_default_graph()

plt.plot(abs(W_x))
plt.show()  
#print ('Tan1=',sor[0],'Tan2=',sor[1])

#for i in range(np1):
#    print(sor[0][i]-(sor[1].reshape((1,n))*data_p1[i]).sum(axis=1).max())
    
#%%   FORME 2 NON ENCORE IMPLEMENTEE
