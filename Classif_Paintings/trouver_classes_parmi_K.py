#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:47:54 2018

@author: said
"""

import tensorflow as tf
import tf_old
import numpy as np
npt=np.float32
tft=np.float32
import time
import matplotlib.pyplot as plt
#from misvm.util import partition, BagSplitter, spdiag, rand_convex, slices
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score
import time
import multiprocessing
# On genere des vecteurs selon deux distributions p1 et p2 dans R^n
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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_reshape(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

class MILSVM():
    """
    The MIL-SVM approach of Said Ladjal
    We advise you to normalized you input data ! by 
    """

    def __init__(self,LR=0.01,C=1.0,C_finalSVM=1.0,restarts=0, max_iters=300,
                 symway=True,all_notpos_inNeg=True,gridSearch=False,n_jobs=-1,
                 final_clf='LinearSVC',verbose=True):
        """
        
        @param LR : Learning rate : pas de gradient de descente [default: 0.01]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param C_finalSVM : the loss/regularization  term fo the final SVM training [default: 1.0]
        @param restarts : the number of random restarts [default: 0]
        @param max_iters : the maximum number of iterations in the inter loop of
                           the optimization procedure [default: 300]
        @param symway : If positive and negative bag are treated as the same way 
            or not [default: True]
        @param all_notpos_inNeg : All the element of the positive bag that are 
            not positive element are put in the negative class [default: True]
        @param gridSearch :GridSearch of the final SVC classifier [default: False]
        @param n_jobs : number of parallel jobs during the gridsearch -1 means = 
            number of cores  [default: -1]
        @param final_clf : final classifier used after the determination of the 
            element [choice : defaultSGD, linearSVC] [default : linearSVC]
        @param verbose : print optimization status messages [default: True]
        """
        tf.reset_default_graph()
        self.LR = LR
        self.C = C
        self.C_finalSVM = C_finalSVM
        self.restarts = restarts
        self.max_iters = max_iters
        self.verbose = verbose
        self.symway= symway
        self.all_notpos_inNeg = all_notpos_inNeg
        self.gridSearch = gridSearch
        self._bags = None
        self._bag_predictions = None
        self.n_jobs = n_jobs
        self.final_clf = final_clf
        self.PositiveRegions = None
        self.PositiveRegionsScore = None
        
    def fit(self,data_pos,data_neg):
        """
        @param data_pos : a numpy array of the positive bag of size number of positive bag
            * number of max element in one baf * dim features
        @param data_neg : a numpy array of the positive bag of size number of negative bag
            * number of max element in one baf * dim features
        """

        LR = self.LR # Regularisation loss
        np1,k,n = data_pos.shape
        np2,_,_ = data_neg.shape
        if self.verbose :print("Shapes :",np1,np2,k,n)
        X1=tf.constant(data_pos,dtype=tft)
        X2=tf.constant(data_neg,dtype=tft)
        W=tf.placeholder(tft,[n])
        b=tf.placeholder(tft,[1])

        W1=tf.reshape(W,(1,1,n))
        
        Prod1=tf.reduce_sum(tf.multiply(W1,X1),axis=2)+b # Caseof the positive data
        Max1=tf.reduce_max(Prod1,axis=1) # We take the max because we have at least one element of the bag that is positive
        Tan1=tf.reduce_sum(tf.tanh(Max1))/np1 # Sum on all the positive exemples 
#        Tan1=tf.reduce_sum(Max1)/np1 # Sum on all the positive exemples 
        
        Prod2=tf.reduce_sum(tf.multiply(W1,X2),axis=2)+b
        if self.symway :
            Max2=tf.reduce_max(Prod2,axis=1)
        else:
            Max2=tf.reduce_mean(Prod2,axis=1) # TODO Il faut que tu check cela avec Said quand meme
        Tan2=tf.reduce_sum(tf.tanh(Max2))/np2
#        Tan2=tf.reduce_sum(Max2)/np2
        # TODO peut on se passer de la tangente ??? 
        # TODO ne faudait il pas faire reduce_min pour le cas negative pour forcer a etre eloigne de tous ?
        
        loss=Tan1-Tan2-self.C*tf.reduce_sum(W*W)  #ceci peut se résoudre par la methode classique des multiplicateurs de Lagrange
         
        gr=tf.gradients(loss,[W,b])
        #print("Grad defined")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        
        bestloss=-1
        for essai in range(self.restarts+1): #on fait 5 essais et on garde la meilleur loss
            if self.verbose : print("essai",essai)
            if self.verbose : t0 = time.time()
            sess.run(tf.global_variables_initializer())
            W_init=npt(np.random.randn(n))
            W_init=W_init/np.linalg.norm(W_init)
            b_init=npt(np.random.randn(1))            
            W_x=W_init.copy()
            b_x=b_init.copy()
#            #LR=0.01
            for i in range(self.max_iters): 
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
                if self.verbose : print("bestloss",bestloss) # La loss est maximale a 2 
                dicobest={W:W_best,b:b_best} 
                #Record of the best SVM         
            if self.verbose:
                t1 = time.time()
                print("durations :",str(t1-t0))
        sor1=sess.run([Prod1],feed_dict=dicobest)
        sor2=sess.run([Prod2],feed_dict=dicobest)
        sess.close()
        tf.reset_default_graph()
        
        pr1=sor1[0]
        mei=pr1.argmax(axis=1) # Indexes of the element that below to the right class normally
        score_mei=pr1.max(axis=1) # Indexes of the element that below to the right class normally
        self.PositiveExScoreAll = pr1
        self.PositiveRegions = mei
        self.PositiveRegionsScore = score_mei
          
        pr2=sor2[0]
        mei=pr2.argmax(axis=1) # Indexes of the element that below to the right class normally
        score_mei=pr2.max(axis=1) # Indexes of the element that below to the right class normally
        self.NegativeRegions = mei
        self.NegativeRegionsScore = score_mei
        
        if (self.final_clf is None) or (self.final_clf == 'None'):
            # We don t return a final classifier
             if self.verbose : print("We don't return a final classifier !!!")
             return(None)
        else:
            full_positives =  np.zeros((np1,n))
            if self.all_notpos_inNeg:
                if self.verbose : print("All element that are not the positive example are considered as negative")
                full_neg = np.zeros((np1*(k-1),n))
                for i in range(np1):
                     index = mei[i]
                     data = data_pos[i,index,:] 
                     full_positives[i,:] = data     
                     data = np.concatenate([data_pos[i,0:index,:],data_pos[i,index:-1,:]])
                     full_neg[i*(k-1):(i+1)*(k-1),:] = data
                data_p2_reshaped =  np.reshape(data_neg,(np2*k,n))
                print(data_p2_reshaped.shape)
                print(full_neg.shape)
                full_neg_all = np.vstack([full_neg,data_p2_reshaped])
                print(full_neg_all.shape)
            else:     
                if self.verbose : print("All element that are not the positive example are not considered as negative,they are ignored")
                full_neg_all =  np.reshape(data_neg,(np2*k,n))
                for i in range(np1):
                     index = mei[i]
                     data = data_pos[i,index,:] 
                     full_positives[i,:] = data
                     
            X = np.vstack((full_positives,full_neg_all))
            y_pos = np.ones((np1,1))
            y_neg = np.zeros((len(full_neg_all),1))
            y = np.vstack((y_pos,y_neg)).ravel()
            
            if len(X) > 30000 and self.verbose and self.final_clf == 'LinearSVC':
                print("We advise you to use an online classification method as SGD")
            
            if self.verbose : 
                print("Shape of X",X.shape)
                print("number of positive examples :",len(y_pos),"number of negative example :",len(y_neg))
            if self.verbose : print("Retrain a new SVM")
                
            classifier = TrainClassif(X,y,clf='LinearSVC',gridSearch=self.gridSearch,n_jobs=self.n_jobs
                     ,C_finalSVM=self.C_finalSVM)
            if self.verbose :
                labels_test_predited = classifier.predict(X)
                print('Number of positive prediction :',np.sum(labels_test_predited))
                training_precision = precision_score(y,labels_test_predited)
                y_predict_confidence_score_classifier = classifier.decision_function(X)
                AP = average_precision_score(y,y_predict_confidence_score_classifier,average=None)
                print("Training precision of the final LinearSVC",training_precision,'training AP:',AP)
            return(classifier)

  
    def get_PositiveRegions(self):
        return(self.PositiveRegions.copy())
     
    def get_PositiveRegionsScore(self):
        return(self.PositiveRegionsScore.copy())
     
    def get_PositiveExScoreAll(self):
        return(self.PositiveExScoreAll.copy())
        
    def get_NegativeRegions(self):
        return(self.NegativeRegions.copy())
     
    def get_NegativeRegionsScore(self):
        return(self.NegativeRegionsScore.copy())


class tf_MILSVM():
    """
    The MIL-SVM approach of Said Ladjal, try to get a tf version that used the 
    different tf optimizer
    We advise you to normalized your input data !  Because we don't do it inside 
    this function
    """

    def __init__(self,LR=0.01,C=1.0,C_finalSVM=1.0,restarts=0, max_iters=300,
                 symway=True,all_notpos_inNeg=True,gridSearch=False,n_jobs=-1,
                 final_clf='LinearSVC',verbose=True,Optimizer='GradientDescent',
                  optimArg=None,mini_batch_size=200,buffer_size=10000,num_features=2048,
                  num_rois=300,num_classes=10,max_iters_sgdc=None,debug=False,
                  is_betweenMinus1and1=False,CV_Mode=None,num_split=2,with_scores=False,
                  epsilon=0.0):
        # TODOD enelver les trucs inutiles ici
        # TODO faire des tests unitaire sur les differentes parametres
        """
        @param LR : Learning rate : pas de gradient de descente [default: 0.01]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param C_finalSVM : the loss/regularization  term fo the final SVM training [default: 1.0]
        @param restarts : the number of random restarts [default: 0]
        @param max_iters : the maximum number of iterations in the inter loop of
                           the optimization procedure [default: 300]
        @param symway : If positive and negative bag are treated as the same way 
            or not [default: True]
        @param all_notpos_inNeg : All the element of the positive bag that are 
            not positive element are put in the negative class [default: True]
        @param gridSearch :GridSearch of the final SVC classifier [default: False]
        @param n_jobs : number of parallel jobs during the gridsearch -1 means = 
            number of cores  [default: -1]
        @param final_clf : final classifier used after the determination of the 
            element [choice : defaultSGD, linearSVC] [default : linearSVC]
        @param verbose : print optimization status messages [default: True]
        @param mini_batch_size : taille des mini_batch_size
        @param buffer_size : taille du buffer
        @param num_features : pnumbre de features
        @param num_rois : nombre de regions d interet
        @param num_classes : numbre de classes dans la base
        @param max_iters_sgdc : Nombre d iterations pour la descente de gradient stochastique classification
        @param debug : default False : if we want to debug 
        @param is_betweenMinus1and1 : default False : if we have the label value alreaddy between -1 and 1
        @param CV_Mode : default None : cross validation mode in the MILSVM : possibility ; None, CV in k split or LA for Leave apart one of the split
        @param num_split : default 2 : the number of split/fold used in the cross validation method
        @param with_scores : default False : Multiply the scalar product before the max by the objectness score from the FasterRCNN
        """
        self.LR = LR
        self.C = C
        self.C_finalSVM = C_finalSVM
        self.restarts = restarts
        self.max_iters = max_iters
        if max_iters_sgdc is None:
            self.max_iters_sgdc = max_iters
        else:
            self.max_iters_sgdc = max_iters_sgdc
        self.verbose = verbose
        self.symway= symway
        self.all_notpos_inNeg = all_notpos_inNeg
        self.gridSearch = gridSearch
        self._bags = None
        self._bag_predictions = None
        self.n_jobs = n_jobs
        self.final_clf = final_clf
        self.PositiveRegions = None
        self.PositiveRegionsScore = None
        self.Optimizer = Optimizer
        self.optimArg =  optimArg# GradientDescent, 
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size
        self.num_features = num_features
        self.num_rois = num_rois
        self.num_classes = num_classes
        self.debug = debug 
        self.is_betweenMinus1and1 = is_betweenMinus1and1
        self.np_pos_value = 1
        self.np_neg_value = 1 # Ces elements peuvent etre des matrices ou des vecteurs selon les cas
        self.CV_Mode = CV_Mode
        if not(CV_Mode is None):
            assert(num_split>1) # Il faut plus d un folder pour separer
            self.num_split = num_split # Only useful if CrossVal==True
            if num_split>2:
                print('The use of more that 2 spits seem to slow a lot the computation with the use of shard')
        self.with_scores = with_scores 
        self.epsilon = epsilon# Used to avoid having a zero score
                
                
    def fit_w_CV(self,data_pos,data_neg):
        kf = KFold(n_splits=3) # Define the split - into 2 folds 
        # KFold From sklearn.model_selection 
        kf.get_n_splits(data_pos) # returns the number of splitting iterations in the cross-validator
        return(0)
        
    # From http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    def parser(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([5*self.num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        #tf.Print(label,[label])
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7, label
    def parser_wRoiScore(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([5*self.num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        roi_scores = parsed['roi_scores']
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7,roi_scores, label
    
    def parser_all_classes(self,record):
        # Perform additional preprocessing on the parsed data.
#        keys_to_features={
#                    'height': tf.FixedLenFeature([], tf.int64),
#                    'width': tf.FixedLenFeature([], tf.int64),
#                    'num_regions':  tf.FixedLenFeature([], tf.int64),
#                    'num_features':  tf.FixedLenFeature([], tf.int64),
#                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
#                    'rois': tf.FixedLenFeature([5*self.num_rois],tf.float32),
#                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
#                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
#                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32),
#                    'name_img' : tf.FixedLenFeature([],tf.string)}
        keys_to_features={
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7, label
    
    def parser_all_classes_wRoiScore(self,record):
        keys_to_features={
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        roi_scores = parsed['roi_scores']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7,roi_scores, label
    
    def parser_w_rois(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([5*self.num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [self.num_rois,5])           
        return fc7,rois, label,name_img

    def parser_w_mei(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'score_mei': tf.FixedLenFeature([1], tf.float32),
                    'mei': tf.FixedLenFeature([1], tf.int64),
                    'rois': tf.FixedLenFeature([self.num_rois*5],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'fc7_selected': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([1],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        label_300 = tf.tile(label,[self.num_rois])
        mei = parsed['mei']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        fc7_selected = parsed['fc7_selected']
        fc7_selected = tf.reshape(fc7_selected, [300,self.num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [self.num_rois,5])           
        return fc7_selected,fc7,mei,label_300
    
    def parser_w_mei_reduce(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'score_mei': tf.FixedLenFeature([1], tf.float32),
                    'mei': tf.FixedLenFeature([1], tf.int64),
                    'rois': tf.FixedLenFeature([self.num_rois*5],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'fc7_selected': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([1],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        label_300 = tf.tile(label,[self.num_rois])
        fc7_selected = parsed['fc7_selected']
        fc7_selected = tf.reshape(fc7_selected, [self.num_rois,self.num_features])         
        return fc7_selected,label_300
    
    def tf_dataset_use_per_batch(self,train_dataset):
        
        if tf.__version__ > '1.6' and self.performance:
            dataset_batch = train_dataset.apply(tf.contrib.data.map_and_batch(
                map_func=self.first_parser, batch_size=self.mini_batch_size))
        else:
            train_dataset = train_dataset.map(self.first_parser,
                                          num_parallel_calls=self.cpu_count)
            dataset_batch = train_dataset.batch(self.mini_batch_size)
        dataset_batch = dataset_batch.cache()
        dataset_batch = dataset_batch.prefetch(1)
        iterator_batch = dataset_batch.make_initializable_iterator()
        return(iterator_batch)
     
    def def_SVM_onMean(self,X_, y_):
        X_mean = tf.reduce_mean(X_,axis=1) 
        # Definition of the graph 
        if self.class_indice==-1:
            if self.restarts_paral:
                W_local=tf.Variable(tf.random_normal([self.paral_number_W,self.num_classes,self.num_features], stddev=1.),name="weights")
                b_local=tf.Variable(tf.random_normal([self.paral_number_W,self.num_classes,1,1], stddev=1.), name="bias")
                if tf.__version__ >= '1.8':
                    normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,axis=[0,1])) 
                else:
                    normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,dim=[0,1]))
                W_r=tf.reshape(W_local,(self.paral_number_W,self.num_classes,1,1,self.num_features))
            else:
                W_local=tf.Variable(tf.random_normal([self.num_classes,self.num_features], stddev=1.),name="weights")
                b_local=tf.Variable(tf.random_normal([self.num_classes,1,1], stddev=1.), name="bias")
                if tf.__version__ >= '1.8':
                    normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,axis=0)) 
                else:
                    normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,dim=0))
                W_r=tf.reshape(W_local,(self.num_classes,1,1,self.num_features))
            
            Prod=tf.add(tf.reduce_sum(tf.multiply(W_r,X_mean),axis=-1),b_local)
            if self.is_betweenMinus1and1:
                weights_bags_ratio = -tf.divide(tf.add(y_,1.),tf.multiply(2.,self.np_pos_value)) + tf.divide(tf.add(y_,-1.),tf.multiply(-2.,self.np_neg_value))
                # Need to add 1 to avoid the case 
                # The wieght are negative for the positive exemple and positive for the negative ones !!!
            else:
                weights_bags_ratio = -tf.divide(y_,self.np_pos_value) + tf.divide(-tf.add(y_,-1),self.np_neg_value)
            weights_bags_ratio = tf.transpose(weights_bags_ratio,[1,0])
            Tan= tf.reduce_sum(tf.multiply(tf.tanh(Prod),weights_bags_ratio),axis=-1) 
            # Sum on all the exemples 
            loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.pow(W_r,2),axis=[-2,-1])))
            # Shape 20
        else:
            W_local=tf.Variable(tf.random_normal([self.num_features], stddev=1.),name="weights")
            b_local=tf.Variable(tf.random_normal([1], stddev=1.), name="bias")
            if tf.__version__ >= '1.8':
                normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,axis=0)) 
            else:
                normalize_W = W_local.assign(tf.nn.l2_normalize(W_local,dim=0)) 
            W_local=tf.reshape(W_local,(1,self.num_features))
            Prod=tf.reduce_sum(tf.multiply(W_local,X_mean),axis=1)+b_local
            Max=tf.reduce_max(Prod,axis=1) # We take the max because we have at least one element of the bag that is positive
            weights_bags_ratio = -tf.divide(y_,self.np_pos_value) + tf.divide(-tf.add(y_,-1),self.np_neg_value) # Need to add 1 to avoid the case 
            Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio)) # Sum on all the positive exemples 
            loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.multiply(W_local,W_local))))
            
        if(self.Optimizer == 'GradientDescent'):
            optimizer_local = tf.train.GradientDescentOptimizer(self.LR) 
        elif(self.Optimizer == 'Momentum'):
            optimizer_local = tf.train.MomentumOptimizer(self.optimArg['learning_rate'],self.optimArg['momentum']) 
        elif(self.Optimizer == 'Adam'):
            if self.optimArg is None:
                optimizer = tf.train.AdamOptimizer(self.LR) # Default value  : beta1=0.9,beta2=0.999,epsilon=1e-08, maybe epsilon should be 0.1 or 1
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=\
                self.optimArg['learning_rate'],beta1=self.optimArg['beta1'],\
                beta2=self.optimArg['beta2'],epsilon=self.optimArg['epsilon']) 
        elif(self.Optimizer == 'Adagrad'):
            optimizer_local = tf.train.AdagradOptimizer(self.LR) 
        else:
            print("The optimizer is unknown",self.Optimizer)
          
        train_local = optimizer_local.minimize(loss) 
        return(W_local,b_local,train_local)

    def compute_STD_all(self,X_batch,iterator_batch):
        """
        Compute the mean and variance per feature
        """
        sum_train_set = np.zeros((self.num_features,),dtype=np.float32)
        sumSquared_train_set = np.zeros((self.num_features,),dtype=np.float32)
        X_batch_value_ph = tf.placeholder(tf.float32, shape=(None,self.num_rois,self.num_features))
        Number_elt = 0
        add_sum = tf.reduce_sum(X_batch_value_ph,axis=[0,1])
        add_sumSquared = tf.reduce_sum(tf.pow(X_batch_value_ph,2),axis=[0,1])
                 
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator_batch.initializer)
            while True:
              try:
                  # Attention a chaque fois que l on appelle la fonction iterator on avance
                  X_batch_value = sess.run(X_batch)
                  Number_elt += X_batch_value.shape[0]
                  sum_train_set += sess.run(add_sum, feed_dict = {X_batch_value_ph:X_batch_value})
                  sumSquared_train_set += sess.run(add_sumSquared, feed_dict = {X_batch_value_ph:X_batch_value})
              except tf.errors.OutOfRangeError:
                break
            mean_train_set = sum_train_set/(Number_elt*self.num_rois)
            var_train_set = (sumSquared_train_set/(Number_elt*self.num_rois)) - (mean_train_set)**2
            std_train_set = np.sqrt(var_train_set)
        return(mean_train_set,std_train_set)

    def fit_SVM_onMean(self,sess,W_local,b_local,train_local):

        for step in range(self.max_iters):
            if self.debug: t2 = time.time()
            sess.run(train_local)
            if self.debug:
                t3 = time.time()
                print(step,"durations :",str(t3-t2))

        W_tmp=sess.run(W_local)
        b_tmp=sess.run(b_local)
#        sess.close()
#        tf.reset_default_graph()
        return(W_tmp,b_tmp)
        
    def fit_MILSVM_tfrecords(self,data_path,class_indice,shuffle=True,WR=False,
                             init_by_mean=None,norm=None,performance=False,restarts_paral=False):
        """" 
        This function run per batch on the tfrecords data folder
        @param : data_path : 
        @param : choose of the class to run the optimisation on, if == -1 , then 
        run all the class at once
        @param : shuffle or not the dataset 
        @param : WR : WR=True mean Without Regularisation in the evaluation of the
        loss in order to choose the best vector
        @param : init_by_mean   :  use of an initialisation of the vecteur 
        W and bias b by a optimisation on a classification task on the mean 
        on all the regions of the image, = None, 'First' or 'All'
        @param : norm : normalisation of the data or not : possible : None or ''
            'L2' : normalisation L2 or 'STDall' : Standardisation on all data 
            'STD' standardisation by feature maps
        @param : performance : boolean use or not of optimize function for 
        shuffle and repeat TF dataset
        @param : restarts_paral : run several W vecteur optimisation in parallel 
        """
        # Idees pour ameliorer cette fonction : 
        # Ajouter la possibilite de choisir les regions avec un nms lors de la 
        
        # TODO : faire un score a 0 ou 1 selon les images negtaives et positives et selon un seuil
        # TODO : Faire les vecteurs restarts en parallele
        # TODO : selectionner que les top_k region sby score
        
        # Travailler sur le min du top k des regions sinon 
        self.norm = norm
        self.First_run = True
        self.WR = WR
        self.class_indice = class_indice # If class_indice = -1 we will run on all the class at once ! parallel power
        self.performance = performance
        self.init_by_mean = init_by_mean
        self.restarts_paral = restarts_paral
        if self.init_by_mean and self.restarts_paral: raise(NotImplemented)
        if self.class_indice>-1 and self.restarts_paral: raise(NotImplemented)
        self.paral_number_W = self.restarts +1
        ## Debut de la fonction        
        self.cpu_count = multiprocessing.cpu_count()
        train_dataset_init = tf.data.TFRecordDataset(data_path)
        
        if self.CV_Mode=='CV':
            if self.verbose: print('Use of the Cross Validation with ',self.num_split,' splits')
            train_dataset_tmp = train_dataset_init.shard(self.num_split,0)
            for i in range(1,self.num_split-1):
                train_dataset_tmp2 = train_dataset_init.shard(self.num_split,i)
                train_dataset_tmp = train_dataset_tmp.concatenate(train_dataset_tmp2)
            train_dataset = train_dataset_tmp
        else: # Case where elf.CV_Mode=='LA" or None
            train_dataset = train_dataset_init
            # The second argument is the index of the subset used
        if self.class_indice==-1:
            if self.with_scores:
                self.first_parser = self.parser_all_classes_wRoiScore
            else:
                self.first_parser = self.parser_all_classes
        else:
            if self.with_scores:
                self.first_parser = self.parser_wRoiScore
            else:
                self.first_parser = self.parser
            
        iterator_batch = self.tf_dataset_use_per_batch(train_dataset)
        
        if self.with_scores:
            X_batch,scores_batch, label_batch = iterator_batch.get_next()
        else:
            X_batch, label_batch = iterator_batch.get_next()
        
        # Calcul preliminaire a la definition de la fonction de cout 
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 16
        self.config.inter_op_parallelism_threads = 16
        self.config.gpu_options.allow_growth = True
        
        minus_1 = tf.constant(-1.)

        if class_indice==-1:
            label_vector = tf.placeholder(tf.float32, shape=(None,self.num_classes))
            if self.is_betweenMinus1and1:
                add_np_pos = tf.divide(tf.reduce_sum(tf.add(label_vector,tf.constant(1.))),tf.constant(2.))
                add_np_neg = tf.divide(tf.reduce_sum(tf.add(label_vector,minus_1)),tf.constant(-2.))
            else:
                add_np_pos = tf.reduce_sum(label_vector,axis=0)
                add_np_neg = -tf.reduce_sum(tf.add(label_vector,minus_1),axis=0)
            np_pos_value = np.zeros((self.num_classes,),dtype=np.float32)
            np_neg_value = np.zeros((self.num_classes,),dtype=np.float32)
        else:
            label_vector = tf.placeholder(tf.float32, shape=(None,))
            if self.is_betweenMinus1and1:
                add_np_pos = tf.divide(tf.reduce_sum(tf.add(label_vector,tf.constant(1.))),tf.constant(2.))
                add_np_neg = tf.divide(tf.reduce_sum(tf.add(label_vector,minus_1)),tf.constant(-2.))
            else:
                label_batch
                add_np_pos = tf.reduce_sum(label_vector)
                add_np_neg = -tf.reduce_sum(tf.add(label_vector,minus_1))
            np_pos_value = 0.
            np_neg_value = 0.
            
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator_batch.initializer)
            while True:
              try:
                  # Attention a chaque fois que l on appelle la fonction iterator on avance
                  label_batch_value = sess.run(label_batch)
                  np_pos_value += sess.run(add_np_pos, feed_dict = {label_vector:label_batch_value})
                  np_neg_value += sess.run(add_np_neg, feed_dict = {label_vector:label_batch_value})
              except tf.errors.OutOfRangeError:
                break
            
        if self.norm=='STDall': # Standardization on all the training set https://en.wikipedia.org/wiki/Feature_scaling
            mean_train_set, std_train_set = self.compute_STD_all(X_batch,iterator_batch)
            
        self.np_pos_value = np_pos_value
        self.np_neg_value = np_neg_value
        if self.verbose:print("Finished to compute the proportion of each label :",np_pos_value,np_neg_value)
       
        if self.CV_Mode=='CV':
            train_dataset_tmp = train_dataset_init.shard(self.num_split,0)
            for i in range(1,self.num_split-1):
                train_dataset_tmp2 = train_dataset.shard(self.num_split,i)
                train_dataset_tmp = train_dataset_tmp.concatenate(train_dataset_tmp2)
            train_dataset2 = train_dataset_tmp
            train_dataset = train_dataset_init.shard(self.num_split,self.num_split-1) 
            # The last fold is keep for doing the cross validation
            iterator_batch = self.tf_dataset_use_per_batch(train_dataset)
            if self.with_scores:
                X_batch,scores_batch, label_batch = iterator_batch.get_next()
            else:
                X_batch, label_batch = iterator_batch.get_next() 
        elif self.CV_Mode=='LA':
            if self.verbose: print('Use of the Leave One Aside with ',self.num_split,' splits')
            train_dataset_tmp = train_dataset_init.shard(self.num_split,0)
            for i in range(1,self.num_split-1):
                train_dataset_tmp2 = train_dataset.shard(self.num_split,i)
                train_dataset_tmp = train_dataset_tmp.concatenate(train_dataset_tmp2)
            train_dataset2 = train_dataset_tmp
            # The evaluation of the loss will be on all the dataset
        else:
            # TODO test !
            train_dataset2 = tf.data.TFRecordDataset(data_path) # train_dataset_init ?  A tester
        
        # From at https://www.tensorflow.org/versions/master/performance/datasets_performance
        if tf.__version__ > '1.6' and shuffle and performance:
            train_dataset2 = train_dataset2.apply(tf.contrib.data.map_and_batch(
                    map_func=self.first_parser, batch_size=self.mini_batch_size,
                    num_parallel_batches=self.cpu_count,drop_remainder=False))
            dataset_shuffle = train_dataset2.apply(tf.contrib.data.shuffle_and_repeat(self.buffer_size))
            dataset_shuffle = dataset_shuffle.prefetch(self.mini_batch_size) 
        else:
            train_dataset2 = train_dataset2.map(self.first_parser,
                                            num_parallel_calls=self.cpu_count)
            if shuffle:
                dataset_shuffle = train_dataset2.shuffle(buffer_size=self.buffer_size,
                                                         reshuffle_each_iteration=True) 
            else:
                dataset_shuffle = train_dataset2
            dataset_shuffle = dataset_shuffle.batch(self.mini_batch_size)
            dataset_shuffle = dataset_shuffle.cache() 
            dataset_shuffle = dataset_shuffle.repeat() # ? self.max_iters
            dataset_shuffle = dataset_shuffle.prefetch(self.mini_batch_size) # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/47025850#47025850

        shuffle_iterator = dataset_shuffle.make_initializable_iterator()
        if self.with_scores:
            X_,scores_, y_ = shuffle_iterator.get_next()
        else:
            X_, y_ = shuffle_iterator.get_next()

        if self.norm=='L2':
            if self.debug: print('L2 normalisation')
            X_ = tf.nn.l2_normalize(X_,axis=-1)
            X_batch = tf.nn.l2_normalize(X_batch,axis=-1)
        elif self.norm=='STD_all':
            if self.debug: print('Standardisation')
            X_ = tf.divide(tf.add( X_,-mean_train_set), std_train_set)
            X_batch = tf.divide(tf.add(X_batch,-mean_train_set), std_train_set)

        # Definition of the graph 
        if class_indice==-1:
            if self.restarts_paral:
                W=tf.Variable(tf.random_normal([self.paral_number_W,self.num_classes,self.num_features], stddev=1.),name="weights")
                b=tf.Variable(tf.random_normal([self.paral_number_W,self.num_classes,1,1], stddev=1.), name="bias")
                if tf.__version__ >= '1.8':
                    normalize_W = W.assign(tf.nn.l2_normalize(W,axis=[0,1])) 
                else:
                    normalize_W = W.assign(tf.nn.l2_normalize(W,dim=[0,1]))
                W_r=tf.reshape(W,(self.paral_number_W,self.num_classes,1,1,self.num_features))
            else:
                W=tf.Variable(tf.random_normal([self.num_classes,self.num_features], stddev=1.),name="weights")
                b=tf.Variable(tf.random_normal([self.num_classes,1,1], stddev=1.), name="bias")
                if tf.__version__ >= '1.8':
                    normalize_W = W.assign(tf.nn.l2_normalize(W,axis=0)) 
                else:
                    normalize_W = W.assign(tf.nn.l2_normalize(W,dim=0))
                W_r=tf.reshape(W,(self.num_classes,1,1,self.num_features))
            Prod=tf.add(tf.reduce_sum(tf.multiply(W_r,X_),axis=-1),b)
            if self.with_scores: Prod=tf.multiply(Prod,tf.add(scores_,self.epsilon))
            Max=tf.reduce_max(Prod,axis=-1) # We could try with a softmax or a relaxation version of the max !
            if self.is_betweenMinus1and1:
                weights_bags_ratio = -tf.divide(tf.add(y_,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(y_,-1.),tf.multiply(-2.,np_neg_value))
                # Need to add 1 to avoid the case 
                # The wieght are negative for the positive exemple and positive for the negative ones !!!
            else:
                weights_bags_ratio = -tf.divide(y_,np_pos_value) + tf.divide(-tf.add(y_,-1),np_neg_value)
            weights_bags_ratio = tf.transpose(weights_bags_ratio,[1,0])
            Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio),axis=-1) # Sum on all the positive exemples 
            loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.pow(W_r,2),axis=[-3,-2,-1])))
            # Shape 20 and if self.restarts_paral shape (number_W) x 20
            
            # Definiton du graphe pour la partie evaluation de la loss
            Prod_batch=tf.add(tf.reduce_sum(tf.multiply(W_r,X_batch),axis=-1),b)
            if self.with_scores: Prod_batch=tf.multiply(Prod_batch,tf.add(scores_batch,self.epsilon))
            Max_batch=tf.reduce_max(Prod_batch,axis=-1) # We take the max because we have at least one element of the bag that is positive
            if self.is_betweenMinus1and1:
                weights_bags_ratio_batch = -tf.divide(tf.add(label_batch,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(label_batch,-1.),tf.multiply(-2.,np_neg_value))
                # Need to add 1 to avoid the case 
                # The wieght are negative for the positive exemple and positive for the negative ones !!!
            else:
                weights_bags_ratio_batch = -tf.divide(label_batch,np_pos_value) + tf.divide(-tf.add(label_batch,-1),np_neg_value) # Need to add 1 to avoid the case 
            weights_bags_ratio_batch = tf.transpose(weights_bags_ratio_batch,[1,0])
            Tan_batch= tf.reduce_sum(tf.multiply(tf.tanh(Max_batch),weights_bags_ratio_batch),axis=-1) # Sum on all the positive exemples 
            loss_batch= tf.add(Tan_batch,tf.multiply(self.C,tf.reduce_sum(tf.pow(W_r,2),axis=[-3,-2,-1])))
            
        else:
            # TODO faire le parallele sur les W
            W=tf.Variable(tf.random_normal([self.num_features], stddev=1.),name="weights")
            b=tf.Variable(tf.random_normal([1], stddev=1.), name="bias")
            if tf.__version__ >= '1.8':
                normalize_W = W.assign(tf.nn.l2_normalize(W,axis=0)) 
            else:
                normalize_W = W.assign(tf.nn.l2_normalize(W,dim=0)) 
            W=tf.reshape(W,(1,1,self.num_features))
            Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)+b
            if self.with_scores: Prod=tf.multiply(Prod,tf.add(scores_,self.epsilon))
            Max=tf.reduce_max(Prod,axis=1) # We take the max because we have at least one element of the bag that is positive
            if self.is_betweenMinus1and1:
                weights_bags_ratio = -tf.divide(tf.add(y_,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(y_,-1.),tf.multiply(-2.,np_neg_value))
            else:
                weights_bags_ratio = -tf.divide(y_,np_pos_value) + tf.divide(-tf.add(y_,-1),np_neg_value) # Need to add 1 to avoid the case 
            Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio)) # Sum on all the positive exemples 
            loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.multiply(W,W))))
            
            Prod_batch=tf.reduce_sum(tf.multiply(W,X_batch),axis=2)+b
            if self.with_scores: Prod_batch=tf.multiply(Prod_batch,tf.add(scores_batch,self.epsilon))
            Max_batch=tf.reduce_max(Prod_batch,axis=1) # We take the max because we have at least one element of the bag that is positive
            if self.is_betweenMinus1and1:
                weights_bags_ratio_batch = -tf.divide(tf.add(label_batch,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(label_batch,-1.),tf.multiply(-2.,np_neg_value))
                # Need to add 1 to avoid the case 
                # The wieght are negative for the positive exemple and positive for the negative ones !!!
            else:
                weights_bags_ratio_batch = -tf.divide(label_batch,np_pos_value) + tf.divide(-tf.add(label_batch,-1),np_neg_value) # Need to add 1 to avoid the case 
            
            Tan_batch= tf.reduce_sum(tf.multiply(tf.tanh(Max_batch),weights_bags_ratio_batch)) # Sum on all the positive exemples 
            if self.WR:
                loss_batch= Tan_batch
            else:
                loss_batch= tf.add(Tan_batch,tf.multiply(self.C,tf.reduce_sum(tf.multiply(W,W))))
        
        if(self.Optimizer == 'GradientDescent'):
            optimizer = tf.train.GradientDescentOptimizer(self.LR) 
        elif(self.Optimizer == 'Momentum'):
            optimizer = tf.train.MomentumOptimizer(self.optimArg['learning_rate'],self.optimArg['momentum']) 
        elif(self.Optimizer == 'Adam'):
            if self.optimArg is None:
                optimizer = tf.train.AdamOptimizer(self.LR) 
                # Default value  : beta1=0.9,beta2=0.999,epsilon=1e-08, 
                # maybe epsilon should be 0.1 or 1 cf https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=\
                self.optimArg['learning_rate'],beta1=self.optimArg['beta1'],\
                beta2=self.optimArg['beta2'],epsilon=self.optimArg['epsilon'])
        elif(self.Optimizer == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(self.LR) 
        else:
            print("The optimizer is unknown",self.Optimizer)
        train = optimizer.minimize(loss)  
        
        if class_indice==-1:
            bestloss = np.zeros((self.num_classes,),dtype=np.float32)
        else:
            bestloss = 0.
        
        if not(self.init_by_mean is None) or not(self.init_by_mean =='')  :
            W_onMean,b_onMean,train_SVM_onMean = self.def_SVM_onMean(X_,y_)
            init_op_onMean = tf.group(W_onMean.initializer,b_onMean.initializer)
            placeholder = tf.placeholder(tf.float32, shape=W_onMean.shape)
            assign_op = W.assign(tf.reshape(placeholder,tf.shape(W)))
            placeholder_b = tf.placeholder(tf.float32, shape=b_onMean.shape)
            assign_op_b = b.assign(tf.reshape(placeholder_b,tf.shape(b)))
            
        
        sess = tf.Session(config=self.config)
#        saver = tf.train.Saver()      
        init_op = tf.group(W.initializer,b.initializer,tf.global_variables_initializer()\
                           ,tf.local_variables_initializer())
#        sess.graph.finalize()   
        if self.restarts_paral:
            if self.verbose : 
                print('Start with the restarts in parallel')
                t0 = time.time()
            sess.run(init_op)
            sess.run(shuffle_iterator.initializer)
           
            for step in range(self.max_iters):
                if self.debug: t2 = time.time()
                sess.run(train)
                if self.debug:
                    t3 = time.time()
                    print(step,"durations :",str(t3-t2))
            # TODO a finir ici lalala
            if class_indice==-1:
                loss_value = np.zeros((self.paral_number_W,self.num_classes),dtype=np.float32)
            else:
                loss_value = np.zeros((self.paral_number_W,),dtype=np.float32)
            sess.run(iterator_batch.initializer)
            while True:
                try:
                    loss_value += sess.run(loss_batch)
                    break
                except tf.errors.OutOfRangeError:
                    break
            argmin = np.argmin(loss_value,axis=0)
            W_tmp=sess.run(W)
            b_tmp=sess.run(b)
            if self.class_indice==-1:
                W_best = W_tmp[argmin,np.arange(self.num_classes),:]
                b_best = b_tmp[argmin,np.arange(self.num_classes),:,:]
            else:
                W_best = W_tmp[argmin,:]
                b_best = b_tmp[argmin]
            if self.verbose : 
                print("bestloss",np.min(loss_value,axis=0))
                t1 = time.time()
                print("durations :",str(t1-t0),' s')
            
        else:
            for essai in range(self.restarts+1): #on fait 5 essais et on garde la meilleur loss
                if self.verbose : 
                    t0 = time.time()
                    print("essai",essai)
                # To do need to reinitialiszed : 
                sess.run(init_op)
                if (essai==0):
                    sess.run(shuffle_iterator.initializer)
                if (self.init_by_mean == 'First' and  (essai==0)) or (self.init_by_mean == 'All'):
                    if self.debug: t2 = time.time()
                    sess.run(init_op_onMean)
                    W_init,b_init = self.fit_SVM_onMean(sess,W_onMean,b_onMean,train_SVM_onMean)
                    if self.debug:
                        t3 = time.time()
                        print("Initialisation by classif by mean durations :",str(t3-t2))
                    sess.run(assign_op, {placeholder: W_init})
                    sess.run(assign_op_b, {placeholder_b: b_init})
    
                for step in range(self.max_iters):
                    if self.debug: t2 = time.time()
                    sess.run(train)
                    if self.debug:
                        t3 = time.time()
                        print(step,"durations :",str(t3-t2))
    
                if class_indice==-1:
                    loss_value = np.zeros((self.num_classes,),dtype=np.float32)
                else:
                    loss_value = 0.
                sess.run(iterator_batch.initializer)
                while True:
                    try:
                        loss_value += sess.run(loss_batch)
                        break
                    except tf.errors.OutOfRangeError:
                        break
                if class_indice==-1:
                    W_tmp=sess.run(W)
                    b_tmp=sess.run(b)
                    if (essai==0):
                        W_best=W_tmp
                        b_best=b_tmp
                        bestloss= loss_value
                    else:
                        for i in range(self.num_classes):
                            if(loss_value[i] < bestloss[i]):
                                bestloss[i] = loss_value[i]
                                W_best[i,:]=W_tmp[i,:]
                                b_best[i]=b_tmp[i]
                    if self.verbose : print("bestloss",bestloss) # La loss est minimale est -2 
                else:
                    if (essai==0) | (loss_value<bestloss): 
                        W_best=sess.run(W)
                        b_best=sess.run(b)
                        bestloss= loss_value
                        if self.verbose : print("bestloss",bestloss) # La loss est minimale est -2 
                if self.verbose:
                    t1 = time.time()
                    print("durations :",str(t1-t0),' s')

        saver = tf.train.Saver()
        X_= tf.identity(X_, name="X")
        if self.norm=='L2':
            X_ = tf.nn.l2_normalize(X_,axis=-1, name="L2norm")
        elif self.norm=='STD_all':
            X_ = tf.divide(tf.add( X_,-mean_train_set), std_train_set, name="STD")
        y_ = tf.identity(y_, name="y")
        if self.with_scores:
            print('Here')
            scores_ = tf.identity(scores_,name="scores")
            print(scores_)
        if class_indice==-1:
            Prod_best= tf.add(tf.reduce_sum(tf.multiply(tf.reshape(W_best,(self.num_classes,1,1,self.num_features)),X_),axis=3),b_best,name='Prod')
        else:
            Prod_best= tf.add(tf.reduce_sum(tf.multiply(W_best,X_),axis=2),b_best,name='Prod')
        if self.with_scores: Prod_score=tf.multiply(Prod_best,tf.add(scores_,self.epsilon),name='ProdScore')
        export_dir = ('/').join(data_path.split('/')[:-1])
        export_dir += '/MILSVM/' + str(time.time())
        name_model = export_dir + '/model'
        saver.save(sess,name_model)
        
        sess.close()
        if self.verbose : print("Return MILSVM weights")
        return(name_model) 

    def blabla(self,data_path):
        # TODO could be possible to select all the positive element of the positive 
        # group for the learning of the next step
        
        train_dataset_w_rois = tf.data.TFRecordDataset(data_path)
        train_dataset_w_rois = train_dataset_w_rois.map(self.parser_w_rois,num_parallel_calls=4)
        dataset_batch = train_dataset_w_rois.batch(self.mini_batch_size)
        dataset_batch = dataset_batch.prefetch(1)
        iterator = dataset_batch.make_initializable_iterator()
        next_element = iterator.get_next()
        
        # Definition des operations pour determiner les elements max de chaque cas
        # positif
        Prod_best=tf.reduce_sum(tf.multiply(W_best,X_),axis=2)+b_best
        mei = tf.argmax(Prod_best,axis=1)
        score_mei = tf.reduce_max(Prod_best,axis=1)
        
        # Create a tfRecords for the output
        data_path_output_tab= data_path.split('.')
        data_path_output = ('.').join(data_path.split('.')[:-1]) +'_mei_c'+str(class_indice)+'.'+data_path_output_tab[-1]

        writer = tf.python_io.TFRecordWriter(data_path_output)
        if self.verbose : print("Starting of best regions determination")
        # TODO trouver une maniere d acceler cela 
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator.initializer)
            while True:
                try:
                    fc7,rois, label,name_img = sess.run(next_element)
                    mei_values,score_mei_values = sess.run([mei,score_mei], feed_dict={X_: fc7, y_: label})
                    for i in range(len(mei_values)):
                        fc7_value = fc7[i,:,:]
                        mei_value = mei_values[i]
                        score_mei_value = score_mei_values[i]
                        if label[i] ==1.:
                            fc7_selected = np.zeros_like(fc7_value)
                            fc7_selected[mei_value,:] = fc7_value[mei_value,:]
                        else:
                            fc7_selected = fc7_value
                        features=tf.train.Features(feature={
                            'score_mei': _floats_feature(score_mei_value),
                            'mei': _int64_feature(mei_value),
                            'rois': _floats_feature(rois[i,:,:]),
                            'fc7': _floats_feature(fc7_value),
                            'fc7_selected': _floats_feature(fc7_selected),
                            'label' : _floats_feature(label[i]),
                            'name_img' : _bytes_feature(name_img[i])})
                        example = tf.train.Example(features=features)
                        writer.write(example.SerializeToString())
                except tf.errors.OutOfRangeError:
                        break
            writer.close()
        if self.verbose : print("End of best regions determination")
        # Train an hinge loss with SGD for the final classifier
        # TODO add a validation step for the parameter
        
        w_estimator = ''
        train_dataset_sgdc = tf.data.TFRecordDataset(data_path_output)
        train_dataset_sgdc = train_dataset_sgdc.map(self.parser_w_mei_reduce,
                                                    num_parallel_calls=4)
        if shuffle:
            dataset_sgdc = train_dataset_sgdc.shuffle(buffer_size=self.buffer_size)
        else:
            dataset_sgdc = train_dataset_sgdc
        dataset_sgdc = dataset_sgdc.batch(self.mini_batch_size)
        dataset_sgdc = dataset_sgdc.cache()
        dataset_sgdc = dataset_sgdc.repeat()
        dataset_sgdc = dataset_sgdc.prefetch(1)
        shuffle_iterator_sgdc = dataset_sgdc.make_initializable_iterator() 
        X_shuffle_sgdc,y_shuffle_sgdc = shuffle_iterator_sgdc.get_next()
        
        X_shuffle_sgdc = tf.identity(X_shuffle_sgdc, name="X")
        y_shuffle_sgdc = tf.identity(y_shuffle_sgdc, name="y")
        
        
        if w_estimator=='tfSVM':
            
            
            def input_fn(data_file, num_epochs, shuffle, batch_size,_SHUFFLE_BUFFER):
                # https://www.tensorflow.org/tutorials/wide
                  """Generate an input function for the Estimator."""
                  train_dataset_sgdc = tf.data.TFRecordDataset(data_file)
                  train_dataset_sgdc = train_dataset_sgdc.map(parser_w_mei_reduce,
                                                            num_parallel_calls=4)
                  if shuffle:
                      dataset_sgdc = train_dataset_sgdc.shuffle(buffer_size=_SHUFFLE_BUFFER)
                  else:
                      dataset_sgdc = train_dataset_sgdc
                  dataset_sgdc = dataset_sgdc.batch(batch_size)
                  dataset_sgdc = dataset_sgdc.cache()
                  dataset_sgdc = dataset_sgdc.repeat(num_epochs)
                  dataset_sgdc = dataset_sgdc.prefetch(1)
                  shuffle_iterator_sgdc = dataset_sgdc.make_one_shot_iterator() 
                  features, labels = shuffle_iterator_sgdc.get_next()
                  return features, labels
            
            
            
            # Use of a TF estimator to compute it 
            # Feature columns describe how to use the input.
            my_feature_columns = []
            for key in range(self.num_features):
                my_feature_columns.append(tf.feature_column.numeric_column(key=key))
            classifier = tf.estimator.LinearClassifier(
                    feature_columns=my_feature_columns,
                    optimizer=tf.train.FtrlOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=1.0,
                        l2_regularization_strength=1.0))
            if self.verbose : print("Start SVM LinearClassifier training")
            if self.debug: t2 = time.time()
#            batch0,batch1 = sess.run(shuffle_next_element_sgdc)
            classifier.train(input_fn=input_fn(data_path_output,self.max_iters_sgdc, True, self.mini_batch_size,self.buffer_size))
            if self.debug:
                t3 = time.time()
                print("All training duration :",str(t3-t2))
            
            
        else:
            W_sgdc=tf.Variable(tf.random_normal([self.num_features], stddev=1.),name="weights_sgdc")
            b_sgdc=tf.Variable(tf.random_normal([1], stddev=1.), name="bias_sgdc")
            if tf.__version__ >= '1.8':
                normalize_W_sgdc = W_sgdc.assign(tf.nn.l2_normalize(W_sgdc,axis=0)) 
            else:
                normalize_W_sgdc = W_sgdc.assign(tf.nn.l2_normalize(W_sgdc,dim=0)) 
            # TODO Need to check it in the future
            # TODO need to weight with the number of positive and negative exemple
            W_sgdc=tf.reshape(W_sgdc,(1,1,self.num_features))
            logits = tf.add(tf.reduce_sum(tf.multiply(W_sgdc,X_shuffle_sgdc),axis=2),
                            b_sgdc,name='logits')
            logits_squeeze=tf.squeeze(logits) # Dim size_batch * 300
            labels_plus_minus_1 = tf.add(tf.multiply(2.,tf.squeeze(y_shuffle_sgdc)),minus_1) # Dim size_batch
            hinge_loss = tf.losses.hinge_loss(labels_plus_minus_1,logits_squeeze)
            
            train = optimizer.minimize(hinge_loss)   
            if self.verbose : print("Start SGDC training")
            saver = tf.train.Saver()
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(shuffle_iterator_sgdc.initializer)
            sess.run(normalize_W_sgdc)
            # InternalError: Dst tensor is not initialized. arrive when an other program is running with tensorflow ! 
            for step in range(self.max_iters_sgdc):
                if self.debug: t2 = time.time()
    #            batch0,batch1 = sess.run(shuffle_next_element_sgdc)
                sess.run(train)
#                hinge_loss_value,logits_value,labels_plus_minus_1_value = sess.run(hinge_loss,logits,labels_plus_minus_1)
#                print(hinge_loss_value,logits_value,labels_plus_minus_1_value)
                if self.debug:
                    t3 = time.time()
                    print(step,"durations :",str(t3-t2))
            
        export_dir = ('/').join(data_path.split('/')[:-1])
        export_dir += '/FinalClassifierModels' + str(time.time())
        name_model = export_dir + '/model'
        saver.save(sess,name_model)
        # http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        
        sess.close()
        if self.verbose : print("End SGDC training")
        return(name_model)
        
        
    def fit_Stocha(self,bags,bags_label,shuffle=True):
        """
        bags_label = 1 or 0 
        """
        LR = self.LR # Regularisation loss
        optimArg = self.optimArg
        N,k,d = bags.shape
        
        
        mini_batch_size = self.mini_batch_size
        n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
        X_=tf.placeholder(tf.float32, shape=[None,k, d], name='X_')
        y_=tf.placeholder(tf.float32, shape=[None], name='y_')
        

        
        W=tf.Variable(tf.random_normal([d], stddev=1.),name="weights")
        b=tf.Variable(tf.random_normal([1], stddev=1.), name="bias")
        
        if tf.__version__ >= '1.8':
            normalize_W = W.assign(tf.nn.l2_normalize(W,axis = 0))
        else:
            normalize_W = W.assign(tf.nn.l2_normalize(W,dim = 0))
        W=tf.reshape(W,(1,1,d))
        Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)+b
        Max=tf.reduce_max(Prod,axis=1) # We take the max because we have at least one element of the bag that is positive
        np_pos = np.sum(bags_label)
        np_neg = -np.sum(bags_label-1)
#        np_pos = tf.reduce_sum(y_)
#        np_neg = tf.abs(tf.reduce_sum(tf.add(y_-1)))
        weights_bags_ratio = -tf.divide(y_,(np_pos)) + tf.divide(-tf.add(y_,-1),(np_neg)) # Need to add 1 to avoid the case 
        Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio)) # Sum on all the positive exemples 
        loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.multiply(W,W))))
        
        if(self.Optimizer == 'GradientDescent'):
            optimizer = tf.train.GradientDescentOptimizer(LR) 
        if(self.Optimizer == 'Momentum'):
            optimizer = tf.train.MomentumOptimizer(optimArg['learning_rate'],optimArg['momentum']) 
        if(self.Optimizer == 'Adam'):
            optimizer = tf.train.AdamOptimizer(LR) 
        if(self.Optimizer == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(LR) 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        train = optimizer.minimize(loss)
        init_op = tf.global_variables_initializer()
        
        bestloss=0
        for essai in range(self.restarts+1): #on fait 5 essais et on garde la meilleur loss
            if self.verbose : print("essai",essai)
            sess.run(init_op)
            sess.run(normalize_W)
            indices = np.arange(bags.shape[0])
            if shuffle: # Stochastics case
                np.random.shuffle(indices) # Do we have to shuffle every time
                
                
            # InternalError: Dst tensor is not initialized. arrive when an other program is running with tensorflow ! 
            for step in range(self.max_iters):
                i_batch = (step % n_batch)*mini_batch_size
                #print(i_batch,i_batch+mini_batch_size)
                excerpt = indices[i_batch:i_batch+mini_batch_size]
                batch0 = bags[excerpt,:,:].astype('float32')
                batch1 = bags_label[excerpt].astype('float32')
                sess.run(train, feed_dict={X_: batch0, y_: batch1})
                    #if step % 200 == 0 and self.verbose: print(step, sess.run([Prod,Max,weights_bags_ratio,Tan,loss], feed_dict={X_: batch0, y_: batch1})) 

            loss_value = 0
            for step in range(n_batch):
                 i_batch = (step % n_batch)*mini_batch_size
                 batch0 = bags[i_batch:i_batch+mini_batch_size].astype('float32')
                 batch1 = bags_label[i_batch:i_batch+mini_batch_size].astype('float32')
                 loss_value += sess.run(loss, feed_dict={X_: batch0, y_: batch1})
                
            if (essai==0) | (loss_value<bestloss):
                W_best=sess.run(W)
                b_best=sess.run(b)
                bestloss= loss_value
                if self.verbose : print("bestloss",bestloss) # La loss est minimale est -2 
                dicobest={W:W_best,b:b_best} 
            
#        sess.run(tf.assign(W,W_best))
#        sess.run(tf.assign(b,b_best))
        
        Pos_elt = tf.constant(bags[np.where(bags_label==1)],dtype=tft)
        Prod1 = tf.reduce_sum(tf.multiply(W_best,Pos_elt),axis=2)+b_best
        sor1=sess.run([Prod1])
        #sor2=sess.run([Prod2])
        sess.close()
        tf.reset_default_graph()
        
        pr1=sor1[0]
        mei=pr1.argmax(axis=1) # Indexes of the element that below to the right class normally
        score_mei=pr1.max(axis=1) # Indexes of the element that below to the right class normally
        self.PositiveExScoreAll = pr1
        self.PositiveRegions = mei
        self.PositiveRegionsScore = score_mei
          
#        pr2=sor2[0]
#        mei=pr2.argmax(axis=1) # Indexes of the element that below to the right class normally
#        score_mei=pr2.max(axis=1) # Indexes of the element that below to the right class normally
#        self.NegativeRegions = mei
#        self.NegativeRegionsScore = score_mei
        
        return(None)
        
    def fit(self,data_pos,data_neg):
        """
        Cet version utilise les optimizeurs de tensorflow mais travail sur des 
        petits batchs de donnees passes en argument et non sur des tf_records
        @param data_pos : a numpy array of the positive bag of size number of positive bag
            * number of max element in one baf * dim features
        @param data_neg : a numpy array of the positive bag of size number of negative bag
            * number of max element in one baf * dim features
        """

        LR = self.LR # Regularisation loss
        optimArg = self.optimArg
        Stocha = False
        np1,k,n = data_pos.shape
        np2,_,_ = data_neg.shape
        if self.verbose :print("Shapes :",np1,k,n,np2)
        X1=tf.constant(data_pos,dtype=tft)
        X2=tf.constant(data_neg,dtype=tft)
        W=tf.Variable(tf.random_normal([n], stddev=1.),name="weights")
        b=tf.Variable(tf.random_normal([1], stddev=1.), name="biases")
        if tf.__version__ >= '1.8':
            normalize_W = W.assign(tf.nn.l2_normalize(W,axis = 0))
        else:
            normalize_W = W.assign(tf.nn.l2_normalize(W,dim = 0))
        W1=tf.reshape(W,(1,1,n))
        Prod1=tf.reduce_sum(tf.multiply(W1,X1),axis=2)+b
        Max1=tf.reduce_max(Prod1,axis=1) # We take the max because we have at least one element of the bag that is positive
        Tan1=tf.divide(tf.reduce_sum(tf.tanh(Max1)),np1) # Sum on all the positive exemples 
        Prod2=tf.reduce_sum(tf.multiply(W1,X2),axis=2)+b
        if self.symway :
            Max2=tf.reduce_max(Prod2,axis=1) # TODO attention tu as mis min au lieu de max ici
        else:
            Max2=tf.reduce_mean(Prod2,axis=1) # TODO Il faut que tu check cela avec Said quand meme
        Tan2=tf.divide(tf.reduce_sum(tf.tanh(Max2)),np2)
        loss= tf.add(tf.add(-Tan1,Tan2),tf.multiply(self.C,tf.reduce_sum(tf.multiply(W,W))))  #car on fait une minimization !!! 
        
        if(self.Optimizer == 'GradientDescent'):
            optimizer = tf.train.GradientDescentOptimizer(LR) 
        if(self.Optimizer == 'Momentum'):
            optimizer = tf.train.MomentumOptimizer(optimArg['learning_rate'],optimArg['momentum']) 
        if(self.Optimizer == 'Adam'):
            optimizer = tf.train.AdamOptimizer(LR) 
        if(self.Optimizer == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(LR) 
        #print("Grad defined")
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        train = optimizer.minimize(loss)
        init_op = tf.global_variables_initializer()
        #sess.graph.finalize() # To test if the graph is correct
        bestloss=0
        for essai in range(self.restarts+1): #on fait 5 essais et on garde la meilleur loss
            if self.verbose : print("essai",essai)
#            sess.run(tf.global_variables_initializer())
            sess.run(init_op)
            sess.run(normalize_W)
#            [Tan1value,Tan2value,lossvalue] = sess.run([Tan1,Tan2,loss])
#            print(Tan1value,Tan2value,lossvalue)
#            W_eval=sess.run(W)
#            b_eval=sess.run(b)
            for i in range(self.max_iters): 
                 sess.run(train)
#            [Tan1value,Tan2value,lossvalue] = sess.run([Tan1,Tan2,loss])
#            print(Tan1value,Tan2value,lossvalue)
            loss_value = sess.run(loss)
            #if self.verbose : print("loss_value",loss_value) 
            if (essai==0) | (loss_value<bestloss):
                W_best=sess.run(W)
                b_best=sess.run(b)
                bestloss= loss_value
                if self.verbose : print("bestloss",bestloss) # La loss est maximale a 2 
                dicobest={W:W_best,b:b_best} 
            
        sess.run(W.assign(W_best))
        sess.run(b.assign(b_best))
        sor1=sess.run([Prod1])
        sor2=sess.run([Prod2])
        sess.close()
        tf.reset_default_graph()
        
        pr1=sor1[0]
        mei=pr1.argmax(axis=1) # Indexes of the element that below to the right class normally
        score_mei=pr1.max(axis=1) # Indexes of the element that below to the right class normally
        self.PositiveExScoreAll = pr1
        self.PositiveRegions = mei
        self.PositiveRegionsScore = score_mei
          
        pr2=sor2[0]
        mei=pr2.argmax(axis=1) # Indexes of the element that below to the right class normally
        score_mei=pr2.max(axis=1) # Indexes of the element that below to the right class normally
        self.NegativeRegions = mei
        self.NegativeRegionsScore = score_mei
        
        if (self.final_clf is None) or (self.final_clf == 'None'):
            # We don t return a final classifier
             #if self.verbose : print("We don't return a final classifier !!!")
             return(None)
        else:
            full_positives =  np.zeros((np1,n))
            if self.all_notpos_inNeg:
                if self.verbose : print("All element that are not the positive example are considered as negative")
                full_neg = np.zeros((np1*(k-1),n))
                for i in range(np1):
                     index = mei[i]
                     data = data_pos[i,index,:] 
                     full_positives[i,:] = data     
                     data = np.concatenate([data_pos[i,0:index,:],data_pos[i,index:-1,:]])
                     full_neg[i*(k-1):(i+1)*(k-1),:] = data
                data_p2_reshaped =  np.reshape(data_neg,(np2*k,n))
                print(data_p2_reshaped.shape)
                print(full_neg.shape)
                full_neg_all = np.vstack([full_neg,data_p2_reshaped])
                print(full_neg_all.shape)
            else:     
                if self.verbose : print("All element that are not the positive example are not considered as negative,they are ignored")
                full_neg_all =  np.reshape(data_neg,(np2*k,n))
                for i in range(np1):
                     index = mei[i]
                     data = data_pos[i,index,:] 
                     full_positives[i,:] = data
                     
            X = np.vstack((full_positives,full_neg_all))
            y_pos = np.ones((np1,1))
            y_neg = np.zeros((len(full_neg_all),1))
            y = np.vstack((y_pos,y_neg)).ravel()
            
            if len(X) > 30000 and self.verbose and self.final_clf == 'LinearSVC':
                print("We advise you to use an online classification method as SGD")
            
            if self.verbose : 
                print("Shape of X",X.shape)
                print("number of positive examples :",len(y_pos),"number of negative example :",len(y_neg))
            if self.verbose : print("Retrain a new SVM")
                
            classifier = TrainClassif(X,y,clf='LinearSVC',gridSearch=self.gridSearch,n_jobs=self.n_jobs
                     ,C_finalSVM=self.C_finalSVM)
            if self.verbose :
                labels_test_predited = classifier.predict(X)
                print('Number of positive prediction :',np.sum(labels_test_predited))
                training_precision = precision_score(y,labels_test_predited)
                y_predict_confidence_score_classifier = classifier.decision_function(X)
                AP = average_precision_score(y,y_predict_confidence_score_classifier,average=None)
                print("Training precision of the final LinearSVC",training_precision,'training AP:',AP)
            return(classifier)

  
    def get_PositiveRegions(self):
        return(self.PositiveRegions.copy())
     
    def get_PositiveRegionsScore(self):
        return(self.PositiveRegionsScore.copy())
     
    def get_PositiveExScoreAll(self):
        return(self.PositiveExScoreAll.copy())
        
    def get_NegativeRegions(self):
        return(self.NegativeRegions.copy())
     
    def get_NegativeRegionsScore(self):
        return(self.NegativeRegionsScore.copy()) 
    
    def get_porportions(self):
        return(self.np_pos_value,self.np_neg_value)

#def SGDClassifier(features, labels, mode, params):
#    """Linear Classifier with hinge loss """
#    net = tf.feature_column.input_layer(features, params['feature_columns'])
#    for units in params['hidden_units']:
#        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
#
#    # Compute logits (1 per class).
#    logits = tf.layers.dense(net, params['n_classes'], activation=None)
#
#    # Compute predictions.
#    predicted_classes = tf.argmax(logits, 1)
#    if mode == tf.estimator.ModeKeys.PREDICT:
#        predictions = {
#            'class_ids': predicted_classes[:, tf.newaxis],
#            'probabilities': tf.nn.softmax(logits),
#            'logits': logits,
#        }
#        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
#
#    # Compute loss.
#    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#
#    # Compute evaluation metrics.
#    accuracy = tf.metrics.accuracy(labels=labels,
#                                   predictions=predicted_classes,
#                                   name='acc_op')
#    metrics = {'accuracy': accuracy}
#    tf.summary.scalar('accuracy', accuracy[1])
#
#    if mode == tf.estimator.ModeKeys.EVAL:
#        return tf.estimator.EstimatorSpec(
#            mode, loss=loss, eval_metric_ops=metrics)
#
#    # Create training op.
#    assert mode == tf.estimator.ModeKeys.TRAIN
#
#    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
#    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
#return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def TrainClassif(X,y,clf='LinearSVC',class_weight=None,gridSearch=True,n_jobs=-1,C_finalSVM=1):
    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.01,0.2,1.,2.,10.,100.]))
    param_grid = dict(C=cs)
    # TODO  class_weight='balanced' TODO add this parameter ! 
    if gridSearch:
        if clf == 'LinearSVC':
            clf = LinearSVC(penalty='l2',class_weight=class_weight, 
                            loss='squared_hinge',max_iter=1000,dual=True)
            param_grid = dict(C=cs)
        elif clf == 'defaultSGD':
            clf = SGDClassifier(max_iter=1000, tol=0.0001)
            param_grid = dict(alpha=cs)
        elif clf == 'SGDsquared_hinge':
            clf = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
            param_grid = dict(alpha=cs)
    
        classifier = GridSearchCV(clf, refit=True,
                                  scoring =make_scorer(average_precision_score,
                                                       needs_threshold=True),
                                  param_grid=param_grid,n_jobs=n_jobs)
    else:
        # ,class_weight='balanced'
        if clf == 'LinearSVC':
            classifier = LinearSVC(penalty='l2',class_weight=class_weight,
                                   loss='squared_hinge',max_iter=1000,dual=True,C=C_finalSVM)
        elif clf == 'defaultSGD':
            classifier = SGDClassifier()
        elif clf == 'SGDsquared_hinge':
            classifier = SGDClassifier(max_iter=1000, tol=0.0001,loss='squared_hinge')
    
    classifier.fit(X,y)
    
    return(classifier)
    
def test_Stocha():
    print('Start Test Selection')
    #%% Variables communes 
    n=2048
    k=30
    # la classe p1
    np1=201
    np2=4005
    
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
    
    print(data_p1.shape) #(200, 30, 2048)
    print(data_p2.shape) #(4000, 30, 2048)
    
    bags = np.vstack((data_p1,data_p2))
    y_pos = np.ones((len(data_p1),1))
    y_neg = np.zeros((len(data_p2),1))
    labels = np.vstack((y_pos,y_neg)).ravel()
    
    max_iters_wt_minibatch = 300
    mini_batch_size = 4000
    N,k,d = bags.shape
    n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
    max_iters = n_batch*max_iters_wt_minibatch
    classifier= tf_MILSVM(LR=0.02,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=max_iters,
                 symway=True,final_clf=None,verbose=True,Optimizer='Adam',
                 mini_batch_size=mini_batch_size)
    
#
#    t0=time.time()
#    classifier.fit(data_p1,data_p2)
#    t1=time.time()
#    temps = t1 - t0
#    mei = classifier.get_PositiveRegions()
#    scor=0
#    for i in range(np1):
#        if mei[i]==choisi[i]:
#            scor+=1
#    score = scor/np1
#    print('score final sur la selection : %.1f%%' % ( 100 * score))
#    print('Temps ecoulé :',temps)
    
    t0=time.time()
    classifier.fit_Stocha(bags,labels,shuffle=False)
    t1=time.time()
    temps = t1 - t0
    mei = classifier.get_PositiveRegions()
    scor=0
    for i in range(np1):
        if mei[i]==choisi[i]:
            scor+=1
    score = scor/np1
    print('score final sur la selection : %.1f%%' % (100 * score))
    print('Temps ecoulé :',temps)
#    score final sur la selection : 99.5%
#    Temps ecoulé : 3731.9643445014954 # For Gradient descent
    
    t0=time.time()
    classifier.fit_Stocha(bags,labels,shuffle=True)
    t1=time.time()
    temps = t1 - t0
    mei = classifier.get_PositiveRegions()
    scor=0
    for i in range(np1):
        if mei[i]==choisi[i]:
            scor+=1
    score = scor/np1
    print('score final sur la selection : %.1f%%' % (100 * score))
    print('Temps ecoulé :',temps)
    
#    score final sur la selection : 100.0%
#    Temps ecoulé : 3744.938458919525 : For gradient descent


def test_selection():
    print('Start Test Selection')
    #%% Variables communes 
    n=2048
    k=30
    # la classe p1
    np1=200
    np2=4000
    
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
    
    print(data_p1.shape) #(200, 30, 2048)
    print(data_p2.shape) #(4000, 30, 2048)
    
    classifiers = {}
    
    classifiers['classifMIL'] = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,all_notpos_inNeg=True,final_clf=None,verbose=True)
    classifiers['GradientDescent'] = tf_MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,final_clf=None,verbose=True,Optimizer='GradientDescent')
    classifiers['Adam'] = tf_MILSVM(LR=0.005,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,final_clf=None,verbose=True,Optimizer='Adam')
    classifiers['Momentum'] = tf_MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,final_clf=None,verbose=True,Optimizer='Momentum',
                 optimArg={'learning_rate':0.01,'momentum': 0.01})
    classifiers['Adagrad'] = tf_MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,final_clf=None,verbose=True,Optimizer='Adagrad')

    score = {}
    temps = {}
    for algorithm, classifier in classifiers.items():
        print(algorithm)
        t0=time.time()
        classifier.fit(data_p1,data_p2)
        t1=time.time()
        temps[algorithm] = t1 - t0
        mei = classifier.get_PositiveRegions()
        scor=0
        for i in range(np1):
            if mei[i]==choisi[i]:
                scor+=1
        score[algorithm] = scor/np1

    for algorithm, scor in score.items():
        print('\n%s score final sur la selection : %.1f%%' % (algorithm, 100 * scor))
        print('Temps ecoulé :',temps[algorithm])
#        
#        classifMIL score final sur la selection : 98.5%
#Temps ecoulé : 99.69157981872559
#
#GradientDescent score final sur la selection : 99.0%
#Temps ecoulé : 119.68762302398682
#
#Adam score final sur la selection : 98.0%
#Temps ecoulé : 120.27855134010315
#
#Momentum score final sur la selection : 96.0%
#Temps ecoulé : 119.71065711975098
#
#Adagrad score final sur la selection : 98.0%
#Temps ecoulé : 120.94558811187744
    
def test_classif():
    print('Start Test')
    #%% Variables communes 
    n=2048
    k=30
    # la classe p1
    np1=200
    np2=4000
    
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
    
    print(data_p1.shape) #(200, 30, 2048)
    print(data_p2.shape) #(4000, 30, 2048)
    
    classifMIL = MILSVM(LR=0.01,C=1.0,C_finalSVM=1.0,restarts=10, max_iters=300,
                 symway=True,all_notpos_inNeg=True,final_clf='LinearSVC',verbose=True)
    classifier = classifMIL.fit(data_p1,data_p2)
    
def test():
    
    #%% Variables communes 
    n=2048
    k=30
    # la classe p1
    np1=200
    np2=4000
    
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
    
    print(data_p1.shape) #(200, 30, 2048)
    print(data_p2.shape) #(4000, 30, 2048)
    #%%   Le graphe de calcul pour le cas où les exemples positifs et negatifs
    # sont traites de maniere symetrique pour traiter les exemples de maniere non 
    # symetrique il faut remplacer le reduce_max par reduce_sum
    for m in range(1): #juste pour refaire m fois l'experience
        nb_essai = 5
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
        #Max2=tf.reduce_sum(Prod2,axis=1)
        Tan2=tf.reduce_sum(tf.tanh(Max2))/np2
        
        
        loss=Tan1-Tan2-tf.reduce_sum(W*W)
        
        gr=tf.gradients(loss,[W,b])
        
        sess=tf.InteractiveSession()
        
        bestloss=-1
        for essai in range(nb_essai): #on fait 5 essais et on garde la meilleur loss
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
    
    
#%% Code pour tf_MILSVM autre facon    
    
    # #            tab_W = [[] for _ in range(self.num_classes)]
#            tab_W_r = [[] for _ in range(self.num_classes)]
#            tab_b = [[] for _ in range(self.num_classes)]
#            tab_loss = [[] for _ in range(self.num_classes)]
#            tab_normalize_W =  [[] for _ in range(self.num_classes)]
#            for i in range(self.num_classes):
##                tab_W[i]=tf.slice(W,[i,0],[1,self.num_features])
##                tab_b[i]=tf.slice(b,[i],[1])
#                tab_W[i]=tf.Variable(tf.random_normal([self.num_features], stddev=1.))
#                tab_b[i]=tf.Variable(tf.random_normal([1], stddev=1.))
#                if tf.__version__ >= '1.8':
#                    tab_normalize_W[i] = tab_W[i].assign(tf.nn.l2_normalize(tab_W[i],axis=0)) 
#                else:
#                    tab_normalize_W[i] = tab_W[i].assign(tf.nn.l2_normalize(tab_W[i],dim=0)) 
#                tab_W_r[i]=tf.reshape(tab_W[i],(1,1,self.num_features))
#                print(tab_W[i])
#                print(tab_b[i])
#                Prod=tf.add(tf.reduce_sum(tf.multiply(tab_W_r[i],X_),axis=2),tab_b[i])
#                Max=tf.reduce_max(Prod,axis=1) # We take the max because we have at least one element of the bag that is positive
#                weights_bags_ratio = -tf.divide(y_,np_pos_value[i]) + tf.divide(-tf.add(y_,-1),np_neg_value[i]) # Need to add 1 to avoid the case 
#                Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio)) # Sum on all the positive exemples 
#                print(Tan)
#                tab_loss[i]= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.multiply(tab_W_r[i],tab_W_r[i]))))
#                print(tab_loss[i])
#                Prod_batch=tf.add(tf.reduce_sum(tf.multiply(tab_W_r[i],X_batch),axis=2),tab_b[i])
#                Max_batch=tf.reduce_max(Prod_batch,axis=1) # We take the max because we have at least one element of the bag that is positive
#                if self.is_betweenMinus1and1:
#                    weights_bags_ratio_batch = -tf.divide(tf.add(label_batch,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(label_batch,-1.),tf.multiply(-2.,np_neg_value))
#                    # Need to add 1 to avoid the case 
#                    # The wieght are negative for the positive exemple and positive for the negative ones !!!
#                else:
#                    weights_bags_ratio_batch = -tf.divide(label_batch,np_pos_value[i]) + tf.divide(-tf.add(label_batch,-1),np_neg_value[i]) # Need to add 1 to avoid the case 
#                
#                Tan_batch= tf.reduce_sum(tf.multiply(tf.tanh(Max_batch),weights_bags_ratio_batch)) # Sum on all the positive exemples 
#                loss_batch= tf.add(Tan_batch,tf.multiply(self.C,tf.reduce_sum(tf.multiply(tab_W_r[i],tab_W_r[i])))
    
    
if __name__ == '__main__':
#    test()
#    test_classif()
#    test_selection()
    test_Stocha()