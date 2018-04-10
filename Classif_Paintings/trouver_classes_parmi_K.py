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
#from misvm.util import partition, BagSplitter, spdiag, rand_convex, slices
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score
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
        if self.verbose :print("Shapes :",np1,k,n,np2)
        X1=tf.constant(data_pos,dtype=tft)
        X2=tf.constant(data_neg,dtype=tft)
        W=tf.placeholder(tft,[n])
        b=tf.placeholder(tft,[1])

        W1=tf.reshape(W,(1,1,n))
        
        Prod1=tf.reduce_sum(tf.multiply(W1,X1),axis=2)+b
        Max1=tf.reduce_max(Prod1,axis=1) # We take the max because we have at least one element of the bag that is positive
        Tan1=tf.reduce_sum(tf.tanh(Max1))/np1 # Sum on all the positive exemples 
#        Tan1=tf.reduce_sum(Max1)/np1 # Sum on all the positive exemples 
        
        Prod2=tf.reduce_sum(tf.multiply(W1,X2),axis=2)+b
        if self.symway :
            Max2=tf.reduce_max(Prod2,axis=1) # TODO attention tu as mis min au lieu de max ici
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
                  optimArg=None,mini_batch_size=200):
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
        self.Optimizer = Optimizer
        self.optimArg =  optimArg# GradientDescent, 
        self.mini_batch_size = mini_batch_size
     
    def fit_w_CV(self,data_pos,data_neg):
        kf = KFold(n_splits=3) # Define the split - into 2 folds 
        kf.get_n_splits(data_pos) # returns the number of splitting iterations in the cross-validator
        return(0)
        
    def fit_LatentSVM_tfrecords(self,data_path,class_indice,shuffle=True):
        """" This function run per batch on the tfrecords data folder """
        # From http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
        feature={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': _floats_feature(rois),
                    'roi_scores':tf.VarLenFeature(tf.float32),
                    'fc7': tf.VarLenFeature(tf.float32),
                    'label' : tf.FixedLenFeature([],tf.string),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        fc7 = tf.decode_raw(features['fc7'], tf.float32)
        
        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        label = tf.slice(label,class_indice,1)
        num_regions = tf.cast(features['num_regions'], tf.int64)
        num_features = tf.cast(features['num_features'], tf.int64)
        # Reshape image data into the original shape
        fc7 = tf.reshape(fc7, [-1,num_features])
        
        # Any preprocessing here ...
        
        # Creates batches by randomly shuffling tensors
        fc7s, labels = tf.train.shuffle_batch([fc7, label], batch_size=batch_size, capacity=30, num_threads=4, min_after_dequeue=10)
            
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
       
#        LR = self.LR # Regularisation loss
#        optimArg = self.optimArg
#        #k = 
#        N,k,d = bags.shape
#        
#        
#        mini_batch_size = self.mini_batch_size
#        n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
#        X_=tf.placeholder(tf.float32, shape=[None,k, d], name='X_')
#        y_=tf.placeholder(tf.float32, shape=[None], name='y_')
#        
#
#        
#        W=tf.Variable(tf.random_normal([d], stddev=1.),name="weights")
#        b=tf.Variable(tf.random_normal([1], stddev=1.), name="bias")
#        
#        normalize_W = W.assign(tf.nn.l2_normalize(W,dim = 0))
#        W=tf.reshape(W,(1,1,d))
#        Prod=tf.reduce_sum(tf.multiply(W,X_),axis=2)+b
#        Max=tf.reduce_max(Prod,axis=1) # We take the max because we have at least one element of the bag that is positive
#        np_pos = np.sum(bags_label)
#        np_neg = -np.sum(bags_label-1)
#        weights_bags_ratio = -tf.divide(y_,(np_pos)) + tf.divide(-tf.add(y_,-1),(np_neg)) # Need to add 1 to avoid the case 
#        Tan= tf.reduce_sum(tf.multiply(tf.tanh(Max),weights_bags_ratio)) # Sum on all the positive exemples 
#        loss= tf.add(Tan,tf.multiply(self.C,tf.reduce_sum(tf.multiply(W,W))))
#        
#        if(self.Optimizer == 'GradientDescent'):
#            optimizer = tf.train.GradientDescentOptimizer(LR) 
#        if(self.Optimizer == 'Momentum'):
#            optimizer = tf.train.MomentumOptimizer(optimArg['learning_rate'],optimArg['momentum']) 
#        if(self.Optimizer == 'Adam'):
#            optimizer = tf.train.AdamOptimizer(LR) 
#        if(self.Optimizer == 'Adagrad'):
#            optimizer = tf.train.AdagradOptimizer(LR) 
#            
#        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = True
#        sess=tf.Session(config=config)
#        train = optimizer.minimize(loss)
##        init_op = tf.global_variables_initializer()
#        
#        sess.run(init_op)
#        # Create a coordinator and run all QueueRunner objects
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(coord=coord)
#        
#        bestloss=0
#        for essai in range(self.restarts+1): #on fait 5 essais et on garde la meilleur loss
#            if self.verbose : print("essai",essai)
#            # To do need to reinitialiszed
#            sess.run(normalize_W)
#            indices = np.arange(bags.shape[0])
#            if shuffle: # Stochastics case
#                np.random.shuffle(indices) # Do we have to shuffle every time
#                
#                
#            # InternalError: Dst tensor is not initialized. arrive when an other program is running with tensorflow ! 
#            for step in range(self.max_iters):
#                i_batch = (step % n_batch)*mini_batch_size
#                #print(i_batch,i_batch+mini_batch_size)
#                excerpt = indices[i_batch:i_batch+mini_batch_size]
#                batch0 = bags[excerpt,:,:].astype('float32')
#                batch1 = bags_label[excerpt].astype('float32')
#                sess.run(train, feed_dict={X_: batch0, y_: batch1})
#                    #if step % 200 == 0 and self.verbose: print(step, sess.run([Prod,Max,weights_bags_ratio,Tan,loss], feed_dict={X_: batch0, y_: batch1})) 
#
#            loss_value = 0
#            for step in range(n_batch):
#                 i_batch = (step % n_batch)*mini_batch_size
#                 batch0 = bags[i_batch:i_batch+mini_batch_size].astype('float32')
#                 batch1 = bags_label[i_batch:i_batch+mini_batch_size].astype('float32')
#                 loss_value += sess.run(loss, feed_dict={X_: batch0, y_: batch1})
#                
#            if (essai==0) | (loss_value<bestloss):
#                W_best=sess.run(W)
#                b_best=sess.run(b)
#                bestloss= loss_value
#                if self.verbose : print("bestloss",bestloss) # La loss est minimale est -2 
#                dicobest={W:W_best,b:b_best} 
#            
##        sess.run(tf.assign(W,W_best))
##        sess.run(tf.assign(b,b_best))
#        
#        Pos_elt = tf.constant(bags[np.where(bags_label==1)],dtype=tft)
#        Prod1 = tf.reduce_sum(tf.multiply(W_best,Pos_elt),axis=2)+b_best
#        sor1=sess.run([Prod1])
#        #sor2=sess.run([Prod2])
#        sess.close()
#        tf.reset_default_graph()
#        
#        pr1=sor1[0]
#        mei=pr1.argmax(axis=1) # Indexes of the element that below to the right class normally
#        score_mei=pr1.max(axis=1) # Indexes of the element that below to the right class normally
#        self.PositiveExScoreAll = pr1
#        self.PositiveRegions = mei
#        self.PositiveRegionsScore = score_mei
        
        
        return(0)
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
    
if __name__ == '__main__':
#    test()
#    test_classif()
#    test_selection()
    test_Stocha()