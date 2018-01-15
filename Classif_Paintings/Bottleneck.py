#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:39:43 2018

@author: gonthier
"""

import cv2
import tensorflow as tf
import resnet_152_keras
import inception_resnet_v2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score,label_ranking_average_precision_score,classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from Custom_Metrics import ranking_precision_score,VOCevalaction,computeAveragePrecision
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
import vgg19
import vgg16
import random
from numpy import linalg as LA
from utils import result_page_gen
import webbrowser
import base64
import os
import Classifier_Evaluation

depicts_depictsLabel = {'Q235113_verif':'angel'}

def compute_InceptionResNetv2_features(kind='1536D',database='Paintings',L2=True,augmentation=True):
    """
    Inception ResNet v2 take RGB image as input
    """
    
    concate = False
    
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        databasetxt = path_data + database + '.txt'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/299/'
    else:
        item_name = 'image'
        databasetxt = path_data + database + '.txt'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/299/'
    df_label = pd.read_csv(databasetxt,sep=",")
    if augmentation:
        N = 50
    else: 
        N=1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    checkpoint_file = '/media/HDD/models/inception_resnet_v2_2016_08_30.ckpt'
    name_img = df_label[item_name][0]
    i = 0
    itera = 1000
    
    if(kind=='output'):
        size_output = 1001   
    elif(kind=='1536D'):
        size_output = 1536
        
    name_pkl_values = path_data+'Values_' +'InceptionResNetv2_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +'InceptionResNetv2_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    if not(concate):
        features_resnet = None
    else:
        features_resnet = pickle.load(open(name_pkl_values, 'rb'))
        im_resnet = pickle.load(open(name_pkl_im, 'rb'))
    
    with tf.Graph().as_default():
      # The Inception networks expect the input image to have color channels scaled from [-1, 1]
      #Load the model
      sess = tf.Session()
      arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
      slim = tf.contrib.slim # The TF-Slim library provides common abstractions which enable users to define models quickly and concisely, while keeping the model architecture transparent and its hyperparameters explicit.
      
      input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3), name='input_image')
      scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
      scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
      scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
      
      with slim.arg_scope(arg_scope):
          if(kind=='output'):
              net, end_points = inception_resnet_v2.inception_resnet_v2(scaled_input_tensor, is_training=False)
          elif(kind=='1536D'):
              net, end_points = inception_resnet_v2.inception_resnet_v2_PreLogitsFlatten(scaled_input_tensor,is_training=False)
          saver = tf.train.Saver()
          saver.restore(sess, checkpoint_file)
 
          for i,name_img in  enumerate(df_label[item_name]):
            if not(concate) or not(name_img in  im_resnet):
                if i%itera==0:
                    print(i,name_img)
                
                if database=='VOC12' or database=='Paintings':
                    complet_name = path_to_img + name_img + '.jpg'
                    im = cv2.imread(complet_name)
                    im = im[:,:,[2,1,0]] # To shift from BGR to RGB
                    # normalily it is 299
                    if augmentation:
                        sizeIm = 335
                    else:
                        sizeIm = 299
                    if(im.shape[0] < im.shape[1]):
                        dim = (sizeIm, int(im.shape[1] * sizeIm / im.shape[0]),3)
                    else:
                        dim = (int(im.shape[0] * sizeIm / im.shape[1]),sizeIm,3)
                    tmp = (dim[1],dim[0])
                    dim = tmp
                    resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA) # INTER_AREA
                else:
                    name_sans_ext = os.path.splitext(name_img)[0]
                    complet_name = path_to_img +name_sans_ext + '.jpg'
                    im = Image.open(complet_name)
                    resized = np.array(im)
                    
                if augmentation:
                    list_im = np.stack(Classifier_Evaluation.get_Stretch_augmentation(resized,N=N,portion_size = (299, 299)))
                    out = sess.run(net, feed_dict={input_tensor: list_im})
                    out_average = np.mean(out,axis=0)
                    if L2:
                        out_norm = LA.norm(out_average) # L2 norm 
                        out= out_average/out_norm
                else:
                    # Crop the center
                    im = resized[int(resized.shape[0]/2 - 149):int(resized.shape[0]/2 +150),int(resized.shape[1]/2-149):int(resized.shape[1]/2+150),:]
                    im = im.reshape(-1,299,299,3)
                    net_values, _ = sess.run([net,end_points], feed_dict={input_tensor: im})
                    out = np.array(net_values[0])
                    if L2:
                        out_norm = LA.norm(out) 
                        out /= out_norm
                        
                if features_resnet is not None:
                    features_resnet = np.vstack((features_resnet,np.array(out)))
                    im_resnet += [name_img]
                else:
                    features_resnet =np.array(out)
                    im_resnet = [name_img]
               
    #print(features_resnet.head(5))
    with open(name_pkl_values, 'wb') as pkl:
        pickle.dump(features_resnet,pkl)
        
    with open(name_pkl_im, 'wb') as pkl:
        pickle.dump(im_resnet,pkl)
        
    return(features_resnet)
    
def TransferLearning_onRawFeatures(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings',L2=False,augmentation=False):
    """
    kindnetwork in  [InceptionResNetv2,ResNet152]
    """
    plot_fp_fn = True
    print('===>',kindnetwork,kind)
    # Multilabel classification assigns to each sample a set of target labels. 
    # This can be thought as predicting properties of a data-point that are not mutually exclusive
    if augmentation:
        N = 50
    else:
        N =1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    path_data = 'data/'
    databasetxt = path_data + database + '.txt'
    df = pd.read_csv(databasetxt,sep=",")
    depicts = ['Q235113_verif']
    colums_selected = ['image']+depicts
    df_reduc = df[colums_selected]
    
    name_pkl_values = path_data+'Values_' +'InceptionResNetv2_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +'InceptionResNetv2_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
   
    X = pd.DataFrame(pickle.load(open(name_pkl_values, 'rb')))
    name_im_order = pickle.load(open(name_pkl_im, 'rb'))
    name_im_order_df = pd.DataFrame(pd.Series(name_im_order),columns=['image'])
    #print(name_im_order_df.head(3))
    
    df_reduc = pd.merge(df, name_im_order_df, on=['image'], how='inner')
    print(len(df_reduc['image']),'images gardees')
    #print(df_reduc.head(3))
    
#    indice_train = df['set']==0
#    indice_test = abs(df['set'])==1
#    X_test = X[indice_test,:]
#    X_trainval = X[indice_train,:]
    
    
    k_tab = [5,10,20,50,100]
    classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []
    
    P_per_class = []
    
    R_per_class = []

    P20_per_class = []

    
    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1,2]))
    cs  =[1]
    param_grid = dict(C=cs)  
    
    #np_classes = df.as_matrix(columns=depictsAll)
    #index_image_with_at_least_one_of_the_class = np.setdiff1d(np.where(np.sum(np_classes,axis=1) > 0),np.where(indice_train))
    
    for i,classe in enumerate(depicts):
        print(classe,depicts_depictsLabel[classe])
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=-1)
        y = df_reduc[classe]
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        print("Number of element in the training set : ",len(X_trainval))
        #print(y)
#        y_trainval = y[indice_train]
#        y_test= y[indice_test]
#        y_test[y_test < 0] = 0 # Put to 0 the image where we don't know
        print("Number of training exemple :",len(y_trainval),"number of positive ones :",np.sum(y_trainval))
        print("Number of test exemple :",len(y_test),"number of positive ones :",np.sum(y_test))
        if 2*np.sum(y_test) < len(y_test):
            prediction_by_chance = 100*(len(y_test)-np.sum(y_test))/len(y_test)
            recall = 0.0
        else:
            prediction_by_chance = 100*(np.sum(y_test))/len(y_test)
            recall = 1.0
        F1 = 2 * (prediction_by_chance * recall) / (prediction_by_chance + recall)
        print("Precision by Chance = {0:.2f}, recall = {1:.2f}, F1 = {1:.2f}".format(prediction_by_chance,recall,F1))
        # Warning it is not the only metric to consider especially in unbalanced case like this one
        grid.fit(X_trainval,y_trainval)  
        y_predict_confidence_score = grid.decision_function(X_test)
        y_predict_test = grid.predict(X_test) 
        y_predict_trainval = grid.predict(X_trainval) 
        training_precision = precision_score(y_trainval,y_predict_trainval)
        print("Training precision :{0:.2f}".format(training_precision))
        # Warning ! predict provide class labels for samples whereas decision_function provide confidence scores for samples.
        AP = average_precision_score(y_test,y_predict_confidence_score,average=None)
        AP_per_class += [AP]
        print("Average Precision on all the data for",depicts_depictsLabel[classe]," = ",AP)   
        test_precision = precision_score(y_test,y_predict_test)
        test_recall = recall_score(y_test,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {1:.2f}".format(test_precision,test_recall,F1))
        #precision_at_k_tab = []
        for k in k_tab:
            precision_at_k = ranking_precision_score(np.array(y_test), y_predict_confidence_score,k)
            if k==20:
                P20_per_class += [precision_at_k]
            print("Precision on all the data @ ",k,":",precision_at_k)
            

    
        if plot_fp_fn:
            path_output = '/media/HDD/output_exp/html_output/'
            path_to_img= '/media/HDD/data/Wikidata_Paintings/340/'
            name_trainval_tab = name_im_order
#    print(len(indice_train),len(y_trainval),len(name_trainval_tab))
            name_test_tab = name_im_order
            
            name_html_fp = path_output+'Training_false_positif' + kindnetwork + '_' +kind +'_' + classe + '.html'
            message_training_fp = """<html><head></head><body><p>False Positif during training for """ + classe + """ </p></body>""" 
            f_fp = open(name_html_fp,'w')
            name_html_fn = 'html_output/Training_false_negatif' + kindnetwork + '_' +kind +'_' + classe + '.html'
            message_training_fn = """<html><head></head><body><p>False Negatif during training for """ + classe + """ </p></body>""" 
            f_fn = open(name_html_fn,'w')
            index= y_trainval.index
            for j,elt in enumerate(np.array(y_trainval)):
                if(elt!=y_predict_trainval[j]):
                    name_tab = name_trainval_tab[index[j]].split('.')
                    name_tab[-1] = 'jpg'
                    namejpg = ".".join(name_tab)
                    name_img = path_to_img + namejpg 
                    data_uri = base64.b64encode(open(name_img, 'rb').read()).decode('utf-8').replace('\n', '')
                    img_tag = '<img src="data:image/png;base64,%s \n">' % data_uri
                    if(elt==0): # cad que y_predict_trainval[j]==1 donc on a un Faux positif
                         message_training_fp += name_trainval_tab[index[j]] + '\n' + img_tag
                    else:
                        message_training_fn +=  name_trainval_tab[index[j]] + '\n' + img_tag
            message_training_fp += """</html>"""
            message_training_fn += """</html>"""  
            f_fp.write(message_training_fp)
            f_fp.close()
            f_fn.write(message_training_fn)
            f_fn.close()
            
                    
            name_html_fp = path_output+'Test_false_positif' + kindnetwork + '_' +kind +'_' + classe + '.html'
            message_training_fp = """<html><head></head><body><p>False Positif during test for """ + classe + """ </p></body>""" 
            f_fp = open(name_html_fp,'w')
            name_html_fn = path_output+'Test_false_negatif' + kindnetwork + '_' +kind +'_' + classe + '.html'
            message_training_fn = """<html><head></head><body><p>False Negatif during test for """ + classe + """ </p></body>""" 
            f_fn = open(name_html_fn,'w')
            index= y_test.index
            for j,elt in enumerate(y_test):
                if(elt!=y_predict_test[j]):
                    name_tab = name_test_tab[index[j]].split('.')
                    name_tab[-1] = 'jpg'
                    namejpg = ".".join(name_tab)
                    name_img = path_to_img + namejpg 
                    data_uri = base64.b64encode(open(name_img, 'rb').read()).decode('utf-8').replace('\n', '')
                    img_tag = '<img src="data:image/png;base64,%s \n">' % data_uri
                    if(elt==0): # cad que y_predict_trainval[j]==1 donc on a un Faux positif
                         message_training_fp += name_test_tab[index[j]] + '\n' + img_tag
                    else:
                        message_training_fn +=  name_test_tab[index[j]] + '\n' + img_tag
            message_training_fp += """</html>"""
            message_training_fn += """</html>"""  
            f_fp.write(message_training_fp)
            f_fp.close()
            f_fn.write(message_training_fn)
            f_fn.close()   
        
    print("mean Average Precision for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
    print("mean Precision for all the data = {0:.3f}".format(np.mean(P_per_class)))  
    print("mean Recall for all the data = {0:.3f}".format(np.mean(R_per_class)))  
    print("mean Precision @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    
    print(AP_per_class)
    
    return(0)
    
if __name__ == '__main__':
    
    #compute_InceptionResNetv2_features(kind='1536D',database='Wikidata_Paintings_Q235113_',L2=False,augmentation=False)
    TransferLearning_onRawFeatures(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings_Q235113_',L2=False,augmentation=False)
    