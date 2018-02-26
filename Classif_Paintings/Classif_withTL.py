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
from pandas import Index
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score,label_ranking_average_precision_score,classification_report
from sklearn.metrics import matthews_corrcoef,f1_score
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
import base64
import os
import Classifier_Evaluation

depicts_depictsLabel = {'Q942467_verif': 'Jesus Child','Q235113_verif':'angel/Cupidon ','Q345_verif' :'Mary','Q109607_verif':'ruins','Q10791_verif': 'nudity or breast'}

def compute_InceptionResNetv2_features(kind='1536D',database='Paintings',concate = True,L2=True,augmentation=True):
    """
    Inception ResNet v2 take RGB image as input
    """
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/299/'
    else:
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/299/'
    databasetxt = path_data + database + '.txt'
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
    
def compute_VGG_features(VGG='19',kind='fuco7',database='Paintings',concate = True,L2=False,augmentation=False):
    """
    Inception ResNet v2 take RGB image as input
    """
    
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    else:
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    databasetxt = path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")

    if augmentation:
        N = 50
    else: 
        N=1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    if(kind=='fuco8'):
        size_output = 1000
    elif(kind in ['fuco6','relu7','relu6','fuco7']):
        size_output = 4096    
    
    name_img = df_label[item_name][0]
    i = 0
    itera = 1000
        
    name_network = 'VGG'+VGG+'_'
    name_pkl_values = path_data+'Values_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    if not(concate):
        features_resnet = None
    else:
        features_resnet = pickle.load(open(name_pkl_values, 'rb'))
        im_resnet = pickle.load(open(name_pkl_im, 'rb'))

    with tf.Graph().as_default():
      # The Inception networks expect the input image to have color channels scaled from [-1, 1]
      #Load the model
      sess = tf.Session()
      if VGG=='19':
          vgg_layers = vgg19.get_vgg_layers()
      elif VGG=='16':
          vgg_layers = vgg16.get_vgg_layers()
      input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
      if VGG=='19':
          net = vgg19.net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME')
      elif VGG=='16':
          net = vgg16.net_preloaded(vgg_layers,input_tensor,pooling_type='max',padding='SAME')

      sess.run(tf.global_variables_initializer())

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
                         sizeIm = 256
                    else:
                        sizeIm = 224
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
            
                resizedf = resized.astype(np.float32)
                # Remove train image mean
                resizedf[:,:,0] -= 103.939
                resizedf[:,:,1] -= 116.779
                resizedf[:,:,2] -= 123.68
    
                    
                if(augmentation==True):
                    list_im =  np.stack(Classifier_Evaluation.get_Stretch_augmentation(resizedf,N=N,portion_size = (224, 224)))
                    out = sess.run(net[kind], feed_dict={input_tensor: list_im})
                    out_average = np.mean(out,axis=0)
                    if L2:
                        out_norm = LA.norm(out_average) # L2 norm 
                        out= out_average/out_norm
                else:
                    crop = resizedf[int(resized.shape[0]/2 - 112):int(resized.shape[0]/2 +112),int(resized.shape[1]/2-112):int(resized.shape[1]/2+112),:]        
                    ims = np.expand_dims(crop, axis=0)
                    net_values = sess.run(net[kind], feed_dict={input_tensor: ims})
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
    
def Compute_ResNet(kind='2048D',database='Paintings',concate = True,L2=False,augmentation=False):
    """
    Inception ResNet v2 take RGB image as input
    """

    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    else:
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    databasetxt = path_data + database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    weights_path = '/media/HDD/models/resnet152_weights_tf.h5'
    if augmentation:
        N = 50
    else: 
        N=1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    if(kind=='output'):
        model = resnet_152_keras.resnet152_model(weights_path)
        size_output = 1000
    elif(kind=='2048D'):
        model = resnet_152_keras.resnet152_model_2018output(weights_path)
        size_output = 2048
    name_img = df_label[item_name][0]
    i = 0
    itera = 1000
        
    name_network = 'ResNet152_'
    name_pkl_values = path_data+'Values_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +name_network+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    if not(concate):
        features_resnet = None
    else:
        features_resnet = pickle.load(open(name_pkl_values, 'rb'))
        im_resnet = pickle.load(open(name_pkl_im, 'rb'))



    for i,name_img in  enumerate(df_label[item_name]):
        if not(concate) or not(name_img in  im_resnet):
            if i%itera==0:
                print(i,name_img)
            
            if database=='VOC12' or database=='Paintings':
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
                # normalily it is 299
                if augmentation:
                    sizeIm = 256
                else:
                    sizeIm = 224
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
                resized = resized[:,:,[2,1,0]] # To shift from RGB to BGR
        
            resizedf = resized.astype(np.float32)
            # Remove train image mean
            resizedf[:,:,0] -= 103.939
            resizedf[:,:,1] -= 116.779
            resizedf[:,:,2] -= 123.68

                
            if(augmentation==True):
                list_im =  np.stack(Classifier_Evaluation.get_Stretch_augmentation(resizedf,N=N,portion_size = (224, 224)))
                out = model.predict(list_im)
                out_average = np.mean(out,axis=0)
                if L2:
                    out_norm = LA.norm(out_average) # L2 norm 
                    out= out_average/out_norm
            else:
                crop = resizedf[int(resized.shape[0]/2 - 112):int(resized.shape[0]/2 +112),int(resized.shape[1]/2-112):int(resized.shape[1]/2+112),:]        
                crop = np.expand_dims(crop, axis=0)
                out = model.predict(crop)[0]
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
    database_verif = 'Wikidata_Paintings_subset_verif' # TODO changer cela

    databasetxt = path_data + database_verif + '.txt'

    df = pd.read_csv(databasetxt,sep=',')
    depicts = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    colums_selected = ['image','BadPhoto']+depicts
    df_reduc = df[colums_selected]
    name_pkl_values = path_data+'Values_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
   
    X = pd.DataFrame(pickle.load(open(name_pkl_values, 'rb')))
    X = X[df_reduc['BadPhoto'] <= 0.0]
    name_im_order = pickle.load(open(name_pkl_im, 'rb'))
    name_im_order_df = pd.DataFrame(pd.Series(name_im_order),columns=['image'])
    #print(name_im_order_df.head(3))
    
    df_reduc = pd.merge(df, name_im_order_df, on=['image'], how='inner')
    df_reduc = df_reduc[df_reduc['BadPhoto'] <= 0.0]
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
    cs = np.hstack((cs,[0.2,1.,2.]))
    param_grid = dict(C=cs)  
    
    #np_classes = df.as_matrix(columns=depictsAll)
    #index_image_with_at_least_one_of_the_class = np.setdiff1d(np.where(np.sum(np_classes,axis=1) > 0),np.where(indice_train))
    
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        print(classe,classestr)
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=-1)
        y = df_reduc[classe]
        random_state = 0
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.6, random_state=random_state)
        number_of_positif_train_exemple = np.sum(y_trainval)
        print("Number of training exemple :",len(y_trainval),"number of positive ones :",number_of_positif_train_exemple)
        print("Number of test exemple :",len(y_test),"number of positive ones :",np.sum(y_test))
        
        # First number of training exemple 
        
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
        F1 = f1_score(y_test,y_predict_test)
        print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}".format(test_precision,test_recall,F1))
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
            name_html_fn = path_output+'Training_false_negatif' + kindnetwork + '_' +kind +'_' + classe + '.html'
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
    
def index_place(a,b):
    """ Test if elt a are in b (bigger) return index of b
    """
    tab = []
    for elt_a in a['image']:
        index = b.index[b['image'] == elt_a].tolist()
        tab += [index[0]]
    return(tab)
  
def TransferLearning_onRawFeatures_protocol(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings',L2=False,augmentation=False):
    """
    Cette fonction trace les courbes de performances en fonction de 
    kindnetwork in  [InceptionResNetv2,ResNet152]
    """
    # TODO : faire une courbe de performance en fonction du nombre de positif exemples mais avec les none en exemples négatifs
    # TODO : faire une courbe de performance en fonction du nombre de positif exemples mais avec d'autres negatifs exemples du neme ordre que precedement
    
    return(0)
    
def TransferLearning_onRawFeatures_protocol(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings',L2=False,augmentation=False):
    """
    Cette fonction trace les courbes de performances en fonction de 
    kindnetwork in  [InceptionResNetv2,ResNet152]
    """
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
    database_verif = 'Wikidata_Paintings_miniset_verif' # TODO changer cela

    databasetxt = path_data + database_verif + '.txt'

    df = pd.read_csv(databasetxt,sep=',')
    depicts = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    colums_selected = ['image','BadPhoto']+depicts
    df_reduc = df[colums_selected]
    name_pkl_values = path_data+'Values_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
   
    X_brut = pd.DataFrame(pickle.load(open(name_pkl_values, 'rb')))
    name_im_order = pickle.load(open(name_pkl_im, 'rb'))
    name_im_order_df = pd.DataFrame(pd.Series(name_im_order),columns=['image'])
    df_reduc = df_reduc[df_reduc['BadPhoto'] <= 0.0]
    index = index_place(df_reduc,name_im_order_df)
    X = X_brut.loc[index,:]
    print(len(df_reduc['image']),'images gardees')
    
    
    k_tab = [5,10,20,50,100]
    classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []

    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1.,2.]))
    param_grid = dict(C=cs)  
    
    tab_values = [5,10,50]
    
    dict_increase_posEx = {}
    dict_increase_per ={}
    dict_noisy_per = {}
    
    min_num_of_posEx = len(df_reduc['image'])
    
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        print(classe,classestr)
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=-1)
        y = df_reduc[classe]
        random_state = 0
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.6, random_state=random_state)
        number_of_positif_train_exemple = np.sum(y_trainval)
        min_num_of_posEx = np.min([min_num_of_posEx,number_of_positif_train_exemple])
        print("Number of training exemple :",len(y_trainval),"number of positive ones :",number_of_positif_train_exemple)
        print("Number of test exemple :",len(y_test),"number of positive ones :",np.sum(y_test))
       
        ### We keep the negative exemples the same and we increase the positive ones
        print('Test on the number of positives exemples')
        index_posEx = np.where(y_trainval>= 1)[0]
        index_negEx = np.where(y_trainval < 1)[0]

        tab_values_tmp = np.concatenate((tab_values,np.arange(100,number_of_positif_train_exemple,100)))
        tab_values_tmp = np.concatenate((tab_values_tmp,[number_of_positif_train_exemple]))
        AP_tab = []
        f1_tab = []
        MCC_tab= []
        #X_trainval = X_trainval.as_matrix()
        for number_of_posEx_selected in tab_values_tmp:
            #print(number_of_posEx_selected)
            index_posEx_keep = random.sample(list(index_posEx),int(number_of_posEx_selected))
            #print(len(index_posEx_keep))
            index_keep = np.concatenate((index_negEx,index_posEx_keep))
            X_trainval_reduc = X_trainval.copy()
            X_trainval_reduc = X_trainval_reduc.as_matrix()
            X_trainval_reduc = X_trainval_reduc[index_keep,:]
            y_trainval_reduc = y_trainval.copy()
            y_trainval_reduc = y_trainval_reduc.as_matrix()
            y_trainval_reduc= y_trainval_reduc[index_keep]
            grid.fit(X_trainval_reduc,y_trainval_reduc)  
            y_predict_confidence_score = grid.decision_function(X_test)
            AP = average_precision_score(y_test,y_predict_confidence_score,average=None)
            AP_tab += [AP]
            y_predict_test = grid.predict(X_test)
            f1 = f1_score(y_test, y_predict_test)
            f1_tab += [f1]
            MCC = matthews_corrcoef(y_test, y_predict_test)
            MCC_tab += [MCC]
        dict_increase_posEx[classestr] = (AP_tab,tab_values_tmp,f1_tab,MCC_tab)    
        print(classe,AP)
        
        ### We will consider of the data to be positive exemple
        print('Test on the percentage of positive exemples')
        AP_tab = []
        f1_tab = []
        MCC_tab= []
        number_of_train_exemple = number_of_positif_train_exemple
        percentages = np.concatenate(([0.05],np.arange(0.1,1.0,0.1),[0.95]))
        for per in percentages:
            number_of_pos_Ex = int(per*number_of_train_exemple)
            #print(len(index_posEx_keep))
            number_of_neg_Ex = int((1-per)*number_of_train_exemple)
            index_posEx_keep = random.sample(list(index_posEx),int(number_of_pos_Ex))
            index_negEx_keep = random.sample(list(index_negEx),int(number_of_neg_Ex))
            index_keep = np.concatenate((index_negEx_keep,index_posEx_keep))
            X_trainval_reduc = X_trainval.copy()
            X_trainval_reduc = X_trainval_reduc.as_matrix()
            X_trainval_reduc = X_trainval_reduc[index_keep,:]
            y_trainval_reduc = y_trainval.copy()
            y_trainval_reduc = y_trainval_reduc.as_matrix()
            y_trainval_reduc= y_trainval_reduc[index_keep]
            grid.fit(X_trainval_reduc,y_trainval_reduc)  
            y_predict_confidence_score = grid.decision_function(X_test)
            AP = average_precision_score(y_test,y_predict_confidence_score,average=None)
            AP_tab += [AP]
            y_predict_test = grid.predict(X_test)
            f1 = f1_score(y_test, y_predict_test)
            f1_tab += [f1]
            MCC = matthews_corrcoef(y_test, y_predict_test)
            MCC_tab += [MCC]
        dict_increase_per[classestr] = (AP_tab,percentages,f1_tab,MCC_tab) 
        
        ### Noise influence : we increase the percentage of positive exemples
        print('Percentage of noisy positive exemples')
        AP_tab = []
        f1_tab = []
        MCC_tab= []
        number_of_train_exemple = number_of_positif_train_exemple
        percentages = np.arange(0.0,1.,0.1)
        for per in percentages:
            number_of_pos_Ex_shift = int(per*number_of_train_exemple)
            index_posEx_keep_shift = random.sample(list(index_posEx),number_of_pos_Ex_shift)
            #print(len(index_posEx_keep_shift))
            X_trainval_reduc = X_trainval.copy()
            X_trainval_reduc = X_trainval_reduc.as_matrix()
            y_trainval_reduc = y_trainval.copy()
            y_trainval_reduc = y_trainval_reduc.as_matrix()
            y_trainval_reduc[index_posEx_keep_shift] = 0
            grid.fit(X_trainval_reduc,y_trainval_reduc)  
            y_predict_confidence_score = grid.decision_function(X_test)
            AP = average_precision_score(y_test,y_predict_confidence_score,average=None)
            AP_tab += [AP]
            y_predict_test = grid.predict(X_test)
            f1 = f1_score(y_test, y_predict_test)
            f1_tab += [f1]
            MCC = matthews_corrcoef(y_test, y_predict_test)
            MCC_tab += [MCC]
        dict_noisy_per[classestr] = (AP_tab,percentages,f1_tab,MCC_tab) 
    
    # Plots AP
    plt.ion()
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)   = dict_increase_posEx[classestr]
        plt.plot(tab_values_tmp, AP_tab,linestyle='--', marker='o')
    plt.title('AP function of number of positive exemples')
    plt.xlabel('Number of positive exemples')
    plt.ylabel('AP')
    plt.legend(dict_increase_posEx.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' AP function of number of positive exemples.png'
    fig.savefig(savename)
        
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab) = dict_increase_per[classestr]
        plt.plot(tab_values_tmp, AP_tab,linestyle='--', marker='o')
    plt.title('AP function of percentage of positive exemples')
    plt.xlabel('Percentage of positive exemples')
    plt.ylabel('AP')
    plt.legend(dict_increase_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' AP function of percentage of positive exemples.png'
    fig.savefig(savename)
    
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)  = dict_noisy_per[classestr]
        plt.plot(tab_values_tmp, AP_tab,linestyle='--', marker='o')
    plt.title('AP function of percentage of noisy positive exemples')
    plt.xlabel('Percentage of noisy positive exemples')
    plt.ylabel('AP')
    plt.legend(dict_noisy_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' AP function of percentage of noisy positive exemples.png'
    fig.savefig(savename)
    
    ## F1
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)  = dict_noisy_per[classestr]
        plt.plot(tab_values_tmp, f1_tab,linestyle='--', marker='o')
    plt.title('F1 function of percentage of noisy positive exemples')
    plt.xlabel('Percentage of noisy positive exemples')
    plt.ylabel('F1')
    plt.legend(dict_noisy_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' F1 function of percentage of noisy positive exemples.png'
    fig.savefig(savename)

    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)   = dict_increase_posEx[classestr]
        plt.plot(tab_values_tmp, f1_tab,linestyle='--', marker='o')
    plt.title('F1 function of number of positive exemples')
    plt.xlabel('Number of positive exemples')
    plt.ylabel('F1')
    plt.legend(dict_increase_posEx.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' F1 function of number of positive exemples.png'
    fig.savefig(savename)
        
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab) = dict_increase_per[classestr]
        plt.plot(tab_values_tmp, f1_tab,linestyle='--', marker='o')
    plt.title('F1 function of percentage of positive exemples')
    plt.xlabel('Percentage of positive exemples')
    plt.ylabel('F1')
    plt.legend(dict_increase_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' F1 function of percentage of positive exemples.png'
    fig.savefig(savename)
    
    ## MCC
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)   = dict_increase_posEx[classestr]
        plt.plot(tab_values_tmp, MCC_tab,linestyle='--', marker='o')
    plt.title('MCC function of number of positive exemples')
    plt.xlabel('Number of positive exemples')
    plt.ylabel('MCC')
    plt.legend(dict_increase_posEx.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' MCC function of number of positive exemples.png'
    fig.savefig(savename)
        
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab) = dict_increase_per[classestr]
        plt.plot(tab_values_tmp, MCC_tab,linestyle='--', marker='o')
    plt.title('MCC function of percentage of positive exemples')
    plt.xlabel('Percentage of positive exemples')
    plt.ylabel('MCC')
    plt.legend(dict_increase_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' MCC function of percentage of positive exemples.png'
    fig.savefig(savename)
   
    fig = plt.figure()
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        (AP_tab,tab_values_tmp,f1_tab,MCC_tab)  = dict_noisy_per[classestr]
        plt.plot(tab_values_tmp, MCC_tab,linestyle='--', marker='o')
    plt.title('MCC function of percentage of noisy positive exemples')
    plt.xlabel('Percentage of noisy positive exemples')
    plt.ylabel('MCC')
    plt.legend(dict_noisy_per.keys(), loc='best')
    savename = 'fig/' + kindnetwork +' MCC function of percentage of noisy positive exemples.png'
    fig.savefig(savename)
    
    plt.show()
    
    
    return(0)
    
def TransferLearning_onRawFeatures_JustAP(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings',L2=False,augmentation=False):
    """
    kindnetwork in  [InceptionResNetv2,ResNet152]
    """
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
    database_verif = 'Wikidata_Paintings_miniset_verif' # TODO changer cela

    databasetxt = path_data + database_verif + '.txt'

    df = pd.read_csv(databasetxt,sep=',')
    depicts = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    colums_selected = ['image','BadPhoto']+depicts
    df_reduc = df[colums_selected]
    name_pkl_values = path_data+'Values_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    name_pkl_im =   path_data+'Name_' +kindnetwork+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
   
    X_brut = pd.DataFrame(pickle.load(open(name_pkl_values, 'rb')))
    name_im_order = pickle.load(open(name_pkl_im, 'rb'))
    name_im_order_df = pd.DataFrame(pd.Series(name_im_order),columns=['image'])
    df_reduc = df_reduc[df_reduc['BadPhoto'] <= 0.0]
    index = index_place(df_reduc,name_im_order_df)
    X = X_brut.loc[index,:]
    print(len(df_reduc['image']),'images gardees')
    classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []

    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1.,2.]))
    param_grid = dict(C=cs)  
    indices = np.arange(len(df_reduc['image']))
    for i,classe in enumerate(depicts):
        classestr = depicts_depictsLabel[classe]
        print(classe,classestr)
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=-1)
        y = df_reduc[classe]
        random_state = 0
        X_trainval, X_test, y_trainval, y_test, id_trainval,id_test = train_test_split(X, y,indices, test_size=0.6, random_state=random_state)
        print(df['image'][id_trainval][0:10])
        number_of_positif_train_exemple = np.sum(y_trainval)
        print("Number of training exemple :",len(y_trainval),"number of positive ones :",number_of_positif_train_exemple)
        print("Number of test exemple :",len(y_test),"number of positive ones :",np.sum(y_test))
        grid.fit(X_trainval.copy(), y_trainval.copy())  
        y_predict_confidence_score = grid.decision_function(X_test)
        AP = average_precision_score(y_test,y_predict_confidence_score,average=None)
        AP_per_class += [AP]  
        print(classe,AP)
    print('MeanAP',np.mean(AP_per_class))    
    return(0)

def vision_of_data():
    import math
    database_verif = 'Wikidata_Paintings_miniset_verif' # TODO changer cela
    path_data = 'data/'
    databasetxt = path_data + database_verif + '.txt'
    databaseArtist = path_data +'Dates_Artists_rewied.csv'
    df_artists = pd.read_csv(databaseArtist)
    df = pd.read_csv(databasetxt,sep=',')
    depicts = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    df_reduc = df[df['BadPhoto'] <= 0.0]
    indices = np.arange(len(df_reduc['image']))
    print("We have ",len(df_reduc['image']),"images")
    depicts_dict = {}
    for classe in depicts:
        depicts_dict[classe] = []
    depicts_dict['None'] = []
    im_w_na = 0
    im_wt_class = 0
    for index, image in df_reduc.iterrows():
        isInOneClass = False
        Add = True
        date = image['year']
        if math.isnan(date):
            artist = image['createur']
            date = df_artists.loc[df_artists['peintre']==artist]['year_merge'].astype(float)
            if  len(date)<=0 or math.isnan(date):
                im_w_na += 1
                Add = False
            else:
                date = float(date)
        if Add:
            for classe in depicts:
                if (image[classe]==1.0):
                    tab = depicts_dict[classe]
                    tab += [date]
                    depicts_dict[classe] = tab
                    isInOneClass = True
            if not(isInOneClass):
                 tab = depicts_dict['None']
                 tab += [date]
                 depicts_dict['None'] = tab
        isInOneClass = False
        for classe in depicts:
            if (image[classe]==1.0):
                isInOneClass = True
        if not(isInOneClass):
            im_wt_class += 1
        
    
    print("We have ",im_w_na," images without date")
    print("We have ",im_wt_class,"images without class")
    random_state = 0
    id_trainval,id_test = train_test_split(indices, test_size=0.6, random_state=random_state)
      
    for j,classe in enumerate(depicts):
        print("==>",depicts_depictsLabel[classe])
        itera = 0
        for index, image in df_reduc.iterrows():
            if itera < 6 and index in id_trainval:
                stringtext = image['image'] + " " + depicts_depictsLabel[classe]
                isInOneClass = False
                if (image[classe]==1.0):
                    itera += 1
                    for jj,classe2 in enumerate(depicts):
                        if not(classe2==classe):
                            if (image[classe2]==1.0):
                                stringtext += " " + depicts_depictsLabel[classe2]
                    print(stringtext)
    print("==>",'None')
    itera = 0
    for index, image in df_reduc.iterrows():
        if itera < 6 and index in id_trainval:
            stringtext = image['image']
            isInOneClass = False
            for jj,classe2 in enumerate(depicts):
                if (image[classe2]==1.0):
                    isInOneClass = True
            if not(isInOneClass):
                print(stringtext)
                itera +=1
                    
    plt.ion()
    fig = plt.figure()
    legends = []
    for classe in depicts:
        classestr = depicts_depictsLabel[classe]
        legends += [classestr]
        date_per_class = depicts_dict[classe]
        plt.hist(date_per_class, 100, normed=0, alpha=0.5)
    classestr='None'
    legends += [classestr]
    date_per_class = depicts_dict[classestr]
    if len(date_per_class)>0:
        plt.hist(date_per_class, 100, normed=0, alpha=0.5)    
        
    plt.legend(legends, loc='best')
    plt.title('Histograms of the classes')
    plt.xlabel('date')
    plt.show()
    savename = 'fig/HistogramOfYearOfTheMiniset_notNorm.png'
    fig.savefig(savename)
    
    fig = plt.figure()
    legends = []
    for classe in depicts:
        classestr = depicts_depictsLabel[classe]
        legends += [classestr]
        date_per_class = depicts_dict[classe]
        print(np.min(date_per_class),np.max(date_per_class))
        plt.hist(date_per_class, 100, normed=1, alpha=0.5)
    
    classestr='None'
    legends += [classestr]
    date_per_class = depicts_dict[classestr]
    if len(date_per_class)>0:
        plt.hist(date_per_class, 100, normed=1, alpha=0.5)  
        print(np.min(date_per_class),np.max(date_per_class))
    
    plt.legend(legends, loc='best')
    plt.title('Histograms of the classes')
    plt.xlabel('date')
    plt.show()
    savename = 'fig/HistogramOfYearOfTheMiniset_Norm.png'
    fig.savefig(savename)
    
    input('wait')
    return(0)
    
if __name__ == '__main__':
    
    #compute_InceptionResNetv2_features(kind='1536D',database='Paintings',concate = False,L2=False,augmentation=False)
    #Compute_ResNet(kind='2048D',database='Wikidata_Paintings',L2=False,augmentation=False)
    #compute_VGG_features(VGG='19',kind='fuco7',database='Wikidata_Paintings',L2=False,augmentation=False)
    #TransferLearning_onRawFeatures(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings_MiniSet',L2=False,augmentation=False)
    #TransferLearning_onRawFeatures_JustAP(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings_MiniSet',L2=False,augmentation=False)
    #TransferLearning_onRawFeatures_JustAP(kind='2048D',kindnetwork='ResNet152',database='Wikidata_Paintings',L2=False,augmentation=False)
    #TransferLearning_onRawFeatures_JustAP(kind='relu7',kindnetwork='VGG19',database='Wikidata_Paintings',L2=False,augmentation=False)


    TransferLearning_onRawFeatures_protocol(kind='1536D',kindnetwork='InceptionResNetv2',database='Wikidata_Paintings_MiniSet',L2=False,augmentation=False)
    TransferLearning_onRawFeatures_protocol(kind='2048D',kindnetwork='ResNet152',database='Wikidata_Paintings',L2=False,augmentation=False)
#    TransferLearning_onRawFeatures_protocol(kind='fuco7',kindnetwork='VGG19',database='Wikidata_Paintings',L2=False,augmentation=False)
    #compute_VGG_features(VGG='19',kind='relu7',database='Wikidata_Paintings',concate = False,L2=False,augmentation=False)
    TransferLearning_onRawFeatures_protocol(kind='relu7',kindnetwork='VGG19',database='Wikidata_Paintings',L2=False,augmentation=False)
    #vision_of_data()
    # TODO plot 
    # TODO test with FasterRCNN 
    