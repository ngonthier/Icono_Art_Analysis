#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:51:16 2017

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

def TestPerformanceSurVOC12(kind='2048D',kindnetwork='ResNet152'):
    """
    Tester la meanAP en s entrainant sur la partie train de VOC12 et en testant 
    sur la partie validation
    """
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    name_pkl = kindnetwork +'_' + kind +'_VOC12_N1.pkl'
    [X_trainval,y_trainval,_,_,X_test,y_test] = pickle.load(open(name_pkl, 'rb'))
    classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []
    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1,2]))
    param_grid = dict(C=cs)
    for i,classe in enumerate(classes):
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=3)
        grid.fit(X_trainval,y_trainval[:,i])  
        y_predict_confidence_score = grid.decision_function(X_test)
        # Warning ! predict provide class labels for samples whereas decision_function provide confidence scores for samples.
        AP = average_precision_score(y_test[:,i],y_predict_confidence_score,average=None)
        AP_per_class += [AP]
        print("Average Precision for",classe," = ",AP)
    
    print("mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))
    return(0)
    

def get_Stretch_augmentation(im,N=50,portion_size = (224, 224)):
    """
    N is the number of image that you need to return
    """
    image_size = im.shape
    list_im = []
    
    for i in range(N):
        x1 = random.randint(0, image_size[0]-portion_size[0]-1)
        y1 = random.randint(0, image_size[1]-portion_size[1]-1)
        x2, y2 = x1+portion_size[0], y1+portion_size[1]
        im_tmp = im[x1:x2,y1:y2,:]
        flip = random.randint(0,1)
        if flip:
            im_tmp = np.fliplr(im_tmp)
        list_im += [im_tmp]
    
    return(list_im)

def HowAreSupport_of_SVM(kind='2048D',kindnetwork='InceptionResNetv2',database='Paintings',L2=True,augmentation=True):
    """
    kindnetwork in  [InceptionResNetv2,ResNet152]
    
    Cas de la base de Crowley ou il manque le label avion = 'http://ichef.bbci.co.uk/arts/yourpaintings/images/paintings/dpe/624x544/ss_dpe_pcf_04_624x544.jpg'

    
    """
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
    
    if database=='Paintings':
        path_to_img = '/media/HDD/data/Painting_Dataset/'
    elif database=='VOC12':
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
    databasetxt = database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_train = df_label[df_label['set']=='train']
    df_val =  df_label[df_label['set']=='validation']
    df_trainval =  pd.concat([df_train,df_val])
    name_trainval_tab = np.array(df_trainval['name_img'])
    df_test =  df_label[df_label['set']=='test']
    name_test_tab = np.array(df_test['name_img'])
    
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    if database == 'Paintings':
        name_pkl = kindnetwork +'_' + kind +'_'+database+'_N'+str(N)+extL2+'.pkl'
        [X_train,y_train,X_test,y_test,X_val,y_val] = pickle.load(open(name_pkl, 'rb'))
    elif database == 'VOC12':
        name_pkl = kindnetwork +'_' + kind +'_'+database+'_N'+str(N)+extL2+'.pkl'
        [X_train,y_train,_,_,X_val,y_val] = pickle.load(open(name_pkl, 'rb'))
        name_pkl = kindnetwork +'_' + kind +'_Paintings_N'+str(N)+extL2+'.pkl'
        [_,_,X_test,y_test,_,_] = pickle.load(open(name_pkl, 'rb'))
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
        
    classifier = SVC(kernel='linear', max_iter=-1)
    #LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
    AP_per_class = []
    cs = np.logspace(-5, 7, 40)
    cs = np.logspace(-5, -2, 20)
    cs = np.hstack((cs,[0.2,1,2]))
    param_grid = dict(C=cs)
    custom_cv = PredefinedSplit(np.hstack((-np.ones((1,X_train.shape[0])),np.zeros((1,X_val.shape[0])))).reshape(-1,1)) # For example, when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.

    for i,classe in enumerate(classes):
        grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=3,
                        cv=custom_cv)
        grid.fit(X_trainval,y_trainval[:,i])  
        best_classifier = grid.best_estimator_ 
        y_predict_confidence_score = best_classifier.decision_function(X_test)
        y_predict_trainval = best_classifier.predict(X_trainval)
        y_predict_test = best_classifier.predict(X_test)
        training_precision = precision_score(y_trainval[:,i],y_predict_trainval)
        
        
        print("Training precision :{0:.2f}".format(training_precision))
        AP = average_precision_score(y_test[:,i],y_predict_confidence_score,average=None)
        AP_per_class += [AP]
        print("Average Precision for",classe," = ",AP)
        
        name_html_fp = 'html_output/Training_false_positif' + kindnetwork + '_' +kind +'_' + classe + '.html'
        message_training_fp = """<html><head></head><body><p>False Positif during training for """ + classe + """ </p></body>""" 
        f_fp = open(name_html_fp,'w')
        name_html_fn = 'html_output/Training_false_negatif' + kindnetwork + '_' +kind +'_' + classe + '.html'
        message_training_fn = """<html><head></head><body><p>False Negatif during training for """ + classe + """ </p></body>""" 
        f_fn = open(name_html_fn,'w')
        for j,elt in enumerate(y_trainval[:,i]):
            if(elt!=y_predict_trainval[j]):
                name_img = path_to_img + name_trainval_tab[j] + '.jpg'
                data_uri = base64.b64encode(open(name_img, 'rb').read()).decode('utf-8').replace('\n', '')
                img_tag = '<img src="data:image/png;base64,%s \n">' % data_uri
                if(elt==0): # Faux positif
                     message_training_fp += name_trainval_tab[j] + '\n' + img_tag
                else:
                    message_training_fn +=  name_trainval_tab[j] + '\n' + img_tag
        message_training_fp += """</html>"""
        message_training_fn += """</html>"""  
        f_fp.write(message_training_fp)
        f_fp.close()
        f_fn.write(message_training_fn)
        f_fn.close()
        
        test_precision = precision_score(y_test[:,i],y_predict_test)
        test_recall = recall_score(y_test[:,i],y_predict_test)
        print("Test precision = {0:.2f}, recall = {1:.2f}".format(test_precision,test_recall))
        support = best_classifier.support_
        name_html = 'html_output/Support_Vectors_' + kindnetwork + '_' +kind +'_' + classe + '.html'
        f = open(name_html,'w')
        message = """<html><head></head><body><p>Support Vectors (""" +str(len(support)) + """) for """ + classe + """ </p></body>""" 
        for index in support:
            name_img = path_to_img + name_trainval_tab[index] + '.jpg'
            data_uri = base64.b64encode(open(name_img, 'rb').read()).decode('utf-8').replace('\n', '')
            img_tag = '<img src="data:image/png;base64,%s \n">' % data_uri
            message +=   name_trainval_tab[index] + '\n' + img_tag
        message += """</html>"""

        f.write(message)
        f.close()

        name_html_fp = 'html_output/Test_false_positif' + kindnetwork + '_' +kind +'_' + classe + '.html'
        message_training_fp = """<html><head></head><body><p>False Positif during test for """ + classe + """ </p></body>""" 
        f_fp = open(name_html_fp,'w')
        name_html_fn = 'html_output/Test_false_negatif' + kindnetwork + '_' +kind +'_' + classe + '.html'
        message_training_fn = """<html><head></head><body><p>False Negatif during test for """ + classe + """ </p></body>""" 
        f_fn = open(name_html_fn,'w')
        for j,elt in enumerate(y_test[:,i]):
            if(elt!=y_predict_test[j]):
                name_img = path_to_img + name_test_tab[j] + '.jpg'
                data_uri = base64.b64encode(open(name_img, 'rb').read()).decode('utf-8').replace('\n', '')
                img_tag = '<img src="data:image/png;base64,%s \n">' % data_uri
                if(elt==0): # Faux positif
                     message_training_fp += name_test_tab[j] + '\n' + img_tag
                else:
                    message_training_fn +=  name_test_tab[j] + '\n' + img_tag
        message_training_fp += """</html>"""
        message_training_fn += """</html>"""  
        f_fp.write(message_training_fp)
        f_fp.close()
        f_fn.write(message_training_fn)
        f_fn.close()

        #webbrowser.open_new_tab(name_html)
        
    print("mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))

    
    return(0)

def Classification_evaluation(kind='2048D',kindnetwork='InceptionResNetv2',database='Paintings',L2=True,augmentation=True,classifier_name='LinearSVM'):
    """
    kindnetwork in  [InceptionResNetv2,ResNet152]
    """
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
    
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    if database == 'Paintings':
        name_pkl = kindnetwork +'_' + kind +'_'+database+'_N'+str(N)+extL2+'.pkl'
        [X_train,y_train,X_test,y_test,X_val,y_val] = pickle.load(open(name_pkl, 'rb'))
    elif database == 'VOC12':
        name_pkl = kindnetwork +'_' + kind +'_'+database+'_N'+str(N)+extL2+'.pkl'
        [X_train,y_train,_,_,X_val,y_val] = pickle.load(open(name_pkl, 'rb'))
        name_pkl = kindnetwork +'_' + kind +'_Paintings_N'+str(N)+extL2+'.pkl'
        [_,_,X_test,y_test,_,_] = pickle.load(open(name_pkl, 'rb'))
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
        
#    X_trainval =  normalize(X_trainval, axis=1, norm='l2')
#    X_test =  normalize(X_test, axis=1, norm='l2')

    k_tab = [5,10,20,50,100]
    if classifier_name=='LinearSVM':
        classifier = LinearSVC(penalty='l2', loss='squared_hinge',max_iter=1000,dual=True)
        AP_per_class = []
        cs = np.logspace(-5, 7, 40)
        cs = np.logspace(-5, -2, 20)
        cs = np.hstack((cs,[0.2,1,2]))
        #cs = np.logspace(-10,0,40)
        param_grid = dict(C=cs)
        custom_cv = PredefinedSplit(np.hstack((-np.ones((1,X_train.shape[0])),np.zeros((1,X_val.shape[0])))).reshape(-1,1)) # For example, when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
        ###custom_cv = PredefinedSplit(np.hstack((np.zeros((1,X_train.shape[0])),np.ones((1,X_val.shape[0])))).reshape(-1,1)) # For example, when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
        #custom_cv = None
        # For custom_cv = zip(train_indices, test_indices) that's is used by Crowley but a better cross validation method is possible 
        # TODOcv=ShuffleSplit(train_size=train_size,n_splits=250, random_state=1)
        #y_predict_all_label = np.zeros_like(y_test)
        
        # The C that produces the highest mAP
        
        
        for i,classe in enumerate(classes):
            grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score,needs_threshold=True), param_grid=param_grid,n_jobs=3,
                            cv=custom_cv)
            grid.fit(X_trainval,y_trainval[:,i])  
            y_predict_confidence_score = grid.decision_function(X_test)
            y_predict_test = grid.predict(X_test) 
            #y_predict_trainval = grid.predict(X_trainval) 
            # Warning ! predict provide class labels for samples whereas decision_function provide confidence scores for samples.
            AP = average_precision_score(y_test[:,i],y_predict_confidence_score,average=None)
            AP_per_class += [AP]
            print("Average Precision for",classe," = ",AP)
            
#            recalls, precisions = VOCevalaction(y_test[:,i], y_predict_confidence_score) # To compare the AP computation
#            AP_VOC12 = computeAveragePrecision(recalls, precisions, use_07_metric=False)
#            AP_VOC07 = computeAveragePrecision(recalls, precisions, use_07_metric=True)
#            print("Average Precision for {0:s} : AP = {1:.2f}, VOC12 = {2:.2f}, VOC07 = {3:.2f}".format(classe,AP,AP_VOC12,AP_VOC07))
            
#            training_precision = precision_score(y_trainval[:,i],y_predict_trainval)
#            print("Training precision :{0:.2f}".format(training_precision))
            test_precision = precision_score(y_test[:,i],y_predict_test)
            test_recall = recall_score(y_test[:,i],y_predict_test)
            print("Test precision = {0:.2f}, recall = {1:.2f}".format(test_precision,test_recall))
            #y_predict_all_label[:,i] =  y_predict_confidence_score
            #print("Number of elt predicted  {0} and true {1}".format(np.sum(y_predict_test),np.sum(y_test[:,i])))
#            precision_at_k_tab = []
#            for k in k_tab:
#                precision_at_k = ranking_precision_score(y_test[:,i], y_predict_confidence_score,k)
#                precision_at_k_tab += [precision_at_k]
#                print("Precision @ ",k,":",precision_at_k)
            
        print("mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))
#        lr_AP = label_ranking_average_precision_score(y_test,y_predict_all_label)
#        print("label Ranking Average Precision {0:.2f}".format(lr_AP))
        
    elif classifier_name=='RF': 
        
        # Random Forest that can deal with multilabel
        classifier = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1)
        classifier.fit(X_trainval,y_trainval)
        y_predict = classifier.predict(X_test)
        AP = average_precision_score(y_test,y_predict)
        print('Average precision in multilabel ',AP)
        print("Accuray in multilabel",classifier.score(X_test,y_test))
        
        # Now in One vs Rest
        AP_tab = []
        for i,classe in enumerate(classes):
            classifier = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1)
            classifier.fit(X_trainval,y_trainval[:,i])
            proba = classifier.predict_proba(X_test)
            y_predict = proba[:,1]
            AP = average_precision_score(y_test[:,i],y_predict)
            AP_tab += [AP]
            print("Average Precision for",classe," = ",AP)
        print('Average precision in one VS rest ',np.mean(AP_tab))
    
    return(0)
   
def Compute_ResNet(kind='2048D',database='Paintings',L2=True,augmentation=True):
    """
    ResNet take BGR image as input
    """
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        databasetxt = path_data + database + '.txt'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    df_label = pd.read_csv(databasetxt,sep=",")
    if augmentation:
        N = 50
    else: 
        N=1
    sLength = len(df_label[item_name])
    #df_label_augmented = df_label.assign(resnet_output=pd.Series(np.ones(sLength)))
    #df_label_augmented = df_label.assign(resnet_output=pd.Series(np.ones(sLength)).values)
    
    # Load the ResNet 152
    weights_path = '/media/HDD/models/resnet152_weights_tf.h5'
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
    name_pkl = path_data+'ResNet152_'+ kind +'_'+database+'_N'+str(N)+extL2+'.pkl'
    name_img = df_label[item_name][0]
    i = 0
    
    features_resnet = np.ones((sLength,size_output))
    classes_vectors = np.zeros((sLength,10))
    
    for i,name_img in  enumerate(df_label[item_name]):
        if i%1000==0:
            print(i,name_img)
        if database=='VOC12' or database=='Paintings':
                complet_name = path_to_img + name_img + '.jpg'
                im = cv2.imread(complet_name)
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
            list_im =  np.stack(get_Stretch_augmentation(resizedf,N=N,portion_size = (224, 224)))
            out = model.predict(list_im)
            out_average = np.mean(out,axis=0)
            if L2:
                out_norm = LA.norm(out_average) # L2 norm 
                out= out_average/out_norm
            #set_im = df_label['set'][i]
            #classes_vectors_tmp = np.zeros(10)
            #for j in range(10):
                #if( classes[j] in df_label['classe'][i]):
                    #classes_vectors_tmp[j] = 1
            
            #set_im = df_label['set'][i]
            #classes_vectors_tmp = np.zeros((N,10))
            #for j in range(10):
                #if( classes[j] in df_label['classe'][i]):
                    #classes_vectors_tmp[:,j] = 1
            #if(X_train is None) and (set_im=='train'):
                #X_train= out
                #y_train = classes_vectors_tmp
            #if(X_test is None) and (set_im=='test'):
                #X_test=  out
                #y_test = classes_vectors_tmp                
            #if(X_val is None) and (set_im=='validation'):
                #X_val=  out
                #y_val = classes_vectors_tmp          
            #if(set_im=='train'):
                #X_train = np.vstack((X_train,out))
                #y_train = np.vstack((y_train,classes_vectors_tmp))
            #if(set_im=='test'):
                #X_test= np.vstack((X_test,out))
                #y_test = np.vstack((y_test,classes_vectors_tmp))
            #if(set_im=='validation'):
                #X_val= np.vstack((X_val,out))
                #y_val = np.vstack((y_val,classes_vectors_tmp))       
        else:
            crop = resizedf[int(resized.shape[0]/2 - 112):int(resized.shape[0]/2 +112),int(resized.shape[1]/2-112):int(resized.shape[1]/2+112),:]        
            crop = np.expand_dims(crop, axis=0)
            out = model.predict(crop)[0]
            if L2:
                out_norm = LA.norm(out) 
                out /= out_norm
        features_resnet[i,:] = np.array(out)
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if( classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
     
    if database=='VOC12' or database=='Paintings':

        X_train = features_resnet[df_label['set']=='train',:]
        y_train = classes_vectors[df_label['set']=='train',:]
        
        X_test= features_resnet[df_label['set']=='test',:]
        y_test = classes_vectors[df_label['set']=='test',:]
        
        X_val = features_resnet[df_label['set']=='validation',:]
        y_val = classes_vectors[df_label['set']=='validation',:]
        
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
        
        Data = [X_train,y_train,X_test,y_test,X_val,y_val]
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(Data,pkl)
        
        return(X_train,y_train,X_test,y_test,X_val,y_val,df_label)
    
    else:
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(features_resnet,pkl)
        
        return(features_resnet)

def compute_InceptionResNetv2_features(kind='1536D',database='Paintings',L2=True,augmentation=True):
    """
    Inception ResNet v2 take RGB image as input
    """
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif(database=='Wikidata_Paintings'):
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
    sLength = len(df_label[item_name])
    checkpoint_file = '/media/HDD/models/inception_resnet_v2_2016_08_30.ckpt'
    name_img = df_label[item_name][0]
    i = 0
    classes_vectors = np.zeros((sLength,10))
    itera = 1000
    
    if(kind=='output'):
        size_output = 1001   
    elif(kind=='1536D'):
        size_output = 1536
    features_resnet = np.ones((sLength,size_output))
    name_pkl = path_data +'InceptionResNetv2_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    
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
                list_im = np.stack(get_Stretch_augmentation(resized,N=N,portion_size = (299, 299)))
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
            features_resnet[i,:] = np.array(out)
            if database=='VOC12' or database=='Paintings':
                for j in range(10):
                    if( classes[j] in df_label['classe'][i]):
                        classes_vectors[i,j] = 1

    if database=='VOC12' or database=='Paintings':

        X_train = features_resnet[df_label['set']=='train',:]
        y_train = classes_vectors[df_label['set']=='train',:]
        
        X_test= features_resnet[df_label['set']=='test',:]
        y_test = classes_vectors[df_label['set']=='test',:]
        
        X_val = features_resnet[df_label['set']=='validation',:]
        y_val = classes_vectors[df_label['set']=='validation',:]
        
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
        
        Data = [X_train,y_train,X_test,y_test,X_val,y_val]
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(Data,pkl)
        
        return(X_train,y_train,X_test,y_test,X_val,y_val,df_label)
    
    else:
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(features_resnet,pkl)
        
        return(features_resnet)
    
def compute_VGG_features(VGG='19',kind='fuco7',database='Paintings',L2=True,augmentation=False):
    """
    VGG take BGR image as input
    """
    path_data = 'data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        databasetxt = path_data + database + '.txt'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/224/'
    df_label = pd.read_csv(databasetxt,sep=",")
    if augmentation:
        N = 50
    else: 
        N=1
    if L2:
        extL2 = '_L2'
    else:
        extL2 = ''
    sLength = len(df_label[item_name])
    name_img = df_label[item_name][0]
    i = 0
    classes_vectors = np.zeros((sLength,10))
    name_pkl = path_data+'VGG'+VGG+'_'+ kind +'_'+database +'_N'+str(N)+extL2+'.pkl'
    if(kind=='fuco8'):
        size_output = 1000
    elif(kind in ['fuco6','relu7','relu6','fuco7']):
        size_output = 4096
    
    features_resnet = np.ones((sLength,size_output))
    
    
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
        if i%1000==0:
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
        resizedf[:,:,2] -= 103.939
        resizedf[:,:,1] -= 116.779
        resizedf[:,:,0] -= 123.68
        
        if(augmentation==True):
            list_im =  np.stack(get_Stretch_augmentation(resizedf,N=N,portion_size = (224, 224)))
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
        features_resnet[i,:] = np.array(out)
        if database=='VOC12' or database=='Paintings':
            for j in range(10):
                if( classes[j] in df_label['classe'][i]):
                    classes_vectors[i,j] = 1
        
    if database=='VOC12' or database=='Paintings':

        X_train = features_resnet[df_label['set']=='train',:]
        y_train = classes_vectors[df_label['set']=='train',:]
        
        X_test= features_resnet[df_label['set']=='test',:]
        y_test = classes_vectors[df_label['set']=='test',:]
        
        X_val = features_resnet[df_label['set']=='validation',:]
        y_val = classes_vectors[df_label['set']=='validation',:]
        
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_val.shape,y_val.shape)
        
        Data = [X_train,y_train,X_test,y_test,X_val,y_val]
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(Data,pkl)
        
        return(X_train,y_train,X_test,y_test,X_val,y_val,df_label)
    
    else:
        
        with open(name_pkl, 'wb') as pkl:
            pickle.dump(features_resnet,pkl)
        
        return(features_resnet)
    
if __name__ == '__main__':
    
    ## ResNet 152
#    Compute_ResNet(kind='2048D',database='VOC12',L2=True,augmentation=True)
#    print("ResNet")
#    Classification_evaluation('2048D',kindnetwork='ResNet152',database='VOC12',L2=True,augmentation=True)
#    
#    ## InceptionResnetV2 
#    print("InceptionResNet")
#    #compute_InceptionResNetv2_features(kind='1536D',database='VOC12',L2=False,augmentation=False)
#    #Classification_evaluation('1536D',kindnetwork='InceptionResNetv2',database='VOC12',L2=False,augmentation=False)
#    compute_InceptionResNetv2_features(kind='1536D',database='VOC12',L2=True,augmentation=True)
#    Classification_evaluation('1536D',kindnetwork='InceptionResNetv2',database='VOC12',L2=True,augmentation=True)
#    
#    
#    ## VGG16
#    kind = 'relu7'
#    VGGnum = '16'
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='VOC12',L2=False,augmentation=False)
#    Classification_evaluation(kind=kind,kindnetwork='VGG16',database='VOC12',L2=False,augmentation=False)
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='VOC12',L2=True,augmentation=True)
#    Classification_evaluation(kind=kind,kindnetwork='VGG16',database='VOC12',L2=True,augmentation=True)
#    
#    ## VGG19
#    print("VGG19")
#    kind = 'relu7'
#    VGGnum = '19'
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='VOC12',L2=True,augmentation=True)
#    Classification_evaluation(kind=kind,database='VOC12',kindnetwork='VGG19',L2=True,augmentation=True)
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='VOC12',L2=False,augmentation=False)
#    Classification_evaluation(kind=kind,database='VOC12',kindnetwork='VGG19',L2=False,augmentation=False)
#    
#    kind = 'relu6'
#    VGGnum = '19'
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='VOC12',L2=False,augmentation=False)
#    Classification_evaluation(kind=kind,database='VOC12',kindnetwork='VGG19',L2=False,augmentation=False)
#      
#    kind = 'relu7'
#    VGGnum = '16'
#    compute_VGG_features(VGG=VGGnum,kind=kind,database='Paintings',L2=True,augmentation=True)
#    Classification_evaluation(kind=kind,kindnetwork='VGG16',L2=True,augmentation=True)
    
    ## Computation on VOC12
    #Compute_ResNet(kind='2048D',database='VOC12',L2=True,augmentation=False)
    #Compute_ResNet(kind='2048D',database='Paintings',L2=True,augmentation=False)
    
    # Plot SVM support and error
    #HowAreSupport_of_SVM(kind='2048D',kindnetwork='ResNet152',database='Paintings',L2=False,augmentation=False)
    
    # Compute bottleNeck for Wikidata
    compute_InceptionResNetv2_features(kind='1536D',database='Wikidata_Paintings',L2=False,augmentation=False)
    kind = 'relu7'
    VGGnum = '19'
    compute_VGG_features(VGG=VGGnum,kind=kind,database='Wikidata_Paintings',L2=False,augmentation=False)
    Compute_ResNet(kind='2048D',database='Wikidata_Paintings',L2=False,augmentation=False)