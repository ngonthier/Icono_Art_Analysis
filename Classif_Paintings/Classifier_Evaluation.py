#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:51:16 2017

@author: gonthier
"""

import cv2
import resnet_152_keras
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.metrics import average_precision_score,recall_score,make_scorer,precision_score,label_ranking_average_precision_score,classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from Custom_Metrics import ranking_precision_score,VOCevalaction,computeAveragePrecision

def Classification_evaluation(kind='2048D',classifier_name='LinearSVM'):
    # Multilabel classification assigns to each sample a set of target labels. 
    # This can be thought as predicting properties of a data-point that are not mutually exclusive
    
    if kind =='output':
        name_pkl = 'ResNet_output.pkl'
    elif kind == '2048D':
        name_pkl = 'ResNet_2048D.pkl'
    [X_train,y_train,X_test,y_test,X_val,y_val] = pickle.load(open(name_pkl, 'rb'))
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
    k_tab = [5,10,20,50,100]
    if classifier_name=='LinearSVM':
        classifier = LinearSVC()
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
        AP_per_class = []
        cs = np.logspace(-5, 7, 40)
        cs = np.logspace(-5, -2, 15)
        param_grid = dict(C=cs)
        custom_cv = PredefinedSplit(np.hstack((np.zeros((1,X_train.shape[0])),np.ones((1,X_val.shape[0])))).reshape(-1,1))
        # For custom_cv = zip(train_indices, test_indices) that's is used by Crowley but a better cross validation method is possible 
        # TODOcv=ShuffleSplit(train_size=train_size,n_splits=250, random_state=1)
        y_predict_all_label = np.zeros_like(y_test)
        for i,classe in enumerate(classes):
            grid = GridSearchCV(classifier, refit=True,scoring =make_scorer(average_precision_score), param_grid=param_grid,n_jobs=-1,
                            cv=custom_cv)
            grid.fit(X_trainval,y_trainval[:,i]) 

#            best_AP = 0
#            for c in cs:
#                classifier = ap = computeAveragePrecision(rec, prec, use_07_metric) LinearSVC(C=c)
#                classifier.fit(X_train,y_train[:,i])
#                y_predict_val = classifier.decision_function(X_val)
#                AP = average_precision_score(y_val[:,i],y_predict_val)
#                if(AP > best_AP):
#                    grid = LinearSVC(C=c)
#                    grid.fit(X_trainval,y_trainval[:,i])
            y_predict_confidence_score = grid.decision_function(X_test)
            y_predict_test = grid.predict(X_test) 
            y_predict_trainval = grid.predict(X_trainval) 
            # Warning ! predict provide class labels for samples whereas decision_function provide confidence scores for samples.
            AP = average_precision_score(y_test[:,i],y_predict_confidence_score)
            AP_per_class += [AP]
            print("Average Precision for",classe," = ",AP)
            
#            recalls, precisions = VOCevalaction(y_test[:,i], y_predict_confidence_score) # To compare the AP computation
#            AP_VOC12 = computeAveragePrecision(recalls, precisions, use_07_metric=False)
#            AP_VOC07 = computeAveragePrecision(recalls, precisions, use_07_metric=True)
#            print("Average Precision for {0:s} : AP = {1:.2f}, VOC12 = {2:.2f}, VOC07 = {3:.2f}".format(classe,AP,AP_VOC12,AP_VOC07))
            
#            training_precision = precision_score(y_trainval[:,i],y_predict_trainval)
#            print("Training precision :{0:.2f}".format(training_precision))
#            test_precision = precision_score(y_test[:,i],y_predict_test)
#            test_recall = recall_score(y_test[:,i],y_predict_test)
#            print("Test precision = {0:.2f}, recall = {1:.2f}".format(test_precision,test_recall))
            y_predict_all_label[:,i] =  y_predict_confidence_score
#            precision_at_k_tab = []
#            for k in k_tab:
#                precision_at_k = ranking_precision_score(y_test[:,i], y_predict_confidence_score,k)
#                precision_at_k_tab += [precision_at_k]
#                print("Precision @ ",k,":",precision_at_k)
            
        print("mean Average Precision = {0:.2f}".format(np.mean(AP_per_class)))
#        lr_AP = label_ranking_average_precision_score(y_test,y_predict_all_label)
#        print("label Ranking Average Precision {0:.2f}".format(lr_AP))
        
    elif classifier_name=='RF': # Random Forest that can deal with multilabel
        classifier = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1)
        classifier.fit(X_trainval,y_trainval)
        y_predict = classifier.predict(X_test)
        AP = average_precision_score(y_test,y_predict)
        print('Average precision',AP)
        print("Accuray ",classifier.score(X_test,y_test))
    
    return(0)
   
def Compute_ResNet(kind='output'):
    classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    df_label = pd.read_csv('Paintings.txt',sep=",")
    
    sLength = len(df_label['name_img'])
    #df_label_augmented = df_label.assign(resnet_output=pd.Series(np.ones(sLength)))
    #df_label_augmented = df_label.assign(resnet_output=pd.Series(np.ones(sLength)).values)
    
    # Load the ResNet 152
    weights_path = '/media/HDD/models/resnet152_weights_tf.h5'
    if(kind=='output'):
        model = resnet_152_keras.resnet152_model(weights_path)
        size_output = 1000
        name_pkl = 'ResNet_output.pkl'
    elif(kind=='2048D'):
        model = resnet_152_keras.resnet152_model_2018output(weights_path)
        size_output = 2048
        name_pkl = 'ResNet_2048D.pkl'
    name_img = df_label['name_img'][0]
    i = 0
    
    features_resnet = np.ones((sLength,size_output))
    classes_vectors = np.zeros((sLength,10))
    
    for i,name_img in  enumerate(df_label['name_img']):
        print(i,name_img)
        complet_name = path_to_img + name_img + '.jpg'
        
        # Crowley : each image is downsized so that its smallest dimension is 224 pixels 
        # and then a 224 × 224 frame is extracted from the centre
        # To shrink an image, it will generally look best with CV_INTER_AREA interpolation
        im = cv2.imread(complet_name)
        if(im.shape[0] < im.shape[1]):
            dim = (224, int(im.shape[1] * 224.0 / im.shape[0]))
        else:
            dim = (int(im.shape[0] * 224.0 / im.shape[1]),224)
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR) # INTER_AREA
        crop = resized[int(resized.shape[0]/2 - 112):int(resized.shape[0]/2 +112),int(resized.shape[1]/2-112):int(resized.shape[1]/2+112),:].astype(np.float32)
        #im = cv2.resize(cv2.imread(complet_name), (224, 224)).astype(np.float32)
        # Remove train image mean
        crop[:,:,0] -= 103.939
        crop[:,:,1] -= 116.779
        crop[:,:,2] -= 123.68
        crop = np.expand_dims(crop, axis=0)
        out = model.predict(crop)
        features_resnet[i,:] = np.array(out[0])
        for j in range(10):
            if( classes[j] in df_label['classe'][i]):
                classes_vectors[i,j] =  1
        #print(out[0].tolist())
        #print(out)
        #df_label_augmented['resnet_output'][df_label_augmented['name_img']==name_img] = out[0].tolist()
        
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
#        
#    # Generate the RF 
#    
#    classe = 'bird'
#    
#    (df_label['classe'].str.contains(classe)).sum()
#    
#    df_label.select(df_label['classe'] == classe)
#    
#    out = model.predict(im)
#    
#    
#    im = cv2.resize(cv2.imread(), ).astype(np.float32)


if __name__ == '__main__':
    Compute_ResNet('2048D')
    Classification_evaluation('2048D')
#Average Precision for aeroplane  =  0.667269500467
#Average Precision for aeroplane : AP = 0.67, VOC12 = 0.68, VOC07 = 0.65
#Training precision :1.00
#Test precision = 0.84, recall = 0.54
#Average Precision for bird  =  0.43263129913
#Average Precision for bird : AP = 0.43, VOC12 = 0.43, VOC07 = 0.44
#Training precision :0.97
#Test precision = 0.71, recall = 0.20
#Average Precision for boat  =  0.919813881584
#Average Precision for boat : AP = 0.92, VOC12 = 0.92, VOC07 = 0.89
#Training precision :0.92
#Test precision = 0.88, recall = 0.78
#Average Precision for chair  =  0.723004427855
#Average Precision for chair : AP = 0.72, VOC12 = 0.73, VOC07 = 0.71
#Training precision :0.90
#Test precision = 0.70, recall = 0.62
#Average Precision for cow  =  0.588617184568
#Average Precision for cow : AP = 0.59, VOC12 = 0.59, VOC07 = 0.59
#Training precision :0.92
#Test precision = 0.76, recall = 0.33
#Average Precision for diningtable  =  0.688512318176
#Average Precision for diningtable : AP = 0.69, VOC12 = 0.69, VOC07 = 0.68
#Training precision :0.88
#Test precision = 0.73, recall = 0.51
#Average Precision for dog  =  0.52862709804
#Average Precision for dog : AP = 0.53, VOC12 = 0.53, VOC07 = 0.53
#Training precision :0.94
#Test precision = 0.69, recall = 0.33
#Average Precision for horse  =  0.760910214053
#Average Precision for horse : AP = 0.76, VOC12 = 0.76, VOC07 = 0.74
#Training precision :0.93
#Test precision = 0.80, recall = 0.60
#Average Precision for sheep  =  0.670122581957
#Average Precision for sheep : AP = 0.67, VOC12 = 0.67, VOC07 = 0.66
#Training precision :0.92
#Test precision = 0.77, recall = 0.41
#Average Precision for train  =  0.847026669653
#Average Precision for train : AP = 0.85, VOC12 = 0.85, VOC07 = 0.81
#Training precision :1.00
#Test precision = 0.89, recall = 0.69
#mean Average Precision = 0.68
