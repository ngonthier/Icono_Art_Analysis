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
from sklearn.metrics import average_precision_score

def RF_on_output():
    # Multilabel classification assigns to each sample a set of target labels. 
    # This can be thought as predicting properties of a data-point that are not mutually exclusive
    
    [X_train,y_train,X_test,y_test,X_val,y_val] = pickle.load(open('ResNet_output.pkl', 'rb'))
    
    myForest = RandomForestClassifier(n_estimators=100)
    
    X_trainval = np.append(X_train,X_val,axis=0)
    y_trainval = np.append(y_train,y_val,axis=0)
    
    
    X_trainval[X_trainval > 0.1] = 1
    X_trainval[X_trainval <= 0.1] = 0
    print('start learning')
    myForest.fit(X_trainval,y_trainval)
    print('End learning')
    X_test[X_test > 0.1] = 1
    X_test[X_test <= 0.1] = 0
    
    y_predict = myForest.predict(X_test)
    AP = average_precision_score(y_test,y_predict)
    print('Average precision',AP)
    print("Accuray ",myForest.score(X_test,y_test))
    
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
        
        im = cv2.resize(cv2.imread(complet_name), (224, 224)).astype(np.float32)
        # Remove train image mean
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = np.expand_dims(im, axis=0)
        out = model.predict(im)
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
    #RF_on_output()