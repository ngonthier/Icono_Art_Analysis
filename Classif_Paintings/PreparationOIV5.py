#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:17:07 2019

@author: gonthier
"""
import pandas as pd
import numpy as np
from shutil import copyfile
#trainset = 'challenge-2019-train-detection-human-imagelabels_expanded.csv'
path = '/media/gonthier/HDD/data/OIV5/'
#df = pd.read_csv(path+trainset)
#nameID = df['ImageID']
#nameIDuniqueTrain = np.unique(nameID)
#LabelName = df['LabelName']

trainbox = 'challenge-2019-train-detection-bbox.csv'
valbox = 'challenge-2019-validation-detection-bbox.csv'
dftrainbox = pd.read_csv(path+trainbox)
dfvalbox = pd.read_csv(path+valbox)
trainboxexpanded = 'challenge-2019-train-detection-bbox_expanded.csv'
dftrainboxexpanded = pd.read_csv(path+trainboxexpanded)
nameID = dftrainboxexpanded['ImageID']
nameIDuniqueTrain = np.unique(nameID)
LabelName = dftrainboxexpanded['LabelName']
LabelNameunique = np.unique(LabelName)

nameIDval = dfvalbox['ImageID']
uniquenameIDval = np.unique(nameIDval)


# To produce the validation set !
columns = ['item'] + list(LabelNameunique) + ['set']
df_full = pd.DataFrame(columns=columns)

index = 0 
for row in dfvalbox.iterrows():
    iteme = row[1]['ImageID']
    label = row[1]['LabelName']
    test = df_full['item']==iteme
    if  not(test.any()):
        num_labels = np.zeros((len(LabelNameunique),))
        num_labels[np.where(LabelNameunique==label)[0]] =1.
        new_row = [iteme] + list(num_labels) + ['test']
        df_full.loc[index] = new_row
        index += 1
        if index%1000==0:
            print('Val :',index)
    else:
        df_full.loc[np.where(df_full['item']==iteme)[0][0],label] = 1.

df_test = df_full
df_full.to_csv(path+'OIV5_test.csv',index=False) 
   
index = 0 
limit_r = 30000 
df_full = pd.DataFrame(columns=columns)
for row in dftrainbox.iterrows():
    if index > limit_r:
        break
    else:
        iteme = row[1]['ImageID']
        label = row[1]['LabelName']
        test = df_full['item']==iteme
        if len(test)==0 or not(test.any()):
            num_labels = np.zeros((len(LabelNameunique),))
            num_labels[np.where(LabelNameunique==label)[0]] =1.
            new_row = [iteme] + list(num_labels) + ['train']
            df_full.loc[index] = new_row
            index += 1
            if index%1000==0:
                print('Test :',index)
        else:
            df_full.loc[np.where(df_full['item']==iteme)[0][0],label] = 1.

df_full.to_csv(path+'OIV5_train'+str(limit_r)+'.csv',index=False)     
   
print('Number of images  :',index) 

number_elts = 3000
df_train = df_full[df_full['set']=='train']
number_img_intrain = 0
num_elt_per_class = np.zeros(shape=(len(LabelNameunique),))
elt_per_class = 0
new_df_train = None
df_train_drop = df_train.drop(['item','set'], axis=1)
suma = df_train_drop.sum(axis=0)
print('The label without images : ',np.where(suma==0.))
while number_img_intrain < number_elts:
    for num_label , label in enumerate(LabelNameunique):
        if num_elt_per_class[num_label] < elt_per_class or True:
            a = df_train[label]==1.
            if a.any():
                df_subset = df_train[a].sample(1)
                df_subset_drop = df_subset.drop(['item','set'], axis=1)
                num_elt_per_class += df_subset_drop.values.reshape((len(LabelNameunique),))
                df_train = df_train.drop(df_subset.index)
                if new_df_train is None:
                    new_df_train = df_subset
                else:
                    new_df_train = new_df_train.append(df_subset)
                number_img_intrain += 1
#        print(new_df_train)
    elt_per_class += 1
new_df_train_old = new_df_train
new_df_train = new_df_train_old
#b = new_df_train.drop(['item','set'], axis=1).sum()
label_missing = LabelNameunique[np.where(new_df_train.drop(['item','set'], axis=1).sum()==0.)]
new_index = len(new_df_train) +1
for label in label_missing:
    dflocal = dftrainboxexpanded[dftrainboxexpanded['LabelName']==label]
    #print(label,dflocal.head())
    nameIm = dflocal.values[0,0]
    labels_on_Im = dftrainboxexpanded[dftrainboxexpanded['ImageID']==nameIm]['LabelName'].values
    num_labels = np.zeros((len(LabelNameunique),))
    num_labels[np.where(LabelNameunique==label)[0]] =1.
    for l in labels_on_Im:
        num_labels[np.where(LabelNameunique==l)[0]] =1.
    new_row = [nameIm] + list(num_labels) + ['train']
    new_df_train.loc[new_index] = new_row
    new_index += 1
    
number_elts = len(new_df_train)
print(np.min(np.sum(new_df_train.drop(['item','set'], axis=1))))

new_df_train.to_csv(path+'OIV5_smalltrain_'+str(number_elts)+'.csv',index=False) 

new_df_full = new_df_train.append(df_test)
new_df_full.to_csv(path+'OIV5_small_'+str(number_elts)+'.csv',index=False) 

path_source = '/media/gonthier/Lacie Grey Gonthier/OIv5/'
path_source_bis = '/media/gonthier/Lacie Grey Gonthier/OIv5/validationV5/validation/'
import glob
list_im_val = glob.glob(path_source_bis+'*.jpg')
new_index = len(new_df_full) +1
for elt in list_im_val:
    et = elt.split('/')[-1]
    e = et.split('.')[0]
    a  = new_df_full['item']==e
    if not(a.any()):
        num_labels = np.zeros((len(LabelNameunique),))
        new_row = [nameIm] + list(num_labels) + ['test']
        new_df_full.loc[new_index] = new_row
        new_index += 1
new_df_full.to_csv(path+'OIV5_small_'+str(number_elts)+'.csv',index=False) 
    
path_target = '/media/gonthier/HDD/data/OIV5/Images/'
list_FileNotFoundError = []
import os
for row in new_df_full.iterrows():
    iteme = row[1]['item']
    set_ = row[1]['set']
    if set_=='train':
        str_set_ = 'train'
        src = path_source + str_set_ +'/' + iteme +'.jpg'
        
    else:
        str_set_ = 'validation'
        src = path_source_bis + iteme +'.jpg'

    
    dst = path_target + iteme +'.jpg'
    try:
        if  not(os.path.isfile(dst)):
            copyfile(src, dst)
    except FileNotFoundError:
        list_FileNotFoundError += [iteme]
        print(set_,iteme)
        pass        
    