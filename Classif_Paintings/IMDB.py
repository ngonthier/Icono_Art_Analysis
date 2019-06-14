#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:54:36 2019

@author: gonthier
"""

import pandas as pd
import os

def get_database(database):
    ext = '.txt'
    default_path_imdb = '/media/gonthier/HDD/data/'
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/gonthier/HDD/data/Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = 'VOCdevkit/VOC2012/JPEGImages/'
    elif database=='VOC2007':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'VOCdevkit/VOC2007/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif database=='watercolor':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'cross-domain-detection/datasets/watercolor/JPEGImages/'
        classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
    elif database=='PeopleArt':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'PeopleArt/JPEGImages/'
        classes =  ["person"]
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        ext = '.csv'
        item_name = 'item'
        path_to_img = 'Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
    elif 'IconArt_v1' in database:
            ext='.csv'
            item_name='item'
            classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien']
            path_to_img = 'Wikidata_Paintings/IconArt_v1/JPEGImages/'
    elif(database=='RMN'):
            ext='.csv'
            item_name='item'
            classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien']
            path_to_img = 'RMN/JPEGImages/'
    elif database=='clipart':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'cross-domain-detection/datasets/clipart/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = 'data/Wikidata_Paintings/600/'
        print(database,' is not implemented yet')
        raise NotImplementedError # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = 'Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    else:
        print('This database don t exist :',database)
        raise NotImplementedError
    num_classes = len(classes)
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    Not_on_NicolasPC = False
    if not(os.path.exists(path_data)): # Thats means you are not on the Nicolas Computer
        # Modification of the path used
        Not_on_NicolasPC = True
        print('you are not on the Nicolas PC, so I think you have the data in the data folder')
        path_tmp = 'data/' 
        path_to_img = path_tmp + path_to_img
        path_data = path_tmp + 'ClassifPaintings/'
        if 'IconArt_v1' in database:
            path_data_csvfile = path_tmp+'Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
        elif database=='RMN':
            path_data_csvfile = path_tmp+'RMN/ImageSets/Main/'
        else:
            path_data_csvfile = path_data
    else:
        path_to_img = default_path_imdb + path_to_img
#        path_to_img = '/media/gonthier/HDD/data/' + path_to_img
#        dataImg_path = '/media/gonthier/HDD/data/'
        if 'IconArt_v1' in database:
            path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
        elif database=='RMN':
            path_data_csvfile = '/media/gonthier/HDD/data/RMN/ImageSets/Main/'
        else:
            path_data_csvfile = path_data
    
    databasetxt = path_data_csvfile + database + ext

    if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                      'turban':int,'set':str,'Anno':int}
    elif 'IconArt_v1' in database or 'IconArt_v1'==database:
        dtypes = {0:str,'item':str,'angel':int,\
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,\
                      'ruins':int,'Saint_Sebastien':int,\
                      'set':str,'Anno':int}
    elif database=='VOC2007':
        dtypes = {0:str,'name_img':str,'aeroplane':int, 'bicycle':int, 'bird':int, 'boat':int,\
                  'bottle':int, 'bus':int, 'car':int, 'cat':int, 'chair':int,\
                  'cow':int, 'diningtable':int, 'dog':int, 'horse':int,\
                  'motorbike':int, 'person':int, 'pottedplant':int,\
                  'sheep':int, 'sofa':int, 'train':int, 'tvmonitor':int}
         
    elif database=='RMN':
        dtypes = {'item':str,'Saint_Sebastien':int}
    else:
        dtypes = {}
        dtypes[item_name] =  str
        for c in classes:
            dtypes[c] = int
    df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
    str_val = 'val'
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
        str_val = 'validation'
    elif database=='Paintings':
        str_val = 'validation'
    elif database in ['VOC2007','watercolor','clipart','PeopleArt']:
        str_val = 'val'
        df_label[classes] = df_label[classes].apply(lambda x:(x + 1.0)/2.0)
    
    return(item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC)