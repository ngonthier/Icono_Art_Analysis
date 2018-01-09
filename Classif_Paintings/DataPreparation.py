#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:03:42 2017

@author: gonthier
"""

import pandas as pd
from pandas import Series
import urllib.request
import numpy as np
import random


def f(x):
     return Series(dict(name_img = x['name_img'].min(), 
                        set = x['set'].min(), 
                        classe = "%s" % ' '.join(x['classe'])))

def fusion_wikidata(x):
     return Series(dict(item = x['item'].min(), itemLabel = x['itemLabel'].min(), 
                        itemDescription = x['itemDescription'].min(), image = x['image'].min(),  
                        depictsIconoclass = "%s" % ' '.join(x['depictsIconoclass']),
                        createur = x['createur'].min(),
                        depicts = "%s" % ' '.join(x['depicts']),
                        country = x['country'].min(),
                        year = x['year'].min(),depictsLabel =  "%s" % ' '.join(x['depictsLabel'])
                        ))

def prepareVOC12():
     classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
     path_to_VOC12_imageset = '/media/HDD/data/VOCdevkit/VOC2012/ImageSets/Main'
     set_list = ['train','validation']

     frames = []

     for classe in classes:
         for type_set in set_list:
             if type_set == 'train':
                 postfix = '_train.txt'
             elif type_set == 'validation':
                 postfix = '_val.txt'
             name_train = path_to_VOC12_imageset +'/'+classe+postfix
             data = pd.read_csv(name_train, sep="\s+|\t+|\s+\t+|\t+\s+", header=None)
             data.columns = ["name_img", "classe"]
             data =  data.loc[data['classe'].isin(['1'])]
             num_samples,_ = data.shape
             data.insert(1, "set", [type_set]*num_samples)
             data['classe'] = data['classe'].apply(lambda x: classe)
             frames += [data]
    
     result = pd.concat(frames)
     result = result.groupby('name_img').apply(f)
     #result.groupby('name_img').agg(dict(name_img = 'min', set = 'min', classe = lambda x: '%s'%', '.join(x)))
     #result = result.groupby('name_img').apply(f)
     
     result.to_csv('VOC12.txt', index=None, sep=',', mode='w')
     
         # Test
     df_test = pd.read_csv('VOC12.txt',sep=",")
     print("VOC12")
     print(df_test.head(11))
     print(len(df_test['classe']))

def preparePaintings():
    name_file = '/media/HDD/data/Painting_Dataset/painting_dataset_updated.csv'
    df = pd.read_csv(name_file, sep=",")
    df.columns = ['a','name_img','page','set','classe']
    df = df.drop('a', 1)
    df = df.drop('page', 1)
    df = df[df['name_img'] != '[]']
    df['name_img'] = df['name_img'].apply(lambda a: str.split(a,'/')[-1])
    df['name_img'] = df['name_img'].apply(lambda a: str.split(a,'.')[0])
    
    df['set'] = df['set'].apply(lambda a: a.replace('\'',''))
    
    df['classe'] = df['classe'].apply(lambda a: a.replace('\'',''))
    print((df['classe'].str.contains('bird')).sum())
    df.to_csv('Paintings.txt', index=None, sep=',', mode='w')
    
    # Test
    df_test = pd.read_csv('Paintings.txt',sep=",")
    print("Paintings")
    print(df_test.head(11))
    print(len(df_test['classe'])) # Must be 8621, to count the number of jpeg images find *.jpg | wc -l
    
def prepareWikiData():
    name_file_paitings = '/media/HDD/Wikidata_query/query_paitings_wikidata.csv'
    
    name_file_unique_url_paitings = '/media/HDD/Wikidata_query/paitings_wikidata.csv'
    
    name_file_print = '/media/HDD/Wikidata_query/query_estampe_wikidata.csv'
    df_estampe = pd.read_csv(name_file_print, sep=",")


    
    name_file_class = '/media/HDD/Wikidata_query/query_Depict_class.csv'
    df_class = pd.read_csv(name_file_class, sep=",")

    df = pd.read_csv(name_file_paitings, sep=",",encoding='utf-8')
    
    # Number of paitings without taking into account the double 
    
    get_im = df['image']
    get_im_unique = get_im.drop_duplicates()
    print("Number of paintings with doublons",len(get_im_unique))
    get_im_unique.to_csv(name_file_unique_url_paitings, index=None, sep=',', mode='w')
    get_im = df_estampe['image']
    get_im_unique = get_im.drop_duplicates()
    print("Number of prints with doublons",len(get_im_unique))
    get_im_unique.to_csv(name_file_unique_url_paitings, index=None, sep=',', mode='w')
    
    df_drop_paitings = df.drop_duplicates(subset='item', keep="last")
    print("Number of different item in Paintings",len(df_drop_paitings))
    get_im_unique = df_drop_paitings['image']
    print("Number of paitings",len(get_im_unique))
    #

    df_drop_prints = df_estampe.drop_duplicates(subset='item', keep="last")
    print("Number of different item in Prints",len(df_drop_prints))
    get_im_unique_estampe = df_drop_prints['image']
    print("Number of Prints",len(get_im_unique_estampe))
   
    df['createur'] = df['createur'] .astype('str')
    df['itemDescription'] = df['itemDescription'] .astype('str')
    df['depictsIconoclass'] = df['depictsIconoclass'] .astype('str')
    df['depicts'] = df['depicts'].fillna(value='')
    df['year'] = df['year'].astype('str')
   
    df_estampe['itemDescription'] = df_estampe['itemDescription'] .astype('str')
    df_estampe['createur'] = df_estampe['createur'] .astype('str')
    df_estampe['depictsIconoclass'] = df_estampe['depictsIconoclass'] .astype('str')
    df_estampe['depicts'] = df_estampe['depicts'].fillna(value='')
    df_estampe['year'] = df_estampe['year'].astype('str')
    
    df_copy = df.copy()
    df_estampe_copy = df_estampe.copy()
    
    list_column_to_change = ['item','image','createur','depicts','country']
    for elt  in list_column_to_change:
        #print(elt)
        df_copy[elt] = df[elt].apply(lambda a: str.split(str(a),'/')[-1])
        df_estampe_copy[elt] = df_estampe[elt].apply(lambda a: str.split(str(a),'/')[-1])
        
    df_copy['image'] = df_copy['image'].apply(lambda a: urllib.request.unquote(a))
    df_estampe_copy['image'] = df_estampe_copy['image'].apply(lambda a: urllib.request.unquote(a))
    
    df_class = df_class.drop('count',axis=1)
    df_class =  df_class.append(['',''])
    df_class['depictsLabel'] = df_class['depictsLabel'] .astype('str')
    elt='depicts'
    df_class[elt] = df_class[elt].apply(lambda a: str.split(str(a),'/')[-1]) 
    df_copy = df_copy.join(df_class.set_index(elt), on=elt)
    df_copy['depictsLabel'] = df_copy['depictsLabel'] .astype('str')
    #print(df_copy.head(5))
    df_estampe_copy = df_estampe_copy.join(df_class.set_index(elt), on=elt)
    df_estampe_copy['depictsLabel'] = df_estampe_copy['depictsLabel'] .astype('str') 
    
    df_copy2 = df_copy.groupby('item').apply(fusion_wikidata)
    df_estampe_copy2 = df_estampe_copy.groupby('item').apply(fusion_wikidata)

    

#    df_copy2 = df_copy2.drop_duplicates(subset='item', keep="last")
#    df_estampe_copy2 = df_estampe_copy2.drop_duplicates(subset='item', keep="last")


    df_copy2.to_csv('data/Wikidata_Paintings.txt', index=None, sep=',', mode='w')
    
    # Test
    df_test = pd.read_csv('data/Wikidata_Paintings.txt',sep=",", encoding='utf-8')
    print("Wikidata Paintings")
    print(df_test.head(2))
    print("Number of Paintings : ",len(df_test['image']))
    
    
    df_estampe_copy2.to_csv('data/Wikidata_Prints.txt', index=None, sep=',', mode='w')
    
    # Test
    df_test = pd.read_csv('data/Wikidata_Prints.txt',sep=",", encoding='utf-8')
    print("Prints")
    print(df_test.head(2))
    print("Number of Prints : ",len(df_test['image']))
    
    return(0)
    
def contains_word(s, w):
    return((' ' + w + ' ') in (' ' + s + ' '))
    
def prepareWikidataSetsPaitings():
    """
    The goal of this file is to create a training and a test sets
    """
    name_file_class = '/media/HDD/Wikidata_query/query_Depict_paintings.csv'
    df_class = pd.read_csv(name_file_class, sep=",")
    df_class['depicts'] = df_class['depicts'].apply(lambda a: str.split(str(a),'/')[-1]) 
    number_elt = 500
    df_reduc = df_class[df_class['count']>number_elt]
    df_test = pd.read_csv('data/Wikidata_Paintings.txt',sep=",", encoding='utf-8')
    df_test = df_test.drop_duplicates(subset='item', keep="last")
    print("Number of Paitings",len(df_test['item']))

    number_paitings = len(df_test['image'])
    number_elt_in_training = int(number_elt/2)
    depicts = df_reduc['depicts'][::-1]
    #number_class = len(depictsLabel)
    df_copy = df_test.copy()
    df_copy['set'] = Series(np.ones(number_paitings), index=df_copy.index)
    #df_copy['class'] = Series(-1*np.ones(number_paitings), index=df_copy.index)
    classes = df_reduc['depictsLabel']
    list_im_in_training = []
    for depict in depicts:
        df_copy[depict] = Series(np.zeros(number_paitings), index=df_copy.index)
        print(df_class[df_class['depicts']==depict])
        # Create the list of the image with the depict elt
        list_image_with_it = []
        for i in range(number_paitings):
            depicts_elts = df_copy.iloc[i]['depicts']
            #print(depicts_elts)
            if not(str(depicts_elts) == 'nan'):
                if contains_word(depicts_elts,depict):
                    #print(depicts_elts)
                    list_image_with_it += [i]
                    df_copy.loc[i, depict] = 1
            else: # If the lsit is empty
                df_copy.loc[i, depict] = -1
        #print(list_image_with_it)
        print("Number of image with this depict",len(list_image_with_it))
        #print(df_class[df_class['depicts']==depict]['count'])
        #print(np.array(df_class[df_class['depicts']==depict]['count']))
        if not(len(list_image_with_it)==np.array(df_class[df_class['depicts']==depict]['count'])):
            print("For ",depict,"element missings")
        
        already_choiced  =  np.intersect1d(list_im_in_training,list_image_with_it)
        number_to_choose = number_elt_in_training - len(already_choiced)
        #print("Number images to choose",number_to_choose)
        not_choiced_yet = np.setdiff1d(list_image_with_it,list_im_in_training)
        #print(not_choiced_yet)
        index_choosed = random.sample(list(not_choiced_yet), k=number_to_choose)
        #print(random_index)
        #index_choosed = not_choiced_yet[random_index]
        #print(len(index_choosed))
        #print(index_choosed)
        for k in index_choosed:
            df_copy.loc[k, 'set'] = 0

        if len(list_im_in_training)==0:
            list_im_in_training = np.asarray(index_choosed)
        else:
            list_im_in_training = np.asarray(index_choosed)
        print("list_im_in_training",len(list_im_in_training))
          
    # Petit test avant la fin
    print("Number of Images in the training set ",np.sum(df_copy['set']==0))
            
    df_copy.to_csv('data/Wikidata_Paintings_sets.txt', index=None, sep=',', mode='w')             
    
#    # Test
    df_test = pd.read_csv('data/Wikidata_Paintings_sets.txt',sep=",", encoding='utf-8')
    print("Wikidata Paintings Sets")
    print(df_test.head(11))
    print("Number of Paintings : ",len(df_test['image']))
    
    return(0)
    
if __name__ == '__main__':
    #prepareVOC12()
    #preparePaintings()
    #prepareWikiData()
    prepareWikidataSetsPaitings()
