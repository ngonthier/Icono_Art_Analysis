#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:03:42 2017

@author: gonthier
"""

import pandas as pd
from pandas import Series
import urllib.request
 
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
    df = pd.read_csv(name_file_paitings, sep=",")
    get_im = df['image']
    get_im_unique = get_im.drop_duplicates()
    name_file_unique = '/media/HDD/Wikidata_query/paitings_wikidata.csv'
    get_im_unique.to_csv(name_file_unique, index=None, sep=',', mode='w')
    name_file_print = '/media/HDD/Wikidata_query/query_estampe_wikidata.csv'
    df_estampe = pd.read_csv(name_file_print, sep=",")
    get_im = df_estampe['image']
    get_im_unique_estampe = get_im.drop_duplicates()
    name_file_unique = '/media/HDD/Wikidata_query/estampe_wikidata.csv'
    get_im_unique_estampe.to_csv(name_file_unique, index=None, sep=',', mode='w')
    name_file_class = '/media/HDD/Wikidata_query/query_Depict_class.csv'
    df_class = pd.read_csv(name_file_class, sep=",")
    
    
    # Read again !
    df = pd.read_csv(name_file_paitings, sep=",",encoding='utf-8')
    df_estampe = pd.read_csv(name_file_print, sep=",",encoding='utf-8')
    list_column_to_change = ['item','image','createur','depicts','country']
    df['image'] = df['image'].apply(lambda a: urllib.request.unquote(a))
    df_estampe['image'] = df_estampe['image'].apply(lambda a: urllib.request.unquote(a))
    #print(df['image'][0])
    #print(df['depictsIconoclass'][0])
    
    df['createur'] = df['createur'] .astype('str')
    df['itemDescription'] = df['itemDescription'] .astype('str')
    df['depictsIconoclass'] = df['depictsIconoclass'] .astype('str')
    df['depicts'] = df['depicts'].fillna(value='')
    df['year'] = df['year'].astype('str')
   
    df_estampe['createur'] = df_estampe['createur'] .astype('str')
    df_estampe['depictsIconoclass'] = df_estampe['depictsIconoclass'] .astype('str')
    df_estampe['depicts'] = df_estampe['depicts'].fillna(value='')
    df_estampe['year'] = df_estampe['year'].astype('str')
    
    df_copy = df.copy()
    df_estampe_copy = df_estampe.copy()
    for elt  in list_column_to_change:
        #print(elt)
        df_copy[elt] = df[elt].apply(lambda a: str.split(str(a),'/')[-1])
        df_estampe_copy[elt] = df_estampe[elt].apply(lambda a: str.split(str(a),'/')[-1])
        
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
    
    df_copy = df_copy.groupby('image').apply(fusion_wikidata)
    df_estampe_copy = df_estampe_copy.groupby('image').apply(fusion_wikidata)
    
    
    
    df_copy.to_csv('data/Wikidata_Paintings.txt', index=None, sep=',', mode='w')
    
    # Test
    df_test = pd.read_csv('Wikidata_Paintings.txt',sep=",", encoding='utf-8')
    print("Wikidata Paintings")
    print(df_test.head(11))
    print("Number of Paintings : ",len(df_test['image']))
    
    
    df_estampe_copy.to_csv('data/Wikidata_Prints.txt', index=None, sep=',', mode='w')
    
    # Test
    df_test = pd.read_csv('Wikidata_Prints.txt',sep=",", encoding='utf-8')
    print("Prints")
    print(df_test.head(11))
    print("Number of Prints : ",len(df_test['image']))
    
    return(0)
    
if __name__ == '__main__':
    #prepareVOC12()
    #preparePaintings()
    prepareWikiData()
