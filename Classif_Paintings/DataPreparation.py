#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:03:42 2017

@author: gonthier
"""

import pandas as pd
from pandas import Series

def f(x):
     return Series(dict(name_img = x['name_img'].min(), 
                        set = x['set'].min(), 
                        classe = "\"%s\"" % ' '.join(x['classe'])))

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
     
     result.to_csv(r'VOC12.txt', index=None, sep=' ', mode='a')
     
         # Test
     df_test = pd.read_csv('VOC12.txt')
     print("VOC12")
     print(df_test.head(11))

def preparePaintings():
    name_file = '/media/HDD/data/Painting Dataset/painting_dataset_updated.csv'
    df = pd.read_csv(name_file, sep=",")
    df.columns = ['a','name_img','page','set','classe']
    df = df.drop('a', 1)
    df = df.drop('page', 1)
    df['name_img'] = df['name_img'].apply(lambda a: str.split(a,'/')[-1])
    df['name_img'] = df['name_img'].apply(lambda a: str.split(a,'.')[0])
    
    df['set'] = df['set'].apply(lambda a: a.replace('\'',''))
    
    df['classe'] = df['classe'].apply(lambda a: a.replace('\'',''))
    
    df.to_csv(r'Paintings.txt', index=None, sep=' ', mode='a')
    
    # Test
    df_test = pd.read_csv('Paintings.txt')
    print("Paintings")
    print(df_test.head(11))
    
if __name__ == '__main__':
    prepareVOC12()
    preparePaintings()