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
from shutil import copyfile
import os.path
import os

depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']


def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)
    
def f(x):
     return Series(dict(name_img = x['name_img'].min(), 
                        set = x['set'].min(), 
                        classe = "%s" % ' '.join(x['classe'])))
def fusion_wikidata_withurl(x):
     return Series(dict(item = x['item'].min(), itemLabel = x['itemLabel'].min(), 
                        itemDescription = x['itemDescription'].min(), image = x['image'].min(), 
                        image_url = x['image_url'].min(),
                        depictsIconoclass = "%s" % ' '.join(x['depictsIconoclass']),
                        createur = x['createur'].min(),
                        depicts = "%s" % ' '.join(x['depicts']),
                        country = x['country'].min(),
                        year = x['year'].min(),depictsLabel =  "%s" % ' '.join(x['depictsLabel'])
                        ))
def fusion_wikidata(x):
     return Series(dict(item = x['item'].min(), itemLabel = x['itemLabel'].min(), 
                        itemDescription = x['itemDescription'].min(), image = x['image'].min(),  
                        depictsIconoclass = "%s" % ' '.join(x['depictsIconoclass']),
                        createur = x['createur'].min(),
                        depicts = "%s" % ' '.join(x['depicts']),
                        country = x['country'].min(),
                        year = x['year'].min(),depictsLabel =  "%s" % ' '.join(x['depictsLabel'])
                        ))
depicts_depictsLabel = {'Q18281':'embroidery','Q148993':'broad_leaved_tree','Q10884':'tree','Q193893':'capital','Q4817':'column','Q12511':'stairs','Q3289701':'step','Q42804':'beard','Q3575260':'jewellery','Q467': 'woman','Q8441' : 'man','Q345': 'Mary','Q527':	'sky','Q10884'	: 'tree','Q942467'	: 'Child Jesus','Q8074':	'cloud','Q3010' :	'boy','Q302':	'Jesus Christ','Q1144593':'sitting','Q10791':	'nudity','Q7569':'child','Q726':	'horse','Q14130':	'long hair','Q7560': 'mother','Q107425':	'landscape','Q144': 'dog','Q8502':'mountain','Q235113':	'angel'}


# Chercher des chapitaux dans stairs et steps, column 

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
    name_file_unique_url_prints = '/media/HDD/Wikidata_query/estampe_wikidata.csv'
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
    get_im_unique.to_csv(name_file_unique_url_prints, index=None, sep=',', mode='w')
    
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
    
def prepare_Dates_WikiData():
    name_file= '/media/HDD/Wikidata_query/Dates_Artists.csv'
    df = pd.read_csv(name_file, sep=",",encoding='utf-8')
    print(df.head(3))    
    df_drop = df.drop_duplicates(subset='peintre', keep="last")
    list_column_to_change = ['peintre','prenom','nom','famillenom']
    for elt  in list_column_to_change:
        #print(elt)
        df_drop[elt] = df_drop[elt].apply(lambda a: str.split(str(a),'/')[-1])
        
    year = Series(df_drop.mean(axis=1, skipna=True, level=None, numeric_only=True))
    df_drop['year_merge']= year
    print(df_drop.head(3)) 
    df_drop.to_csv('data/Dates_Artists_rewied.csv')
    
    return(0)
    
def prepareWikiDataWithSubSet():
    already = False
    
    name_file_paitings = '/media/HDD/Wikidata_query/query_paitings_wikidata.csv'
    name_file_unique_url_paitings = '/media/HDD/Wikidata_query/paitings_wikidata.csv'
    name_file_class = '/media/HDD/Wikidata_query/query_Depict_class.csv'
    df_class = pd.read_csv(name_file_class, sep=",")
    df_class = df_class.drop('count',axis=1)
    df_class =  df_class.append(['',''])
    df_class['depictsLabel'] = df_class['depictsLabel'] .astype('str')
    elt='depicts'
    df_class[elt] = df_class[elt].apply(lambda a: str.split(str(a),'/')[-1]) 
    
    if not(already):
        
        df = pd.read_csv(name_file_paitings, sep=",",encoding='utf-8')
        get_im = df['image']
        get_im_unique = get_im.drop_duplicates()
        print("Number of paintings with doublons",len(get_im_unique))
        get_im_unique.to_csv(name_file_unique_url_paitings, index=None, sep=',', mode='w')
       
        
        df_drop_paitings = df.drop_duplicates(subset=['item'], keep="last")
        print("Number of different item in Paintings",len(df_drop_paitings))
        get_im_unique = df_drop_paitings['image']
        print("Number of paitings",len(get_im_unique))
    
        df['createur'] = df['createur'] .astype('str')
        df['itemDescription'] = df['itemDescription'] .astype('str')
        df['depictsIconoclass'] = df['depictsIconoclass'] .astype('str')
        df['depicts'] = df['depicts'].fillna(value='')
        df['year'] = df['year'].astype('str')
        
        df_copy = df.copy()
        
        list_column_to_change = ['item','image','createur','depicts','country']
        df_copy['image_url'] = df_copy['image']
        for elt  in list_column_to_change:
            #print(elt)
            df_copy[elt] = df[elt].apply(lambda a: str.split(str(a),'/')[-1])
            
        df_copy['image'] = df_copy['image'].apply(lambda a: urllib.request.unquote(a))
    
        
        df_copy = df_copy.join(df_class.set_index('depicts'), on='depicts')
        df_copy['depictsLabel'] = df_copy['depictsLabel'] .astype('str')
        
        df_copy2 = df_copy.groupby('item').apply(fusion_wikidata_withurl)
        df_copy2 = df_copy2.drop_duplicates(subset=['image'], keep="last")
        df_copy2.to_csv('data/Wikidata_Paintings_modify.txt', index=None, sep=',', mode='w')
    # Test
    df_copy2 = pd.read_csv('data/Wikidata_Paintings_modify.txt',sep=",", encoding='utf-8')
    #print("Wikidata Paintings")
    print(df_copy2.head(2))

    Labels = ['Q42804','Q345','Q942467','Q302','Q10791','Q235113','Q109607','Q63070','Q40662','Q179718','Q1698874','Q47652','Q183332','Q328804','Q83772','Q998','Q44015','Q51636','Q15223957','Q81710','Q13147','Q35500','Q488841','Q132543','Q618057','Q80513']
    number_paitings = len(df_copy2['image'])
    print("Number of Paintings : ",number_paitings)
    list_image_with_it = []
    for depict in Labels:
        df_copy2[depict] = Series(np.zeros(number_paitings), index=df_copy2.index)
        print(df_class[df_class['depicts']==depict])
        # Create the list of the image with the depict elt
        for i in range(number_paitings):
            depicts_elts = df_copy2.iloc[i]['depicts']
            #print(depicts_elts)
            if not('nan' in str(depicts_elts)):
                if contains_word(depicts_elts,depict):
                    #print(depicts_elts)
                    list_image_with_it += [i]
                    df_copy2.loc[i, depict] = 1 # There are a element of that on that image
            else: # If the list is empty
                df_copy2.loc[i, depict] = -1
    list_image_with_it = np.unique(list_image_with_it) 
    df_new = df_copy2.loc[list_image_with_it]
    print("Number of image in this subset",len(df_new['image_url']))
    df_new.to_csv('data/Wikidata_Paintings_miniset10c.txt', index=None, sep=',', mode='w') 
    
    
#    Labels = ['Q345','Q942467','Q302','Q10791','Q235113','Q109607','Q63070','Q40662','Q179718','Q1698874','Q47652','Q183332','Q328804','Q83772','Q998','Q44015','Q51636','Q15223957','Q81710','Q13147','Q35500','Q488841','Q132543','Q618057','Q80513']
#    number_paitings = len(df_copy2['image'])
#    print("Number of Paintings : ",number_paitings)
#    list_image_with_it = []
#    for depict in Labels:
#        df_copy2[depict] = Series(np.zeros(number_paitings), index=df_copy2.index)
#        print(df_class[df_class['depicts']==depict])
#        # Create the list of the image with the depict elt
#        for i in range(number_paitings):
#            depicts_elts = df_copy2.iloc[i]['depicts']
#            #print(depicts_elts)
#            if not('nan' in str(depicts_elts)):
#                if contains_word(depicts_elts,depict):
#                    #print(depicts_elts)
#                    list_image_with_it += [i]
#                    df_copy2.loc[i, depict] = 1 # There are a element of that on that image
#            else: # If the list is empty
#                df_copy2.loc[i, depict] = -1
#    list_image_with_it = np.unique(list_image_with_it) 
#    df_new = df_copy2.loc[list_image_with_it]
#    print("Number of image in this subset",len(df_new['image_url']))
#    df_new.to_csv('data/Wikidata_Paintings_subset.txt', index=None, sep=',', mode='w') 
#    
#    # Number of image in this subset  5491 this is miniset label creation !!! 
#    Labels = ['Q107425', 'Q10791', 'Q10884', 'Q109607', 'Q1144593', 'Q13147',
#       'Q132543', 'Q14130', 'Q144', 'Q15223957', 'Q1698874', 'Q179718',
#       'Q183332', 'Q235113', 'Q3010', 'Q302', 'Q328804', 'Q345', 'Q35500',
#       'Q40662', 'Q44015', 'Q467', 'Q47652', 'Q488841', 'Q51636', 'Q527',
#       'Q618057', 'Q63070', 'Q726', 'Q7560', 'Q7569', 'Q80513', 'Q8074',
#       'Q81710', 'Q83772', 'Q8441', 'Q8502', 'Q942467', 'Q998'] 
#    # Number of image in this subset 14738 
#    # Et  9247dans ce qui reste ! 
#    number_paitings = len(df_copy2['image'])
#    print("Number of Paintings : ",number_paitings)
#    list_image_with_it = []
#    for depict in Labels:
#        df_copy2[depict] = Series(np.zeros(number_paitings), index=df_copy2.index)
#        print(df_class[df_class['depicts']==depict])
#        # Create the list of the image with the depict elt
#        for i in range(number_paitings):
#            depicts_elts = df_copy2.iloc[i]['depicts']
#            #print(depicts_elts)
#            if not('nan' in str(depicts_elts)):
#                if contains_word(depicts_elts,depict):
#                    #print(depicts_elts)
#                    list_image_with_it += [i]
#                    df_copy2.loc[i, depict] = 1 # There are a element of that on that image
#            else: # If the list is empty
#                df_copy2.loc[i, depict] = -1
#    list_image_with_it = np.unique(list_image_with_it) 
#    df_new = df_copy2.loc[list_image_with_it]
#    print("Number of image in this subset",len(df_new['image_url']))
#    df_new.to_csv('data/Wikidata_Paintings_subset2.txt', index=None, sep=',', mode='w') 
    #df_new['image_url']
    
#    np_classes = df_copy2.as_matrix(columns=Labels)
#    #print(np_classes.shape)
#    index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
#    name_tab_index =  index_image_with_at_least_one_of_the_class[0]
    
    
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
    print("Number of Paintings",len(df_test['item']))

    number_paitings = len(df_test['image'])
    number_elt_in_training = int(number_elt/2)
    print("number of element per class, at least :",number_elt_in_training)
    depicts = df_reduc['depicts']
    #[::-1] # to reverse
    #number_class = len(depictsLabel)
    df_copy = df_test.copy()
    df_copy['set'] = Series(np.ones(number_paitings), index=df_copy.index)
    #df_copy['class'] = Series(-1*np.ones(number_paitings), index=df_copy.index)
    #classes = df_reduc['depictsLabel']
    list_im_in_training = []
    depicts2 = []
    for depict in depicts:
        depicts2 += [depict]
        df_copy[depict] = Series(np.zeros(number_paitings), index=df_copy.index)
        print(df_class[df_class['depicts']==depict])
        # Create the list of the image with the depict elt
        list_image_with_it = []
        for i in range(number_paitings):
            depicts_elts = df_copy.iloc[i]['depicts']
            #print(depicts_elts)
            if not('nan' in str(depicts_elts)):
                if contains_word(depicts_elts,depict):
                    #print(depicts_elts)
                    list_image_with_it += [i]
                    df_copy.loc[i, depict] = 1 # There are a element of that on that image
            else: # If the list is empty
                df_copy.loc[i, depict] = -1 # There are no annotation on that image
        #print(list_image_with_it)
        print("Number of image with this depict",len(list_image_with_it))
        #print(df_class[df_class['depicts']==depict]['count'])
        #print(np.array(df_class[df_class['depicts']==depict]['count']))
        if not(len(list_image_with_it)==np.array(df_class[df_class['depicts']==depict]['count'])):
            print("For ",depict,"element missings")
        
        already_choiced  =  np.intersect1d(list_im_in_training,list_image_with_it)
        print("Already",len(already_choiced))
        number_to_choose = number_elt_in_training - len(already_choiced)
        print("Number images to choose",number_to_choose)
        if number_to_choose > 0:
            not_choiced_yet = np.setdiff1d(list_image_with_it,list_im_in_training) 
            #Return the sorted, unique values in ar1 that are not in ar2.
            print("not_choiced_yet",len(not_choiced_yet))
            index_choosed = random.sample(list(not_choiced_yet), k=number_to_choose)
            if(len(np.intersect1d(index_choosed,list_im_in_training))>0):
                print("Problem")
            #Used for random sampling without replacement.
            #print(random_index)
            #index_choosed = not_choiced_yet[random_index]
            print("index_choosed",len(index_choosed))
            #print(index_choosed)
            for k in index_choosed:
                assert(df_copy.loc[k, 'set'] == 1)
                df_copy.loc[k, 'set'] = 0 # Le set de training est note 0
    
            if len(list_im_in_training)==0:
                list_im_in_training = np.asarray(index_choosed)
            else:
                list_im_in_training = np.append(list_im_in_training,np.asarray(index_choosed))
        print("Number of image in the training set for the moment :",len(list_im_in_training))
        for depict2 in depicts2:
            print("Number of exemple for",depicts_depictsLabel[depict2],'in train set :',np.sum(df_copy[depict2][df_copy['set']==0]),'and not in train set :',np.sum(df_copy[depict2][df_copy['set']==1]))
          
    # Petit test avant la fin
    print("Number of Images in the training set ",np.sum(df_copy['set']==0))
    for depict in depicts:
        print("Number of exemple for",depict,np.sum(df_copy[depict][df_copy['set']==0]))
    df_copy.to_csv('data/Wikidata_Paintings_sets.txt', index=None, sep=',', mode='w')             
    
#    # Test
    df_test = pd.read_csv('data/Wikidata_Paintings_sets.txt',sep=",", encoding='utf-8')
    print("Wikidata Paintings Sets")
    print(df_test.head(11))
    print("Number of Paintings : ",len(df_test['image']))
    
    return(0)
    
def MiseDeCote2():
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset2.txt'
    minisubset = False
    df = pd.read_csv(databasetxt,sep=",")
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    Labels = ['Q107425', 'Q10791', 'Q10884', 'Q109607', 'Q1144593', 'Q13147',
       'Q132543', 'Q14130', 'Q144', 'Q15223957', 'Q1698874', 'Q179718',
       'Q183332', 'Q235113', 'Q3010', 'Q302', 'Q328804', 'Q345', 'Q35500',
       'Q40662', 'Q44015', 'Q467', 'Q47652', 'Q488841', 'Q51636', 'Q527',
       'Q618057', 'Q63070', 'Q726', 'Q7560', 'Q7569', 'Q80513', 'Q8074',
       'Q81710', 'Q83772', 'Q8441', 'Q8502', 'Q942467', 'Q998']
    if not (minisubset):
        minisubset_labels =  ['Q345','Q942467','Q302','Q10791','Q235113','Q109607','Q63070','Q40662','Q179718','Q1698874','Q47652','Q183332','Q328804','Q83772','Q998','Q44015','Q51636','Q15223957','Q81710','Q13147','Q35500','Q488841','Q132543','Q618057','Q80513']
        np_classes_depictsAll = df.as_matrix(columns=minisubset_labels)
        np_classes_Labels = df.as_matrix(columns=Labels)
        #print(np_classes.shape)
        index_to_remove_becauseAlready_inMiniset = np.where(np.sum(np_classes_depictsAll,axis=1) > 0)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes_Labels,axis=1) > 0)
        print("Number of image in Labels :",len(index_image_with_at_least_one_of_the_class[0]))
        print("Number of image in depictsAll :",len(index_to_remove_becauseAlready_inMiniset[0]))
        name_tab_index = np.setdiff1d(index_image_with_at_least_one_of_the_class[0],index_to_remove_becauseAlready_inMiniset[0])
        print("Number of image :",len(name_tab_index))
        print("Number of image :",len(np.unique(name_tab_index)))
    else:
        name_tab_index = np.array(df.index)
    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'Subset1/' # This subset contain 9 247  images !! 
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    diff_folder = False
    if diff_folder:
        numberOfFolder = 0
        numberOfFolder_str = str(0)+ '/'
    else:
        numberOfFolder_str = ''
    for i in name_tab_index:
        if numberIm%500 == 0:
            print("numberIm",numberIm)
            dstfoldernew = dstfolder + numberOfFolder_str 
            print(dstfoldernew)
            do_mkdir(dstfoldernew)
            if diff_folder:
                numberOfFolder_str = str(numberOfFolder)+ '/'
                numberOfFolder += 1
        name = df['image'][i]
        if name in list_of_im:
            print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        list_of_im += [name]
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        src = read_data + namejpg
        dst = dstfoldernew + namejpg
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)

def CreationMiniSet10C():
    """
    Creation du Miniset 10 classes
    Les classes seront Mary, Jesus Child, Ruins, nudite, ange,
    barbe, Chapiteau, bijou,arbre feuillu, broderie
    
    
    """
    
    LabelsFor10c = ['Q3575260','Q42804','Q3289701','Q12511','Q4817','Q193893',
                    'Q148993','Q18281','Q345','Q942467','Q302','Q10791','Q235113',
                    'Q109607','Q63070','Q40662','Q179718','Q1698874','Q47652',
                    'Q183332','Q328804','Q83772','Q998','Q44015','Q51636','Q15223957',
                    'Q81710','Q13147','Q35500','Q488841','Q132543','Q618057','Q80513']
    already = False
    
    name_file_paitings = '/media/HDD/Wikidata_query/query_paitings_wikidata.csv'
    name_file_unique_url_paitings = '/media/HDD/Wikidata_query/paitings_wikidata.csv'
    name_file_class = '/media/HDD/Wikidata_query/query_Depict_class.csv'
    df_class = pd.read_csv(name_file_class, sep=",")
    df_class = df_class.drop('count',axis=1)
    df_class =  df_class.append(['',''])
    df_class['depictsLabel'] = df_class['depictsLabel'] .astype('str')
    elt='depicts'
    df_class[elt] = df_class[elt].apply(lambda a: str.split(str(a),'/')[-1]) 
    
    if not(already):
        
        df = pd.read_csv(name_file_paitings, sep=",",encoding='utf-8')
        get_im = df['image']
        get_im_unique = get_im.drop_duplicates()
        print("Number of paintings with doublons",len(get_im_unique))
        get_im_unique.to_csv(name_file_unique_url_paitings, index=None, sep=',', mode='w')
       
        
        df_drop_paitings = df.drop_duplicates(subset=['item'], keep="last")
        print("Number of different item in Paintings",len(df_drop_paitings))
        get_im_unique = df_drop_paitings['image']
        print("Number of paitings",len(get_im_unique))
    
        df['createur'] = df['createur'] .astype('str')
        df['itemDescription'] = df['itemDescription'] .astype('str')
        df['depictsIconoclass'] = df['depictsIconoclass'] .astype('str')
        df['depicts'] = df['depicts'].fillna(value='')
        df['year'] = df['year'].astype('str')
        
        df_copy = df.copy()
        
        list_column_to_change = ['item','image','createur','depicts','country']
        df_copy['image_url'] = df_copy['image']
        for elt  in list_column_to_change:
            #print(elt)
            df_copy[elt] = df[elt].apply(lambda a: str.split(str(a),'/')[-1])
            
        df_copy['image'] = df_copy['image'].apply(lambda a: urllib.request.unquote(a))
    
        
        df_copy = df_copy.join(df_class.set_index('depicts'), on='depicts')
        df_copy['depictsLabel'] = df_copy['depictsLabel'] .astype('str')
        
        df_copy2 = df_copy.groupby('item').apply(fusion_wikidata_withurl)
        df_copy2 = df_copy2.drop_duplicates(subset=['image'], keep="last")
        df_copy2.to_csv('/media/HDD/output_exp/ClassifPaintings/Wikidata_Paintings_modify.txt', index=None, sep=',', mode='w')
    # Test
    df_copy2 = pd.read_csv('/media/HDD/output_exp/ClassifPaintings/Wikidata_Paintings_modify.txt',sep=",", encoding='utf-8')
    #print("Wikidata Paintings")
    print(df_copy2.head(2))
    number_paitings = len(df_copy2['image'])
    print("Number of Paintings : ",number_paitings)
    list_image_with_it = []
    for depict in LabelsFor10c:
        df_copy2[depict] = Series(np.zeros(number_paitings), index=df_copy2.index)
        print(df_class[df_class['depicts']==depict])
        # Create the list of the image with the depict elt
        for i in range(number_paitings):
            depicts_elts = df_copy2.iloc[i]['depicts']
            #print(depicts_elts)
            if not('nan' in str(depicts_elts)):
                if contains_word(depicts_elts,depict):
                    #print(depicts_elts)
                    list_image_with_it += [i]
                    df_copy2.loc[i, depict] = 1 # There are a element of that on that image
            else: # If the list is empty
                df_copy2.loc[i, depict] = -1
    list_image_with_it = np.unique(list_image_with_it) 
    df_new = df_copy2.loc[list_image_with_it]
    print("Number of image in this subset",len(df_new['image_url']))
    df_new.to_csv('/media/HDD/output_exp/ClassifPaintings/Wikidata_Paintings_miniset10c.txt', index=None, sep=',', mode='w')
    
    # TODO finir ici 
    
    database='Wikidata_Paintings'
    allelt = False
    df = df_new.copy()
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    if not (allelt):
        depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']
        np_classes = df.as_matrix(columns=depictsAll)
        print(np_classes.shape)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
        name_tab_index =  index_image_with_at_least_one_of_the_class[0]
        print("Number of image :",len(name_tab_index))
        print("Number of image :",len(np.unique(name_tab_index)))
    else:
        name_tab_index = np.array(df.index)
    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'MiniSet10c/' # This subset is THE MiniSet and contain 5 491 images
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    numberOfFolder = 0
    diff_folder = False
    if diff_folder:
        numberOfFolder = 0
        numberOfFolder_str = str(0)+ '/'
    else:
        numberOfFolder_str = ''
    for i in name_tab_index:
        if numberIm%500 == 0:
            print("numberIm",numberIm)
            dstfoldernew = dstfolder + numberOfFolder_str 
            print(dstfoldernew)
            do_mkdir(dstfoldernew)
            if diff_folder:
                numberOfFolder_str = str(numberOfFolder)+ '/'
                numberOfFolder += 1
        name = df['image'][i]
        if name in list_of_im:
            print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        list_of_im += [name]
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        src = read_data + namejpg
        dst = dstfoldernew + namejpg
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
    
    
    
    return(0)

def MiseDeCote():
    """
    Creation du MiniSet
    """
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset.txt' 
    allelt = False
    df = pd.read_csv(databasetxt,sep=",")
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    if not (allelt):
        depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']
        np_classes = df.as_matrix(columns=depictsAll)
        print(np_classes.shape)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
        name_tab_index =  index_image_with_at_least_one_of_the_class[0]
        print("Number of image :",len(name_tab_index))
        print("Number of image :",len(np.unique(name_tab_index)))
    else:
        name_tab_index = np.array(df.index)
    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'MiniSet/' # This subset is THE MiniSet and contain 5 491 images
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    numberOfFolder = 0
    diff_folder = False
    if diff_folder:
        numberOfFolder = 0
        numberOfFolder_str = str(0)+ '/'
    else:
        numberOfFolder_str = ''
    for i in name_tab_index:
        if numberIm%500 == 0:
            print("numberIm",numberIm)
            dstfoldernew = dstfolder + numberOfFolder_str 
            print(dstfoldernew)
            do_mkdir(dstfoldernew)
            if diff_folder:
                numberOfFolder_str = str(numberOfFolder)+ '/'
                numberOfFolder += 1
        name = df['image'][i]
        if name in list_of_im:
            print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        list_of_im += [name]
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        src = read_data + namejpg
        dst = dstfoldernew + namejpg
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
def MiseDeCotePourAnnotationRapideBadPhoto():
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset.txt'
    df = pd.read_csv(databasetxt,sep=",")
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    classe_a_annotee = 'BadPhoto'
    print(df.head(2))
    name_tab_index =  df.index

    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'Not_' + classe_a_annotee + '/'
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    for i in name_tab_index:
        name = df['image'][i]
        nameoutput = df['item'][i]
        #if name in list_of_im:
            #print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            #print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        list_of_im += [namejpg]
        src = read_data + namejpg
        dst = dstfolder + nameoutput + '.jpg'
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
    input("Press input when you have remove all the image containing this class...")
    
    name_tab_index = np.array(df.index)
    classe_a_annotee_verif = classe_a_annotee
    list_elt= os.listdir(dstfolder)
    #print(list_elt)
    number_paitings = len(df['image'])
    df[classe_a_annotee_verif] = Series(np.ones(number_paitings), index=df.index)
    for name in list_elt:
        #print(name_tab)
        name_tab = name.split('.') 
        item_name = name_tab[0]
        df.loc[df['item']==item_name,classe_a_annotee_verif] = 0
    print('Number of ',classe_a_annotee,' element',np.sum(df[classe_a_annotee_verif]))
            
    print(df.head(2))
    namefile = 'data/Wikidata_Paintings_'+classe_a_annotee+'_.txt'
    df.to_csv(namefile, index=None, sep=',', mode='w') 
  
def stringInOtherList(list_a,list_b):
    for a in list_a:
        for b in list_b:
            if(a in b):
                return(True)
    return(False)

def AnnotationJesusChild():
    path_data = 'data/'
    databasetxt = path_data +  'Wikidata_Paintings_Q345_.txt'
    df = pd.read_csv(databasetxt,sep=",")
    classe_a_annotee = 'Q942467'
    classe_a_annotee_verif = classe_a_annotee + '_verif'
    number_paitings = len(df['image'])
    df[classe_a_annotee_verif] = Series(np.zeros(number_paitings), index=df.index)
    np_classes = df.as_matrix(columns=[classe_a_annotee])
    print("Number of images in this class :",len(np.where(np.sum(np_classes,axis=1) > 0)[0]))
    list_elt = df.as_matrix(columns=['item'])
    elementDescriptif = ['child','enfant','holy family','sainte famille','adoration des rois mages','adoration of the kings','nativity of jesus','nativité']
    for item_name in list_elt:
        row = df[df['item']==item_name[0]] 
        if row.iloc[0]['Q942467'] > 0:
            df.loc[df['item']==item_name[0],classe_a_annotee_verif] = 1
        elif row.iloc[0]['Q345_verif']> 0:
             # They are Mary
             name_im = str.lower(str(row.iloc[0]['image']))
             descrip = str.lower(str(row.iloc[0]['itemDescription']))
             list_dec = [name_im,descrip]
             if stringInOtherList(elementDescriptif,list_dec):
                 df.loc[df['item']==item_name[0],classe_a_annotee_verif] = 1
    
    print('Number of ',classe_a_annotee,' element',np.sum(df[classe_a_annotee_verif]))
            
    print(df.head(2))
    namefile = 'data/Wikidata_Paintings_'+classe_a_annotee+'_.txt'
    df.to_csv(namefile, index=None, sep=',', mode='w') 
    
def MiseDeCotePourAnnotationRapide():
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset.txt'
    allelt = True
    df = pd.read_csv(databasetxt,sep=",")
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    classe_a_annotee = 'Q235113'
    classe_a_annotee = 'Q345'
    classe_a_annotee = 'Q10791'
    classe_a_annotee = 'Q109607'
    print(df.head(2))
    if not (allelt):
        depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']
        np_classes = df.as_matrix(columns=depictsAll)
        print(np_classes.shape)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
        name_tab_index =  index_image_with_at_least_one_of_the_class[0]
        print("Number of image :",len(name_tab_index),'==?',len(np.unique(name_tab_index)))
    else:
        np_classes = df.as_matrix(columns=[classe_a_annotee])
        print(np_classes.shape)
        name_tab_index = np.where(np.sum(np_classes,axis=1) <= 0)[0]

    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'Not_' + classe_a_annotee + '/'
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    for i in name_tab_index:
        name = df['image'][i]
        nameoutput = df['item'][i]
        #if name in list_of_im:
            #print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            #print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        list_of_im += [namejpg]
        src = read_data + namejpg
        dst = dstfolder + nameoutput + '.jpg'
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
    input("Press input when you have remove all the image containing this class...")
    
    name_tab_index = np.array(df.index)
    classe_a_annotee_verif = classe_a_annotee + '_verif'
    list_elt= os.listdir(dstfolder)
    #print(list_elt)
    number_paitings = len(df['image'])
    df[classe_a_annotee_verif] = Series(np.ones(number_paitings), index=df.index)
    for name in list_elt:
        #print(name_tab)
        name_tab = name.split('.') 
        item_name = name_tab[0]
        df.loc[df['item']==item_name,classe_a_annotee_verif] = 0
    print('Number of ',classe_a_annotee,' element',np.sum(df[classe_a_annotee_verif]))
            
    print(df.head(2))
    namefile = 'data/Wikidata_Paintings_'+classe_a_annotee+'_.txt'
    df.to_csv(namefile, index=None, sep=',', mode='w') 
    
def AnnoterCapital_in_ruins():
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset.txt'
    allelt = True
    df = pd.read_csv(databasetxt,sep=",")
    classe_a_annotee = 'Q193893'
    ruins_Q = 'Q10791'
    print(df.head(2))
    if not (allelt):
        depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']
        np_classes = df.as_matrix(columns=depictsAll)
        print(np_classes.shape)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
        name_tab_index =  index_image_with_at_least_one_of_the_class[0]
        print("Number of image :",len(name_tab_index),'==?',len(np.unique(name_tab_index)))
    else:
        np_classes = df.as_matrix(columns=[classe_a_annotee])
        print(np_classes.shape)
        name_tab_index = np.where(np.sum(np_classes,axis=1) <= 0)[0]

    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'Not_' + classe_a_annotee + '/'
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    for i in name_tab_index:
        name = df['image'][i]
        nameoutput = df['item'][i]
        #if name in list_of_im:
            #print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            #print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        list_of_im += [namejpg]
        src = read_data + namejpg
        dst = dstfolder + nameoutput + '.jpg'
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
    input("Press input when you have remove all the image containing this class...")
    
    name_tab_index = np.array(df.index)
    classe_a_annotee_verif = classe_a_annotee + '_verif'
    list_elt= os.listdir(dstfolder)
    #print(list_elt)
    number_paitings = len(df['image'])
    df[classe_a_annotee_verif] = Series(np.ones(number_paitings), index=df.index)
    for name in list_elt:
        #print(name_tab)
        name_tab = name.split('.') 
        item_name = name_tab[0]
        df.loc[df['item']==item_name,classe_a_annotee_verif] = 0
    print('Number of ',classe_a_annotee,' element',np.sum(df[classe_a_annotee_verif]))
            
    print(df.head(2))
    namefile = 'data/Wikidata_Paintings_'+classe_a_annotee+'_.txt'
    df.to_csv(namefile, index=None, sep=',', mode='w')
    
def createSubset2():
    path_data = 'data/'
    database='Wikidata_Paintings'
    databasetxt = path_data +  'Wikidata_Paintings_subset.txt'
    allelt = True
    df = pd.read_csv(databasetxt,sep=",")
    #df = df.drop_duplicates(subset='image', keep="last")
    #print(df)
    classe_a_annotee = 'Q235113'
    #print(df.head(2))
    if not (allelt):
        depictsAll = ['Q467','Q8441','Q345','Q527','Q10884','Q942467','Q8074','Q3010','Q302','Q1144593','Q10791','Q7569','Q726','Q14130','Q7560','Q107425','Q144','Q8502','Q235113']
        np_classes = df.as_matrix(columns=depictsAll)
        print(np_classes.shape)
        index_image_with_at_least_one_of_the_class = np.where(np.sum(np_classes,axis=1) > 0)
        name_tab_index =  index_image_with_at_least_one_of_the_class[0]
        print("Number of image :",len(name_tab_index),'==?',len(np.unique(name_tab_index)))
    else:
        
        np_classes = df.as_matrix(columns=[classe_a_annotee])
        name_tab_index = np.where(np.sum(np_classes,axis=1) <= 0)[0]

    folder=database +'/'
    target_path = '/media/HDD/data/'
    write_data = target_path + folder 
    bigger_size = 600
    read_data = write_data + str(bigger_size) + '/'
    dstfolder=  write_data + 'Not_' + classe_a_annotee + '/'
    numberIm = 0
    do_mkdir(dstfolder)
    list_of_im  =[]
    for i in name_tab_index:
        name = df['image'][i]
        nameoutput = df['item'][i]
        #if name in list_of_im:
            #print(name," already read! indice = ",i,"Q index = ",df['item'][i])
            #print(df[df['image']==name])
            #print(df[df['item']==df['item'][i]])
        
        name_tab = name.split('.')
        name_tab[-1] = 'jpg'
        namejpg = ".".join(name_tab)
        list_of_im += [namejpg]
        src = read_data + namejpg
        dst = dstfolder + nameoutput + '.jpg'
        copyfile(src, dst)
        if os.path.exists(dst):
            numberIm +=1 
        else:
            print("Image not copied",dst)
    print("Number of image copied",numberIm)
    
    input("Press input when you have remove all the image containing this class...")
    
    name_tab_index = np.array(df.index)
    classe_a_annotee_verif = classe_a_annotee + '_verif'
    list_elt= os.listdir(dstfolder)
    #print(list_elt)
    number_paitings = len(df['image'])
    df[classe_a_annotee_verif] = Series(np.ones(number_paitings), index=df.index)
    for name in list_elt:
        #print(name_tab)
        name_tab = name.split('.') 
        item_name = name_tab[0]
        df.loc[df['item']==item_name,classe_a_annotee_verif] = 0
    print('Number of ',classe_a_annotee,' element',np.sum(df[classe_a_annotee_verif]))
            
    print(df.head(2))
    namefile = 'data/Wikidata_Paintings_'+classe_a_annotee+'_.txt'
    df.to_csv(namefile, index=None, sep=',', mode='w')     
        
if __name__ == '__main__':
    #prepareVOC12()
    #preparePaintings()
    prepareWikiData()
    #prepareWikidataSetsPaitings()
    #MiseDeCote()
    #prepareWikiDataWithSubSet()
    #MiseDeCote()
    #MiseDeCote2()
#    MiseDeCotePourAnnotationRapide()
#    MiseDeCotePourAnnotationRapideBadPhoto()
#    AnnotationJesusChild()
#Number of Images in the training set  2878
#Number of exemple for Q467 1284.0
#Number of exemple for Q8441 790.0
#Number of exemple for Q345 618.0
#Number of exemple for Q527 561.0
#Number of exemple for Q10884 471.0
#Number of exemple for Q942467 436.0
#Number of exemple for Q8074 433.0
#Number of exemple for Q3010 395.0
#Number of exemple for Q302 297.0
#Number of exemple for Q1144593 350.0
#Number of exemple for Q10791 310.0
#Number of exemple for Q7569 287.0
#Number of exemple for Q726 275.0
#Number of exemple for Q14130 264.0
#Number of exemple for Q7560 260.0
#Number of exemple for Q107425 266.0
#Number of exemple for Q144 254.0
#Number of exemple for Q8502 251.0
#Number of exemple for Q235113 250.0