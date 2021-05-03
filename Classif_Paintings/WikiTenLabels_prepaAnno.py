#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:43:31 2018

the goal of this script is to compute the statistiques on your database WikiTenLabels

@author: gonthier
"""

import pandas as pd
import cv2
import os
from pascal_voc_writer import Writer
from tf_faster_rcnn.lib.datasets import voc_eval
import numpy as np 
def StatsOnWikiTenLabels():
    annotations_folder = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    name_file = path_data + 'WikiTenLabels.csv'
    classes_a_garder = ['angel','Child_Jesus','crucifixion_of_Jesus','Mary','nudity', 'ruins','Saint_Sebastien']
    df = pd.read_csv(name_file,sep=',')
    df_test = df[df['set']=='test']
    df_train = df[df['set']=='train']
    df_test[classes_a_garder] = df_test[classes_a_garder]
    print("For test")
    df_test['sum'] = df_test[classes_a_garder].sum(axis=1)
    for i in range(len(classes_a_garder)):
        print(i,len(np.where(np.array(df_test.as_matrix(['sum']).ravel(),dtype=int)==i)[0]))
    df_train[classes_a_garder] = df_train[classes_a_garder]
    print("For train")
    df_train['sum'] = df_train[classes_a_garder].sum(axis=1)
    for i in range(len(classes_a_garder)):
        print(i,len(np.where(np.array(df_train.as_matrix(['sum']).ravel(),dtype=int)==i)[0]))
        
    print('Statistiques sur le test set')
    print(df_test.sum())
    print()    
    print('Statistiques sur le train set')
    print(df_train.sum())
    
#    classes = ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
#                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
#    list_elt= os.listdir(annotations_folder)
#    file_test = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/ImageSets/Main/test.txt'
#    file = open(file_test,"w") 
#    for elt in list_elt:
#        elt_wt_jpg = elt.split('.')[0]
#        str_w = elt_wt_jpg +' \n'
#        file.write(str_w) 
#    file.close()

    size_min = 25*25 # 15*15

    path_b ='/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/ImageSets/Main/test.txt'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    
    dict_elts_total = {}
    dict_elts_sizemin = {}
    for c in classes_a_garder:
        pd_b[c] = 0
        dict_elts_total[c] = 0
        dict_elts_sizemin[c] = 0
    without_class = 0
    numberofIm = 0
    for index, row in pd_b.iterrows():
        numberofIm += 1
        i = row['item']
        path_i = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/%s.xml'%(i)
        read_file = voc_eval.parse_rec(path_i)
        with_class = False
        for element in read_file:
            classe_elt = element['name']
            bbox = element['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            area = (xmax -xmin)*(ymax-ymin) 
#            print(area)
            
            for c in classes_a_garder:
                if classe_elt==c: # We got an element from 
                    with_class = True
                    pd_b.loc[pd_b['item']==row['item'],c] = 1
                    dict_elts_total[c] += 1
                    if area > size_min:
                        dict_elts_sizemin[c] += 1
        if not(with_class):
            without_class += 1
    print('Statistiques au niveaux du nombre de classes avec les labels dans la partie annotee en detection')
    print(pd_b.sum())
    print('Nombre d instances des differentes classes')
    num_obj = 0
    for c in classes_a_garder:
        print(c,' : ',dict_elts_total[c])
        num_obj+=dict_elts_total[c]
    print('Nombre d images totales',numberofIm)
    print('Nombre d instances totales',num_obj)
    print("Nombre d images sans classes",without_class)
    print('Nombre d instances des differentes classes avec une taille superieur a :',size_min,'pixels',num_obj)
    num_obj = 0
    for c in classes_a_garder:
        print(c,' : ',dict_elts_sizemin[c])
        num_obj+=dict_elts_sizemin[c]
    print('Nombre d instances de taille superieur a ',size_min,'pixels',num_obj)


def Stats_and_testFile():
    annotations_folder = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/'
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    name_file = path_data + 'WikiTenLabels.csv'
    df = pd.read_csv(name_file,sep=',')
    df_test = df[df['set']=='test']
    
    print('Statistiques sur le test set')
    print(df_test.sum())
    
    classes = ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
    list_elt= os.listdir(annotations_folder)
    write_test_file = False
    if write_test_file:
        file_test = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Main/ImageSets/test.txt'
        file = open(file_test,"w") 
        for elt in list_elt:
            elt_wt_jpg = elt.split('.')[0]
            str_w = elt_wt_jpg +' \n'
            file.write(str_w) 
        file.close()

    size_min = 25*25 # 15*15

    path_b ='/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/ImageSets/Main/test.txt'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    
    dict_elts_total = {}
    dict_elts_sizemin = {}
    for c in classes:
        pd_b[c] = 0
        dict_elts_total[c] = 0
        dict_elts_sizemin[c] = 0
        
    for index, row in pd_b.iterrows():
        i = row['item']
        path_i = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/%s.xml'%(i)
        read_file = voc_eval.parse_rec(path_i)
        for element in read_file:
            classe_elt = element['name']
            bbox = element['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            area = (xmax -xmin)*(ymax-ymin) 
#            print(area)
            for c in classes:
                if classe_elt==c: # We got an element from 
                    pd_b.loc[pd_b['item']==row['item'],c] = 1
                    dict_elts_total[c] += 1
                    if area > size_min:
                        dict_elts_sizemin[c] += 1
    print('Statistiques au niveaux du nombre de classes avec les labels dans la partie annotee en detection')
    print(pd_b.sum())
    print('Nombre d instances des differentes classes')
    num_obj = 0
    for c in classes:
        print(c,' : ',dict_elts_total[c])
        num_obj+=dict_elts_total[c]
    print('Nombre d instances totales',num_obj)
    print('Nombre d instances des differentes classes avec une taille superieur a :',size_min,'pixels',num_obj)
    num_obj = 0
    for c in classes:
        print(c,' : ',dict_elts_sizemin[c])
        num_obj+=dict_elts_sizemin[c]
    print('Nombre d instances de taille superieur a ',size_min,'pixels',num_obj)
 
def addColumns_IfAnnotation():
    """ This function add a column to the csv files about the fact that we have annotation or not
    """
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    name_file = path_data + 'WikiTenLabels.csv'
    df = pd.read_csv(name_file,sep=',')  
    path_b ='/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Main/test.txt'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    df['Anno'] = 0.0
    for index, row in pd_b.iterrows():
        i = row['item']
        df.loc[df['item']==i,'Anno'] = 1.0
    
    df  = df.sort_values(by=['item'])    
    print(df.sum())
    df.to_csv(name_file, index=None, sep=',')               
    
def WriteDifficultsBoxes():
    """
    This function will mark as difficult all the tiny objects in the xml files 
    """
    size_min = 25*25 # 20*20 Au moins un truc de taille superieur a 17*17
    path_b ='/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Main/test.txt'
    path_to_im = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    for index, row in pd_b.iterrows():
        Erase = False
        i = row['item']
        path_i = path_to_im + i +'.jpg'
        im = cv2.imread(path_i)
        height = im.shape[0]
        width = im.shape[1]
        writer = Writer(path_i, width, height)
        pathxml = '/media/gonthier/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/%s.xml'%(i)
        read_file = voc_eval.parse_rec(pathxml)
        for element in read_file:
            classe_elt = element['name']
            bbox = element['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            area = (xmax -xmin)*(ymax-ymin)
            if area <= size_min:
                # Marked as difficult
                element['difficult']=1
                Erase = True
                writer.addObject(classe_elt, xmin, ymin, xmax, ymax, difficult=1)
            else:
                writer.addObject(classe_elt, xmin, ymin, xmax, ymax)
        if Erase:
             writer.save(annotation_path=pathxml)
             print('Modified :',i)
    return(0)
    
if __name__ == '__main__':
#    WriteDifficultsBoxes()
#     Stats_and_testFile()
     StatsOnWikiTenLabels()
    
