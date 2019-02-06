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

def createNewXML_files():
	size_min = 25*25
	annotations_folder = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/Annotations/'
	path_data = '/media/HDD/output_exp/ClassifPaintings/'
	name_file = path_data + 'WikiTenLabels.csv'
	df = pd.read_csv(name_file,sep=',')
	df_test = df[df['set']=='test']
	df_train = df[df['set']=='train']
    
    list_elt= os.listdir(annotations_folder)
    #list_elt = df_test.as_array(['item'])
    file_test = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/test.txt'
    file = open(file_test,"w") 
    for elt in list_elt:
        elt_wt_jpg = elt.split('.')[0]
        str_w = elt_wt_jpg +' \n'
        file.write(str_w) 
    file.close()

    file_train = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/train.txt'
    file = open(file_train,"w") 
    list_elt = df_train.as_matrix(['item']).ravel()
    print(list_elt)
    for elt in list_elt:
        elt_wt_jpg = elt.split('.')[0]
        str_w = elt_wt_jpg +' \n'
        file.write(str_w) 
    file.close()

    classes_a_garder = ['angel','Child_Jesus','crucifixion_of_Jesus','Mary','nudity', 'ruins','Saint_Sebastien'] 
    path_b ='/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/test.txt'
    path_to_im = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    # Creation of the XML annotations files for the test set
    for index, row in pd_b.iterrows():
        Erase = False
        i = row['item']
        print(index,i)
        path_i = path_to_im + i +'.jpg'
        im = cv2.imread(path_i)
        height = im.shape[0]
        width = im.shape[1]
        writer = Writer(path_i, width, height,database='IconArt_v1')
        pathxml = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/Annotations/%s.xml'%(i)
        read_file = voc_eval.parse_rec(pathxml)
        for element in read_file:
            classe_elt = element['name']
            bbox = element['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            area = (xmax -xmin)*(ymax-ymin)
            if classe_elt in classes_a_garder:
                if area <= size_min:
                    element['difficult']=1
                    writer.addObject(classe_elt, xmin, ymin, xmax, ymax, difficult=1)
                else:
                    writer.addObject(classe_elt, xmin, ymin, xmax, ymax)
                    
        writer.save(annotation_path=pathxml) 
        
    # Creation of the XML annotations for the train set
	path_b ='/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/train.txt'
	path_to_im = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
	pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
	for index, row in pd_b.iterrows():
        Erase = False
        i = row['item']
        path_i = path_to_im + i +'.jpg'
        im = cv2.imread(path_i)
        height = im.shape[0]
        width = im.shape[1]
        writer = Writer(path_i, width, height,database='IconArt_v1')
        pathxml = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/Annotations/%s.xml'%(i)
        labels = np.array(df_train[df_train['item']==i][classes_a_garder])[0]
        
        for element,classe_elt in zip(labels,classes_a_garder):
            xmin = 0
            ymin = 0
            xmax = im.shape[0]
            ymax = im.shape[1]
            writer.addObject(classe_elt, xmin, ymin, xmax, ymax)
                    
        writer.save(annotation_path=pathxml) 
        
    return(0)
    
    
if __name__ == '__main__':
     createNewXML_files()
    
