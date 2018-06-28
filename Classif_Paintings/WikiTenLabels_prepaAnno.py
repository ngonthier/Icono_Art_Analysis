#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:43:31 2018

the goal of this script is to compute the statistiques on your database WikiTenLabels

@author: gonthier
"""

import pandas as pd
import os
from tf_faster_rcnn.lib.datasets import voc_eval

def Stats_and_testFile():
    annotations_folder = '/media/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/'
    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    name_file = path_data + 'WikiTenLabels.csv'
    df = pd.read_csv(name_file,sep=',')
    df_test = df[df['set']=='test']
    
    print('Statistiques sur le test set')
    print(df_test.sum())
    
    classes = ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
    list_elt= os.listdir(annotations_folder)
    file_test = '/media/HDD/data/Wikidata_Paintings/WikiTenLabels/Main/test.txt'
    file = open(file_test,"w") 
    for elt in list_elt:
        elt_wt_jpg = elt.split('.')[0]
        str_w = elt_wt_jpg +' \n'
        file.write(str_w) 
    file.close()

    size_min = 400 # 15*15

    path_b ='/media/HDD/data/Wikidata_Paintings/WikiTenLabels/Main/test.txt'
    pd_b = pd.read_csv(path_b,sep=r"\s*",names=['item'],dtype=str)
    
    dict_elts_total = {}
    dict_elts_sizemin = {}
    for c in classes:
        pd_b[c] = 0
        dict_elts_total[c] = 0
        dict_elts_sizemin[c] = 0
        
    for index, row in pd_b.iterrows():
        i = row['item']
        path_i = '/media/HDD/data/Wikidata_Paintings/WikiTenLabels/Annotations/%s.xml'%(i)
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
    print('Nombre d instances des differentes classes avec une taille superieur a :',size_min,'pixels')
    num_obj = 0
    for c in classes:
        print(c,' : ',dict_elts_sizemin[c])
        num_obj+=dict_elts_total[c]
    print('Nombre d instances de taille superieur a ',size_min,'pixels',num_obj)
                        
    
    
if __name__ == '__main__':
     Stats_and_testFile()
    
