#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:56:51 2018

The goal of this script is to create a panda dataframe for VOC 2007

Inspired https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
@author: gonthier
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pandas as pd
from tf_faster_rcnn.lib.datasets import voc_eval
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

path = '/media/gonthier/HDD/data/'

path_output = '/media/gonthier/HDD/output_exp/ClassifPaintings/'




def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('/media/gonthier/HDD/data/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('/media/gonthier/HDD/data/VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def VOC2007():
        
    df = None
    years = ['2007']
    
    for y in years:
        for year, image_set in sets:
            if year==y:
                image_ids = open('/media/gonthier/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
                pd_c = None
                for c in classes: 
                    print(year,image_set,c)
                    path_c = '/media/gonthier/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year,c,image_set)
                    if pd_c is None:
                        pd_c = pd.read_csv(path_c,sep=r"\s*",names=['name_img',c],dtype=str)
                        print(pd_c.head(5))
                    else:
                        pd_c_tmp =  pd.read_csv(path_c,sep=r"\s*",names=['name_img',c],dtype=str)
                        pd_c = pd_c.merge(pd_c_tmp,on='name_img', how='outer')
                pd_c['set'] = image_set
            if df is None:
                df = pd_c
            else:
                df = df.append(pd_c)
        output_name = path_output + 'VOC%s_all'%y + '.csv'
        print(df.iloc[[45,46,47]])
        df.to_csv(output_name)
        
        output_name = path_output + 'VOC%s'%y + '.csv'
        # On remplace les 0 par des 1  ! les cas difficiles par des certitudes
        for c in classes: 
            df.loc[df[c]=='0',c] = '1'
        print(df.iloc[[45,46,47]])
        df.to_csv(output_name)
    
        df=pd.read_csv(output_name,dtype=str)
        print(df.iloc[[45,46,47]])
        
def Watercolor():
    df = None
    classes = ["bicycle", "bird","car", "cat", "dog", "person"]
    sets = [('watercolor','train'),('watercolor','test')]
    for base,image_set in sets:
        path_b = '/media/gonthier/HDD/data/cross-domain-detection/datasets/watercolor/ImageSets/Main/%s.txt'%(image_set)
        pd_b = pd.read_csv(path_b,sep=r"\s*",names=['name_img'],dtype=str)
        for c in classes:
            pd_b[c] = -1
        print(pd_b.head(5))
        for index, row in pd_b.iterrows():
            i = row['name_img']
            path_i = '/media/gonthier/HDD/data/cross-domain-detection/datasets/watercolor/Annotations/%s.xml'%(i)
            read_file = voc_eval.parse_rec(path_i)
            for element in read_file:
                classe_elt = element['name']
                for c in classes:
                    if classe_elt==c:
                        pd_b.loc[pd_b['name_img']==row['name_img'],c] = 1
        pd_b['set'] = image_set
        if df is None:
            df = pd_b
        else:
            df = df.append(pd_b)
    output_name = path_output + 'watercolor_all' + '.csv'
    print(df.iloc[[45,46,47]])
    df.to_csv(output_name)
    
    output_name = path_output + 'watercolor' + '.csv'
    # On remplace les 0 par des 1  ! les cas difficiles par des certitudes
    for c in classes: 
        df.loc[df[c]==0,c] = 1
    print(df.iloc[[45,46,47]])
    df.to_csv(output_name)

    df=pd.read_csv(output_name,dtype=str)
    print(df.iloc[[45,46,47]])
    
def PeopleArt():
    df = None
    classes = ["person"]
    sets = [('PeopleArt','train'),('PeopleArt','test'),('PeopleArt','val')]
    for base,image_set in sets:
        path_b = '/media/gonthier/HDD/data/PeopleArt/ImageSets/Main/person_%s.txt'%(image_set)
        pd_b = pd.read_csv(path_b,sep=r"\s*",names=['name_img','person'],dtype=str)
        pd_b['set'] = image_set
        if df is None:
            df = pd_b
        else:
            df = df.append(pd_b)
    import numpy as np
    print(np.sum(df['person']==0))
    output_name = path_output + 'PeopleArt' + '.csv'
    # On remplace les 0 par des 1  ! les cas difficiles par des certitudes
    for c in classes: 
        df.loc[df[c]==0,c] = 1
    df.to_csv(output_name,index=None)

#    df=pd.read_csv(output_name,dtype=str)
    
    

def Clipart():
    df = None
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    sets = [('clipart','train'),('clipart','test')]
    for base,image_set in sets:
        path_b = '/media/gonthier/HDD/data/cross-domain-detection/datasets/clipart/ImageSets/Main/%s.txt'%(image_set)
        pd_b = pd.read_csv(path_b,sep=r"\s*",names=['name_img'],dtype=str)
        for c in classes:
            pd_b[c] = -1
        print(pd_b.head(5))
        for index, row in pd_b.iterrows():
            i = row['name_img']
            path_i = '/media/gonthier/HDD/data/cross-domain-detection/datasets/clipart/Annotations/%s.xml'%(i)
            read_file = voc_eval.parse_rec(path_i)
            for element in read_file:
                classe_elt = element['name']
                for c in classes:
                    if classe_elt==c:
                        pd_b.loc[pd_b['name_img']==row['name_img'],c] = 1
        pd_b['set'] = image_set
        if df is None:
            df = pd_b
        else:
            df = df.append(pd_b)
    output_name = path_output + 'clipart_all' + '.csv'
    print(df.iloc[[45,46,47]])
    df.to_csv(output_name)
    
    output_name = path_output + 'clipart' + '.csv'
    # On remplace les 0 par des 1  ! les cas difficiles par des certitudes
    for c in classes: 
        df.loc[df[c]==0,c] = 1
    print(df.iloc[[45,46,47]])
    df.to_csv(output_name)

    df=pd.read_csv(output_name,dtype=str)
    print(df.iloc[[45,46,47]])
        
if __name__ == '__main__':
    Clipart()
#        cls_label = open('/media/gonthier/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year,c,image_set)).read().strip().split()
#        print(cls_label)
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    print(image_ids)
#
#
##wd = getcwd()
##print(wd)
##
#for year, image_set in sets:
#    if not os.path.exists('/media/gonthier/HDD/data/VOCdevkit/VOC%s/labels/'%(year)):
#        os.makedirs('/media/gonthier/HDD/data/VOCdevkit/VOC%s/labels/'%(year))
#    image_ids = open('/media/gonthier/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    for image_id in image_ids:
#        list_file.write('/media/gonthier/HDD/data/%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#        convert_annotation(year, image_id)
#    list_file.close()

#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")