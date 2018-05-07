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

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

path = '/media/HDD/data/'

path_output = '/media/HDD/output_exp/ClassifPaintings/'




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
    in_file = open('/media/HDD/data/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('/media/HDD/data/VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
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
        
df = None
years = ['2007']

for y in years:
    for year, image_set in sets:
        if year==y:
            image_ids = open('/media/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
            pd_c = None
            for c in classes: 
                print(year,image_set,c)
                path_c = '/media/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year,c,image_set)
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
    output_name = path_output + 'VOC%s'%y + '.csv'
    print(df.head(5))
    df.to_csv(output_name)

    df=pd.read_csv(output_name,dtype=str)
    print(df.head(5))
    
    
#        cls_label = open('/media/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year,c,image_set)).read().strip().split()
#        print(cls_label)
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    print(image_ids)
#
#
##wd = getcwd()
##print(wd)
##
#for year, image_set in sets:
#    if not os.path.exists('/media/HDD/data/VOCdevkit/VOC%s/labels/'%(year)):
#        os.makedirs('/media/HDD/data/VOCdevkit/VOC%s/labels/'%(year))
#    image_ids = open('/media/HDD/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    for image_id in image_ids:
#        list_file.write('/media/HDD/data/%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#        convert_annotation(year, image_id)
#    list_file.close()

#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")