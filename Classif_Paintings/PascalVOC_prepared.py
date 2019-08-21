#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:56:51 2018

The goal of this script is to create a panda dataframe for VOC 2007 and others
datasets used in this project such as Watercolor or PeopleArt


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
from shutil import copyfile
import glob
from sklearn.model_selection import train_test_split
from pascal_voc_writer import Writer # pip install pascal-voc-writer
import matplotlib.pyplot as plt

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
    
def Comic():
    df = None
    classes = ['bicycle','bird','car','cat','dog','person']
    sets = [('comic','train'),('comic','test')]
    for base,image_set in sets:
        path_b = '/media/gonthier/HDD/data/cross-domain-detection/datasets/comic/ImageSets/Main/%s.txt'%(image_set)
        pd_b = pd.read_csv(path_b,sep=r"\s*",names=['name_img'],dtype=str)
        for c in classes:
            pd_b[c] = -1
        print(pd_b.head(5))
        for index, row in pd_b.iterrows():
            i = row['name_img']
            path_i = '/media/gonthier/HDD/data/cross-domain-detection/datasets/comic/Annotations/%s.xml'%(i)
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
    output_name = path_output + 'comic_all' + '.csv'
    df.to_csv(output_name)
    
    output_name = path_output + 'comic' + '.csv'
    df.to_csv(output_name)

def WriteNewXMLfile(pathxml,path_img,width, height,list_bd,size_min = 225):
    writer = Writer(path_img, width, height)
    for element in list_bd:
        classe_elt,xmin,ymin,xmax,ymax = element
        area = (xmax -xmin)*(ymax-ymin)
        if area <= size_min:
            # Marked as difficult
            writer.addObject(classe_elt, xmin, ymin, xmax, ymax, difficult=1)
        else:
            writer.addObject(classe_elt, xmin, ymin, xmax, ymax)
    writer.save(annotation_path=pathxml)
   
def CASPApaintings(copyFile=False,convertXML=False,copyIm=False):
    """
    Creation of the CASPA paintings dataset for WSOD training and evaluation
    """
    df = None
    classes = ["bear", "bird", "cat", "cow", "dog", "elephant", "horse", "sheep"]
    old_names = ["Bear", "bird", "cat", "Cattle Cow Bull", "dog", "elephant", "horse", "sheep shepherd"]
    
    old_img_folder = 'Images_Per_Class'
    default_path_imdb = '/media/gonthier/HDD2/data/'
    
    if convertXML:
        for old_name in old_names :
            ff = old_name.lower()
            ff = ff.replace(' ','_')
            for folder in ['non','realistic']:
                f = ff +'_'+folder
                old_path= default_path_imdb + 'CASPApaintings/Annotations/'+f 
                list_xml = glob.glob(old_path+'/*.xml')
                for elt in list_xml:
                    dst = default_path_imdb + 'CASPApaintings/Annotations/' + elt.split('/')[-1]
                    path_img = default_path_imdb + 'CASPApaintings/JPEGImages/' + elt.split('/')[-1].replace('xml','jpg')
                    # read the old xml file
                    old_tree=ET.parse(elt)
                    root = old_tree.getroot()
                    size = root.find('imagesize')
                    width = int(size.find('ncols').text)
                    height = int(size.find('nrows').text)
                    list_bd =  []
                    for obj in root.iter('object'):
                        cls = obj.find('name').text
                        
                        if cls not in classes:
                            continue
                        #cls_id = classes.index(cls)
                        polygon = obj.find('polygon')
                        points = polygon.iter('pt')
                        xmin = None
                        ymin = None
                        xmax = None
                        ymax = None
                        for i,pt in enumerate(points):
                            x = int(pt.find('x').text)
                            y = int(pt.find('y').text)
                            if i == 0:
                                xmin = x
                                ymin = y
                            elif i ==1:
                                xmax = x
                            elif i==2:
                                ymax = y
                        list_bd += [[cls,xmin,ymin,xmax,ymax]]
                    WriteNewXMLfile(dst,path_img,width,height,list_bd)
    
    if copyIm:
        # Copie of images in the new folders
        for f_name in old_names:
            for folder in ['non','realistic']:
                old_path= default_path_imdb + 'CASPApaintings/' +  old_img_folder +'/'+f_name+'/'+folder
                list_im = glob.glob(old_path+'/*.jpg')
                for elt in list_im:
                    lower_name_img = elt.split('/')[-1]
                    lower_name_img = lower_name_img.lower()
                    print(lower_name_img)
                    dst = default_path_imdb + 'CASPApaintings/JPEGImages/' + lower_name_img
                    copyfile(elt,dst)
    if copyFile:  
        # Creation of the train and set file per class
        df_test= None
        df_train = None
        for new_name,old_name in zip(classes,old_names):
            #print(new_name,old_name)
            path_c = default_path_imdb + 'CASPApaintings/' +  old_img_folder+'/' + old_name + '.txt'
            pd_c =  pd.read_csv(path_c,names=['name_img'],dtype=str, encoding='utf-8')
            pd_c_train, pd_c_test = train_test_split(pd_c,test_size=0.5, random_state=0)
            pd_c_train['name_img'] = pd_c_train['name_img'].str.replace('.jpg','')
            pd_c_train['name_img'] = pd_c_train['name_img'].str.lower()
    #        pd_c_train = pd_c_train.apply((lambda x : x.replace('.jpg','')),axis=0)
            pd_c_test['name_img'] = pd_c_test['name_img'].str.replace('.jpg','')
            pd_c_test['name_img'] = pd_c_test['name_img'].str.lower()
    #        pd_c_test =pd_c_test.apply((lambda x : x.replace('.jpg','')),axis=0)
            path_c_train =  default_path_imdb + 'CASPApaintings/ImageSets/Main/' + new_name + '_train.txt'
            path_c_test =  default_path_imdb + 'CASPApaintings/ImageSets/Main/' + new_name + '_test.txt'
            pd_c_train.to_csv(path_c_train,header=False,index=False)
            pd_c_test.to_csv(path_c_test,header=False,index=False)
            #print(pd_c_test.head(5))
            if df_train is None:
                df_train = pd_c_train
            else:
                df_train = df_train.append(pd_c_train)
            if df_test is None:
                df_test = pd_c_test
            else:
                df_test = df_test.append(pd_c_test)
            
        df_test.drop_duplicates(subset ='name_img', 
                     keep = False, inplace = True) 
        df_train.drop_duplicates(subset ='name_img', 
                     keep = False, inplace = True)
        df_train = df_train['name_img']
        df_test = df_test['name_img']
        #print(df_test.head(5))
        num_test_drop = 0
        for row in df_test.values:
            name_im = default_path_imdb + 'CASPApaintings/JPEGImages/' + row +'.jpg'
            if not os.path.isfile(name_im):
                #print(row,name_im)
                df_test = df_test[df_test!=row]
                num_test_drop += 1
        num_train_drop = 0
        for row in df_train.values:
            name_im = default_path_imdb + 'CASPApaintings/JPEGImages/' + row +'.jpg'
            if not os.path.isfile(name_im):
                print(row,name_im)
                df_train = df_train[df_train!=row]
                num_train_drop += 1
        print('number of drop images for train :',num_train_drop,'for test',num_test_drop)
        path_train =default_path_imdb + 'CASPApaintings/ImageSets/Main/train.txt'
        path_test =default_path_imdb + 'CASPApaintings/ImageSets/Main/test.txt'
        df_train.to_csv(path_train,header=False,index=False)
        df_test.to_csv(path_test,header=False,index=False)
        
        # Copie of the xml files
        # Besoin de convertir en Pascal voc format
    
        
    oneNotFound = False
    list_mising_xml = []
    sets = [('CASPApaintings','train'),('CASPApaintings','test')]
    for base,image_set in sets:
        path_b = '/media/gonthier/HDD2/data/CASPApaintings/ImageSets/Main/%s.txt'%(image_set)
        pd_b = pd.read_csv(path_b,names=['name_img'],dtype=str, encoding='utf-8')
        for c in classes:
            pd_b[c] = -1
        for index, row in pd_b.iterrows():
            i = row['name_img']
            path_i = '/media/gonthier/HDD2/data/CASPApaintings/Annotations/%s.xml'%(i)
            try:
                read_file = voc_eval.parse_rec(path_i)
                for element in read_file:
                    classe_elt = element['name']
                    for c in classes:
                        if classe_elt==c:
                            pd_b.loc[pd_b['name_img']==row['name_img'],c] = 1
            except FileNotFoundError:
                print(path_i,'not found')
                oneNotFound = True
                list_mising_xml += [i]
        pd_b['set'] = image_set
        if df is None:
            df = pd_b
        else:
            df = df.append(pd_b)
    print(list_mising_xml)
    for  i in list_mising_xml:
        name_img = default_path_imdb + 'CASPApaintings/JPEGImages/' + i +'.jpg'
        plt.figure()
        imreadoutput = plt.imread(name_img)
        plt.imshow(imreadoutput)
    input('wait')
    
    c_missing = [['bear','horse'],['bear'],['bear'],['elephant'],['horse'],['horse'],['horse'],['horse'],['bear'],['bear','sheep'],['bear','dog'],['bear'],['bear'],['bear'],['horse']]
      
    print('Modification of missing elements')       
    for classes,name in zip(c_missing,list_mising_xml):
        print(name,df[df['name_img']==name]['set'])
        for c in classes:
            df.loc[df['name_img']==name,c] = 1
        df.loc[df['name_img']==name,'set'] = 'train'
    print('Size of datasets : ',len(df),'size train set :',len(df[df['set']=='train']),'size test set :',len(df[df['set']=='test']))
    oneNotFound = False
    
    if not(oneNotFound):
        output_name = path_output + 'CASPApaintings_all' + '.csv'
        df.to_csv(output_name,index=False)
        
        output_name = path_output + 'CASPApaintings' + '.csv'
        df.to_csv(output_name,index=False)
        output_name = default_path_imdb + 'CASPApaintings/ImageSets/Main/CASPApaintings.csv'
        df.to_csv(output_name,index=False)
    else:
        output_name = path_output + 'CASPApaintings_tmp' + '.csv'
        df.to_csv(output_name,index=False)
        print('XML not found, not saved')
        
    # Il faut extraire les  fichiers test et etc

        
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