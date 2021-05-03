#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:37:06 2020

Goal do a small test with the Ukiyo-e images to see if we can do something og not

@author: gonthier
"""
import pywt # Wavelet
import pandas as pd
path_data = '/media/gonthier/HDD2/data/Ukiyoe/'

databasetxt = path_data+'SmallUkiyoe_dataset_bigImages_and_dateClasses.csv'
df_label = pd.read_csv(databasetxt,sep=",", encoding="utf-8")
    # A  refaire ce code la

wavelet_db4 = pywt.Wavelet('db4') # Daubechies D4 : lenght filter = 8
# In this experiment, we employed the conventional pyramid
# wavelet decomposition with three levels using the Daubechies’
# maximally flat orthogonal filters of length 8 ( filters)
for row in df_label.iterrows():
    rowData = row[1]
    name_img = rowData['item_name']
    date = rowData['DateClasses']    
    complet_name = path_data+'Images/'+name_img+'.jpg'
    
   
    
    # Multilevel 2D Discrete Wavelet Transform. if level = None take the maximum



