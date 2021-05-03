#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:34:59 2020

The goal of this script is to create a usable vaersion of the Rijksmuseum_challenge
dataset for my code 

@author: gonthier
"""
import scipy.io

path_data = '/media/gonthier/HDD2/data/Rijksmuseum_challenge/'
mat_file = path_data +'rijksgt.mat'
mat = scipy.io.loadmat(mat_file)
gt = mat['gt']
gt00 = gt[0][0]
for i in range(len(gt00)):
    print(i,len(gt00[i]),gt00[i][0])
M = gt00['M'] # Varie entre 0 et 29
Mnames = gt00['Mnames'] # Materiaux name
C = gt00['C'] # Varie entre 1 et 6622
Cnames = gt00['Cnames'] # authors name / artist
T = gt00['T'] # Varie entre 0 et 15
Tnames = gt00['Tnames'] # Type
set_ = gt00['set'] # Set of the challenge ???

