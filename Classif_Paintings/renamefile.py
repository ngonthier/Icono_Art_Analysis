#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:23:29 2020

@author: gonthier
"""

import os
import glob
output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata')

list_all_files = glob.glob(output_path+'/**/*', recursive=True)

for file in list_all_files:
    
    if '_BestOnVal' in file:
        new_name = file.replace('_BestOnVal','')
        print(file,new_name)
        if os.path.isfile(new_name):
            os.remove(file)
        else:
            os.rename(file,new_name)
        
