#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:56:56 2018

@author: gonthier
"""

import time
import pathlib

def create_param_id_file_and_dir(param_dir,arrayParam,arrayParamStr):

    ts = time.time()
    #find if id already exists
    param_name = str(ts)
    dir_path = param_dir + param_name +'/'
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)  # Creation of the folder
    file_param = dir_path + 'param.txt'
    f= open(file_param,"w+")
    for paramstr,param in zip(arrayParamStr,arrayParam):
        str_to_write = paramstr + ' : ' + str(param) +'\n'
        f.write(str_to_write)
    f.close() 
    return(param_name,dir_path,file_param)
    
def write_results(file_param,list_result,list_result_str):
    f= open(file_param,"a")
    for paramstr,param in zip(list_result_str,list_result):
        str_to_write = paramstr + ' : ' + str(param) +'\n'
        f.write(str_to_write)
    f.close()
    
def tabs_to_str(list_result,list_result_str):
    str_to_write = ''
    for paramstr,param in zip(list_result_str,list_result):
        str_to_write += paramstr + ' : ' + str(param) +','
    return(str_to_write)