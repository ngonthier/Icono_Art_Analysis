# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:52:12 2020

@author: gonthier
"""

import shutil
import os
import glob

source = os.path.join('C:\\','Users','gonthier','Travail','V3DHNORD','im_old','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200')
destination = os.path.join('C:\\','Users','gonthier','ownCloud','Mes_Latex','2021_PhD_Thesis','im','RASTA_big0001_modif_adam_unfreeze50_RandForUnfreezed_SmallDataAug_ep200')

#def recursive_copy_files(source_path, destination_path, override=False):
#    """
#    Recursive copies files from source  to destination directory.
#    :param source_path: source directory
#    :param destination_path: destination directory
#    :param override if True all files will be overridden otherwise skip if file exist
#    :return: count of copied files
#    """
#    files_count = 0
#    if not os.path.exists(destination_path):
#        os.mkdir(destination_path)
#    items = glob.glob(os.path.join(source_path,'*')
#    for item in items:
#        if os.path.isdir(item):
#            path = os.path.join(destination_path, item.split('/')[-1])
#            files_count += recursive_copy_files(source_path=item, destination_path=path, override=override)
#        else:
#            file = os.path.join(destination_path, item.split('/')[-1])
#            if not os.path.exists(file) or override:
#                shutil.copyfile(item, file)
#                files_count += 1
#    return files_count
#
#recursive_copy_files(source, destination)



# the destination folder must not exist !!!
shutil.copytree(source, destination)
