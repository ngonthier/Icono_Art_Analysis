# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:13:08 2020

Image modification simple one

@author: gonthier
"""

from PIL import Image
import os
import numpy as np

def change_from_BRG_to_RGB(img_name_path,output_path=None,ext_name='toRGB'):
    """
    This function will go from RGB to BGR or the opposite
    """
    
    head,tail = os.path.split(img_name_path) 
    tail_split = tail.split('.')
    ext = tail_split[-1]
    name_im = '.'.join(tail_split[:-1])
    new_name_im = name_im + '_'+ext_name+'.' + ext
    if output_path is None:
        output_path = head
    
    img = Image.open(img_name_path)
    img = np.array(img) 
#    blue, green, red  = data.T 
#    data = np.array([red, green, blue])
#    data = data.transpose()
#    
#    
    img_new_channel = img[:,:,[2,1,0]]
    img_new_channel = Image.fromarray(img_new_channel)
    new_path_im = os.path.join(output_path,new_name_im)
    
    img_new_channel.save(new_path_im)
    
if __name__ == '__main__': 
    img_name_path='C:/Users/gonthier/ownCloud/Mes Presentations Latex/2020-04 Feature Visualisation/im/mixed3a_5x5_bottleneck_pre_reluConv2D_8_RASTA_small01_modif.png'
    change_from_BRG_to_RGB(img_name_path,output_path=None,ext_name='toRGB')
    img_name_path='C:/Users/gonthier/ownCloud/Mes Presentations Latex/2020-04 Feature Visualisation/im/mixed3a_5x5_bottleneck_pre_reluConv2D_8_ImagnetVGG.png'
    change_from_BRG_to_RGB(img_name_path,output_path=None,ext_name='toRGB')
