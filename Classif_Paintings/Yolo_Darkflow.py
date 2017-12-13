#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:58:31 2017

@author: gonthier
"""

from darkflow.net.build import TFNet
import cv2

def compute_YOLO_output():  
    """
    Load YOLO model and compute the output : 
    Possible model : yolov1
    """
    path_model ='/media/HDD/models/'
    model = 'yolov1'
    model = 'yolov2-voc'
    cfg = path_model + model + ".cfg"
    weights = path_model + model +".weights"
    options = {"model": cfg, "load": weights, "threshold": 0.1}

    tfnet = TFNet(options)

    imgcv = cv2.imread("loulou.jpg")
    result = tfnet.return_predict(imgcv)
    print(result)
    
if __name__ == '__main__':
    compute_YOLO_output()