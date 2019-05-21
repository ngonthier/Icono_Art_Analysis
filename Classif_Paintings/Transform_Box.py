#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:59:41 2018

@author: gonthier
"""

import numpy as np

def py_cpu_modif(dets,kind='SumPond'):
    """ Modification of the bounding box."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    dets_copy = dets.copy()
    dets_copy[:, 4] = 0
    for i in range(len(dets_copy)):

        xx1 = np.maximum(x1[i], x1[:])
        yy1 = np.maximum(y1[i], y1[:])
        xx2 = np.minimum(x2[i], x2[:])
        yy2 = np.minimum(y2[i], y2[:])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[:] - inter)
        if kind=='SumPond': # Sum ponderate of the score
            dets_copy[i,4] = np.sum(ovr*scores)
        elif kind=='Inter': # Only the intersection bring point
            dets_copy[i,4] = np.sum(inter*scores)

    

    return dets_copy