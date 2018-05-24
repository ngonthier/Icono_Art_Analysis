#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:05:40 2018

@author: gonthier
"""

import numpy as np

def arrayToLatex(a,dtype=np.float32):
    if dtype==np.float32:
        stra = ' & '
        for i in range(len(a)):
            stra += "{0:.3f} & ".format(a[i])
        stra += "{0:.3f} \\\ \hline".format(np.mean(a))
        return(stra)
    elif dtype==str:
        stra = ' & '
        for i in range(len(a)):
            stra +=a[i] +" & "
        stra += "mean \\\ \hline"
        return(stra)