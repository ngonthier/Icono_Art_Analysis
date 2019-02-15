#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:08:15 2019

@author: gonthier
"""

import cv2 as cv
import numpy as np
im = cv.imread('data/000001.jpg')
rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
model = 'model/model.yml'
edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
orimap = edge_detection.computeOrientation(edges)
edges = edge_detection.edgesNms(edges, orimap)
edge_boxes = cv.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(300)
boxes = edge_boxes.getBoundingBoxes(edges, orimap)
for b in boxes:
    x, y, w, h = b
    cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
    
cv.imshow("edges", edges)
cv.imshow("edgeboxes", im)
edge_boxes.setMaxBoxes(30)
boxes = edge_boxes.getBoundingBoxes(edges, orimap)
scores_boxes = edge_detection.scoreAllBoxes(boxes)
