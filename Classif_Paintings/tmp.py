#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:08:15 2019

@author: gonthier
"""

import cv2 as cv
import numpy as np
im = cv.imread('data/000001.jpg')
cv.imshow("im", im)
rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
print(rgb_im.shape)
model = 'model/model.yml'
edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
print(edges.shape)
orimap = edge_detection.computeOrientation(edges)
print(orimap.shape)
edges = edge_detection.edgesNms(edges, orimap)
edge_boxes = cv.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(5)
boxes = edge_boxes.getBoundingBoxes(edges, orimap)
(newx,newy) = (256,256)
for i,b in enumerate(boxes):
    x, y, w, h = b
    crop_img = im[y:y+h,x:x+w,:]
    if crop_img.shape[0] ==0 or crop_img.shape[1]==0:
         print(rgb_im.shape)
         print(b)
         print(x,y,x+h,y+h)
         print(crop_img.shape)
         resized_img = cv.resize(crop_img,(newx,newy))
    cv.imshow(str(i), crop_img)
    cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

cv.imshow("edges", edges)
cv.imshow("edgeboxes", im)
cv.waitKey(0)
cv.destroyAllWindows()
#edge_boxes.setMaxBoxes(300)
#boxes = edge_boxes.getBoundingBoxes(edges, orimap)
#(newx,newy) = (256,256)
#for boxe in boxes:
#    x, y, w, h = boxe
#    print(rgb_im.shape)
#    print(boxe)
#    print(x,y,x+h,y+h)
#    crop_img = im[x:x+w,y:y+h,:]
#    print(crop_img.shape)
#    resized_img = cv.resize(crop_img,(newx,newy))
