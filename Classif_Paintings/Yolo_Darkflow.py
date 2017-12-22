#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:58:31 2017

@author: gonthier


voc_models = ['yolo-full', 'yolo-tiny', 'yolo-small',  # <- v1
              'yolov1', 'tiny-yolov1', # <- v1.1 
              'tiny-yolo-voc', 'yolo-voc'] # <- v2

coco_models = ['tiny-coco', 'yolo-coco',  # <- v1.1
               'yolo', 'tiny-yolo'] # <- v2

"""

from darkflow.net.build import TFNet
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CLASSESVOC = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSESCOCO =  ('person', 'bicycle','car','motorcycle', 'aeroplane','bus','train','truck','boat',
 'traffic light','fire hydrant', 'stop sign', 'parking meter','bench','bird',
 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
 'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball', 'kite',
 'baseball bat','baseball glove','skateboard', 'surfboard','tennis racket','bottle', 
 'wine glass','cup','fork', 'knife','spoon','bowl', 'banana', 'apple','sandwich', 'orange', 
'broccoli','carrot','hot dog','pizza','donut','cake','chair', 'couch','potted plant','bed',
 'diningtable','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
 'oven','toaster','sink','refrigerator', 'book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush')
CLASSES_SET ={'VOC' : CLASSESVOC,
              'COCO' : CLASSESCOCO }

NETS_Pretrained =  ('yolo-voc','yolo-full','yolo')
NETS_Pretrained =  ('yolo-full','yolo')
def expit(x):
	return( 1. / (1. + np.exp(-x)))
    
def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def compute_YOLO_Perf_Paintings():
    classes_paitings = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    path_to_img = '/media/HDD/data/Painting_Dataset/'
    database = 'Paintings'
    databasetxt = database + '.txt'
    df_label = pd.read_csv(databasetxt,sep=",")
    df_test = df_label[df_label['set']=='test']
    sLength = len(df_test['name_img'])
    name_img = df_test['name_img'][0]
    i = 0
    y_test = np.zeros((sLength,10))
    path_model ='/media/HDD/models/yolo/'

    for model in NETS_Pretrained:
        print(model)
        if model=='yolo-voc' or model=='yolo-full':
            CLASSES = CLASSES_SET['VOC']
        elif model=='yolo':
            CLASSES = CLASSES_SET['COCO']
        nbClasses = len(CLASSES)
        
        cfg = path_model + model + ".cfg"
        weights = path_model + model +".weights"
        options = {"model": cfg, "load": weights, "threshold": 0.1,
                   "gpu" : 1.0}
        tfnet = TFNet(options)
        
        scores_all_image = np.zeros((len(df_test),nbClasses))
        
        for i,name_img in  enumerate(df_test['name_img']):
            if i%1000==0:
                print(i,name_img)
            complet_name = path_to_img + name_img + '.jpg'
            im = cv2.imread(complet_name)
            result = tfnet.return_predict_gonthier(im) # Arguments: im (ndarray): a color image in BGR order
            if(model=='yolo-full'):
                C,B,S = 20,2,7
                probs = get_probs(result, C, B, S)
                probs_per_classe = np.max(probs,axis=0)
            if(model=='yolo-voc'):
                C,B= 20,5
                H = 13
                W = 13
                probs = get_probs_v2(result,H,W,C,B)
                probs_per_classe = np.max(probs,axis=(0,1,2))
            if(model=='yolo'):
                C,B= 80,5
                H = 19
                W = 19
                probs = get_probs_v2(result,H,W,C,B)
                probs_per_classe = np.max(probs,axis=(0,1,2))
            scores_all_image[i,:] = probs_per_classe
            for j in range(10):
                if(classes_paitings[j] in list(df_test['classe'][df_test['name_img']==name_img])[0]):
                    y_test[i,j] = 1
            
        AP_per_class = []
        for k,classe in enumerate(classes_paitings):
            index_classe = np.where(np.array(CLASSES)==classe)[0][0]
            scores_per_class = scores_all_image[:,index_classe]
            #print(scores_per_class)
            #print(y_test[:,k],np.sum(y_test[:,k]))
            AP = average_precision_score(y_test[:,k],scores_per_class,average=None)
            AP_per_class += [AP]
            print("Average Precision for",classe," = ",AP)
        print(model," mean Average Precision = {0:.3f}".format(np.mean(AP_per_class)))

def get_probs(net_out, C, B, S):
    """ Come from  yolo_box_constructor in cy_yolo_fndboxes.pyx"""
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell
    probs =  np.ascontiguousarray(net_out[0 : prob_size]).reshape([SS,C])
    confs =  np.ascontiguousarray(net_out[prob_size : (prob_size + conf_size)]).reshape([SS,B])
    #coords =  np.ascontiguousarray(net_out[(prob_size + conf_size) : ]).reshape([SS, B, 4])
    #final_probs = np.zeros([SS,B,C],dtype=np.float32)
 
    for grid in range(SS):
        for b in range(B):
#            coords[grid, b, 0] = (coords[grid, b, 0] + grid %  S) / S
#            coords[grid, b, 1] = (coords[grid, b, 1] + grid // S) / S
#            coords[grid, b, 2] =  coords[grid, b, 2] ** sqrt
#            coords[grid, b, 3] =  coords[grid, b, 3] ** sqrt
            for class_loop in range(C):
                probs[grid, class_loop] = probs[grid, class_loop] * confs[grid, b]
    return(probs)
    
def get_probs_v2(net_out_in,H,W,C,B):
    """ Come from  yolo_box_constructor in cy_yolo2_fndboxes.pyx"""
    #int(net_out_in.shape[2]/B)
    net_out = net_out_in.reshape([H, W, B, int(net_out_in.shape[2]/B)])
    Classes = net_out[:, :, :, 5:]
    #print(Classes.shape)
    Bbox_pred =  net_out[:, :, :, :5]
    
    probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit(Bbox_pred[row, col, box_loop, 4])
                for class_loop in range(C):
                    arr_max=np.max((arr_max,Classes[row,col,box_loop,class_loop]))
                
                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=np.exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]
                
                for class_loop in range(C):
                    probs[row, col, box_loop, class_loop] = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum
    return(probs)

def compute_YOLO_output():  
    """
    Load YOLO model and compute the output : 
    Possible model : yolov1
    """
    path_model ='/media/HDD/models/yolo/'
    model = 'yolo-full' # Version 1 de YOLO trained on VOC12+07
    model = 'yolo-voc' # YOLOv2 	VOC 2007+2012 	2007
    model = 'yolo' #YOLOv2 608x608 	COCO trainval
    cfg = path_model + model + ".cfg"
    weights = path_model + model +".weights"
    options = {"model": cfg, "load": weights, "threshold": 0.1,
               "gpu" : 1.0}
    tfnet = TFNet(options)

    imgcv = cv2.imread("loulou.jpg")
    result = tfnet.return_predict_gonthier(imgcv) # Return the best prediction
    print(result.shape)
    #print(result)
    
    if(model=='yolo-full'):
        C,B,S = 20,2,7
        probs = get_probs(result, C, B, S)
        probs_per_classe = np.max(probs,axis=0)
        print(CLASSESVOC[np.argmax(probs_per_classe)])
    if(model=='yolo-voc'):
        C,B= 20,5
        H = 13
        W = 13
        probs = get_probs_v2(result,H,W,C,B)
        probs_per_classe = np.max(probs,axis=(0,1,2))
        print(CLASSESVOC[np.argmax(probs_per_classe)])
    if(model=='yolo'):
        C,B= 80,5
        H = 19
        W = 19
        probs = get_probs_v2(result,H,W,C,B)
        probs_per_classe = np.max(probs,axis=(0,1,2))
        print(CLASSESCOCO[np.argmax(probs_per_classe)])

    result = tfnet.return_predict(imgcv) # Return the best prediction
    print(result)
    
    
if __name__ == '__main__':
#    compute_YOLO_output()
    compute_YOLO_Perf_Paintings()