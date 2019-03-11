#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:26:50 2019

@author: gonthier
"""

import pickle
import tensorflow as tf
#from tf_faster_rcnn.lib.nets.vgg16 import vgg16
#from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
import resnet_152_keras
#import sys
import os
import cv2 # Need the contrib :  pip install opencv-contrib-python
# Echec de la Tentative de build avec tes modifications !!! https://gist.github.com/jarle/8336eb9cd140ad95f26a54f1572fc2fd
import pandas as pd
import os.path
from tool_on_Regions import reduce_to_k_regions
import numpy as np
from FasterRCNN import _int64_feature,_bytes_feature,_floats_feature


CLASSESVOC = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSESCOCO = ('__background__','person', 'bicycle','car','motorcycle', 'aeroplane','bus',
               'train','truck','boat',
 'traffic light','fire hydrant', 'stop sign', 'parking meter','bench','bird',
 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
 'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball', 'kite',
 'baseball bat','baseball glove','skateboard', 'surfboard','tennis racket','bottle', 
 'wine glass','cup','fork', 'knife','spoon','bowl', 'banana', 'apple','sandwich', 'orange', 
'broccoli','carrot','hot dog','pizza','donut','cake','chair', 'couch','potted plant','bed',
 'diningtable','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
 'oven','toaster','sink','refrigerator', 'book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush')


NETS = {'res152' : ('resnet152_weights_tf.h5',)}

DATASETS= {'coco': ('coco_2014_train+coco_2014_valminusminival',),'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

NETS_Pretrained = {'vgg16_VOC07' :'vgg16_faster_rcnn_iter_70000.ckpt',
                   'vgg16_VOC12' :'vgg16_faster_rcnn_iter_110000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt',
                   'res101_VOC07' :'res101_faster_rcnn_iter_70000.ckpt',
                   'res101_VOC12' :'res101_faster_rcnn_iter_110000.ckpt',
                   'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'
                   }
CLASSES_SET ={'VOC' : CLASSESVOC,
              'COCO' : CLASSESCOCO }

def resize(im,b,demonet,augmentation,list_of_crop,rois):
    x, y, w, h = b
    crop_img = im[x:x+w,y:y+h,:].astype(np.float32) # The network need an image in BGR
    if not(crop_img.shape[0]==0) and not(crop_img.shape[1]==0):
        if demonet=='res152':
            if augmentation:
                sizeIm = 256
            else:
                sizeIm = 224
        else:
            raise(NotImplemented)
#            if(crop_img.shape[0] < crop_img.shape[1]):
#                dim = (sizeIm, int(crop_img.shape[1] * sizeIm / crop_img.shape[0]),3)
#            else:
#                dim = (int(crop_img.shape[0] * sizeIm / crop_img.shape[1]),sizeIm,3)
        dim = (sizeIm,sizeIm)
        resized_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        resized_img[:,:,0] -= 103.939
        resized_img[:,:,1] -= 116.779
        resized_img[:,:,2] -= 123.68
        list_of_crop += [resized_img]
        rois += [b]
    return(list_of_crop,rois)
    
def get_crops(complet_name,edge_detection,k_regions,demonet,augmentation=False):
    im = cv2.imread(complet_name) # Load image in BGR
    rgb_im = im[:,:,[2,1,0]] # To shift from BGR to RGB
#    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(k_regions)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    list_of_crop = []
    rois = []
    for b in boxes:
       list_of_crop,rois = resize(im,b,demonet,augmentation,list_of_crop,rois)

    if len(rois)==0:
        b= [0,0,im.shape[0],im.shape[1]]
        list_of_crop,rois = resize(im,b,demonet,augmentation,list_of_crop,rois)
    rois = np.stack(rois).astype(np.float32)

    list_im =  np.stack(list_of_crop)
    return(list_im,rois)

def Compute_EdgeBoxesAndCNN_features(demonet='res152',nms_thresh = 0.7,database='IconArt_v1',
                                 augmentation=False,L2 =False,
                                 saved='all',verbose=True,filesave='tfrecords',k_regions=300,
                                 testMode=False):
    """
    The goal of this function is to compute 
    @param : demonet : teh kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : nms_thresh : the nms threshold on the Region Proposal Network
    
    /!\ Pour le moment la version de EdgeBoxes dans les contribs ne permet pas 
    d'avoir de scores 
        
    """

    path_data = '/media/HDD/output_exp/ClassifPaintings/'
    
    if database=='Paintings':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/Painting_Dataset/' 
        num_classes = 10
        ext = '.txt'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2012/JPEGImages/'
        num_classes = 20
        ext = '.txt'
        raise(NotImplementedError)
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        ext = '.csv'
        item_name = 'item'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        classes = ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
        num_classes = 10
    elif database=='VOC2007':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/VOCdevkit/VOC2007/JPEGImages/'
        num_classes = 20
        ext = '.csv'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif(database=='IconArt_v1'):
        ext='.csv'
        item_name='item'
        classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
        'Mary','nudity', 'ruins','Saint_Sebastien']
        path_to_img = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/JPEGImages/'
        num_classes = 7
    elif database=='PeopleArt':
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/PeopleArt/JPEGImages/'
        num_classes = 1
        ext = '.csv'
    elif database=='watercolor':
        num_classes = 6
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/cross-domain-detection/datasets/watercolor/JPEGImages/'
        classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
    elif database=='clipart':
        num_classes = 20
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = '/media/HDD/data/cross-domain-detection/datasets/clipart/JPEGImages/'
        raise(NotImplementedError)
    elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = '/media/HDD/data/Wikidata_Paintings/600/'
        num_classes = 5
        ext = '.txt'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    else:
        print(database,'is unknown')
        raise(NotImplemented)
        
    if database=='IconArt_v1':
        path_data_csvfile = '/media/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
    else:
        path_data_csvfile = path_data
    
    databasetxt = path_data_csvfile + database + ext 
    if database=='VOC2007' or database=='watercolor' or database=='clipart':
        df_label = pd.read_csv(databasetxt,sep=",",dtype=str)
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                  'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                  'turban':int,'set':str,'Anno':int}
        df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
    else:
        df_label = pd.read_csv(databasetxt,sep=",")
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
    
    if augmentation:
        raise NotImplementedError
        N = 50
    else: 
        N=1
    if L2:
        raise NotImplementedError
        extL2 = '_L2'
    else:
        extL2 = ''
    if saved=='all':
        savedstr = '_all'
    elif saved=='fc7':
        savedstr = ''
    elif saved=='pool5':
        savedstr = '_pool5'
    
    tf.reset_default_graph() # Needed to use different nets one after the other
    if verbose: print('=== EdgeBoxes net',demonet,'database',database,' ===')
    
    if demonet=='res152':
        weights_path = '/media/HDD/models/resnet152_weights_tf.h5'
        model = resnet_152_keras.resnet152_model_2048output(weights_path)
        size_output = 2048
    else:
        raise(NotImplemented)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
#    sess = tf.Session(config=tfconfig)
    
    features_resnet_dict= {}
    
    sets = ['train','val','trainval','test']
    
    if filesave == 'pkl':
        name_pkl_all_features = path_data+'EdgeBoxes_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+'.pkl'
        pkl = open(name_pkl_all_features, 'wb')
    elif filesave =='tfrecords':
        if k_regions==300:
            k_per_bag_str = ''
        else:
            k_per_bag_str = '_k'+str(k_regions)
        dict_writers = {}
        for set_str in sets:
            name_pkl_all_features = path_data
            if testMode: name_pkl_all_features+= 'TestMode_'
            name_pkl_all_features += 'EdgeBoxes_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
            dict_writers[set_str] = tf.python_io.TFRecordWriter(name_pkl_all_features)
     
    model_edgeboxes = 'model/model.yml'
    # Need of  pip install opencv-contrib-python
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model_edgeboxes)
    
    Itera = 1000
    if testMode:
        Itera = 1
    for i,name_img in  enumerate(df_label[item_name]):
        if testMode and i>1:
            break
        if filesave=='pkl':
            if not(k_regions==300):
                raise(NotImplemented)
            if i%Itera==0:
                if verbose : print(i,name_img)
                if not(i==0):
                    pickle.dump(features_resnet_dict,pkl) # Save the data
                    features_resnet_dict= {}
            if database in ['IconArt_v1','VOC2007','clipart','Paintings','watercolor','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                complet_name = path_to_img + name_img + '.jpg'
            elif database=='PeopleArt':
                complet_name = path_to_img + name_img
                name_sans_ext = os.path.splitext(name_img)[0]
            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                name_sans_ext = os.path.splitext(name_img)[0]
                complet_name = path_to_img +name_sans_ext + '.jpg'
            
            list_im, rois = get_crops(complet_name,edge_detection,k_regions,demonet,augmentation=False)
            fc7 = model.predict(list_im)
            roi_scores = np.ones((len(list_im,)))
#            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
            if saved=='fc7':
                features_resnet_dict[name_img] = fc7
#            elif saved=='pool5':
#                features_resnet_dict[name_img] = pool5
            elif saved=='all':
                features_resnet_dict[name_img] = rois,roi_scores,fc7
                
        elif filesave=='tfrecords':
            if i%Itera==0:
                if verbose : print(i,name_img)
            if database in ['IconArt_v1','VOC2007','clipart','Paintings','watercolor','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                complet_name = path_to_img + name_img + '.jpg'
                name_sans_ext = name_img
            elif database=='PeopleArt':
                complet_name = path_to_img + name_img
                name_sans_ext = os.path.splitext(name_img)[0]
            elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                name_sans_ext = os.path.splitext(name_img)[0]
                complet_name = path_to_img +name_sans_ext + '.jpg'

            im = cv2.imread(complet_name)
            
            height = im.shape[0]
            width = im.shape[1]
            
            list_im, rois = get_crops(complet_name,edge_detection,k_regions,demonet,augmentation=False)
            fc7 = model.predict(list_im)
            roi_scores = np.ones((len(list_im,)))
#            cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
            
            if(len(fc7) >= k_regions):
                rois = rois[0:k_regions,:]
                roi_scores =roi_scores[0:k_regions,]
                fc7 = fc7[0:k_regions,:]
            else:
                number_repeat = k_regions // len(fc7)  +1
                f_repeat = np.repeat(fc7,number_repeat,axis=0)
                roi_scores_repeat = np.repeat(roi_scores,number_repeat,axis=0)
                rois_repeat = np.repeat(rois,number_repeat,axis=0)
                rois = rois_repeat[0:k_regions,:]
                roi_scores =roi_scores_repeat[0:k_regions,]
                fc7 = f_repeat[0:k_regions,:]
            num_regions = fc7.shape[0]
            num_features = fc7.shape[1]
            dim1_rois = rois.shape[1]
            classes_vectors = np.zeros((num_classes,1),dtype=np.float32)
            
            print('fc7.shape',fc7.shape)
            print('rois.shape',rois.shape)
            
            if database=='Paintings':
                for j in range(num_classes):
                    if(classes[j] in df_label['classe'][i]):
                        classes_vectors[j] = 1
            if database in ['VOC2007','clipart','watercolor','PeopleArt']:
                for j in range(num_classes):
                    value = int((int(df_label[classes[j]][i])+1.)/2.)
                    #print(value)
                    classes_vectors[j] = value
            if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training','IconArt_v1']:
                for j in range(num_classes):
                    value = int(df_label[classes[j]][i])
                    classes_vectors[j] = value
            
            #features_resnet_dict[name_img] = fc7[np.concatenate(([0],np.random.randint(1,len(fc7),29))),:]
            if saved=='fc7':
                print('It is possible that you need to replace _bytes_feature by _floats_feature in this function')
                print('!!!!!!!!!!!!!!!!!!!!!')
                raise(NotImplementedError)
                # TODO : modifier cela !
                features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'num_regions': _int64_feature(num_regions),
                    'num_features': _int64_feature(num_features),
                    'fc7': _bytes_feature(tf.compat.as_bytes(fc7.tostring())),
                    'label' : _bytes_feature(tf.compat.as_bytes(classes_vectors.tostring())),
                    'name_img' : _bytes_feature(str.encode(name_sans_ext))})
            elif saved=='pool5':
                raise(NotImplementedError)
            elif saved=='all':
                feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'num_regions': _int64_feature(num_regions),
                    'num_features': _int64_feature(num_features),
                    'dim1_rois': _int64_feature(dim1_rois),
                    'rois': _floats_feature(rois),
                    'roi_scores': _floats_feature(roi_scores),
                    'fc7': _floats_feature(fc7),
                    'label' : _floats_feature(classes_vectors),
                    'name_img' : _bytes_feature(str.encode(name_sans_ext))}
                features=tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)    
#            print(len(feature['rois']))
            if database=='VOC2007' or database=='PeopleArt':
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='val').any():
                    dict_writers['val'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
            if (database=='Wikidata_Paintings_miniset') or database=='Paintings':
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='validation').any():
                    dict_writers['val'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
            if database in ['IconArt_v1','watercolor','clipart','WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
                if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                    dict_writers['train'].write(example.SerializeToString())
                    dict_writers['trainval'].write(example.SerializeToString())
                elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                    dict_writers['test'].write(example.SerializeToString())
                    
    if filesave=='pkl':
        pickle.dump(features_resnet_dict,pkl)
        pkl.close()
    elif filesave=='tfrecords':
        for set_str  in sets:
            dict_writers[set_str].close()
    
    if testMode:
        from TL_MIL import parser_w_rois_all_class
        import os
        sets = ['train','test','trainval','val']
        dim_rois = 4
        for set_str in sets:
            name_pkl_all_features = path_data
            if testMode: name_pkl_all_features+= 'TestMode_'
            name_pkl_all_features += 'EdgeBoxes_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords'
            print(name_pkl_all_features)
            train_dataset = tf.data.TFRecordDataset(name_pkl_all_features)
            sess = tf.Session()
            train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
                num_classes=num_classes,with_rois_scores=True,num_features=num_features,num_rois=k_regions,
                dim_rois=dim_rois))
            mini_batch_size = 1
            dataset_batch = train_dataset.batch(mini_batch_size)
            dataset_batch.cache()
            iterator = dataset_batch.make_one_shot_iterator()
            next_element = iterator.get_next()
            
            print(next_element)
            nx = sess.run(next_element)
            print(nx)
            os.remove(name_pkl_all_features)
            
if __name__ == '__main__':
#    Compute_EdgeBoxesAndCNN_features()
    Compute_EdgeBoxesAndCNN_features(database='watercolor')
    Compute_EdgeBoxesAndCNN_features(database='VOC2007')
