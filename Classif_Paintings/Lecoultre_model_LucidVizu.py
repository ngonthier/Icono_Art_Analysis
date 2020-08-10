# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:20:04 2020

The goal of this script is to run lucid on the RASTA model from Lecoutre et al. 2017 

@author: gonthier
"""

import os
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
import platform
import pathlib
import matplotlib
#
from tensorflow.python.keras.backend import set_session
import lucid_utils
import CompNet_FT_lucidIm 

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense

from RASTApython.models.tf_custom_resnets import resnet_trained,custom_resnet

from StatsConstr_ClassifwithTL import predictionFT_net,evaluationScoreRASTA
from IMDB import get_database

def get_weights_and_name_layers_forPurekerasModel(keras_net):
        
    net_layers = keras_net.layers
       
    list_weights = []
    list_name_layers = []
    for original_layer in net_layers:
        #print(original_layer.name,original_layer,isinstance(original_layer, keras.layers.convolutional.Conv2D))
        # check for convolutional layer
        layer_name = original_layer.name
        if isinstance(original_layer, keras.layers.convolutional.Conv2D) :
            # get filter weights
            o_weights = original_layer.get_weights() # o_filters, o_biases
            list_weights +=[o_weights]
            list_name_layers += [layer_name]
    return(list_weights,list_name_layers)

def seeNameLayers_KerasModel():
    import keras
    import os
    path_to_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','rasta_models','resnet_2017_7_31-19_9_45','model.h5')
    keras_net_finetuned = keras.models.load_model(path_to_model,compile=True)
    keras_net_finetuned.summary()
    for layer in keras_net_finetuned.layers:
        trainable_l = layer.trainable
        name_l = layer.name
        print(name_l,trainable_l)

def testDiversVaries():
    #tf.keras.backend.clear_session()
    #tf.reset_default_graph()
    #K.set_learning_phase(0)
    
    sess = tf.Session()
    #graph = tf.get_default_graph()
    #keras.backend.set_session(sess)
    # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
    # Otherwise, their weights will be unavailable in the threads after the session there has been set
    set_session(sess)
    
    
    original_model = resnet_trained(n_retrain_layers=20)
     # Cela va charger un tf.keras model
    
    base_model = resnet_trained(20)
    predictions = Dense(25, activation='softmax')(base_model.output)
    net_finetuned = Model(inputs=base_model.input, outputs=predictions)
    
    net_finetuned.predict(np.random.rand(1,224,224,3))
    trainable_layers_name = []
    for original_layer in original_model.layers:
        if original_layer.trainable:
            trainable_layers_name += [original_layer.name]
    #C:\media\gonthier\HDD2\output_exp\rasta_models\resnet_2017_7_31-19_9_45
    
    path_to_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','rasta_models','resnet_2017_7_31-19_9_45','model.h5')
    
    constrNet = 'LResNet50' # For Lecoutre ResNet50 version
    model_name = 'Lecoutre2017'
    input_name_lucid = 'input_1'
    
    tf.keras.backend.set_image_data_format('channels_last')
    
    net_finetuned.load_weights(path_to_model,by_name=True)
    net_finetuned.build((224,224,3))
    net_finetuned.summary()
    net_finetuned.predict(np.random.rand(1,224,224,3))
    #net_finetuned = keras.models.load_model(path_to_model,compile=True)
    #net_finetuned = load_model(path_to_model,compile=True)
    
    number_of_trainable_layer = 20 
    #
    #list_layer_index_to_print = []
    #for layer in model.layers:
    #    trainable_l = layer.trainable
    #    name_l = layer.name
    #    if trainable_l and 'res' in name_l:
    #        print(name_l,trainable_l)
    #        num_features = tf.shape(layer.bias).eval(session=sess)[0]
    #        list_layer_index_to_print += [name_l,np.arange(0,num_features)]
    #        
    #for layer in original_model.layers:
    #    print(layer)
    #    trainable_l = layer.trainable
    #    name_l = layer.name
    #    if trainable_l and 'res' in name_l:
    #        print(name_l,trainable_l)
    #        num_features = tf.shape(layer.bias).eval(session=sess)[0]
    #        list_layer_index_to_print += [name_l,np.arange(0,num_features)]
            
    #list_weights,list_name_layers = get_weights_and_name_layers_forPurekerasModel(original_model)
    list_weights,list_name_layers = CompNet_FT_lucidIm.get_weights_and_name_layers(original_model)
    
    dict_layers_relative_diff,dict_layers_argsort = CompNet_FT_lucidIm.get_gap_between_weights(list_name_layers,\
                                                                                    list_weights,net_finetuned)
    
    layer_considered_for_print_im = []
    for layer in net_finetuned.layers:
        trainable_l = layer.trainable
        name_l = layer.name
        print(name_l,trainable_l)
        if trainable_l and (name_l in trainable_layers_name):
            layer_considered_for_print_im += [name_l]
    num_top = 3
    list_layer_index_to_print_base_model = []
    list_layer_index_to_print = []
    #print(layer_considered_for_print_im)
    for key in dict_layers_argsort.keys():
        #print(key)
        if not(key in layer_considered_for_print_im):
            continue
        for k in range(num_top):
             topk = dict_layers_argsort[key][k]
             list_layer_index_to_print += [[key,topk]]
             list_layer_index_to_print_base_model += [[key,topk]]
    
    print('list_layer_index_to_print',list_layer_index_to_print)
    #dict_list_layer_index_to_print_base_model[model_name+suffix] = list_layer_index_to_print_base_model
    
    #dict_layers_relative_diff,dict_layers_argsort = CompNet_FT_lucidIm.get_gap_between_weights(list_name_layers,\
    #                                                    list_weights,model)
    
    # For the fine-tuned model !!!
    path_lucid_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','Lucid_model')
    path = path_lucid_model
    if path=='':
        os.makedirs('./model', exist_ok=True)
        path ='model'
    else:
        os.makedirs(path, exist_ok=True)
    
    frozen_graph = lucid_utils.freeze_session(sess,
                              output_names=[out.op.name for out in net_finetuned.outputs])
    
    name_pb = 'tf_graph_'+constrNet+model_name+'.pb'
    
    #nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(nodes_tab)
    tf.io.write_graph(frozen_graph,logdir= path,name= name_pb, as_text=False)
    
    if platform.system()=='Windows': 
        output_path = os.path.join('CompModifModel',constrNet)
    else:
        output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata','CompModifModel',constrNet)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    
    matplotlib.use('Agg')
    output_path_with_model = os.path.join(output_path,model_name)
    pathlib.Path(output_path_with_model).mkdir(parents=True, exist_ok=True)
    
#    global sess
#    global graph
#    with graph.as_default():
#        set_session(sess)
#        net_finetuned.predict(np.random.rand(1,224,224,3))
    net_finetuned.predict(np.random.rand(1,224,224,3))
    lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
             ,path_output=output_path_with_model,prexif_name=model_name,input_name=input_name_lucid,Net=constrNet)
    
    # For the original one !!! 
    original_model.predict(np.random.rand(1,224,224,3))
    #sess = keras.backend.get_session()
    #sess.run()
    frozen_graph = lucid_utils.freeze_session(sess,
                              output_names=[out.op.name for out in original_model.outputs])
    
    name_pb = 'tf_graph_'+constrNet+'PretrainedImageNet.pb'
    tf.io.write_graph(frozen_graph,logdir= path,name= name_pb, as_text=False)
    lucid_utils.print_images(model_path=path_lucid_model+'/'+name_pb,list_layer_index_to_print=list_layer_index_to_print\
         ,path_output=output_path_with_model,prexif_name=model_name,input_name=input_name_lucid,Net=constrNet)
    
def perf_test_RASTAweights():
    """
    Test the performance of the RASTA weights provide by Lecoultre et al.
    """
    dataset = 'RASTA'
    sess = tf.Session()
    set_session(sess)
    tf.keras.backend.set_image_data_format('channels_last')
    
    base_model = resnet_trained(20)
    predictions = Dense(25, activation='softmax')(base_model.output)
    net_finetuned = Model(inputs=base_model.input, outputs=predictions)
    #net_finetuned = custom_resnet() # Ce model a 87 layers 
    
    path_to_model = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','rasta_models','resnet_2017_7_31-19_9_45','model.h5')
    #ce model a 107 layers
    constrNet = 'LResNet50' # For Lecoutre ResNet50 version
    model_name = 'Lecoutre2017'
    input_name_lucid = 'input_1'
    
    net_finetuned.load_weights(path_to_model) # ,by_name=True
    net_finetuned.build((224,224,3))
    print(net_finetuned.summary())
    print(net_finetuned.predict(np.random.rand(1,224,224,3)))
    
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
            path_data,Not_on_NicolasPC = get_database(dataset)
    
    sLength = len(df_label[item_name])
    classes_vectors =  df_label[classes].values
    df_label_test = df_label[df_label['set']=='test']
    y_test = classes_vectors[df_label['set']=='test',:]
    
    cropCenter = False
    randomCrop = False
    imSize = 224
    predictions = predictionFT_net(net_finetuned,df_test=df_label_test,x_col=item_name,\
                                           y_col=classes,path_im=path_to_img,Net=constrNet,\
                                           cropCenter=cropCenter,randomCrop=randomCrop,\
                                           imSize=imSize)
    with sess.as_default():
        metrics = evaluationScoreRASTA(y_test,predictions) 
    top_k_accs,AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class,acc_per_class= metrics

    for k,top_k_acc in zip([1,3,5],top_k_accs):
        print('Top-{0} accuracy : {1:.2f}%'.format(k,top_k_acc*100))
