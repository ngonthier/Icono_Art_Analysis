# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:06:48 2020

For the InceptionV1 from Lucid (based on old TF it seems)

@author: gonthier
"""

def getInceptionV1layersName():
    liste =  ['input_1',
     'zero_padding2d',
     'conv2d0_pre_relu',
     'conv2d0',
     'zero_padding2d_1',
     'pool_helper',
     'maxpool0',
     'localresponsenorm0',
     'conv2d1_pre_relu',
     'conv2d1',
     'conv2d2_pre_relu',
     'conv2d2',
     'localresponsenorm1',
     'zero_padding2d_2',
     'pool_helper_1',
     'maxpool1',
     'mixed3a_3x3_bottleneck_pre_relu',
     'mixed3a_5x5_bottleneck_pre_relu',
     'mixed3a_3x3_bottleneck',
     'mixed3a_5x5_bottleneck',
     'zero_padding2d_3',
     'zero_padding2d_4',
     'mixed3a_pool_pre_relu',
     'mixed3a_1x1_pre_relu',
     'mixed3a_3x3_pre_relu',
     'mixed3a_5x5_pre_relu',
     'mixed3a_pool_reduce_pre_relu',
     'mixed3a_pre_relu',
     'mixed3a',
     'mixed3b_3x3_bottleneck_pre_relu',
     'mixed3b_5x5_bottleneck_pre_relu',
     'mixed3b_3x3_bottleneck',
     'mixed3b_5x5_bottleneck',
     'zero_padding2d_5',
     'zero_padding2d_6',
     'mixed3b_pool',
     'mixed3b_1x1_pre_relu',
     'mixed3b_3x3_pre_relu',
     'mixed3b_5x5_pre_relu',
     'mixed3b_pool_reduce_pre_relu',
     'mixed3b_pre_relu',
     'mixed3b',
     'zero_padding2d_7',
     'pool_helper_2',
     'maxpool2',
     'mixed4a_3x3_bottleneck_pre_relu',
     'mixed4a_5x5_bottleneck_pre_relu',
     'mixed4a_3x3_bottleneck',
     'mixed4a_5x5_bottleneck',
     'zero_padding2d_8',
     'zero_padding2d_9',
     'mixed4a_pool',
     'mixed4a_1x1_pre_relu',
     'mixed4a_3x3_pre_relu',
     'mixed4a_5x5_pre_relu',
     'mixed4a_pool_reduce_pre_relu',
     'mixed4a_pre_relu',
     'mixed4a',
     'mixed4b_3x3_bottleneck_pre_relu',
     'mixed4b_5x5_bottleneck_pre_relu',
     'mixed4b_3x3_bottleneck',
     'mixed4b_5x5_bottleneck',
     'zero_padding2d_10',
     'zero_padding2d_11',
     'mixed4b_pool',
     'mixed4b_1x1_pre_relu',
     'mixed4b_3x3_pre_relu',
     'mixed4b_5x5_pre_relu',
     'mixed4b_pool_reduce_pre_relu',
     'mixed4b_pre_relu',
     'mixed4b',
     'mixed4c_3x3_bottleneck_pre_relu',
     'mixed4c_5x5_bottleneck_pre_relu',
     'mixed4c_3x3_bottleneck',
     'mixed4c_5x5_bottleneck',
     'zero_padding2d_12',
     'zero_padding2d_13',
     'mixed4c_pool',
     'mixed4c_1x1_pre_relu',
     'mixed4c_3x3_pre_relu',
     'mixed4c_5x5_pre_relu',
     'mixed4c_pool_reduce_pre_relu',
     'mixed4c_pre_relu',
     'mixed4c',
     'mixed4d_3x3_bottleneck_pre_relu',
     'mixed4d_5x5_bottleneck_pre_relu',
     'mixed4d_3x3_bottleneck',
     'mixed4d_5x5_bottleneck',
     'zero_padding2d_14',
     'zero_padding2d_15',
     'mixed4d_pool',
     'mixed4d_1x1_pre_relu',
     'mixed4d_3x3_pre_relu',
     'mixed4d_5x5_pre_relu',
     'mixed4d_pool_reduce_pre_relu',
     'mixed4d_pre_relu',
     'mixed4d',
     'mixed4e_3x3_bottleneck_pre_relu',
     'mixed4e_5x5_bottleneck_pre_relu',
     'mixed4e_3x3_bottleneck',
     'mixed4e_5x5_bottleneck',
     'zero_padding2d_16',
     'zero_padding2d_17',
     'mixed4e_pool',
     'mixed4e_1x1_pre_relu',
     'mixed4e_3x3_pre_relu',
     'mixed4e_5x5_pre_relu',
     'mixed4e_pool_reduce_pre_relu',
     'mixed4e_pre_relu',
     'mixed4e',
     'zero_padding2d_18',
     'pool_helper_3',
     'maxpool3',
     'mixed5a_3x3_bottleneck_pre_relu',
     'mixed5a_5x5_bottleneck_pre_relu',
     'mixed5a_3x3_bottleneck',
     'mixed5a_5x5_bottleneck',
     'zero_padding2d_19',
     'zero_padding2d_20',
     'mixed5a_pool',
     'mixed5a_1x1_pre_relu',
     'mixed5a_3x3_pre_relu',
     'mixed5a_5x5_pre_relu',
     'mixed5a_pool_reduce_pre_relu',
     'mixed5a_pre_relu',
     'mixed5a',
     'mixed5b_3x3_bottleneck_pre_relu',
     'mixed5b_5x5_bottleneck_pre_relu',
     'mixed5b_3x3_bottleneck',
     'mixed5b_5x5_bottleneck',
     'zero_padding2d_21',
     'zero_padding2d_22',
     'mixed5b_pool',
     'mixed5b_1x1_pre_relu',
     'mixed5b_3x3_pre_relu',
     'mixed5b_5x5_pre_relu',
     'mixed5b_pool_reduce_pre_relu',
     'head0_pool',
     'head1_pool',
     'mixed5b_pre_relu',
     'head0_bottleneck_pre_relu',
     'head1_bottleneck_pre_relu',
     'mixed5b',
     'head0_bottleneck',
     'head1_bottleneck',
     'avgpool']
    return(liste)
    
def get_trainable_layers_name():
    liste = ['conv2d0_pre_relu' ,
    'conv2d1_pre_relu' ,
    'conv2d2_pre_relu' ,
    'mixed3a_3x3_bottleneck_pre_relu' ,
    'mixed3a_5x5_bottleneck_pre_relu' ,
    'mixed3a_1x1_pre_relu' ,
    'mixed3a_3x3_pre_relu' ,
    'mixed3a_5x5_pre_relu' ,
    'mixed3a_pool_reduce_pre_relu' ,
    'mixed3b_3x3_bottleneck_pre_relu',
    'mixed3b_5x5_bottleneck_pre_relu' ,
    'mixed3b_1x1_pre_relu' ,
    'mixed3b_3x3_pre_relu' ,
    'mixed3b_5x5_pre_relu' ,
    'mixed3b_pool_reduce_pre_relu' ,
    'mixed4a_3x3_bottleneck_pre_relu' ,
    'mixed4a_5x5_bottleneck_pre_relu' ,
    'mixed4a_1x1_pre_relu' ,
    'mixed4a_3x3_pre_relu' ,
    'mixed4a_5x5_pre_relu' ,
    'mixed4a_pool_reduce_pre_relu' ,
    'mixed4b_3x3_bottleneck_pre_relu' ,
    'mixed4b_5x5_bottleneck_pre_relu' ,
    'mixed4b_1x1_pre_relu' ,
    'mixed4b_3x3_pre_relu' ,
    'mixed4b_5x5_pre_relu' ,
    'mixed4b_pool_reduce_pre_relu' ,
    'mixed4c_3x3_bottleneck_pre_relu' ,
    'mixed4c_5x5_bottleneck_pre_relu' ,
    'mixed4c_1x1_pre_relu' ,
    'mixed4c_3x3_pre_relu' ,
    'mixed4c_5x5_pre_relu' ,
    'mixed4c_pool_reduce_pre_relu' ,
    'mixed4d_3x3_bottleneck_pre_relu' ,
    'mixed4d_5x5_bottleneck_pre_relu' ,
    'mixed4d_1x1_pre_relu' ,
    'mixed4d_3x3_pre_relu' ,
    'mixed4d_5x5_pre_relu' ,
    'mixed4d_pool_reduce_pre_relu' ,
    'mixed4e_3x3_bottleneck_pre_relu' ,
    'mixed4e_5x5_bottleneck_pre_relu' ,
    'mixed4e_1x1_pre_relu' ,
    'mixed4e_3x3_pre_relu' ,
    'mixed4e_5x5_pre_relu' ,
    'mixed4e_pool_reduce_pre_relu' ,
    'mixed5a_3x3_bottleneck_pre_relu' ,
    'mixed5a_5x5_bottleneck_pre_relu' ,
    'mixed5a_1x1_pre_relu' ,
    'mixed5a_3x3_pre_relu' ,
    'mixed5a_5x5_pre_relu' ,
    'mixed5a_pool_reduce_pre_relu' ,
    'mixed5b_3x3_bottleneck_pre_relu' ,
    'mixed5b_5x5_bottleneck_pre_relu' ,
    'mixed5b_1x1_pre_relu' ,
    'mixed5b_3x3_pre_relu' ,
    'mixed5b_5x5_pre_relu' ,
    'mixed5b_pool_reduce_pre_relu' ,
    'head0_bottleneck_pre_relu' ,
    'head1_bottleneck_pre_relu']
    return(liste)
    
def get_dico_layers_type():
  dico =  {'zero_padding2d':'Pad',
     'conv2d0_pre_relu':'Conv2D',
     'conv2d0':'Relu',
     'zero_padding2d_1':'Pad',
#     'pool_helper',
#     'maxpool0',
#     'localresponsenorm0':None,
     'conv2d1_pre_relu':'Conv2D',
     'conv2d1':'Relu',
     'conv2d2_pre_relu':'Conv2D',
     'conv2d2':'Relu',
#     'localresponsenorm1',
     'zero_padding2d_2':'Pad',
#     'pool_helper_1',
#     'maxpool1',
     'mixed3a_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed3a_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed3a_3x3_bottleneck':'Conv2D',
     'mixed3a_5x5_bottleneck':'Conv2D',
     'zero_padding2d_3':'Pad',
     'zero_padding2d_4':'Pad',
     'mixed3a_pool_pre_relu':'Conv2D',
     'mixed3a_1x1_pre_relu':'Conv2D',
     'mixed3a_3x3_pre_relu':'Conv2D',
     'mixed3a_5x5_pre_relu':'Conv2D',
     'mixed3a_pool_reduce_pre_relu':'Conv2D',
     'mixed3a_pre_relu':'concat',
     'mixed3a':'Relu',
     'mixed3b_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed3b_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed3b_3x3_bottleneck':'Relu',
     'mixed3b_5x5_bottleneck':'Relu',
     'zero_padding2d_5':'Pad',
     'zero_padding2d_6':'Pad',
#     'mixed3b_pool',
     'mixed3b_1x1_pre_relu':'Conv2D',
     'mixed3b_3x3_pre_relu':'Conv2D',
     'mixed3b_5x5_pre_relu':'Conv2D',
     'mixed3b_pool_reduce_pre_relu':'Conv2D',
     'mixed3b_pre_relu':'concat',
     'mixed3b':'Relu',
     'zero_padding2d_7':'Pad',
#     'pool_helper_2',
#     'maxpool2',
     'mixed4a_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed4a_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed4a_3x3_bottleneck':'Relu',
     'mixed4a_5x5_bottleneck':'Relu',
     'zero_padding2d_8':'Pad',
     'zero_padding2d_9':'Pad',
#     'mixed4a_pool',
     'mixed4a_1x1_pre_relu':'Conv2D',
     'mixed4a_3x3_pre_relu':'Conv2D',
     'mixed4a_5x5_pre_relu':'Conv2D',
     'mixed4a_pool_reduce_pre_relu':'Conv2D',
     'mixed4a_pre_relu':'concat',
     'mixed4a':'Relu',
     'mixed4b_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed4b_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed4b_3x3_bottleneck':'Relu',
     'mixed4b_5x5_bottleneck':'Relu',
     'zero_padding2d_10':'Pad',
     'zero_padding2d_11':'Pad',
#     'mixed4b_pool',
     'mixed4b_1x1_pre_relu':'Conv2D',
     'mixed4b_3x3_pre_relu':'Conv2D',
     'mixed4b_5x5_pre_relu':'Conv2D',
     'mixed4b_pool_reduce_pre_relu':'Conv2D',
     'mixed4b_pre_relu':'concat',
     'mixed4b':'Relu',
     'mixed4c_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed4c_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed4c_3x3_bottleneck':'Relu',
     'mixed4c_5x5_bottleneck':'Relu',
     'zero_padding2d_12':'Pad',
     'zero_padding2d_13':'Pad',
#     'mixed4c_pool',
     'mixed4c_1x1_pre_relu':'Conv2D',
     'mixed4c_3x3_pre_relu':'Conv2D',
     'mixed4c_5x5_pre_relu':'Conv2D',
     'mixed4c_pool_reduce_pre_relu':'Conv2D',
     'mixed4c_pre_relu':'concat',
     'mixed4c':'Relu',
     'mixed4d_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed4d_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed4d_3x3_bottleneck':'Relu',
     'mixed4d_5x5_bottleneck':'Relu',
     'zero_padding2d_14':'Pad',
     'zero_padding2d_15':'Pad',
#     'mixed4d_pool',
     'mixed4d_1x1_pre_relu':'Conv2D',
     'mixed4d_3x3_pre_relu':'Conv2D',
     'mixed4d_5x5_pre_relu':'Conv2D',
     'mixed4d_pool_reduce_pre_relu':'Conv2D',
     'mixed4d_pre_relu':'concat',
     'mixed4d':'Relu',
     'mixed4e_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed4e_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed4e_3x3_bottleneck':'Relu',
     'mixed4e_5x5_bottleneck':'Relu',
     'zero_padding2d_16':'Pad',
     'zero_padding2d_17':'Pad',
#     'mixed4e_pool',
     'mixed4e_1x1_pre_relu':'Conv2D',
     'mixed4e_3x3_pre_relu':'Conv2D',
     'mixed4e_5x5_pre_relu':'Conv2D',
     'mixed4e_pool_reduce_pre_relu':'Conv2D',
     'mixed4e_pre_relu':'concat',
     'mixed4e':'Relu',
     'zero_padding2d_18':'Pad',
#     'pool_helper_3',
#     'maxpool3',
     'mixed5a_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed5a_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed5a_3x3_bottleneck':'Relu',
     'mixed5a_5x5_bottleneck':'Relu',
     'zero_padding2d_19':'Pad',
     'zero_padding2d_20':'Pad',
#     'mixed5a_pool',
     'mixed5a_1x1_pre_relu':'Conv2D',
     'mixed5a_3x3_pre_relu':'Conv2D',
     'mixed5a_5x5_pre_relu':'Conv2D',
     'mixed5a_pool_reduce_pre_relu':'Conv2D',
     'mixed5a_pre_relu':'concat',
     'mixed5a':'Relu',
     'mixed5b_3x3_bottleneck_pre_relu':'Conv2D',
     'mixed5b_5x5_bottleneck_pre_relu':'Conv2D',
     'mixed5b_3x3_bottleneck':'Relu',
     'mixed5b_5x5_bottleneck':'Relu',
     'zero_padding2d_21':'Pad',
     'zero_padding2d_22':'Pad',
#     'mixed5b_pool',
     'mixed5b_1x1_pre_relu':'Conv2D',
     'mixed5b_3x3_pre_relu':'Conv2D',
     'mixed5b_5x5_pre_relu':'Conv2D',
     'mixed5b_pool_reduce_pre_relu':'Conv2D',
#     'head0_pool',
#     'head1_pool',
     'mixed5b_pre_relu':'concat',
     'head0_bottleneck_pre_relu':'Conv2D',
     'head1_bottleneck_pre_relu':'Conv2D',
     'mixed5b':'Relu',
     'head0_bottleneck':'Relu',
     'head1_bottleneck':'Relu',
#     'avgpool'
     }
  return(dico)
    
def getInceptionV1LayersNumeral(style_layers):

    keras_inceptionv1_layers = getInceptionV1layersName()
    string = ''
    for elt in style_layers:
        try:
            string+= str(keras_inceptionv1_layers.index(elt))+'_'
        except ValueError as e:
            print(e)
    return(string)
    
def getInceptionV1LayersNumeral_bitsVersion(style_layers):
    """
    Return a shorter version for the layer index : maybe not the best way to do because
    if only bn_conv1 layer, we have a very big number because it is 001000000000000000000 full of 0 converted to base 10
    """
    keras_inceptionv1_layers = getInceptionV1layersName()

    list_bool = [False]*len(keras_inceptionv1_layers)
    for elt in style_layers:
        try:
            list_bool[keras_inceptionv1_layers.index(elt)] = True
        except ValueError as e:
            print(e)
    string = 'BV'+ str(int(''.join(['1' if i else '0' for i in list_bool]), 2)) # Convert the boolean version of index list to int
    return(string)