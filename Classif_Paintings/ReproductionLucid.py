#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:21:23 2020

@author: gonthier
"""

import numpy as np
import scipy.ndimage as nd
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
#import lucid.optvis.optimizer as transfor
import tensorflow as tf
import matplotlib.pyplot as plt

model = models.InceptionV1()
model.load_graphdef()

neuron1 = ('mixed4b_pre_relu', 452) 
C = lambda neuron: objectives.channel(*neuron)

out = render.render_vis(model, C(neuron1))
plt.imshow(out[0][0])

JITTER = 1
ROTATE = 5
SCALE  = 1.1

transforms = [
    transform.pad(2*JITTER),
    transform.jitter(JITTER),
    transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    transform.random_rotate(range(-ROTATE, ROTATE+1))
]

imgs = render.render_vis(model, "mixed4b_pre_relu:452", transforms=transforms,
                         param_f=lambda: param.image(64), 
                         thresholds=[2048], verbose=False,
                         relu_gradient_override=True,use_fixed_seed=True)
plt.imshow(imgs[0][0])


# Note that we're doubling the image scale to make artifacts more obvious
show([nd.zoom(img[0], [2,2,1], order=0) for img in imgs])


model = models.InceptionV1_slim()
model.load_graphdef()

out = render.render_vis(model, 'InceptionV1/InceptionV1/Mixed_4c/concat:452')
plt.imshow(out[0][0])


model = models.VGG19_caffe()
model.load_graphdef()
nodes_tab = [n.name for n in tf.get_default_graph().as_graph_def().node]
out = render.render_vis(model, 'conv3_1/conv3_1:1')
plt.figure()
plt.imshow(out[0][0])

LEARNING_RATE = 0.0005

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)      
JITTER = 1
ROTATE = 5
SCALE  = 1.1

transforms = [
    transform.pad(2*JITTER),
    transform.jitter(JITTER),
    transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    transform.random_rotate(range(-ROTATE, ROTATE+1))
]
image_full_tricks = render.render_vis(model, "conv3_1/conv3_1:1", transforms=transforms,
                         param_f=lambda: param.image(64, fft=False, decorrelate=False),
                         optimizer=optimizer,
                         thresholds=[4096], verbose=False)
plt.imshow(image_full_tricks[0][0])
image_full_tricks = render.render_vis(model, "conv5_1/conv5_1:1", transforms=transforms,
                         param_f=lambda: param.image(64, fft=False, decorrelate=False),
                         optimizer=optimizer,
                         thresholds=[4096], verbose=False)
plt.imshow(image_full_tricks[0][0])

param_f = lambda: param.image(128, batch=4)
obj = objectives.channel("conv5_1/conv5_1", 1) - 1e2*objectives.diversity("conv5_1/conv5_1")
imgs = render.render_vis(model, obj, param_f)
for img in imgs[0]:
    plt.figure()
    plt.imshow(img)
    
LEARNING_RATE = 0.005 # Parametres tres important depend des couches considere meme

optimizer = tf.train.AdamOptimizer(LEARNING_RATE) 
out2 =  render.render_vis(model, "conv5_1/conv5_1:1",optimizer=optimizer,thresholds=[4096])
plt.figure()
plt.imshow(out2[0][0])


