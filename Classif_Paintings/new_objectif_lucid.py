# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:35:33 2020

New objectif function for the lucid package 

@author: gonthier
"""

from __future__ import absolute_import, division, print_function

from decorator import decorator
import numpy as np
import tensorflow as tf


from lucid.optvis.objectives_util import _dot, _dot_cossim, _extract_act_pos, _make_arg_str, _T_force_NHWC, _T_handle_batch
from lucid.optvis.objectives import wrap_objective,handle_batch

@wrap_objective(require_format='NHWC')
def autocorr(layer, n_channel, batch=None):
  """Visualize a single channel by maximizing it autocorrelation"""

  @handle_batch(batch)
  def inner(T):
      layer_t = T(layer)
      layer_t_channel_n = layer_t[..., n_channel]
      F_x = tf.fft2d(tf.complex(layer_t_channel_n,0.))
      R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) 
      # Module de la transformee de Fourrier : produit terme a terme
      norm2 = tf.nn.l2_normalize(R_x, axis=[1,2], epsilon=1e-10)
     
      return norm2
#  return inner
#
#  def inner(T):
#    layer_t = T(layer)
#    batch_n, _, _, channels = layer_t.get_shape().as_list()
#
#    flattened = tf.reshape(layer_t, [batch_n, -1, channels])
#    grams = tf.matmul(flattened, flattened, transpose_a=True)
#    grams = tf.nn.l2_normalize(grams, axis=[1,2], epsilon=1e-10)
#
#    return sum([ sum([ tf.reduce_sum(grams[i]*grams[j])
#                      for j in range(batch_n) if j != i])
#                for i in range(batch_n)]) / batch_n
#
#    length_style_layers_int = len(style_layers)
#    length_style_layers = float(length_style_layers_int)
#    weight_help_convergence = (10**9)
#    total_style_loss = 0.
#
#    
#    _, h_a, w_a, N = image_style.shape      
#    sess.run(net['input'].assign(image_style))
#        
#    for layer, weight in style_layers:
#        N = style_layers_size[layer[:5]]
#        M = M_dict[layer]
#        a = sess.run(net[layer])
#        x = net[layer]
#        x = tf.transpose(x, [0,3,1,2])
#        a = tf.transpose(a, [0,3,1,2])
#        F_x = tf.fft2d(tf.complex(x,0.))
#        R_x = tf.real(tf.multiply(F_x,tf.conj(F_x))) # Module de la transformee de Fourrier : produit terme a terme
#        R_x /= tf.to_float(M**2) # Normalisation du module de la TF
#        F_a = tf.fft2d(tf.complex(a,0.))
#        R_a = tf.real(tf.multiply(F_a,tf.conj(F_a))) # Module de la transformee de Fourrier
#        R_a /= tf.to_float(M**2)
#        style_loss = tf.nn.l2_loss(tf.subtract(R_x,R_a))  
#        style_loss *=  gamma_autocorr* weight * weight_help_convergence  / (2.*(N**2)*length_style_layers)
#        total_style_loss += style_loss
#    total_style_loss =tf.to_float(total_style_loss)