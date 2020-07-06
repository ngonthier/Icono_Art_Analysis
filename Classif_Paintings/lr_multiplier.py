#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:12:59 2019

Wrapper of optimizer for having different learning rate during the optimization 
according to the layer. It only works for SGD for the moment

@author:  stante : https://github.com/stante gonthier 
https://github.com/stante/keras-contrib/blob/feature-lr-multiplier/keras_contrib/optimizers/lr_multiplier.py
"""

from tensorflow.python.keras.optimizers import Optimizer
#from tensorflow.python.keras.utils import get_custom_objects
#from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from tensorflow.python.training import training_ops

class LearningRateMultiplier(Optimizer):
    """Optimizer wrapper for per layer learning rate.
    This wrapper is used to add per layer learning rates by
    providing per layer factors which are multiplied with the
    learning rate of the optimizer.
    Note: This is a wrapper and does not implement any
    optimization algorithm.
    # Arguments
        optimizer: An optimizer class to be wrapped.
        lr_multipliers: Dictionary of the per layer factors. For
            example `optimizer={'conv_1/kernel':0.5, 'conv_1/bias':0.1}`.
            If for kernel and bias the same learning rate is used, the
            user can specify `optimizer={'conv_1':0.5}`.
        **kwargs: The arguments for instantiating the wrapped optimizer
            class.
    """
    def __init__(self, optimizer, lr_multipliers=None, **kwargs):
        self._class = optimizer
        self._optimizer = optimizer(**kwargs)
        config = self._optimizer.get_config()
        self.kind_opt = config['name']
        self._lr_multipliers = lr_multipliers or {}

    def _get_multiplier(self, param):
        for k in self._lr_multipliers.keys():
            if k in param.name:
                return self._lr_multipliers[k]

#    def get_updates(self, loss, params): # Version de base  this version works for small network
#             #ie it works for VGG19 but not for ResNet50
#         #print('self._optimizer',self._optimizer)
#         #print('self.updates ',self._optimizer.updates )
#         mult_lr_params = {p: self._get_multiplier(p) for p in params
#                           if self._get_multiplier(p)}
#         base_lr_params = [p for p in params if self._get_multiplier(p) is None]
#
#         updates = []
#         base_lr = self._optimizer.lr
#         for param, multiplier in mult_lr_params.items():
#             self._optimizer.lr.assign(tf.multiply(base_lr,multiplier))
#             updates.extend(self._optimizer.get_updates(loss, [param]))
#
#         self._optimizer.lr.assign(base_lr)
#         updates.extend(self._optimizer.get_updates(loss, base_lr_params))
#         print('updates diff lr',updates)
#         return updates
    
    def get_updates(self, loss, params):
        if self.kind_opt == 'SGD':
            return(self.get_updates_SGD(loss, params))
        elif self.kind_opt == 'Adam':
            return(self.get_updates_ADAM(loss, params))
        elif self.kind_opt == 'Padam':
            return(self.get_updates_Padam(loss, params))
        elif self.kind_opt == 'RMSprop':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError)
        
    def get_updates_SGD(self, loss, params):
        
        """
        This fonction is inspired by the get_updates function from tensorflow keras optimizer
        https://github.com/tensorflow/tensorflow/blob/a78fa541d83b1e5887be1f1f92d4a946623380ee/tensorflow/python/keras/optimizers.py
        even if keras by tensorflow use the optimizer_v2 fonctions :
        https://github.com/tensorflow/tensorflow/blob/a78fa541d83b1e5887be1f1f92d4a946623380ee/tensorflow/python/keras/optimizer_v2/optimizer_v2.py 
        This only works for SGD optimizer !!
        
        TODO : write on for adam and an other for RMSprop
        """

        mult_lr_params = {p: self._get_multiplier(p) for p in params
                           if self._get_multiplier(p)}
        
        for p in params:
            if self._get_multiplier(p) is None:
                mult_lr_params.update({p : 1.0})
                
        #base_lr_params = [mult_lr_params.update({p : 1.0} for p in params if self._get_multiplier(p) is None]

        base_lr = self._optimizer.lr
        
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self._optimizer.iterations, 1)]
    
        if self._optimizer._initial_decay  > 0:
          base_lr = base_lr * (  # pylint: disable=g-no-augmented-assignment
              1. /
              (1. +
               self._optimizer.decay * math_ops.cast(self._optimizer.iterations, K.dtype(self._optimizer.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        #self._optimizer.weights = [self._optimizer.iterations] + moments
        #self._optimizer.set_weights([self._optimizer.iterations] + moments)
        for p, g, m in zip(params, grads, moments):
          if self._get_multiplier(p) is None:
              multiplier= 1.0
          else:
              multiplier = self._get_multiplier(p)
              
          v = self._optimizer.momentum * m - base_lr*multiplier * g  # velocity
          self.updates.append(state_ops.assign(m, v))
    
          if self._optimizer.nesterov:
            new_p = p + self._optimizer.momentum * v - base_lr* multiplier* g
          else:
            new_p = p + v
    
          # Apply constraints.
          if getattr(p, 'constraint', None) is not None:
            new_p = p.constraint(new_p)
    
          self.updates.append(state_ops.assign(p, new_p))

        return self.updates
    
    def get_updates_ADAM(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []
    
        base_lr = self._optimizer.lr
        if self._optimizer._initial_decay > 0:
          base_lr = base_lr * (  # pylint: disable=g-no-augmented-assignment
              1. /
              (1. +
               self._optimizer.decay * math_ops.cast(self._optimizer.iterations, K.dtype(self._optimizer.decay))))
    
        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
          t = math_ops.cast(self._optimizer.iterations, K.floatx())
        base_lr_t = base_lr * (
            K.sqrt(1. - math_ops.pow(self._optimizer.beta_2, t)) /
            (1. - math_ops.pow(self._optimizer.beta_1, t)))
    
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
          vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
          vhats = [K.zeros(1) for _ in params]
        self.weights = [self._optimizer.iterations] + ms + vs + vhats
    
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
          if self._get_multiplier(p) is None:
              multiplier= 1.0
          else:
              multiplier = self._get_multiplier(p)
            
          m_t = (self._optimizer.beta_1 * m) + (1. - self._optimizer.beta_1) * g
          v_t = (self._optimizer.beta_2 * v) + (1. - self._optimizer.beta_2) * math_ops.square(g)
          if self.amsgrad:
            vhat_t = math_ops.maximum(vhat, v_t)
            p_t = p - base_lr_t *multiplier* m_t / (K.sqrt(vhat_t) + self._optimizer.epsilon)
            self.updates.append(state_ops.assign(vhat, vhat_t))
          else:
            p_t = p - base_lr_t *multiplier* m_t / (K.sqrt(v_t) + self._optimizer.epsilon)
    
          self.updates.append(state_ops.assign(m, m_t))
          self.updates.append(state_ops.assign(v, v_t))
          new_p = p_t
    
          # Apply constraints.
          if getattr(p, 'constraint', None) is not None:
            new_p = p.constraint(new_p)
    
          self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_updates_Padam(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        base_lr = self._optimizer.lr
        if self.initial_decay > 0:
            base_lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = base_lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            if self._get_multiplier(p) is None:
              multiplier= 1.0
            else:
              multiplier = self._get_multiplier(p)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                denom = (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                denom = (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # Partial momentum adaption.
            new_p = p - (lr_t*multiplier* (m_t / (denom ** (self.partial * 2))))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates    

    def get_config(self):
        config = {'optimizer': self._class,
                  'lr_multipliers': self._lr_multipliers}
        base_config = self._optimizer.get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

#    def __setattr__(self, name, value):
#        if name.startswith('_'):
#            super(LearningRateMultiplier, self).__setattr__(name, value)
#        else:
#            self._optimizer.__setattr__(name, value)


#get_custom_objects().update({'LearningRateMultiplier': LearningRateMultiplier})