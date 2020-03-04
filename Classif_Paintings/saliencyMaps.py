#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:30:01 2020

@author: gonthier
"""

from keras import backend as K
import numpy as np
import tensorflow as tf

# Ici je pense que tu as besoin de définir une classe qui instancie la fonction 
# le gradient etc et ensuite tu l'utilise pour calculer le résultats pour une image donnée

def GetSmoothedMask(
      x_value,model,c_i, stdev_spread=.15, nsamples=25,
      magnitude=True):
    """Returns a mask that is smoothed with the SmoothGrad method.
    The average of gradient of noisy image
    Args:
      x_value: Input value, not batched. Ie the input image
      model : the deep model used
      c_i : the index of the class concerned
      stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min). Defaults to 15%. Level of noise
      nsamples: Number of samples to average across to get the smooth gradient.
      magnitude: If true, computes the sum of squares of gradients instead of
                 just the sum. Defaults to true.
    """
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

    total_gradients = np.zeros_like(x_value)
    
    loss_c = model.output[0][c_i]
    grad_symbolic = K.gradients(loss_c, model.input)[0]
    iterate = K.function([model.input], grad_symbolic)
    
    for i in range(nsamples):
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      grad = iterate(x_plus_noise)
      if magnitude:
        total_gradients += (grad * grad)
      else:
        total_gradients += grad

    return total_gradients / nsamples

class SmoothedMask(object):
    def __init__(self,model,c_i, stdev_spread=.15, nsamples=25,
      magnitude=True):
        """
          Define the smoothGrad Mask class to return the smooth grad mask
          model : the deep model used
          c_i : the index of the class concerned
          stdev_spread: Amount of noise to add to the input, as fraction of the
                        total spread (x_max - x_min). Defaults to 15%. Level of noise
          nsamples: Number of samples to average across to get the smooth gradient.
          magnitude: If true, computes the sum of squares of gradients instead of
                     just the sum. Defaults to true.
        """
        self.magnitude = magnitude
        self.c_i = c_i
        self.nsamples = nsamples
        self.stdev_spread = stdev_spread

        loss_c = model.output[0][c_i]
        grad_symbolic = K.gradients(loss_c, model.input)[0]
        self.iterate = K.function([model.input], grad_symbolic)
    
    def GetMask(self,x_value):
        """Returns a mask that is smoothed with the SmoothGrad method.
        The average of gradient of noisy image
        Args:
          x_value: Input value, not batched. Ie the input image
        """
        total_gradients = np.zeros_like(x_value)
        stdev = self.stdev_spread * (np.max(x_value) - np.min(x_value))
    
        total_gradients = np.zeros_like(x_value)
        x_shape = list(x_value.shape)
        x_shape[0] = self.nsamples
        noise = np.random.normal(0, stdev, x_value.shape)
        x_plus_noise = x_value + noise
        grad = self.iterate(x_plus_noise)
        if self.magnitude: # Non teste
            grad = (grad * grad)
        total_gradients = np.mean(grad,axis=0,keepdims=True)
        return(total_gradients)
    
        # for i in range(self.nsamples):
        #   noise = np.random.normal(0, stdev, x_value.shape)
        #   x_plus_noise = x_value + noise
        #   grad = self.iterate(x_plus_noise)
        #   if self.magnitude:
        #     total_gradients += (grad * grad)
        #   else:
        #     total_gradients += grad
    
        # return total_gradients / self.nsamples
    
class IntegratedGradient(object):
    def __init__(self,model,c_i, x_baseline=None, x_steps=50,multiByImBaseline=False):
        """
          Define the Integradted Mask class to return the smooth grad mask
          model : the deep model used
          Args:
          model : the deep model used
          c_i : the index of the class concerned
          x_baseline: Baseline value used in integration. Defaults to 0.
          x_steps: Number of integrated steps between baseline and x.
      
        A SaliencyMask function that implements the integrated gradients method.
        https://arxiv.org/abs/1703.01365
        """
        self.c_i = c_i
        self.x_steps = x_steps
        self.x_baseline = x_baseline

        loss_c = model.output[0][c_i]
        grad_symbolic = K.gradients(loss_c, model.input)[0]
        self.iterate = K.function([model.input], grad_symbolic)
        
        self.multiByImBaseline = multiByImBaseline
    
    def GetMask(self,x_value,Net='ResNet'):
        """Returns a mask that is the integrated gradient between the baseline 
        and the image
        Args:
          x_value: Input value, not batched. Ie the input image
        """
        self.Net = Net
        if self.x_baseline is None:
            self.x_baseline = np.zeros_like(x_value)
        if 'VGG' in self.Net:
            preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif 'ResNet' in self.Net:
            preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        self.x_baseline =  preprocessing_function(self.x_baseline)

        assert self.x_baseline.shape == x_value.shape

        x_diff = x_value - self.x_baseline # Image - baseline
        
        x_steps_x_diff_arr = np.tile(x_diff,(self.x_steps,1,1,1))
        alphas = np.linspace(0, 1, self.x_steps)
        alphas = np.reshape(alphas,(self.x_steps,1,1,1))
        x_step = self.x_baseline + alphas * x_steps_x_diff_arr
        total_gradients = np.mean(self.iterate(x_step),axis=0,keepdims=True)
        if self.multiByImBaseline:
            total_gradients = total_gradients * x_diff
        
        return total_gradients


def GetMask_IntegratedGradients( x_value, model,c_i, x_baseline=None, x_steps=50):
    """Returns a integrated gradients mask.
    Args:
      x_value: input ndarray.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      
    A SaliencyMask function that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """
    loss_c = model.output[0][c_i]
    grad_symbolic = K.gradients(loss_c, model.input)[0]
    iterate = K.function([model.input], grad_symbolic)
    
    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    x_diff = x_value - x_baseline

    total_gradients = np.zeros_like(x_value)

    for alpha in np.linspace(0, 1, x_steps):
      x_step = x_baseline + alpha * x_diff

      total_gradients += iterate(x_step)

    return total_gradients * x_diff / x_steps

def GetMask_RandomBaseline_IntegratedGradients( x_value, model,c_i, x_baseline=None, x_steps=50,\
                                               num_random_trials=10):
    """Returns the average of random baselined of integrated gradients mask
    Args:
      x_value: input ndarray.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      num_random_trials : number of random trial used to averaging the random
      
    A SaliencyMask function that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """
    
    all_intgrads = []
    for i in range(num_random_trials):
        if x_baseline is None:
            baseline= np.max(x_value) *np.random.random(x_value.shape)
        else:
            baseline= x_baseline *np.random.random(x_value.shape)
        integrated_grad = GetMask_IntegratedGradients(x_value, model,c_i, x_baseline=baseline,\
                                                      x_steps=x_steps)
        all_intgrads.append(integrated_grad)
        # print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

def GetMask_IntegratedGradients_noisyImage( x_value, model,c_i, x_baseline=None, x_steps=50,\
                                               num_random_trials=10,stdev_spread=0.15):
    """Returns the average of integrated gradients between a baseline and a 
        noisy version of the image input
    Args:
      x_value: input ndarray.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      num_random_trials : number of random trial used to averaging the random
      
    A SaliencyMask function that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))
    
    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape
    
    all_intgrads = []
    for i in range(num_random_trials):
        noise = np.random.normal(0, stdev, x_value.shape)
        x_plus_noise = x_value + noise
        integrated_grad = GetMask_IntegratedGradients(x_plus_noise, model,c_i, x_baseline=x_baseline,\
                                                      x_steps=x_steps)
        all_intgrads.append(integrated_grad)
        # print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
