#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:01:43 2019

@author: gonthier
"""

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
#from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers  import Layer,InputSpec
#from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils

class _GlobalPooling2D(Layer):
  """Abstract class for different global pooling 2D layers.
  """

  def __init__(self, data_format=None, **kwargs):
    super(_GlobalPooling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape([input_shape[0], input_shape[3]])
    else:
      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(_GlobalPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class GlobalMinPooling2D(_GlobalPooling2D):
  """Global min pooling operation for spatial data.
  Arguments:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    - If `data_format='channels_last'`:
      4D tensor with shape `(batch_size, rows, cols, channels)`.
    - If `data_format='channels_first'`:
      4D tensor with shape `(batch_size, channels, rows, cols)`.
  Output shape:
    2D tensor with shape `(batch_size, channels)`.
  """

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.min(inputs, axis=[1, 2])
    else:
      return backend.min(inputs, axis=[2, 3])