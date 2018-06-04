#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:00:59 2018

@author: gonthier

Sparsemax version by Nicolas Gonthier

"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow import transpose
from tensorflow import meshgrid
import numpy as np

def sparsemax(logits,axis=1,number_dim=2,name=None):
  """Computes sparsemax activations [1].
  For each batch `i` and class `j` we have
    $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
  [1]: https://arxiv.org/abs/1602.02068
  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `logits`.
  """

  with ops.name_scope(name, "sparsemax", [logits]) as name:
    logits = ops.convert_to_tensor(logits, name="Matrix")
    print(logits)
    obs = array_ops.shape(logits)[0]
    obs2 = array_ops.shape(logits)[1]
    dims = array_ops.shape(logits)[2]
    print(obs,dims)
    z = logits - math_ops.reduce_mean(logits, axis=-1)[:, array_ops.newaxis]

    # sort z
    z_sorted, _ = nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = math_ops.cumsum(z_sorted, axis=-1)
    k = math_ops.range(
        1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=-1)

    # calculate tau(z)
    print(k_z)
    mesh = meshgrid(math_ops.range(0, obs))
    print(mesh)
    indices = array_ops.stack([mesh, k_z - 1], axis=-1)
    tau_sum = array_ops.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

    # calculate p
    sparsemax = math_ops.maximum(
            math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])
#    sparsemax = transpose(sparsemax,perm=permut)
    return(sparsemax)