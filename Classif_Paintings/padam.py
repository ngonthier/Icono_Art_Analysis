# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:44:16 2020

padam optimizer from Closing the Generalization Gap of Adaptive Gradient Methods in TrainingDeep Neural Networks
https://arxiv.org/pdf/1806.06763.pdf


Code inspired from : 
https://github.com/stante/keras-contrib/blob/feature-lr-multiplier/keras_contrib/optimizers/padam.py



@author: gonthier
"""

from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

class Padam(Optimizer):
    """Partially adaptive momentum estimation optimizer.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        partial: float, 0 <= partial <= 0.5 . Parameter controlling partial
            momentum adaption. For `partial=0`, this optimizer behaves like SGD,
            for `partial=0.5` it behaves like AMSGrad.

    # References
        - [Closing the Generalization Gap of Adaptive Gradient Methods
        in Training Deep Neural Networks](https://arxiv.org/pdf/1806.06763.pdf)

    """

    def __init__(self, learning_rate=1e-1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., amsgrad=False, partial=1. / 8., **kwargs):
        if partial < 0 or partial > 0.5:
            raise ValueError(
                "Padam: 'partial' must be a positive float with a maximum "
                "value of `0.5`, since higher values will cause divergence "
                "during training."
            )
        super(Padam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.partial = partial
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr* (1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                     (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                denom = (math_ops.sqrt(vhat_t) + self.epsilon)
                self.updates.extend(K.update(vhat, vhat_t))
            else:
                denom = (math_ops.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # Partial momentum adaption.
            new_p = p - (lr_t * (m_t / (math_ops.pow(denom,(self.partial * 2)))))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.extend(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'partial': self.partial,
                  'name': 'Padam'}
        base_config = super(Padam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

