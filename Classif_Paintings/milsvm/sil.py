"""
Implements Single Instance Learning SVM
From https://github.com/garydoranjr/misvm/blob/master/misvm/sil.py
Modified by Nicolas
"""
from __future__ import print_function, division
import numpy as np
import inspect
from sklearn.svm import LinearSVC as SVM
from milsvm.util import slices


class SIL(SVM):
    """
    Single-Instance Learning applied to MI data
    """

    def __init__(self,C=1.0, scale_C=True,
                 verbose=True, sv_cutoff=1e-7, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if False [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
        
        self._bags = None
        self._bag_predictions = None
        self.scale_C = scale_C
        self.verbose = verbose
        self.sv_cutoff = sv_cutoff
        self.C = C

        self._X = None
        self._y = None
        self._objective = None
        self._alphas = None
        self._sv = None
        self._sv_alphas = None
        self._sv_X = None
        self._sv_y = None
        self._b = None
        self._predictions = None
        super(SIL, self).__init__(**kwargs)

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])
        super(SIL, self).fit(svm_X, svm_y)

    def _compute_separator(self, K):
        super(SIL, self)._compute_separator(K)
        self._bag_predictions = _inst_to_bag_preds(self._predictions, self._bags)

    def predict(self, bags, instancePrediction = None):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param instancePrediction : flag to indicate if instance predictions 
                                    should be given as output.
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if instancePrediction is None:
            instancePrediction = False
            
        bags = [np.asmatrix(bag) for bag in bags]
        inst_preds = super(SIL, self).predict(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)

    def get_params(self, deep=True):
        """
        return params
        """
        args, _, _, _ = inspect.getargspec(super(SIL, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}


def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(map(len, bags))])
