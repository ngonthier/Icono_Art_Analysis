"""
Implements Single Instance Learning SVM
"""

import numpy as np
import inspect
from sklearn.svm import SVC as SVM
from misvm.util import slices
from sklearn.svm import OneClassSVM

class MIbyOneClassSVM(SVM):
    """
    Single-Instance Learning applied to MI data
    """

    def __init__(self, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if True [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
        super(MIbyOneClassSVM, self).__init__(**kwargs)
        self._bags = None
        self._bag_predictions = None
#        self.scale_C = scale_C

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
#        svm_X = np.vstack(self._bags)
#        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
#                           for bag, cls in zip(self._bags, y)])
        # Select only the negative Bag :
        list_X_neg =[]
        for bag, cls in zip(self._bags, y):
            if cls==-1:
               list_X_neg +=[bag] 
        X_neg =  np.vstack(list_X_neg)
#        nu=0.00001 # An upper bound on the fraction of training errors
        onClassSVM = OneClassSVM()
        onClassSVM.fit(X_neg)
#        score_samples_X_neg = onClassSVM.score_samples(X_neg) # Positive for the normal value ie negative instance here
#        min_score = np.min(score_samples_X_neg)
        svm_X = [np.asmatrix(X_neg)]
        svm_y = [np.matrix(-np.ones((len(X_neg), 1)))]
        for bag, cls in zip(self._bags, y):
            if cls==1:
               scores_bag = onClassSVM.score_samples(bag)
               local_y = -1.*onClassSVM.predict(bag)
               local_y[np.argmin(scores_bag)] = 1.
               local_y = np.reshape(local_y,(len(bag), 1))
               svm_X += [bag]
               svm_y += [local_y]
        svm_X = np.vstack(svm_X)
        svm_y = np.vstack(svm_y)
#        print('Number of positive instances :',len(np.nonzero(1+svm_y)[0]),'on ',len(np.nonzero(1+y)[0]),' positive bags')
        super(MIbyOneClassSVM, self).fit(svm_X, svm_y)

#    def _compute_separator(self, K):
#        super(SIL, self)._compute_separator(K)
#        self._bag_predictions = _inst_to_bag_preds(self._predictions, self._bags)

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
        inst_preds = super(MIbyOneClassSVM, self).predict(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)
        
    def predict_proba(self, bags, instancePrediction = None):
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
        inst_preds = super(MIbyOneClassSVM, self).predict_proba(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)

    def get_params(self, deep=True):
        """
        return params
        """
        args, _, _, _ = inspect.getargspec(super(MIbyOneClassSVM, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}


def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(list(map(len, bags)))])
