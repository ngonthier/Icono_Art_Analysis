"""
Implements Single Instance Learning SVM
"""

import numpy as np
import inspect
from sklearn.dummy import DummyClassifier
from misvm.util import slices
from sklearn.utils import check_X_y

class MIdummy(DummyClassifier):
    """
    Single-Instance Learning applied to MI data : dummy algorithm
    """

    def __init__(self, **kwargs):
        """
        """
        super(MIdummy, self).__init__(**kwargs)
        self._bags = None
        self._bag_predictions = None

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
        svm_X, svm_y = check_X_y(X=svm_X, y=svm_y)
        super(MIdummy, self).fit(svm_X, svm_y)


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
        bags_stack = np.array(np.vstack(bags))
        print(type(bags_stack))
        print(super(MIdummy, self).predict)
        
#        if self.strategy == "stratified":
#            inst_preds = super(MIdummy, self).predict_proba(np.vstack(bags))
#        else:
#        bags_stack = check_X_y(X=np.vstack(bags))
        inst_preds = super(MIdummy, self).predict(bags_stack)
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
        inst_preds = super(MIdummy, self).predict_proba(np.array(np.vstack(bags)))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)
        
    def get_params(self, deep=True):
        """
        return params
        """
        args = inspect.getfullargspec(super(DummyClassifier, self).__init__).args
        return {key: getattr(self, key, None) for key in args}


def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(list(map(len, bags)))])
