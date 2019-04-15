"""
Implements Multiple Instance Learning Deep neural network or multi-layers perceptron
based on the mi-svm idea
"""

import numpy as np
import inspect
from misvm.util import slices
from sklearn.utils import check_X_y
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.utils import class_weight
import keras

class miDLearlyStop():
    """
    Deep Learning model Multiple-Instance Learning based on mi-SVM idea applied 
    to MI data based on the idea to do early stopping that seems to be reobust 
    to noisy data
    """

    def __init__(self,max_iter=10,verbose=False, **kwargs):
        """
        @param : input_features
        """
        self._bags = None
        self._bag_predictions = None
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, bags, y,epochs=1):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        keras.backend.tensorflow_backend.set_session(sess)
        
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.array(np.asmatrix(y).reshape((-1, 1)))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])
        svm_X, svm_y = check_X_y(X=svm_X, y=svm_y)
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y.ravel())
        input_features = svm_X.shape[1]
        model = Sequential()
        model.add(Dense(64, activation=tf.nn.relu, input_shape=(input_features,)))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.tanh))
        self.model = model
        self.epochs = epochs
        
        self.model.compile(optimizer='Adadelta', 
              loss='mean_squared_error', # binary_crossentropy < mean_squared_error < hinge : seems better
              metrics=['accuracy'])
        if self.verbose:
            verboseNum = 1
        else:
            verboseNum = 0

        iteration = 0
        SelectirVar_haveChanged = True
        while((iteration < self.max_iter) and SelectirVar_haveChanged):
            if self.verbose: print("Iteration number in mi-NN :",iteration)
            iteration +=1
            self.model.fit(svm_X, svm_y, epochs=self.epochs,verbose=verboseNum,
                       class_weight=class_weights,validation_split=0,batch_size=32,
                       shuffle=True)
            
            all_labels = []
            for bag,y_bag in zip(bags,y):
                if y_bag==1:
                    decision_fct = self.model.predict(bag) # positive bag k
                    labels_k = np.sign(decision_fct)
#                    print(labels_k)
                    if len(np.nonzero(labels_k+1)[0])==0:
#                        print("need to assign positive label")
                        argmax_k = np.argmax(decision_fct)
                        labels_k[argmax_k] = 1 # We assign the highest case to the value 1 in each of the positive bag
                    assert(np.max(labels_k)==1)
                    all_labels += [labels_k.ravel()]
                else:   # Negative bags
                    all_labels += [[-1]*len(bag)]
            old_svm_y = svm_y
            svm_y = np.hstack(all_labels)
            class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(svm_y),
                                                 svm_y.ravel())
            yy_equal = old_svm_y==svm_y
            if all(yy_equal):
                SelectirVar_haveChanged=False          
                if self.verbose: print("End of the mi-meta Algo at iteration ",iteration," on ",self.max_iter)
        return(self)


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
        inst_preds = self.model.predict(np.vstack(bags))

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
        inst_preds = self.model.predict(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)
        
    def decision_function(self, bags, instancePrediction = None):
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
        inst_preds = self.model.predict(np.vstack(bags))

        if instancePrediction:        
            return _inst_to_bag_preds(inst_preds, bags), inst_preds
        else:
            return _inst_to_bag_preds(inst_preds, bags)

    def get_params(self, deep=True):
        """
        return params
        """
#        args, _, _, _ = inspect.getargspec(super(SIXGBoost, self).__init__)
#        args.pop(0) # Deprecated
        args = inspect.getfullargspec(self.__init__).args
        return {key: getattr(self, key, None) for key in args}

def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(list(map(len, bags)))])
