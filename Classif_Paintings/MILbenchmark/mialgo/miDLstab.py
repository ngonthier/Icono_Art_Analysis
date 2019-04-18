"""
Implements Multiple Instance Learning Deep neural network or multi-layers perceptron
based on the mi-svm idea but with a full training at the end to stabilize it
"""

import numpy as np
import inspect
from misvm.util import slices
from sklearn.utils import check_X_y
from keras.models import Sequential
from keras.layers import Dense,Dropout,InputLayer
import tensorflow as tf
from sklearn.utils import class_weight
import keras
from keras.callbacks import EarlyStopping

from MILbenchmark.mialgo.lsuv_init import LSUVinit

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_pred_01 = (K.sign(y_pred)+1)/2.
    y_true_01 =(y_true+1)/2.
    precision = precision(y_true_01, y_pred_01)
    recall = recall(y_true_01, y_pred_01)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def matthews_correlation(y_true, y_pred):
    """
    Inputs vectors between -1 and 1
    """
    
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

class miDLstab():
    """
    Deep Learning model Multiple-Instance Learning based on mi-SVM idea applied 
    to MI data based on the idea to do early stopping that seems to be reobust 
    to noisy data
    """


    def __init__(self,max_iter=10,epochs_final=20,dropout_rate=0.0,verbose=False,\
#                 kernel_initializer = 'Orthogonal',bias_initializer='RandomNormal',
                 kernel_initializer = 'glorot_uniform',bias_initializer='zeros',
                 lsuv_init=False,class_weight='balanced',final_earlyStop=False,**kwargs):
        """
        @param : max_iter : maximum number of iteration in the mi algo
        @param : epochs_final number of epochs on the final training
        @param : dropout_rate : dropout rate of the model
        @param : kernel_initializer : kernel initializer
        @param : bias_initializer : biases initializer
        @param : lsuv_init do a lsuv initializaion
        @param : class_weight of the data for the training
        @param : final_earlyStop 
        """
        self._bags = None
        self._bag_predictions = None
        self.verbose = verbose
        self.max_iter = max_iter
        self.epochs_final = epochs_final
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.lsuv_init = lsuv_init
        self.class_weight =class_weight
        self.final_earlyStop = final_earlyStop

    def fit(self, bags, y,epochs=1):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        
        # Parameters of the model : 
        optimizer = 'Adadelta'
        loss = 'mean_squared_error'
        metric = 'accuracy'
        metric = f1
        list_size_layers = [64,128,64]
        batch_size = 32
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        keras.backend.tensorflow_backend.set_session(sess)
        
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.array(np.asmatrix(y).reshape((-1, 1)))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])
        svm_X, svm_y = check_X_y(X=svm_X, y=svm_y)
        if self.class_weight=='balanced':
            class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y),
                                                     y.ravel())
            class_weights = {-1:class_weights[0],1:class_weights[1]}
        else:
            class_weights = {-1:1.,1:1.}
        input_features = svm_X.shape[1]
        model = Sequential()
        model.add(InputLayer(input_shape=(input_features,)))
        for number_neurones in list_size_layers:
            model.add(Dense(number_neurones, activation=tf.nn.relu,kernel_initializer=
                            self.kernel_initializer,bias_initializer=self.bias_initializer))
#        model.add(Dense(64, activation=tf.nn.relu, input_shape=(input_features,)))
#        
#        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.tanh))
        
        if self.lsuv_init:
            model = LSUVinit(model,batch=svm_X[:batch_size,:],verbose=self.verbose)
        
        self.model = model
        self.epochs = epochs
        
        self.model.compile(optimizer=optimizer, 
              loss=loss, # binary_crossentropy < mean_squared_error < hinge : seems better
              metrics=[metric])
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
                       class_weight=class_weights,validation_split=0,batch_size=batch_size,
                       shuffle=True)
            
#            all_labels = []
#            thres_pos_list = []
#            thres_neg_list = []
#            for bag,y_bag in zip(bags,y):
#                decision_fct = self.model.predict(bag)
#                if y_bag==1:
#                    labels_k = np.sign(decision_fct)
##                    print(labels_k)
#                    if len(np.nonzero(labels_k+1)[0])==0:
#                        thres_pos = np.max(decision_fct)
#                        thres_pos_list += [thres_pos]
#                else:
#                    thres_neg = np.max(decision_fct)
#                    thres_neg_list += [thres_neg]
#            thres_neg = np.max(thres_neg_list)
#            if len(thres_pos_list)>0:
#                thres_pos = np.min(thres_pos_list)
#                print('thres_pos',thres_pos)
#            print('thres_neg',thres_neg)
#            
#            for bag,y_bag in zip(bags,y):
#                decision_fct = self.model.predict(bag)
#                if y_bag==1:
#                    labels_k = np.sign(decision_fct)
#                    labels_k[np.where(decision_fct<thres_neg)[0]] = -1
##                    print(labels_k)
#                    if len(np.nonzero(labels_k+1)[0])==0:
##                        print("need to assign positive label")
#                        argmax_k = np.argmax(decision_fct)
#                        labels_k[argmax_k] = 1 # We assign the highest case to the value 1 in each of the positive bag
##                    assert(np.max(labels_k)==1)
#                    all_labels += [labels_k.ravel()]
#                else:   # Negative bags
#                    all_labels += [[-1]*len(bag)]
            all_labels = [] # Classical mi-svm like meta algo
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
            print(iteration,'number of positive instances in training loop',np.sum((svm_y+1)/2.))
            if self.class_weight=='balanced':
                class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(svm_y),
                                                     svm_y.ravel())
                class_weights = {-1:class_weights[0],1:class_weights[1]}
            else:
                class_weights = {-1:1.,1:1.}

            yy_equal = old_svm_y==svm_y
            if all(yy_equal):
                SelectirVar_haveChanged=False          
                if self.verbose: print("End of the mi-meta Algo at iteration ",iteration," on ",self.max_iter)
        
        # Fine-Tuning of the model at the end        
        final_model = Sequential()
        final_model.add(InputLayer(input_shape=(input_features,)))
        for number_neurones in list_size_layers:
            final_model.add(Dense(number_neurones, activation=tf.nn.relu,kernel_initializer=
                            self.kernel_initializer,bias_initializer=self.bias_initializer))
            if self.dropout_rate > 0.0:
                final_model.add(Dropout(self.dropout_rate))

        final_model.add(Dense(1, activation=tf.nn.tanh))
        if self.lsuv_init:
            final_model = LSUVinit(final_model,batch=svm_X[:batch_size,:],verbose=self.verbose)
        # simple early stopping
#        from keras.callbacks import EarlyStopping
        if self.final_earlyStop:
            es = EarlyStopping(monitor='val_f1', mode='max', verbose=1,patience=1)
        
        final_model.compile(optimizer=optimizer, 
              loss=loss, # binary_crossentropy < mean_squared_error < hinge : seems better
              metrics=[metric])

        if not(self.final_earlyStop):
            final_model.fit(svm_X, svm_y, epochs=self.epochs_final,verbose=verboseNum,
                           class_weight=class_weights,validation_split=0,batch_size=batch_size,
                           shuffle=True)
        else:
            final_model.fit(svm_X, svm_y, epochs=self.epochs_final,verbose=verboseNum,
                           class_weight=class_weights,validation_split=0.2,batch_size=batch_size,
                           shuffle=True,callbacks=[es])
        self.model = final_model
        
        from sklearn.metrics import f1_score
        pred_instance_labels = self.model.predict(svm_X)
        f1eval = f1_score(svm_y, np.sign(pred_instance_labels),labels=[-1,1])
        print("F1 score training set",f1eval)
        
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
