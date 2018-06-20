#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  FineTuning_Test.py
#  
#  Copyright 2018 gonthier <gonthier@Morisot>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

Labels = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']

from keras.applications import inception_resnet_v2
import keras.backend as K
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras import metrics
import tensorflow as tf

def average_precision(target,output):
    # sort examples
    #sorted, indices = torch.sort(output, dim=0, descending=True)
    indices = tf.contrib.framework.argsort(output,axis=0,direction='DESCENDING',name=None)
    # Computes prec@i
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.
    for i in indices:
        label = target[i]
        if label == 0:
            continue
        if label == 1:
            pos_count += 1
        total_count += 1
        if label == 1:
            precision_at_i += pos_count / total_count
    precision_at_i /= pos_count
    return precision_at_i

def imageLoader(file_name, batch_size):
    path_csv = '/media/HDD/output_exp/ClassifPaintings/'
    path_images = '/media/HDD/data/Painting_Dataset/' 
    file_csv = path_csv + file_name
    df = pd.read_csv(file_csv,sep=",")
    df['name'] = df['name'].apply(lambda a: path_images + a)
    files = df['name']
    YY = df.as_matrix(columns=Labels)
    L = len(files)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            batch_end_local = batch_start
            i = 0
            X = np.zeros((limit-batch_start,299,299,3),dtype=np.float32)
            while i  < batch_end_local:
                image = df[i,'name']
                im = Image.open(image).resize((299, 299))
                X[i,:,:,:] = im.astype('float32')
                i+= 1
            #X = np.expand_dims(np.array(im), axis=0).astype('float32')
            
            Y = YY[batch_start:limit,:]
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size


def main(args):
    
    file_trainval = 'Paintings_Classification_trainval.csv'
    file_test = 'Paintings_Classification_test.csv'
    demonet = 'inception_resnet_v2'
    if demonet == 'inception_resnet_v2':
        base_model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
    print('Model based loaded')
    outputs = Dense(10, activation='sigmoid')(base_model.output)
    model = Model(base_model.inputs, outputs)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    batch_size= 16
    print('Before compiling')
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=[metrics.categorical_accuracy, average_precision])
    print('model compiled')
    steps_per_epoch = 5000 / batch_size
    model.fit_generator(imageLoader(file_trainval,batch_size),steps_per_epoch=steps_per_epoch, epochs=20,validation_data=imageLoader(file_test,batch_size))
    print('End training')
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
