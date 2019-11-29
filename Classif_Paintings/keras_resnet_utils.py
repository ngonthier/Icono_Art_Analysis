#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:19:18 2019

@author: gonthier
"""

import warnings

import tensorflow as tf
import tensorflow.python.keras.utils as Sequence
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as K

def getResNet50layersName():
    liste = ['input_1',
     'conv1_pad',
     'conv1',
     'bn_conv1',
     'activation',
     'pool1_pad',
     'max_pooling2d',
     'res2a_branch2a',
     'bn2a_branch2a',
     'activation_1',
     'res2a_branch2b',
     'bn2a_branch2b',
     'activation_2',
     'res2a_branch2c',
     'res2a_branch1',
     'bn2a_branch2c',
     'bn2a_branch1',
     'add',
     'activation_3',
     'res2b_branch2a',
     'bn2b_branch2a',
     'activation_4',
     'res2b_branch2b',
     'bn2b_branch2b',
     'activation_5',
     'res2b_branch2c',
     'bn2b_branch2c',
     'add_1',
     'activation_6',
     'res2c_branch2a',
     'bn2c_branch2a',
     'activation_7',
     'res2c_branch2b',
     'bn2c_branch2b',
     'activation_8',
     'res2c_branch2c',
     'bn2c_branch2c',
     'add_2',
     'activation_9',
     'res3a_branch2a',
     'bn3a_branch2a',
     'activation_10',
     'res3a_branch2b',
     'bn3a_branch2b',
     'activation_11',
     'res3a_branch2c',
     'res3a_branch1',
     'bn3a_branch2c',
     'bn3a_branch1',
     'add_3',
     'activation_12',
     'res3b_branch2a',
     'bn3b_branch2a',
     'activation_13',
     'res3b_branch2b',
     'bn3b_branch2b',
     'activation_14',
     'res3b_branch2c',
     'bn3b_branch2c',
     'add_4',
     'activation_15',
     'res3c_branch2a',
     'bn3c_branch2a',
     'activation_16',
     'res3c_branch2b',
     'bn3c_branch2b',
     'activation_17',
     'res3c_branch2c',
     'bn3c_branch2c',
     'add_5',
     'activation_18',
     'res3d_branch2a',
     'bn3d_branch2a',
     'activation_19',
     'res3d_branch2b',
     'bn3d_branch2b',
     'activation_20',
     'res3d_branch2c',
     'bn3d_branch2c',
     'add_6',
     'activation_21',
     'res4a_branch2a',
     'bn4a_branch2a',
     'activation_22',
     'res4a_branch2b',
     'bn4a_branch2b',
     'activation_23',
     'res4a_branch2c',
     'res4a_branch1',
     'bn4a_branch2c',
     'bn4a_branch1',
     'add_7',
     'activation_24',
     'res4b_branch2a',
     'bn4b_branch2a',
     'activation_25',
     'res4b_branch2b',
     'bn4b_branch2b',
     'activation_26',
     'res4b_branch2c',
     'bn4b_branch2c',
     'add_8',
     'activation_27',
     'res4c_branch2a',
     'bn4c_branch2a',
     'activation_28',
     'res4c_branch2b',
     'bn4c_branch2b',
     'activation_29',
     'res4c_branch2c',
     'bn4c_branch2c',
     'add_9',
     'activation_30',
     'res4d_branch2a',
     'bn4d_branch2a',
     'activation_31',
     'res4d_branch2b',
     'bn4d_branch2b',
     'activation_32',
     'res4d_branch2c',
     'bn4d_branch2c',
     'add_10',
     'activation_33',
     'res4e_branch2a',
     'bn4e_branch2a',
     'activation_34',
     'res4e_branch2b',
     'bn4e_branch2b',
     'activation_35',
     'res4e_branch2c',
     'bn4e_branch2c',
     'add_11',
     'activation_36',
     'res4f_branch2a',
     'bn4f_branch2a',
     'activation_37',
     'res4f_branch2b',
     'bn4f_branch2b',
     'activation_38',
     'res4f_branch2c',
     'bn4f_branch2c',
     'add_12',
     'activation_39',
     'res5a_branch2a',
     'bn5a_branch2a',
     'activation_40',
     'res5a_branch2b',
     'bn5a_branch2b',
     'activation_41',
     'res5a_branch2c',
     'res5a_branch1',
     'bn5a_branch2c',
     'bn5a_branch1',
     'add_13',
     'activation_42',
     'res5b_branch2a',
     'bn5b_branch2a',
     'activation_43',
     'res5b_branch2b',
     'bn5b_branch2b',
     'activation_44',
     'res5b_branch2c',
     'bn5b_branch2c',
     'add_14',
     'activation_45',
     'res5c_branch2a',
     'bn5c_branch2a',
     'activation_46',
     'res5c_branch2b',
     'bn5c_branch2b',
     'activation_47',
     'res5c_branch2c',
     'bn5c_branch2c',
     'add_15',
     'activation_48',
     'avg_pool',
     'fc1000']
    return(liste)
    
def getBNlayersResNet50():
    """ Il y en a 53
    """
    liste=['bn_conv1',
     'bn2a_branch2a',
     'bn2a_branch2b',
     'bn2a_branch2c',
     'bn2a_branch1',
     'bn2b_branch2a',
     'bn2b_branch2b',
     'bn2b_branch2c',
     'bn2c_branch2a',
     'bn2c_branch2b',
     'bn2c_branch2c',
     'bn3a_branch2a',
     'bn3a_branch2b',
     'bn3a_branch2c',
     'bn3a_branch1',
     'bn3b_branch2a',
     'bn3b_branch2b',
     'bn3b_branch2c',
     'bn3c_branch2a',
     'bn3c_branch2b',
     'bn3c_branch2c',
     'bn3d_branch2a',
     'bn3d_branch2b',
     'bn3d_branch2c',
     'bn4a_branch2a',
     'bn4a_branch2b',
     'bn4a_branch2c',
     'bn4a_branch1',
     'bn4b_branch2a',
     'bn4b_branch2b',
     'bn4b_branch2c',
     'bn4c_branch2a',
     'bn4c_branch2b',
     'bn4c_branch2c',
     'bn4d_branch2a',
     'bn4d_branch2b',
     'bn4d_branch2c',
     'bn4e_branch2a',
     'bn4e_branch2b',
     'bn4e_branch2c',
     'bn4f_branch2a',
     'bn4f_branch2b',
     'bn4f_branch2c',
     'bn5a_branch2a',
     'bn5a_branch2b',
     'bn5a_branch2c',
     'bn5a_branch1',
     'bn5b_branch2a',
     'bn5b_branch2b',
     'bn5b_branch2c',
     'bn5c_branch2a',
     'bn5c_branch2b',
     'bn5c_branch2c']
    return(liste)
    
def getResNetLayersNumeral(style_layers,num_layers=50):
    if num_layers==50:
        keras_resnet_layers = getResNet50layersName()
    else:
        print('Only Resnet50 is supported')
        raise(NotImplementedError)
    string = ''
    for elt in style_layers:
        try:
            string+= str(keras_resnet_layers.index(elt))+'_'
        except ValueError as e:
            print(e)
    return(string)
    
def getResNetLayersNumeral_bitsVersion(style_layers,num_layers=50):
    """
    Return a shorter version for the layer index : maybe not the best way to do because
    if only bn_conv1 layer, we have a very big number because it is 001000000000000000000 full of 0 converted to base 10
    """
    if num_layers==50:
        keras_resnet_layers = getResNet50layersName()
    else:
        print('Only Resnet50 is supported')
        raise(NotImplementedError)
    list_bool = [False]*len(keras_resnet_layers)
    for elt in style_layers:
        try:
            list_bool[keras_resnet_layers.index(elt)] = True
        except ValueError as e:
            print(e)
    string = 'BV'+ str(int(''.join(['1' if i else '0' for i in list_bool]), 2)) # Convert the boolean version of index list to int
    return(string)
 
def is_sequence(seq):
    """Determine if an object follows the Sequence API.
    # Arguments
        seq: a possible Sequence object
    # Returns
        boolean, whether the object follows the Sequence API.
    From https://github.com/keras-team/keras/blob/1cf5218edb23e575a827ca4d849f1d52d21b4bb0/keras/engine/training_utils.py
    """
    # TODO Dref360: Decide which pattern to follow. First needs a new TF Version.
    return (getattr(seq, 'use_sequence_api', False) or set(dir(Sequence)).issubset(set(dir(seq) + ['use_sequence_api'])))

def iter_sequence_infinite(seq):
    """Iterate indefinitely over a Sequence.
    # Arguments
        seq: Sequence object
    # Returns
        Generator yielding batches.
    """
    while True:
        for item in seq:
            yield item
    
def fit_generator_ForRefineParameters(model,
                  generator,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
#                  callbacks=None,
#                  validation_data=None,
#                  validation_steps=None,
#                  validation_freq=1,
#                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0,controlGPUmemory=True):
    """ The goal of this function is to run the generator to update the parameters 
    of the batch normalisation"""
    
#    if controlGPUmemory:
#        config = tf.ConfigProto()
#        config.gpu_options.per_process_gpu_memory_fraction = 0.9
#        config.gpu_options.visible_device_list = "0"
#        sess = tf.Session(config=config)
#        K.set_session(sess)
#    else:
    sess = K.get_session()
    
    train_fn = K.function(inputs=[model.input], \
        outputs=[model.output], updates=model.updates) # model.output
    #init = tf.global_variables_initializer()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    epoch = initial_epoch

#    do_validation = bool(validation_data)
#    model._make_train_function()
#    if do_validation:
#        model._make_test_function()
#    use_sequence_api = True
#    print('generator',generator)
    use_sequence_api = is_sequence(generator)
#    print('use_sequence_api',use_sequence_api)
    if not use_sequence_api and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the `keras.utils.Sequence'
                        ' class.'))

    # if generator is instance of Sequence and steps_per_epoch are not provided -
    # recompute steps_per_epoch after each epoch
    recompute_steps_per_epoch = use_sequence_api and steps_per_epoch is None

    if steps_per_epoch is None:
        if use_sequence_api:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` '
                             'or use the `keras.utils.Sequence` class.')

#    # python 2 has 'next', 3 has '__next__'
#    # avoid any explicit version checks
#    val_use_sequence_api = is_sequence(validation_data)
#    val_gen = (hasattr(validation_data, 'next') or
#               hasattr(validation_data, '__next__') or
#               val_use_sequence_api)
#    if (val_gen and not val_use_sequence_api and
#            not validation_steps):
#        raise ValueError('`validation_steps=None` is only valid for a'
#                         ' generator based on the `keras.utils.Sequence`'
#                         ' class. Please specify `validation_steps` or use'
#                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
#    out_labels = model.metrics_names
#    callback_metrics = out_labels + ['val_' + n for n in out_labels]
#
#    # prepare callbacks
#    model.history = cbks.History()
#    _callbacks = [cbks.BaseLogger(
#        stateful_metrics=model.metrics_names[1:])]
#    if verbose:
#        _callbacks.append(
#            cbks.ProgbarLogger(
#                count_mode='steps',
#                stateful_metrics=model.metrics_names[1:]))
#    _callbacks += (callbacks or []) + [model.history]
#    callbacks = cbks.CallbackList(_callbacks)
#
#    # it's possible to callback a different model than self:
#    callback_model = model._get_callback_model()
#
#    callbacks.set_model(callback_model)
#    callbacks.set_params({
#        'epochs': epochs,
#        'steps': steps_per_epoch,
#        'verbose': verbose,
#        'do_validation': do_validation,
#        'metrics': callback_metrics,
#    })
#    callbacks._call_begin_hook('train')

    enqueuer = None
    val_enqueuer = None

    try:
#        if do_validation:
#            if val_gen and workers > 0:
#                # Create an Enqueuer that can be reused
#                val_data = validation_data
#                if is_sequence(val_data):
#                    val_enqueuer = OrderedEnqueuer(
#                        val_data,
#                        use_multiprocessing=use_multiprocessing)
#                    validation_steps = validation_steps or len(val_data)
#                else:
#                    val_enqueuer = GeneratorEnqueuer(
#                        val_data,
#                        use_multiprocessing=use_multiprocessing)
#                val_enqueuer.start(workers=workers,
#                                   max_queue_size=max_queue_size)
#                val_enqueuer_gen = val_enqueuer.get()
#            elif val_gen:
#                val_data = validation_data
#                if is_sequence(val_data):
#                    val_enqueuer_gen = iter_sequence_infinite(val_data)
#                    validation_steps = validation_steps or len(val_data)
#                else:
#                    val_enqueuer_gen = val_data
#            else:
#                # Prepare data for validation
#                if len(validation_data) == 2:
#                    val_x, val_y = validation_data
#                    val_sample_weight = None
#                elif len(validation_data) == 3:
#                    val_x, val_y, val_sample_weight = validation_data
#                else:
#                    raise ValueError('`validation_data` should be a tuple '
#                                     '`(val_x, val_y, val_sample_weight)` '
#                                     'or `(val_x, val_y)`. Found: ' +
#                                     str(validation_data))
#                val_x, val_y, val_sample_weights = model._standardize_user_data(
#                    val_x, val_y, val_sample_weight)
#                val_data = val_x + val_y + val_sample_weights
#                if model.uses_learning_phase and not isinstance(K.learning_phase(),
#                                                                int):
#                    val_data += [0.]
#                for cbk in callbacks:
#                    cbk.validation_data = val_data

        if workers > 0:
            if use_sequence_api:
                enqueuer = utils.OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = utils.GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

#        callbacks.model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            #model.reset_metrics()
#            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                generator_output = next(output_generator)

#                if not hasattr(generator_output, '__len__'):
#                    raise ValueError('Output of generator should be '
#                                     'a tuple `(x, y, sample_weight)` '
#                                     'or `(x, y)`. Found: ' +
#                                     str(generator_output))
#
                if len(generator_output) == 1:
                    x = generator_output
                    y = None
                    sample_weight = None
                elif len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    x = generator_output
                    y = None
                    sample_weight = None
#                    raise ValueError('Output of generator should be '
#                                     'a tuple `(x, y, sample_weight)` '
#                                     'or `(x, y) or (x)`. Found: ' +
#                                     str(generator_output))
                if len(x.shape)==4:
                    if x is None or len(x) == 0:
                        # Handle data tensors support when no input given
                        # step-size = 1 for data tensors
                        batch_size = 1
                    elif isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                elif len(x.shape)==3:
                    # In this case we only have one image in the batch
                    x = np.expand_dims(x, axis=0)
                    batch_size = 1

                # build batch logs
#                batch_logs = {'batch': batch_index, 'size': batch_size}
#                callbacks.on_batch_begin(batch_index, batch_logs)
                
                
#                outs = model.train_on_batch(x, y,
#                                            sample_weight=sample_weight,
#                                            class_weight=class_weight,
#                                            reset_metrics=False)
                # in the train_on_batch fct they use 
                # in _make_train_function : updates = self.updates + training_updates (from optimizer)

                # Here x is a numpy array because the datagenerator load numpy array
                
                #print(epoch,steps_done,'BEfore train fn')
                train_fn(x)  # updates property is updated after each call of the layer/model with an input, not before.
                #print('after train fn')
#                print(model.updates)
#                training_updates = model.get_updates_for(tf.convert_to_tensor(x))
#                print('training_updates',training_updates)
#                model.process_update(training_updates)
#                print('model.updates',model.updates)
#
#                outs = to_list(outs)
#                for l, o in zip(out_labels, outs):
#                    batch_logs[l] = o
#
#                callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)
#
                batch_index += 1
                steps_done += 1
#
#                # Epoch finished.
#                if (steps_done >= steps_per_epoch and
#                        do_validation and
#                        should_run_validation(validation_freq, epoch)):
#                    # Note that `callbacks` here is an instance of
#                    # `keras.callbacks.CallbackList`
#                    if val_gen:
#                        val_outs = model.evaluate_generator(
#                            val_enqueuer_gen,
#                            validation_steps,
#                            callbacks=callbacks,
#                            workers=0)
#                    else:
#                        # No need for try/except because
#                        # data has already been validated.
#                        val_outs = model.evaluate(
#                            val_x, val_y,
#                            batch_size=batch_size,
#                            sample_weight=val_sample_weights,
#                            callbacks=callbacks,
#                            verbose=0)
#                    val_outs = to_list(val_outs)
#                    # Same labels assumed.
#                    for l, o in zip(out_labels, val_outs):
#                        epoch_logs['val_' + l] = o
#
#                if callbacks.model.stop_training:
#                    break
#
#            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
#            if callbacks.model.stop_training:
#                break
#
            if use_sequence_api and workers == 0:
                generator.on_epoch_end()

            if recompute_steps_per_epoch:
                if workers > 0:
                    enqueuer.join_end_of_epoch()
#
#                # recomute steps per epochs in case if Sequence changes it's length
#                steps_per_epoch = len(generator)
#
#                # update callbacks to make sure params are valid each epoch
#                callbacks.set_params({
#                    'epochs': epochs,
#                    'steps': steps_per_epoch,
#                    'verbose': verbose,
#                    'do_validation': do_validation,
#                    'metrics': callback_metrics,
#                })
#
    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

#    callbacks._call_end_hook('train')
#    return model.history
    return(model)
