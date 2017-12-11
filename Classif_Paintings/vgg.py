# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze and num_classes is not None:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19

if __name__ == '__main__':
  import numpy as np
  import yaml
  import cv2
  from PIL import Image
  with tf.Graph().as_default():
     # The Inception networks expect the input image to have color channels scaled from [-1, 1]
      
      #Load the model
      sess = tf.Session()
      slim = tf.contrib.slim
      
      input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
      checkpoint_file = '/media/HDD/models/vgg_19.ckpt'
      logits, end_points = vgg_19(input_tensor, is_training=False)
      #checkpoint_file = '/media/HDD/models/vgg_16.ckpt'
      #logits, end_points = vgg_16(input_tensor, is_training=False)
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_file)
      
      # 2 manièeres de preprocesse les images 
#      testImage_string = tf.gfile.FastGFile('lyon.jpg', 'rb').read()
#      testImage = tf.image.decode_jpeg(testImage_string, channels=3)
#      image_size = 299
#      processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
#      im = sess.run(tf.expand_dims(processed_image, 0))
#   
      im = Image.open('loulou.jpg').resize((224,224),Image.NEAREST)
      im = np.array(im).astype(np.float32) 
      im2 = Image.open('cat.jpg').resize((224,224),Image.NEAREST) 
      im2 = np.array(im2).astype(np.float32) 
      
      im[:,:,2] -= 103.9390
      im[:,:,1] -= 116.7790
      im[:,:,0] -= 123.6800
      
      im2[:,:,2] -= 103.9390
      im2[:,:,1] -= 116.7790
      im2[:,:,0] -= 123.6800
#      im = im[:,:,::-1]
#      im2 = im2[:,:,::-1]
      im = im.reshape(-1,224,224,3)
      im2 = im2.reshape(-1,224,224,3)
      ims = np.concatenate((im,im2))
      
      
#      im = cv2.resize(cv2.imread('dog.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
#      im2 = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32) # Read image in BGR !
#      im[:,:,0] -= 103.939
#      im[:,:,1] -= 116.779
#      im[:,:,2] -= 123.68
#       
#      im2[:,:,0] -= 103.939
#      im2[:,:,1] -= 116.779
#      im2[:,:,2] -= 123.68
#      im = np.expand_dims(im, axis=0)
#      im2 = np.expand_dims(im2, axis=0)
#      ims = np.concatenate((im,im2))
      
      dictr = yaml.load(open("imageNet_map.txt").read().replace('\n',''))
      predict_values, logit_values = sess.run([end_points, logits], feed_dict={input_tensor: ims})

      print(logit_values.shape)
      print(logit_values)
      for j in range(2):
          print("Im ",j)
          string = "5 first Predicted class : \n"
          out_sort_arg = np.argsort(logit_values[j,:])[::-1]
          #out_sort_arg = np.flip(np.argsort(predict_values[j,:]),axis=1)[0]
          for i in range(5):
              string += str(out_sort_arg[i]) + ' : ' + dictr[out_sort_arg[i]] + ' : ' + str(logit_values[j,out_sort_arg[i]]) + '\n'
          print(string)
    
          
## VGG19 avec image en RGB
#[[-2.73314333  1.09662306 -4.03027964 ..., -1.74038112  1.5133183
#   4.8088212 ]
# [-1.92644906 -3.12641644 -0.26798081 ..., -1.03142321  3.46962595
#   2.97795582]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#265 : toy poodle
#154 : Pekinese, Pekingese, Peke
#266 : miniature poodle
#850 : teddy, teddy bear
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#287 : lynx, catamount
#282 : tiger cat
#281 : tabby, tabby cat
#478 : carton



## VGG16
#[[-3.47197485  0.19120288 -5.1765132  ..., -2.32900071  1.14006698
#   5.31669807]
# [-3.2320621  -1.92238438 -0.17705512 ..., -2.62692833  4.10695648
#   4.26083612]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#154 : Pekinese, Pekingese, Peke
#152 : Japanese spaniel
#265 : toy poodle
#260 : chow, chow chow
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#281 : tabby, tabby cat
#478 : carton
#287 : lynx, catamount
#282 : tiger cat

#VGG19 de Google avec image en BGR 
#
#[[-1.95166147 -1.05174422 -2.39439106 ..., -2.45173836  0.72128332
#   3.64116573]
# [-2.27594757 -3.95760512  1.48110342 ..., -1.25377035  2.28270459
#   1.37344515]]
#Im  0
#5 first Predicted class : 
#265 : toy poodle
#259 : Pomeranian
#154 : Pekinese, Pekingese, Peke
#266 : miniature poodle
#267 : standard poodle
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#287 : lynx, catamount
#279 : Arctic fox, white fox, Alopex lagopus
#248 : Eskimo dog, husky
#250 : Siberian husky
#
#VGG16 de Google avec image en BGR
#[[-2.69815135 -1.34160721 -3.91112494 ..., -1.79113984  0.84197533
#   3.50710869]
# [-2.74866271 -1.84448302  1.16664696 ..., -3.25384331  2.18476605
#   2.35617161]]
#Im  0
#5 first Predicted class : 
#259 : Pomeranian
#154 : Pekinese, Pekingese, Peke
#151 : Chihuahua
#265 : toy poodle
#152 : Japanese spaniel
#
#Im  1
#5 first Predicted class : 
#285 : Egyptian cat
#248 : Eskimo dog, husky
#281 : tabby, tabby cat
#434 : bath towel
#250 : Siberian husky