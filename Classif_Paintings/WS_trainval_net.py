# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_faster_rcnn.tools import _init_paths
from tf_faster_rcnn.lib.model.WS_train_val import get_training_roidb, trainWS_net
from tf_faster_rcnn.lib.model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from tf_faster_rcnn.lib.datasets.factory import get_imdb
from  tf_faster_rcnn.lib.datasets import imdb as datasetsimdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
#from tf_faster_rcnn.lib.nets.mobilenet_v1 import mobilenetv1

import os

possible_data_path = [os.path.join('media','HDD','data'),os.path.join("C:",os.sep,'Users','gonthier','Travail','data'),os.path.join('data'),os.path.join('data','ClassifPaintings')]

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network in a Weakly Supervised Manner')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str,default=os.path.join("C:",os.sep,'Users','gonthier','Travail','model','res152_faster_rcnn_iter_1190000.ckpt'))
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='IconArt_v1_train', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='IconArt_v1_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res152', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

#  if len(sys.argv) == 1:
#    parser.print_help()
#    sys.exit(1)

  args = parser.parse_args()
  return args

def get_imbdb_deal_withpath(imdb_name):
    for i,data_path in enumerate(possible_data_path):
        try:
            imdb = get_imdb(imdb_name,data_path=data_path)
            break
        except AssertionError:
            if i==(len(possible_data_path)-1):
                print('Data path unknown')
                print('Not in',possible_data_path)
                raise(AssertionError)
    return(imdb)

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imbdb_deal_withpath(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    print(roidb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasetsimdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imbdb_deal_withpath(imdb_names)
  return imdb, roidb

def get_train_labels_dbs(imdb_names):
  """
  Get the labels of the training images
  """

  def get_train_label_db(imdb_name):
    imdb = get_imbdb_deal_withpath(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
#    train_label_db = get_training_roidb(imdb)
#    print(train_label_db)
#    return train_label_db
    return(0)

  train_label_dbs = [get_train_label_db(s) for s in imdb_names.split('+')]
  train_label_db = train_label_dbs[0]
  if len(train_label_dbs) > 1:
    for r in train_label_dbs[1:]:
      train_label_db.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasetsimdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imbdb_deal_withpath(imdb_names)
  return imdb, train_label_db


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
#  imdb, roidb = combined_roidb(args.imdb_name)
#  print('{:d} roidb entries'.format(len(roidb)))
  imdb, train_labels = get_train_labels_dbs(args.imdb_name)

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
#  elif args.net == 'mobile':
#    net = mobilenetv1()
  else:
    raise NotImplementedError
    
  trainWS_net(net, imdb, valroidb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
#  trainWS_net(net, imdb, train_labels, valroidb, output_dir, tb_dir,
#            pretrained_model=args.weight,
#            max_iters=args.max_iters)
