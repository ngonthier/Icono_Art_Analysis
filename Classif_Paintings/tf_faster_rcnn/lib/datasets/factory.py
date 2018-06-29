# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

__sets = {}
from ..datasets.pascal_voc import pascal_voc
from ..datasets.CrossMod_db import CrossMod_db
from ..datasets.WikiTenLabels_db import WikiTenLabels_db
#from ..datasets.coco import coco # Commented by Nicolas because API COCO Python need python27 : it need to be modified problem with _mask

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,devkit_path= '/media/HDD/data/VOCdevkit',test_ext=True))
    
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

for db in ['watercolor','comic','clipart']:
    for split in ['train', 'test']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: CrossMod_db(db,split,devkit_path='/media/HDD/data/cross-domain-detection/datasets',test_ext=True))
 
for db in ['WikiTenLabels']:
    for split in ['test']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: WikiTenLabels_db(db,split,devkit_path='/media/HDD/data/Wikidata_Paintings/',test_ext=True))
 
## Set up coco_2014_<split>
#for year in ['2014']:
  #for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    #name = 'coco_{}_{}'.format(year, split)
    #__sets[name] = (lambda split=split, year=year: coco(split, year))

## Set up coco_2015_<split>
#for year in ['2015']:
  #for split in ['test', 'test-dev']:
    #name = 'coco_{}_{}'.format(year, split)
    #__sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
