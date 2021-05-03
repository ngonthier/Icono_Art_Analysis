#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:57:09 2020

@author: gonthier
"""

from StatsConstr_ClassifwithTL import learn_and_eval
target_dataset = 'Paintings'
#learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
#    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=True,\
#    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
#    regulOnNewLayer=None,optimizer='SGD',opt_option=[0.1,0.01],\
#    epochs=5,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
#    suffix='testdebug',plotConv=True,verbose=True)
    
# Former performance : # & 68.1 & 47.8 & 93.0 & 73.5 & 61.6 & 72.4 & 56.1 & 79.0 & 71.4 & 86.8 & 71.0 \\ 
    
# learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
#    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=True,\
#    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=False,\
#    regulOnNewLayer=None,optimizer='SGD',opt_option=[0.01],\
#    epochs=20,batch_size=16,pretrainingModif=True,\
#    suffix='testdebug',plotConv=True,verbose=True)
    
# learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
#     constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=True,\
#     transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
#     regulOnNewLayer=None,optimizer='SGD',opt_option=[0.1,0.01],\
#     epochs=5,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
#     suffix='testdebug',plotConv=True,verbose=True,clipnorm=10.)


# learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
#     constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
#     transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
#     regulOnNewLayer=None,optimizer='Padam',opt_option=[0.1],\
#     epochs=5,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
#     suffix='testdebug',plotConv=True,verbose=True)

learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
    regulOnNewLayer=None,optimizer='Padam',opt_option=[0.1,0.1],\
    epochs=5,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
    suffix='testdebug',plotConv=True,verbose=True)
# & 70.6 & 49.2 & 93.4 & 74.6 & 64.7 & 73.8 & 58.6 & 81.0 & 73.1 & 86.5 & 72.6 \\ 

learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP2',features='block5_pool',\
    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
    regulOnNewLayer=None,optimizer='Padam',opt_option=[0.1,0.01],\
    epochs=5,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
    suffix='testdebug',plotConv=True,verbose=True)
# & 61.2 & 45.2 & 92.4 & 72.1 & 55.2 & 68.7 & 54.6 & 79.1 & 63.9 & 83.7 & 67.6 \\ 
learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=True,\
    regulOnNewLayer=None,optimizer='Padam',opt_option=[0.1,0.05],\
    epochs=20,SGDmomentum=0.9,decay=1e-3,batch_size=16,pretrainingModif=True,\
    suffix='testdebug',plotConv=True,verbose=True,return_best_model=True)
#& 68.3 & 48.6 & 93.8 & 75.4 & 62.7 & 74.5 & 54.8 & 80.9 & 72.4 & 85.5 & 71.7 \\ 

learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='MLP1',features='block5_pool',\
    constrNet='ResNet50',kind_method='FT',gridSearch=False,ReDo=False,\
    transformOnFinalLayer='GlobalAveragePooling2D',cropCenter=False,\
    regulOnNewLayer=None,optimizer='Padam',opt_option=[0.1,0.1],\
    epochs=20,SGDmomentum=0.9,decay=1e-4,batch_size=16,pretrainingModif=True,\
    suffix='testdebug',plotConv=True,verbose=True,return_best_model=True,randomCrop=True)

