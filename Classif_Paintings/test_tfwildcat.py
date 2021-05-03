# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:28:54 2020

Test of my reimplementation of Wildcat pooling 

@author: gonthier
"""

from StatsConstr_ClassifwithTL import learn_and_eval

learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01],return_best_model=True,
                epochs=20,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten')

learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.1,0.01],return_best_model=True,
                epochs=20,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16)

learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01,0.1],return_best_model=True,
                epochs=20,cropCenter=False,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16)
# ResNet50 noflatten ep :20 & 19.6 & 23.6 & 3.3 & 33.7 & 38.3 & 13.8 & 3.1 & 19.3 \\  # avec 0.1 pour le test set

learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01,0.1],return_best_model=False,
                epochs=20,cropCenter=False,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16)

# Test image size different
learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01],return_best_model=True,
                epochs=1,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16,
                param_wildcat=[8,20,20,0.7],imSize=256) 

# Test return the model weights
a = learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01],return_best_model=True,
                epochs=1,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16,
                param_wildcat=[8,20,20,0.7],imSize=256,returnStatistics=True) 

learn_and_eval('IconArt_v1',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01,0.1],return_best_model=True,
                epochs=20,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten',batch_size=16,
                param_wildcat=[8,20,20,0.7],imSize=448) # 448 batch size = 16, lrp = 0.1, lr = 0.01, epochs = 20, k=20, maps=8 and alpha= 0.7

learn_and_eval('Paintings',source_dataset='ImageNet',final_clf='wildcat',
               features='activation_48',\
                constrNet='ResNet50',kind_method='FT',pretrainingModif=True,\
                optimizer='SGD',opt_option=[0.01,0.1],return_best_model=True,
                epochs=20,cropCenter=True,verbose=True,SaveInit=False,
                transformOnFinalLayer='noflatten')
