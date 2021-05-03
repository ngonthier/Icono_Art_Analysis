#!/bin/bash
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
python -u miperceptron_for_artistist_dataset.py >> results/miperceptron_for_artistist_dataset.txt
