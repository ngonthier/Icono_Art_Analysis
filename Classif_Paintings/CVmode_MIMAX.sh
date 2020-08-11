#!/bin/bash
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
/cal/softs/anaconda/anaconda3/bin/python -u CVmode_MIMAX.py >> results/CVmode_MIMAX.txt
