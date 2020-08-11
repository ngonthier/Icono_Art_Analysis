#!/bin/bash
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
python -u Other_way_to_use_score_MaxOfMax.py >> results/Other_way_to_use_score_MaxOfMax.txt
