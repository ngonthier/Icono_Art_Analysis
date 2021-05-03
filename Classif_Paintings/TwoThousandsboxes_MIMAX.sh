#!/bin/bash
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
python -u TwoThousandsboxes.py >> results/TwoThousandsboxes.txt
