#!/bin/bash
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
python -u PascalVOC_sanity_check_HL1run.py >> results/PascalVOC_sanity_check_HL1run.txt
