#!/bin/zsh
cd /ldaphome/gonthier/IconArtAnalysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate tf18
python -u Other_way_to_use_score_MIMAX.py >> results/Other_way_to_use_score_MIMAX.txt
