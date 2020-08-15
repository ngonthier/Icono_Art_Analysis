#!/bin/bash
cd /ldaphome/gonthier/Classif/Icono_Art_Analysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate tf18
/cal/softs/anaconda/anaconda3/bin/python -u TL_MIL.py >> data/TL_MIL_results2.txt
