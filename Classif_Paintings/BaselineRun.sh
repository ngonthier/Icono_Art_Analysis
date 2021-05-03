#!/bin/bash
cd /ldaphome/gonthier/Classif/Icono_Art_Analysis/Classif_Paintings/
source /cal/softs/anaconda/anaconda3/bin/activate TF
/cal/softs/anaconda/anaconda3/bin/python Baseline_script.py >> data/Baseline_results.txt
