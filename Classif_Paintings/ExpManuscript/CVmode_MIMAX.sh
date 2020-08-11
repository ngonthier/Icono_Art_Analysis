#!/bin/bash
cd /ldaphome/gonthier/Classif/Icono_Art_Analysis/Classif_Paintings/ExpManuscript/
source /cal/softs/anaconda/anaconda3/bin/activate TF
/cal/softs/anaconda/anaconda3/bin/python -u CVmode_MIMAX.py >> results/CVmode_MIMAX.txt
