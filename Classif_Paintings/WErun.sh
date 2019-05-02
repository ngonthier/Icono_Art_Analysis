#!/bin/bash
cd /home/gonthier/Travail_Local/Icono_Art/Icono_Art_Analysis/Classif_Paintings
conda activate tf_cu90
python3 MIbenchmarkage.py
cd /home/gonthier/Travail_Local/Texture_Style/Style_Transfer
python LossFct_Test_Gen.py 



