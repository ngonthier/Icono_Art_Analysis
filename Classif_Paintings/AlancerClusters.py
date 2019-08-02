#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:49:51 2019

@author: gonthier
"""

from TL_MIL import tfR_FRCNN

def main():
    
    # Liste des choses que tu as a faire tourner :
    
    for database,restarts in zip(['RMN'],[3]):
        for with_score in  [False,True]:
            print(database,restarts,with_score)
            try: 
                tfR_FRCNN(database=database,verbose=True,restarts=restarts,ReDo=False,\
                          with_scores=with_score,AddOneLayer=True)
            except Exception as e:
                print(e)
                pass   

if __name__ == '__main__':
    main()