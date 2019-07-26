#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:49:51 2019

@author: gonthier
"""

from TL_MIL import tfR_FRCNN

def main():
    
    # Liste des choses que tu as a faire tourner :
    
    for database,restarts in zip(['OIV5_small_30001','RMN'],[2,11]):
        print(database,restarts)
        for with_score in  [False,True]:
            try: 
                tfR_FRCNN(database=database,verbose=True,restarts=restarts,ReDo=False,with_scores=with_score)
            except Exception as e:
                print(e)
                pass   

if __name__ == '__main__':
    main()