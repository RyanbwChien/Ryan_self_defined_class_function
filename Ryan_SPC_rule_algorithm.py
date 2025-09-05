# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:24:35 2025

@author: USER
"""

from typing import Literal

class SPC_rule_algorithm:
    @staticmethod
    def consequence_n_point_asc_desc(limit_count:int,
                                     direct:Literal["asc", "desc"],
                                     dataset):
        cnt = 0
        all_trigger_loc = []
        
        
        for idx in range(len(dataset)):
            if idx == 0:
                continue
            if direct == "asc":
                if dataset[idx] > dataset[idx-1]:
                    cnt += 1
                else:
                    cnt = 0
            if direct == "desc":
                if dataset[idx] < dataset[idx-1]:
                    cnt += 1        
                else:
                    cnt = 0
            if cnt>=limit_count: 
                all_trigger_loc.append(idx)
                yield idx
        return all_trigger_loc
            
                
dataset = list(range(0,10)) + list(range(9,-2,-1))    
    
    
g = SPC_rule_algorithm.consequence_n_point_asc_desc(7,"asc",dataset)


g.__next__()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    