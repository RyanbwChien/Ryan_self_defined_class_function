# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 11:22:40 2025

@author: USER
"""

from typing import Union

string1 = "ijfkgjfkj;djkfj;"
string2 = "ogkk;kd;kfj;kdj"
pattern1 = "kfj"
pattern2 = "abc"


def check_multi_pattern_in_string(string:Union[list,str], pattern:Union[list,str]) -> bool:
    if isinstance(string, list):
        if isinstance(pattern, list):
            result = [p in s  for p in pattern for s in string]
            output = any(result)
        else:
            result = [pattern in s  for s in [string1,string2]]
            output = any(result)
    else:
        if isinstance(pattern, list):
            result = [p in string  for p in pattern]
            output = any(result)
        else:
            result = [pattern in string]
            output = any(result)
            
    return result, output
            
        
        
check_multi_pattern_in_string([string1,string2],[pattern1,pattern2])            
