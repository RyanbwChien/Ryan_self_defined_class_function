# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:38:34 2025

@author: user
"""

#索引遞迴

def print_element(x:iter, index = 0):
    if index < len(x):
        print(x[index])
        print_element(x,index+1)
       

print_element([1,3,5,69,5])