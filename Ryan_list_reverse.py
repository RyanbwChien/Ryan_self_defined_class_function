# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:13:24 2025

@author: Ryan
"""

l = list(range(1,10))
# data = l
def reverse(data, result= None):
    new = []
    cnt = len(data)
    while cnt>0:
        cur = data[0]
        prev = data[1:]

        new = [cur ] + new
        data = prev
        cnt -= 1
    return new
    
    


a = reverse(l)
print(a)
