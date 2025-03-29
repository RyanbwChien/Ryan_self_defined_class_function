# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 23:21:11 2025

@author: user
"""

from typing import Union, Literal
import numpy as np

def list_find_closest_target(vector:Union[list,np.ndarray], target:int, condition:Literal['greater', 'less'] ) -> int:
    # 原始向量中由左到右找 大於50的元素位置
    vector_arr = np.array(vector)
    
    
    if condition == 'greater':
        greater_than_target_idx = np.where((vector_arr -target)>0)[0]
        
        # 原始向量中由左到右 大於target的元素值 是哪個位置值，
        # VECTOR大於target的元素值列表中 最接近target 哪個位置
        # 將第2個得出的位置給第1個，得到原始向量中大於target且最靠近target的位置值
        closest_idx = greater_than_target_idx[abs(vector_arr[greater_than_target_idx] - target).argmin()]
    else:
        less_than_target_idx = np.where((vector_arr -target)<0)[0]
        
        # 原始向量中由左到右 小於target的元素值 是哪個位置值，
        # VECTOR大於target的元素值列表中 最接近target 哪個位置
        # 將第2個得出的位置給第1個，得到原始向量中大於target且最靠近target的位置值
        closest_idx = less_than_target_idx[abs(vector_arr[less_than_target_idx] - target).argmin()]
        
        
    return closest_idx


if __name__ == 'main':
    vector = [1,6,9,88,101,323,69,5,8,3]
    target = 50
    print(list_find_closest_target(vector, target,'greater'))
    print(list_find_closest_target(vector, target,'less'))
