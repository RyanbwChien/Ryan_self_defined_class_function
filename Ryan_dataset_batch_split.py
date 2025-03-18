# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 01:36:27 2025

@author: user
"""

import pandas as pd
import numpy as np
from typing import Union


def batch_data(data:Union[list, 'pd.DataFrame', 'np.ndarray'], batch_size:int) -> list:
    output = [] # 先宣告了 output 變數，並將變數賦予一個空的列表作為初始值
    num_full_batch = len(data)//batch_size # data 長度除上 batch_size，取得商(表示被batch_size整除)其數量表示 有多少處組數據大小是batch_size            
    last_batch = len(data)%batch_size # data 長度除上 batch_size，取得餘數(表示不能被batch_size整除)其數量表示剩下一組大小是餘數部分
    
    # 將整除的批量，依序從i*batch_size~i*batch_size+batch_size
    for i in range(num_full_batch):        
        output.append(data[i*batch_size:i*batch_size+batch_size])
        
    # 將整除的批量，依序從i*batch_size~i*batch_size+batch_size 
    output.append(data[num_full_batch*batch_size:num_full_batch*batch_size+last_batch])
    return(output)

if __name__ == '__main__':
    data = np.random.normal(0,1,(100,20))
    print(len(batch_data(data, 99)))