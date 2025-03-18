# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 22:36:11 2025

@author: user
"""

import numpy as np


class TimeSeries:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.X_timesplit = []
        self.Y_timesplit = []

    # 2. Python 切片 (array[start:end]) 的特性
    # Python 的 list[start:end] 切片機制允許 end 超過陣列長度，並且 不會報錯，而是返回 實際可用的數據：

    def timestep_split(self, past_period, furture_period):
        
        for i in range( len(self.X)-past_period-furture_period+1 ):
            self.X_timesplit.append(self.X[i:i+past_period])
            self.Y_timesplit.append(self.Y[i+past_period:i+past_period+furture_period])
        self.X_timesplit = np.array(self.X_timesplit)
        self.Y_timesplit = np.array(self.Y_timesplit)
        
        return(self.X_timesplit, self.Y_timesplit)
    
           
        


if __name__ == '__main__':
    X = np.random.normal(0,1,(100,5))
    Y = np.random.normal(0,1,(100,2))
    
    time_obj = TimeSeries(X,Y)
    X_new, Y_new = time_obj.timestep_split(5,3)
    len(Y_new)
    
    Y_new.shape
