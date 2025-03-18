# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:52:47 2025

@author: user
"""

import os
path = r"C:\Users\user\Desktop\TEMP\news"


# 利用可變參數作為遞迴的累積變數
def Ryan_file_abspath_recursive(path, result = []):
    for content in os.listdir(path):
        abspath = os.path.join(path, content)
        if os.path.isdir(abspath):
            result.append(abspath)
            Ryan_file_abspath_recursive(abspath, result)
        else:
            result.append(abspath)
    return result


# 物件屬性紀錄結果，並在主遞迴函數返回
class Ryan_file_abspath_obj:
    def __init__(self):
        self.result = []
        
    # 確保 每次調用都會從零開始計算，不會累積舊結果：    
    def Ryan_file_abspath(self, path):
        self.result = []
        self._recursive(path)
        return self.result

    def _recursive(self, path):
        for content in os.listdir(path):
            abspath = os.path.join(path, content)
            if os.path.isdir(abspath):
                self.result.append(abspath)
                self._recursive(abspath)
            else:
                self.result.append(abspath)

        

if __name__ == '__main__'  : 
    Ryan_file_abspath_recursive(path)
    new = Ryan_file_abspath_obj()
    new.Ryan_file_abspath(path)



