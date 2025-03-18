# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:47:15 2025

@author: user
"""

import os

path = os.path.dirname(__file__)

def list_file_abspath(path):
    result = []
    for directory, folders, files in os.walk(path):
        for file in files:
            result.append(os.path.join(directory,file))
        for folder in folders:
            result.append(os.path.join(directory,folder))
    return result

if __name__ == '__main__':
    print(list_file_abspath(path)) 
        