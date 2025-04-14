# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:54:37 2025

@author: user
"""

class test_div_syntax_sugar:
    def __init__(self, path):
        self.path = path
        
    def __truediv__(self, suffix):
        return(f"{self.path} / suffix")
        
    
new = test_div_syntax_sugar("456")    
new / "789"

        