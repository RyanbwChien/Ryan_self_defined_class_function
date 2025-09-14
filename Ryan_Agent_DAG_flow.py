# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:20:02 2025

@author: Ryan
"""
from typing import Literal, Callable
def start():
    print("this is start")

def funA():
    print("this is A")
    
def funB():
    print("this is B")
    
def funC():
    print("this is C")    
    
    
class node:
    def __init__(self):
        self.current_func = None
        self.next_func = None
        self.cur_result = None




class DAG_flow:
    def __init__(self,node_flow):
        self.node_dict = {}
        self.node_flow = []
        self.cur_result = None
    def add_node(self, name:str, func:Callable):
        self.node_dict[name] = func
    def add_edge(self)    
        


