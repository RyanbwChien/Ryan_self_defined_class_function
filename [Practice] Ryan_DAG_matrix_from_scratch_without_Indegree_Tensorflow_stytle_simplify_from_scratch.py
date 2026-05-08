# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:52:08 2026

@author: USER
"""
import numpy as np

GRAD_MAP = {}
# 模組自動註冊（超重要）
def register_grad(ops):
    def decorator(func):
        GRAD_MAP[ops] = func
        return func
    return decorator
# 目前這寫法不會改變原來函數 只是會去多執行GRAD_MAP[ops] = func
# func = func

@register_grad("Add")
def Add_grad(grad, inputs):   
    return grad , grad

@register_grad("Mul")
def Mul_grad(grad, inputs):   
    return grad*inputs[1] , grad*inputs[0]

class Variable:
    def __init__(self,value):
        self.value = np.array(value)
        self.id = id(self)
    def __add__(self,other):
        return apply_op("Add",self,other)
    
    
def apply_op(ops, left, right):
    if ops == "Add": 
        out = left.value + right.value
        
    out = Variable(out)
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_ops([ops,[left, right], out])
        
    return out





     
class GradientTape:
    def __enter__(self):
        global _CURRENT_TAPE
        _CURRENT_TAPE = self
        self.history = []
        return self
    def __exit__(self):
        global _CURRENT_TAPE
        _CURRENT_TAPE = None
    def record_ops(self, op_name, inputs, output):
        self.history.append([op_name, inputs, output])
    def gradient(self, target, weight):
        grads = {target.id:np.ones_like(target.values)}
        
        for op_name, inputs, output in reversed(self.history):
            GRAD_MAP[op_name]
        
        
        
        
        
        
        
        
        