# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 19:14:38 2025

@author: Ryan
"""

import numpy as np

A = np.random.normal(0,1,(3,3))
A.T
class Variable:
    def __init__(self, value, parent=[], grad_fn=None):
        self.value = np.array(value,dtype=float)
        self.parent = parent
        self.grad_fn = grad_fn
        self.grad = np.zeros_like(self.value )
        
    def backward(self, grad=None):
        
        if grad is None:
            grad = np.ones_like(self.value )
        self.grad += grad
        if self.parent:
            for parent, gf in zip(self.parent,self.grad_fn):

                parent.backward(gf(grad))#gf(grad)當前微分的梯度 chain rule
            
    def __add__(self, other):
        if not isinstance(other,Variable):
            other = Variable(other)
        return Variable(value=self.value+other.value,
                        parent=[self, other],
                        grad_fn=[lambda g:g,lambda g:g])
    def __radd__(self, other):
        return self.__add__(other)
    
    
    def __matmul__(self, other):
        if not isinstance(other,Variable):
            other = Variable(other)
        return Variable(value=self.value @ other.value,
                        parent=[self, other],
                        grad_fn=[lambda g:g @ other.value.T,lambda g:self.value @ g])   
    
    def __mul__(self, other):
        if not isinstance(other,Variable):
            other = Variable(other)
        return Variable(value=self.value*other.value,
                        parent=[self, other],
                        grad_fn=[lambda g:g*other.value,lambda g:g*self.value])   
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
X = Variable(np.random.normal(0,1,(3,3)))     
B = Variable(np.random.normal(0,1,(3,3)))      

Y = X@X + 3*X +B
Y.backward()