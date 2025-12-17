# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 19:14:38 2025

@author: Ryan
"""

import numpy as np

class Variable:
    def __init__(self, value, parent=None, grad_fn=None):
        self.value = np.array(value, dtype=float)
        self.parent = parent or []
        self.grad_fn = grad_fn or []
        self.grad = np.zeros_like(self.value)
        
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)

        # FIX: Check if shapes match. If not, unbroadcast (sum) the gradient.
        if grad.shape != self.grad.shape:
            # Reduce extra dimensions (e.g. matrix gradient -> scalar variable)
            while grad.ndim > self.grad.ndim:
                grad = grad.sum(axis=0)
            # Reduce broadcasted dimensions (e.g. sum over axis where dim is 1)
            for i, dim in enumerate(self.grad.shape):
                if dim == 1 and grad.shape[i] != 1:
                    grad = grad.sum(axis=i, keepdims=True)

        self.grad += grad
        
        if self.parent:
            for parent, gf in zip(self.parent, self.grad_fn):
                parent.backward(gf(grad))
            
    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(value=self.value + other.value,
                        parent=[self, other],
                        grad_fn=[lambda g: g, lambda g: g])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __matmul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(value=self.value @ other.value,
                        parent=[self, other],
                        grad_fn=[lambda g: g @ other.value.T, lambda g: self.value.T @ g])   
    
    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(value=self.value * other.value,
                        parent=[self, other],
                        grad_fn=[lambda g: g * np.array(other.value, dtype=float),
                                 lambda g: g * np.array(self.value, dtype=float)])   
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
# --- Testing ---
X = Variable(np.random.normal(0, 1, (3, 3)))     
B = Variable(np.random.normal(0, 1, (3, 1)))      

# This caused the error before because '3' is a scalar variable
Y = X@X + 3*X + B 

Y.backward()

print("Execution Successful")
print("Gradient of X shape:", X.grad)
print("Gradient of B shape:", B.grad)
