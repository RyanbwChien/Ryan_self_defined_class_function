# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 22:24:41 2025

@author: Ryan
"""


import numpy as np
from collections import deque, defaultdict
class Tensor:
    def __init__(self,value,inputs=[],grad_func=[],require_grad=True):
        self.value = np.array(value,dtype=float)
        self.inputs = inputs
        self.grad_func = grad_func
        self.grad = np.zeros_like(self.value)
        self.require_grad = require_grad
    def detach(self):
        return Tensor(self.value, requires_grad=False)    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad =  grad  
        grad_counts = defaultdict(int)    
        
        node_to_visit = [self]
        visited_nodes = set()
        while node_to_visit:
            # 這個 while node_to_visit: 會結束，發生在你已經把整棵計算圖一路走到所有「葉節點（inputs = []）」為止，
            current_node = node_to_visit.pop()
            if id(current_node) not in visited_nodes:
                visited_nodes.add(id(current_node))
            for i in current_node.inputs:
                grad_counts[id(i)] += 1
                node_to_visit.append(i)
        
        queue = deque([self])
        
        while queue:
            current_node = queue.popleft()
            
            if not current_node.inputs:
                continue
            
            for i,f in zip(current_node.inputs, current_node.grad_func):
                if not i.require_grad:
                    continue
                grad_input = f(current_node.grad) # 都是numpy array做運算
                
                if grad_input.shape != i.grad.shape:
                     while grad_input.ndim > i.grad.ndim:
                        grad_input = grad_input.sum(axis=0)
                     for ii, dim in enumerate(i.grad.shape):
                        if dim == 1 and grad_input.shape[i] != 1:
                            grad_input = grad_input.sum(axis=ii, keepdims=True)
                
                
                
                i.grad += grad_input
                grad_counts[id(i)] -= 1
                if grad_counts[id(i)] == 0:
                    queue.append(i)
        
        
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = self.value @ other.value
        
        return Tensor(out,inputs=[self,other],grad_func=[lambda g:g @ other.value.T,lambda g:self.value.T @ g])
    def __add__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        return Tensor(self.value + other.value, inputs=[self, other], grad_func=[lambda g: g, lambda g: g])
    def __radd__(self, other): 
        return self.__add__(other)
    def __mul__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        return Tensor(self.value * other.value, inputs=[self, other], grad_func=[lambda g: g * other.value, lambda g: g * self.value])
    def __rmul__(self, other): 
        return self.__mul__(other)
    
    
# 測試
X = Tensor(np.random.normal(0,1,(3,3)), require_grad=False)     
B = Tensor(np.random.normal(0,1,(3,3)))      

# 複雜一點的圖：X 被用了兩次 (Diamond Pattern)
Y = X @ X + 3 * X + B
Y.value
Y.backward()

print("執行成功，沒有遞迴")
print("X 的梯度:\n", B.grad)    
    
    
    
    
    