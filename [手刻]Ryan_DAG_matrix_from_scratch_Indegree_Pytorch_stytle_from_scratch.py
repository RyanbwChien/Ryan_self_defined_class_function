# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:49:02 2026

@author: Ryan
"""

# =============================================================================
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import torch.nn.functional as F
# 
# class TORCH_NN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.L1 = nn.Linear(10, 5)
#         
#         self.L2 = nn.Linear(5, 2)
#     def forward(self, x):
#         x = self.L1(x)
#         x = F.gelu(x)
#         x = self.L2(x)
#         return x
# 
# model = TORCH_NN()    
# model.parameters()
# optimizer = Adam(model.parameters())
# =============================================================================

import numpy as np
from collections import deque, defaultdict

class Tensor:
    def __init__(self, value, inputs=[], grad_fn=None):
        self.value = np.array(value)
        self.id = id(self)
        self.grad = np.zeros_like(self.value) #初始化
        self.inputs = inputs
        self.grad_fn = grad_fn #放置目前TENSOR對INPUTS左右個別微分的梯度函數
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Tensor(value=self.value + other.value,
                      inputs=[self,other],
                      grad_fn = lambda outer_grad: [outer_grad,outer_grad]                      
                      )
    
    def __matmul__(self, other):
        # C = A@B 此時  self = A other = B self 不是 C
        # matmul 的「最後兩個維度」必須符合矩陣乘法 (m,n) @ (n,p)；最後兩維不會靠 broadcasting 解決。
        # 但最後兩維之前的 batch dimensions，可以做 broadcasting。
        
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(value=self.value @ other.value,
                      inputs=[self,other],
                      grad_fn = lambda outer_grad: [outer_grad @ other.value.T, 
                                                    self.value.T @ outer_grad]
                      )
    
    def __mul__(self, other):
        # C = A@B 此時  self = A other = B self 不是 C
        
        # 純 Python scalar（例如 3）通常不是計算圖中由前一個運算產生的 Tensor 節點，
        # 而是被當成某個運算的常數輸入；因此反向即使可以計算對它的偏導，也沒有更前面的計算圖可以繼續傳。
        
        # 就算讓純係數算梯度也不會對其他要算梯度的有任何影響，只是如果不需要算，就在系統中部用讓他算減少計算成本
        
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(value=self.value * other.value, # 這邊使用numpy 內建的__mul__方法 內建會幫你向前時做 broadcasting 
                      inputs=[self,other],
                      # 反向傳播，這邊使用numpy 內建的__mul__方法 內建會幫你向前時做 broadcasting，後續再做梯度更新的時候
                      grad_fn = lambda outer_grad: [outer_grad * other.value, 
                                                    self.value * outer_grad]
                      )
    

    
    
    def backward(self):
        # 利用 kahn's algorithm DAG
        
        in_degree = defaultdict(int)
        queue = deque()
        queue.append(self)
        while queue:
            current_node = queue.popleft()
            for p in current_node.inputs:
                in_degree[p.id] += 1
                queue.append(p)
        
        self.grad += np.ones_like(self.value)
        ready_node = deque()
        ready_node.append(self)
        
        while ready_node:
            current_bp_node = ready_node.popleft()
            if current_bp_node.inputs:
                out_grad = current_bp_node.grad
                in_grad = current_bp_node.grad_fn(out_grad)
                
                for i,g in zip(current_bp_node.inputs, in_grad):
                                      
                    
                    # case1 維度不一樣
                    while g.ndim > i.value.ndim:
                        g=g.sum(axis=0)
                    # case2
                    if g.ndim == i.value.ndim:  
                        for dim in range(g.ndim):
                            if i.value.shape[dim] ==1 and g.shape[dim] != 1:
                                g=g.sum(axis=dim, keepdims=True)

        # =============================================================================
        #                     ***正向傳播時是「擴充（廣播）維度」，反向傳播求梯度時，反而變成用 sum 來「縮減（降維）」 
        #                     因為向前時 INPUT 「同一個變數，在單一一個運算內部，因為廣播（Broadcasting）在一次矩陣運算中 同一個變數參與多個運算」 
        #                     除了外層DAG圖節點鏈鎖率/偏微分 將多個方向維到x_input.id的梯度加總起來 內層矩陣廣播 
        #                     其實也是同一個INPUT 透過多個路徑去做運算 所以用sum(axis=...) 
        #                     就是在做這個「多路徑梯度合併」 鏈鎖率/偏微分 FOR 內層矩陣廣播 
        #                     將多個方向維到x_input.id的梯度加總起來
        # =============================================================================
        
                    in_degree[id(i)] -= 1
                    i.grad += g
                    
                    if in_degree[id(i)] ==0:
                        ready_node.append(i)

            
A = np.random.normal(0,1,(2,2))        
A.sum(0).shape
x = Tensor([[10,20],[20,30]])
y = Tensor([[40,30],[60,70]])
b = Tensor([7,8])
b = Tensor([[7],[8]])
res = x+y+x@y+x@x+y@y+y@y+b
res.backward()

y.grad
# =============================================================================
# queue = deque("1")    
# while queue:
#     print("A")
#     queue.popleft()
# =============================================================================


"""
TO-DO
那為什麼 parameters() 還用 generator？

因為兩個層次不同。

Module.parameters()

負責：

找東西

所以：

yield parameter

很合理。
"""
