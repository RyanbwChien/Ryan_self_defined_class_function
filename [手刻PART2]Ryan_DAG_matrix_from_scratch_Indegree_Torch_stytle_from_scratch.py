# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:07:07 2026

@author: USER
"""

class Tensor:
    pass

class Parameter(Tensor):
    def __init__(self, value):
        super(self,Tensor).__init__(value)
        self.require_grad = True        

class Ryan_torch_module:
    def __init__(self):
        self._parameters = {} # 因為 Parameter 是 葉節點（leaf Tensor）。
        self._modules = {}
        
    def __setattr__(self, name, value):
        # 覆寫 __setattr__ 後，取代了原本流程 → 自己加入新邏輯 → 手動呼叫父類別最底層實作
        # 那才是這樣才是在既有方法下新增功能
        if isinstance(value, Parameter):
            self._parameters[name] = value
            
        if isinstance(value, Ryan_torch_module):
            self._modules[name] = value
        
        object.__setattr__(self, name, value)
        # 透過類別直接呼叫實體方法，因為不是由實體來呼叫實體方法顧不會自動綁定self
        # descriptor 在這裡主要就是 Python 區分「存取屬性」和「呼叫方法」時背後的機制。
        
            
    # parameters() 是走「Module 樹」，它本身不會遇到 DAG 的重複節點問題。 每個 Module 屬性只會被走一次。
    def parameters(self):
        for value in self._parameters.values(): 
            yield value 
            
        for module in self._modules.values(): 
            yield from module.parameters() 
        
                
# loss.backward() Tensor DAG 需要避免重複計算
def parameters(loss): 
    params = [] 
    visited = set() 
    def dfs(node): 
        if node.id in visited: 
            return visited.add(node.id) 
        if isinstance(node, Parameter): 
            params.append(node) 
            for p in node.inputs: 
                dfs(p) 
    dfs(loss) 
    return params