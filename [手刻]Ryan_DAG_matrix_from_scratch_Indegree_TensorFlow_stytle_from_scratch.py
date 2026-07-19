# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:16:00 2026

@author: Ryan
"""


"""
Tensorflow 寫法必須在最外面 先定義一個全域變數 _Current_Tape，先給一個初始值 None 
*** 做上述動作原因 是為了讓之後每一次操作，都要能利用TAPE 回傳的那個實體去做紀錄，這個是要事先寫在每一次operator函數

_Current_Tape 是以後當 利用class GradientTape 定義 GradientTape實體時
class GradientTape 需定義實體屬性 history的LIST 來記錄 從頭到尾依照順序拓樸排序的OPS
"""

import numpy as np
# 直接使用python 內建numpy module 可以快速做矩陣運算
import tensorflow as tf


#%%
class Tensor:
    def __init__(self, value):
        """
        仿造原始 tf.Variable 運算節點物件 
        self.value
     
        """
        self.value = np.array(value)
        self._id = id(self) # 回傳這個物件的唯一id記憶體位置，之後要用它來對應梯度grad
    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)        
        return apply_ops("__add__", inputs=[self,other])
    def __matmul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)        
        return apply_ops("__matmul__", inputs=[self,other])
    
def Variable(Tensor):  
    # =============================================================================
    #     只有在「子類別有覆寫 __init__」時，才需要呼叫 super().__init__()
    #     沒有覆寫 __init__ → Python 會自動用父類別的 __init__
    #     
    #     def __init__(self):
    #         super().__init__()
    # =============================================================================
    def assign_sub(self, grad):
        pass



#%%
"""
每一個實際op_name對應的反向傳播梯度計算

"""

def apply_ops(op_name,inputs:list[Variable,Variable]):
    global _Current_Tape
    # 注意 inputs 與 output 裡面都是 Variable，所以要用.value取值
    match op_name:
        case "__add__":
            output = inputs[0].value + inputs[1].value
        case "__matmul__":
            output = inputs[0].value @ inputs[1].value    
    
    output = Variable(output) # 注意要再包成Variable 物件存值存id 可以使用 
    _Current_Tape._record(op_name,inputs,output)
    
    return output   


#%%
"""
利用裝飾器在最一開始定義函數時 就所有op_name 對應的微分函數，包到全域變數_Grad_Map{}字典
之後再 Current_tape ( GradientTape() 實體物件 就可以利用每次ops 紀錄 op_name 利用最外面包的全域變數
_Grad_Map{}字典回傳當次ops，反向傳播微分函數
"""
_Grad_Map = {}

def registor_derivative_ops(op_name):
    def decorator(derivative_func):
        _Grad_Map[op_name] = derivative_func
        return derivative_func
    return decorator
    

@registor_derivative_ops("__add__")
def _(inputs:list[Variable,Variable], grad)->tuple[np.array,np.array]:
    return grad, grad

@registor_derivative_ops("__matmul__")
def _(inputs:list[Variable,Variable], grad)->tuple[np.array,np.array]:
    return grad @ inputs[1].value.T, inputs[0].value.T @ grad   
    
#%%    

    
_Current_Tape = None

class GradientTape:
    def __enter__(self):
        self.ops_history:[]
        # 需定義實體屬性 history的LIST 來記錄 從頭到尾依照順序拓樸排序的OPS
        # 儲存 (op_name:當下該運算是哪種, inputs:list存放左/右2個parent, output:輸出的節點)
        global _Current_Tape
        if _Current_Tape is None:
            _Current_Tape = self
        return self
    
    def __exit__(self,*args):
        # 注意Python 預設Context manager 就是要帶入4個參數，所以寫*arg去接
        # with 結束後自動呼叫
        global _Current_Tape
        _Current_Tape is None
        
    def _record(self, op_name, inputs, output):
        """
        呼叫_record 利用實體屬性 history的LIST 來記錄 從頭到尾依照順序拓樸排序的OPS
        """
        self.ops_history.apped([op_name, inputs, output])
        
    def gradient(self, target:Variable, source:Variable):
        """
        仿造原始 Tape.gradient 算梯度
        with tf.GradientTape() as Tape:
            Tape.gradient(target,sources)
        target: 大部分是最後要反向回去的LOSS
        sources: 要計算那些權重梯度        
        """
        grads = {target.id: np.ones_like(target.value)} 
        #創造一個grads dict，計入所有算梯度的節點ID，對應的值記錄
        #一開始初始化最一開始的target node進入點的梯度
        
        for op_name,inputs,output in reversed(self.ops_history):
            # self.ops_history 從頭到尾依照順序拓樸排序的OPS，故不需向torch graph theory use kahn's algorithm
            # 外層 for op_name,inputs,output in reversed(self.ops_history):
            # 在做鏈鎖率/偏微分 將多個方向維到x_input.id的梯度加總起來
            
            if output.id in grads:
                out_grad = grads[output.id]
            derivative_func = _Grad_Map[op_name]
            in_grads:tuple = derivative_func(inputs, out_grad)
            
            # ***正向傳播時是「擴充（廣播）維度」，反向傳播求梯度時，反而變成用 sum 來「縮減（降維）」
            # 廣播 Broadcasting
            # 當兩個 Tensor（矩陣）要進行逐元素運算（例如 +、-、*、/）時，會依照以下規則進行 Broadcasting。
            # CASE1 兩矩陣SHAPE數量一樣
            # 會根據兩矩陣最後一維 看是不是一樣或是一個是1則會把它廣播成變成 另一矩陣的最後一維，如果不同直接報錯
            # 接下在比兩矩陣倒數2維也是一樣 ，看是不是一樣或是一個是1則把它廣播成另一個 Tensor 倒數2維的大小，如果不同直接報錯
            # 接下來依此類推把所有維度都做該動作，看是不是一樣或是一個是1則把它廣播成另一個 Tensor 在「目前比較的那個維度」的大小，如果不同直接報錯
            # CASE2 兩矩陣SHAPE數量不一樣
            # 會先將維度較少去看維度較多的矩陣，維度較少的矩陣會補齊前面維度 例如(4,) (5,3,4) 則(4,) 會先補成 (1,1,4) vs (5,3,4)
            # 往前補維度 新增的維度都是1維直到兩矩陣SHAPE一樣
            for x_input, g in zip(inputs, in_grads):
                # 1. 處理 CASE 2: 維度數量不同 (把前面多出來的維度 sum 掉)
                while g.ndim > x_input.value.ndim:
                    g = g.sum(axis=0)
                    
                # 2. 處理 CASE 1: 維度數量相同，但大小是 1 的被廣播了 (把被擴充的那個維度 sum 掉，並保持維度)
                for axis, (g_dim, x_dim) in enumerate(zip(g.shape, x_input.value.shape)):
                    if g_dim != x_dim and x_dim == 1:
                        g = g.sum(axis=axis, keepdims=True) # 保持形狀還是 1
                #這處理的是**「同一個變數，在單一一個運算內部，因為廣播（Broadcasting）被複製給了多個元素（Elements）」**。        
                
                grads[x_input.id] = grads.get(x_input.id, 0) + g 
                #最後就是外層DAG圖節點 與內層廣播 鏈鎖率/偏微分 將多個方向維到x_input.id的梯度加總起來
        return [ grads.get(s.id,0) for s in source] # 只回傳要更新梯度的權重，當次反向傳播梯度
        
