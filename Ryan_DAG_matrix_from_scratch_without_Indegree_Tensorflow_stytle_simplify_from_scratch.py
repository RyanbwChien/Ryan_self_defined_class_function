# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 01:10:47 2026

@author: Ryan
"""

import numpy as np
from collections import deque, defaultdict
"""
只有在「子類別有覆寫 __init__」時，才需要呼叫 super().__init__()
沒有覆寫 __init__ → Python 會自動用父類別的 __init__
"""

# 1. 統一的梯度註冊中心
_GRAD_MAP = {}

class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            var.value -= self.lr * grad
class SGD(Optimizer):
    # def __init__(self,lr):
    #     super(SGD,self).__init__(lr)
    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            var.assign_sub(self.lr * grad)




def register_grad(op_name):
    def decorator(f):
        _GRAD_MAP[op_name] = f
        return f
    return decorator

class Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)
        self.id = id(self)

    # --- 運算子重載：讓 TF 像 PyTorch 一樣簡潔的關鍵 ---
    def __add__(self, other): return apply_op("Add", self, other)
    def __matmul__(self, other): return apply_op("MatMul", self, other)
    def __mul__(self, other): return apply_op("Mul", self, other)
    def __sub__(self, other): return apply_op("Sub", self, other)

class Variable(Tensor):
    def assign_sub(self, delta):
        self.value -= delta
# 全域 Tape 指針
_CURRENT_TAPE = None

def apply_op(op_name, *inputs):
    # 執行數值運算
    if op_name == "Add": res = inputs[0].value + inputs[1].value
    elif op_name == "MatMul": res = inputs[0].value @ inputs[1].value
    elif op_name == "Mul": res = inputs[0].value * inputs[1].value
    elif op_name == "Sub": res = inputs[0].value - inputs[1].value
    
    out = Tensor(res)
    # 如果 Tape 開啟中，錄製下來
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record(op_name, inputs, out)
    return out

@register_grad("MatMul")
def _(grad, inputs):
    return grad @ inputs[1].value.T, inputs[0].value.T @ grad

@register_grad("Add")
def _(grad, inputs):
    return grad, grad

class GradientTape:
    def __enter__(self):
        global _CURRENT_TAPE; _CURRENT_TAPE = self
        self.history = [] # 儲存 (op_name, inputs, output)
        return self
    
    def __exit__(self, *args):
        global _CURRENT_TAPE; _CURRENT_TAPE = None

    def record(self, op_name, inputs, output):
        self.history.append((op_name, inputs, output))

    def gradient(self, target, sources):
        grads = {target.id: np.ones_like(target.value)}
        # 從後往前遍歷 Tape
        for op_name, inputs, output in reversed(self.history):
            if output.id in grads:
                out_grad = grads[output.id]
                in_grads = _GRAD_MAP[op_name](out_grad, inputs) #求變數偏微梯度
                if not isinstance(in_grads, tuple): in_grads = (in_grads,)
                
                for x, g in zip(inputs, in_grads):
                    # 廣播處理 (簡化版)
                    while g.ndim > x.value.ndim: g = g.sum(axis=0)
                    grads[x.id] = grads.get(x.id, 0) + g
        return [grads.get(s.id, 0) for s in sources]

# --- 測試：現在看起來跟 PyTorch 幾乎一樣了 ---
X = Tensor(np.random.randn(3, 3))
W = Tensor(np.random.randn(3, 3))
B = Tensor(np.random.randn(1, 3))

with GradientTape() as tape:
    # 這裡的寫法現在非常簡潔！
    Y = X @ W + B
    loss = Y # 假設 Loss 就是 Y 的總和之類的

grads = tape.gradient(loss, [W, B])
print("W gradient shape:", grads[0].shape)

# =============================================================================
# 
# 答案是：在「Tape（錄記）」模式下，不需要手動計算入度（In-degree），因為 Tape 的順序本身就是一種「拓撲排序」。
# 以下是深度解析：
# 1. 為什麼 PyTorch 版（DAG 模式）需要算入度？
# 在你的 PyTorch 實作中，Tensor 對象之間透過指針連結。當你呼叫 Y.backward() 時：
# 程序只知道 
# Y
# Y
#  是終點。
# 它不知道路徑上有哪些節點被「重複使用」了（例如 Diamond Pattern）。
# 如果不算入度，直接用 BFS/DFS 遍歷，你會在某個節點還沒拿到「所有來自上游的梯度累積」時，就提前把它後傳了，導致計算錯誤。
# 結論： 處理一個靜態結構的圖，必須先透過入度分析來確定執行順序。
# 2. 為什麼 TensorFlow 版（Tape 模式）不需要算入度？
# TensorFlow 的 GradientTape 記錄的是執行的時間線（Timeline）。
# 這是一個關鍵邏輯：在 Python 程式執行的過程中，你絕對不可能在定義 A 之前就先用 A 去計算 B。
# 執行順序 = 拓撲順序：
# 當你寫 Y = X * W + B 時，底層錄製的順序一定是：
# MatMul(X, W) -> Temp_Res
# Add(Temp_Res, B) -> Y
# 這個順序（Tape 的 list）本身就已經是一個完美的拓撲排序（Topological Sort）。
# 倒序遍歷 = 反向拓撲順序：
# 當我們執行 reversed(self.history) 時：
# 我們先處理 Add：算出 Temp_Res 的梯度。
# 再處理 MatMul：這時 Temp_Res 的梯度已經被上一步算好了。
# 即使是 Diamond Pattern (
# Y
# =
# X
# ⋅
# X
# +
# X
# Y=X⋅X+X
# )：
# Tape 錄製：op1: X*X -> A, op2: A+X -> Y
# 倒序遍歷：
# 處理 op2 (Add)：X 拿到第一份梯度，
# A
# A
#  拿到一份梯度。
# 處理 op1 (Mul)：X 拿到第二份梯度（累加）。
# 結果： 雖然 
# X
# X
#  被用了兩次，但因為我們是按 Tape 倒著走，我們保證了 
# X
# X
#  所有的「下游」節點（即 
# A
# A
#  和 
# Y
# Y
# ）都會在 
# X
# X
#  本身被處理之前先處理完。
# 3. 兩者設計哲學的對比
# 特性	PyTorch 模式 (DAG-based)	TensorFlow 模式 (Tape-based)
# 圖的本質	物件之間的引用網路	操作的序列（歷史記錄）
# 排序機制	執行時動態計算入度/拓撲排序	依賴執行時的時間順序
# 優點	結構清楚，不依賴全局變數	不需要預計算圖結構，性能開銷低
# 你的代碼邏輯	while queue + grad_counts	for op in reversed(history)
# 4. 你的 TF 原本寫法為什麼要算入度？
# 在你原本提供的 TensorFlow 範例中，你混合了兩種做法：
# 你用了 Gradient_Tape 錄製 ops。
# 但在 gradient 函數裡，你又跑去遍歷 ops 來計算 grad_counts。
# 其實在 Tape 模式下，那段計算入度的代碼是可以刪除的。 只要你保證：
# 錄製時是按計算順序錄入 list。
# 微分時是 reversed 遍歷這個 list。
# 梯度累加時使用 grads[id] += new_grad 而不是直接覆蓋。
# 這樣即使不計算入度，結果也會是完全正確的！
# 總結
# 你之所以覺得 PyTorch 版簡潔，是因為你把「如何微分」寫在了 Tensor 內部；
# 你之所以覺得 TensorFlow 版繁瑣，是因為你手動實作了「註冊機制」與「入度計算」。
# =============================================================================
