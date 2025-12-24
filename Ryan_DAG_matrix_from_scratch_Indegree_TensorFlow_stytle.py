# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:52:40 2025


@author: USER
"""

# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, defaultdict

# =============================================================================
# 1. 全域註冊表 (Registry)
# 這裡存放所有運算的微分邏輯，與 Tensor 物件分離
# =============================================================================
_GRADIENT_REGISTRY = {}

def register_gradient(op_name):
    """裝飾器：用來註冊微分函數"""
    def wrapper(func):
        _GRADIENT_REGISTRY[op_name] = func
        return func
    return wrapper

# =============================================================================
# 2. Tensor (啞巴物件)
# 它只負責存 Data，不包含微分邏輯
# =============================================================================
class Tensor:
    def __init__(self, value, name=None):
        self.value = np.array(value, dtype=float)
        self.name = name
        self.id = id(self) # 用記憶體位址當作唯一 ID
    
    def __repr__(self):
        return f"Tensor(shape={self.value.shape}, id={self.id})"

# =============================================================================
# 3. GradientTape (錄影機 + 執行引擎)
# 這是模仿 TensorFlow 的核心
# =============================================================================
class GradientTape:
    def __init__(self):
        self.ops = [] # 錄影帶：依序記錄發生的操作
        self.active = False
        
    def __enter__(self):
        self.active = True
        # 將自己設為全域 tape (簡化版做法)
        global _CURRENT_TAPE
        _CURRENT_TAPE = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        global _CURRENT_TAPE
        _CURRENT_TAPE = None

    def record_op(self, op_name, inputs, output):
        """錄影：記錄一行操作"""
        if self.active:
            self.ops.append({
                "op_name": op_name,
                "inputs": inputs,   # [Tensor A, Tensor B]
                "output": output    # Tensor C
            })

    def gradient(self, target, sources):
        """
        核心反向傳播引擎 (Topological Sort / Kahn's Algorithm)
        target: loss (Tensor)
        sources: list of params to update (list of Tensors)
        """
        # 1. 建立反向查找表 (Producer Map)
        # 我們需要知道：哪個 Tensor 是由哪個 Op 產生的？
        # key: Tensor ID, value: Op Entry
        producer_map = {}
        for op in self.ops:
            producer_map[op['output'].id] = op
            
        # 2. 計算入度 (Indegree / Dependency Counts)
        # 統計每個 Tensor 被當作 input 使用了幾次
        grad_counts = defaultdict(int)
        for op in self.ops:
            for x in op['inputs']:
                grad_counts[x.id] += 1 # producer_map 是計output但grad_counts是計INPUT
                
        # 3. 初始化梯度字典 (類似 TF 的不沾鍋設計，不汙染 Tensor 本身)
        grads = defaultdict(lambda: 0)
        grads[target.id] = np.ones_like(target.value)
        
        # 4. 準備 Queue (Kahn's Algorithm)
        # 從 Target (Loss) 開始
        queue = deque([target])
        
        # 5. 開始反向拓撲遍歷
        while queue:
            # Pop 出當前節點 (這代表它的梯度已經完全累加完畢)
            current_tensor = queue.popleft()
            current_grad = grads[current_tensor.id]
            
            # 找到是誰產生了這個 Tensor (Producer)
            if current_tensor.id not in producer_map:
                continue # 可能是最源頭的 Input，沒有 Producer
                
            op_entry = producer_map[current_tensor.id]
            op_name = op_entry['op_name']
            inputs = op_entry['inputs']
            
            # --- 關鍵：去註冊表查閱微分方法 ---
            if op_name not in _GRADIENT_REGISTRY:
                raise ValueError(f"Op {op_name} 沒有註冊微分函數！")
            
            grad_func = _GRADIENT_REGISTRY[op_name]
            
            # 計算梯度 (傳入上游梯度 和 原始輸入)
            input_grads = grad_func(current_grad, inputs)
            
            # 確保 input_grads 是 list (為了處理單一輸入的情況)
            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]
            
            # --- 累加梯度並處理 Queue ---
            for x, g in zip(inputs, input_grads):
                # 廣播修正 (Broadcasting Fix)
                if g.shape != x.value.shape:
                    while g.ndim > x.value.ndim:
                        g = g.sum(axis=0)
                    for i, dim in enumerate(x.value.shape):
                        if dim == 1 and g.shape[i] != 1:
                            g = g.sum(axis=i, keepdims=True)
                
                # 累加梯度 (Gradient Accumulation)
                # 注意：這裡我們存在 grads 字典裡，不寫入 x.grad
                grads[x.id] += g #只會把梯度加在上一層元素，
                                 #對哪一變數去微分就是去計算函數的梯度，然後要給那個變數去做梯度下降
                
                # 減少依賴計數
                grad_counts[x.id] -= 1
                
                # 只有當 x 的所有消費者都回傳梯度後，x 才能進入 Queue
                if grad_counts[x.id] == 0:
                    queue.append(x)
                    
        # 6. 最後只回傳使用者要的那些參數的梯度
        return [grads[s.id] for s in sources]

# 全域變數，用來存當前正在錄影的 Tape
_CURRENT_TAPE = None

# =============================================================================
# 4. 定義運算函數 (Operations)
# 這些函數負責：1. 算數值 2. 叫 Tape 錄影
# =============================================================================

def tf_matmul(a, b):
    # 1. Eager Execution (馬上算數值)
    val = a.value @ b.value
    out = Tensor(val)
    
    # 2. 錄影 (如果有 Tape 開著)
    if _CURRENT_TAPE: # 只有實際建立實體的當下GradientTape() 才會設成自己
        _CURRENT_TAPE.record_op("MatMul", [a, b], out)
    return out

def tf_add(a, b):
    # 支援廣播加法
    val = a.value + b.value
    out = Tensor(val)
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Add", [a, b], out)
    return out

def tf_mul(a, b):
    # Element-wise 乘法
    val = a.value * b.value
    out = Tensor(val)
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Mul", [a, b], out)
    return out

# =============================================================================
# 5. 註冊微分邏輯 (Gradient Implementation)
# 這就是 "分離式" 設計，邏輯寫在這裡，而不是 Tensor class 裡
# =============================================================================

@register_gradient("MatMul")
def grad_matmul(grad, inputs):
    A, B = inputs
    # dL/dA = grad @ B.T
    grad_A = grad @ B.value.T
    # dL/dB = A.T @ grad
    grad_B = A.value.T @ grad
    return grad_A, grad_B

@register_gradient("Add")
def grad_add(grad, inputs):
    A, B = inputs
    # 加法的梯度就是 1 * grad，直接複製給兩個輸入
    return grad, grad

@register_gradient("Mul")
def grad_mul(grad, inputs):
    A, B = inputs
    # dL/dA = grad * B
    grad_A = grad * B.value
    # dL/dB = grad * A
    grad_B = grad * A.value
    return grad_A, grad_B

# =============================================================================
# 6. 測試執行
# =============================================================================

# 1. 準備數據
X = Tensor(np.random.normal(0,1,(3,3)), name="X")
W = Tensor(np.random.normal(0,1,(3,3)), name="W")
B = Tensor(np.random.normal(0,1,(3,3)), name="B")

print("X shape:", X.value.shape)

# 2. 開始錄影 (Forward Pass)

"""
1. 在建立實體的當下GradientTape()
2. 透過__enter__ 將外面的全域變數 global _CURRENT_TAPE 把設成自己 CURRENT_TAPE = self
3. 當在呼叫自訂運算函數時，裡面會執行_CURRENT_TAPE.record_op("Add", [a, b], out)
     就是將global _CURRENT_TAPE(GradientTape()) 呼叫 record_op 方法把 自己的屬性 
     self.ops = [] # 錄影帶：依序記錄發生的操作
4. grads = tape.gradient(Loss, [X, W, B])    
"""

with GradientTape() as tape:
    # 模擬: Y = (X @ W) + B
    Z = tf_matmul(X, W)
    Y = tf_add(Z, B)
    
    # 再加一點複雜度: Loss = Sum(Y * Y)
    # 這裡我們手動模擬一個簡單的 Loss 運算，方便演示
    Loss = tf_mul(Y, Y) 
    
    # 注意：這裡的運算過程，Tape 都在偷偷記錄
    # Op1: MatMul (X, W) -> Z
    # Op2: Add (Z, B) -> Y
    # Op3: Mul (Y, Y) -> Loss

print("\n--- Forward 完成，Tape 記錄的長度: ", len(tape.ops), "---")

# 3. 反向傳播 (Backward Pass)
# 我們想要算 X 和 W 的梯度
grads = tape.gradient(Loss, [X, W, B])

grad_X, grad_W, grad_B = grads

print("\n--- Backward 完成 ---")
print("Gradient of X:\n", grad_X)
print("Gradient of W:\n", grad_W)
print("Gradient of B:\n", grad_B)

# 驗證形狀是否正確
assert grad_X.shape == X.value.shape
print("\n形狀檢查通過！")