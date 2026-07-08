# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:29:03 2026

@author: USER
"""

import numpy as np
from collections import defaultdict, deque

# ==========================================
# 1. 核心計算節點：Function (Autograd Node)
# ==========================================
class Function:
    """
    所有的數學運算都要繼承這個基底類別。
    這對應 PyTorch 原始碼中的 torch.autograd.Function
    """
    def __init__(self, *inputs):
        # 記錄這個操作的輸入 Tensors，為了 backward 時知道要傳給誰
        self.inputs = inputs 

    @classmethod
    def apply(cls, *inputs):
        """
        這個方法負責兩件事：
        1. 算出 forward 的數值結果。
        2. 動態建立計算圖 (Dynamic Computation Graph)。
        """
        # 1. 只要有任何一個 input 需要梯度，產生的 output 就需要梯度
        requires_grad = any(isinstance(inp, Tensor) and inp.requires_grad for inp in inputs)
        
        # 2. 提取 numpy 數值準備計算
        input_values = [inp.data if isinstance(inp, Tensor) else inp for inp in inputs]
        
        # 3. 實例化這個 Function (這就是我們計算圖上的 Node)
        ctx = cls(*inputs)
        
        # 4. 執行正向傳播 (呼叫子類別定義的 forward)
        output_data = ctx.forward(*input_values)
        
        # 5. 封裝成新的 Tensor 返回
        if requires_grad:
            # 如果需要梯度，把這顆 Node 綁定到 output tensor 的 grad_fn 上
            return Tensor(output_data, requires_grad=True, grad_fn=ctx)
        else:
            return Tensor(output_data, requires_grad=False)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        """計算並回傳相對於 inputs 的梯度，必須由子類別實作"""
        raise NotImplementedError

# --- 實作具體的數學操作 ---

class Add(Function):
    def forward(self, a, b):
        return a + b
    def backward(self, grad_output):
        # f = a + b, df/da = 1, df/db = 1
        return grad_output, grad_output

class Mul(Function):
    def forward(self, a, b):
        return a * b
    def backward(self, grad_output):
        a_data = self.inputs[0].data if isinstance(self.inputs[0], Tensor) else self.inputs[0]
        b_data = self.inputs[1].data if isinstance(self.inputs[1], Tensor) else self.inputs[1]
        # f = a * b, df/da = b, df/db = a
        return grad_output * b_data, grad_output * a_data

class MatMul(Function):
    def forward(self, a, b):
        return a @ b
    def backward(self, grad_output):
        a_data = self.inputs[0].data
        b_data = self.inputs[1].data
        # f = A @ B, df/dA = grad @ B^T, df/dB = A^T @ grad
        return grad_output @ b_data.T, a_data.T @ grad_output


# ==========================================
# 2. 資料載體：Tensor
# ==========================================
class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        
        # PyTorch 的靈魂屬性：指向創造它的 Function (計算圖上的父節點)
        # 如果是使用者自己創造的變數(Leaf Node)，grad_fn 就是 None
        self.grad_fn = grad_fn 
        
        self.grad = None # 延遲初始化，有需要 backward 才配置記憶體

    # --- 讓 Magic Methods 呼叫 Function.apply 來建立計算圖 ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul.apply(self, other)

    # --- Autograd 引擎 ---
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("不需要梯度的 Tensor 無法執行 backward")

        if grad is None:
            grad = np.ones_like(self.data)

        # 1. 建立入度表 (In-degree) 以處理 Diamond 拓撲 (Kahn's Algorithm)
        tensor_in_degree = defaultdict(int)
        visited = set()
        nodes_to_visit = [self]

        while nodes_to_visit:
            curr = nodes_to_visit.pop()
            if id(curr) not in visited:
                visited.add(id(curr))
                # 如果這個 Tensor 是由某個 Function 創造出來的
                if curr.grad_fn is not None: 
                    for inp in curr.grad_fn.inputs:
                        if isinstance(inp, Tensor) and inp.requires_grad:
                            tensor_in_degree[id(inp)] += 1
                            nodes_to_visit.append(inp)

        # 2. 開始反向傳播
        grad_dict = defaultdict(lambda: 0)
        grad_dict[id(self)] = grad
        queue = deque([self])

        while queue:
            curr_tensor = queue.popleft()
            current_grad = grad_dict[id(curr_tensor)]

            # 如果是葉子節點 (使用者創建的參數，如 Weight, Bias)
            if curr_tensor.grad_fn is None:
                if curr_tensor.grad is None:
                    curr_tensor.grad = np.zeros_like(curr_tensor.data)
                curr_tensor.grad += current_grad
                continue

            # 如果它有 grad_fn，代表它是計算出來的結果，要把梯度透過 Function 往回推
            func = curr_tensor.grad_fn
            
            # 呼叫 Function 的 backward 取得對 inputs 的偏微分
            grads_wrt_inputs = func.backward(current_grad)
            
            # 確保返回的是 tuple (即使只有一個 input)
            if not isinstance(grads_wrt_inputs, tuple):
                grads_wrt_inputs = (grads_wrt_inputs,)

            # 將梯度派發給輸入的 Tensors
            for inp, g_inp in zip(func.inputs, grads_wrt_inputs):
                if isinstance(inp, Tensor) and inp.requires_grad:
                    
                    # 廣播機制 (Broadcasting) 處理 (例如 3 * X -> 變成 scalar 跟矩陣相加的梯度處理)
                    if g_inp.shape != inp.data.shape:
                        while g_inp.ndim > inp.data.ndim:
                            g_inp = g_inp.sum(axis=0)
                        for axis, dim in enumerate(inp.data.shape):
                            if dim == 1:
                                g_inp = g_inp.sum(axis=axis, keepdims=True)

                    # 累積梯度
                    grad_dict[id(inp)] += g_inp
                    
                    # 入度減 1，為 0 時放入 queue (表示它上面所有的分支都傳完梯度了)
                    tensor_in_degree[id(inp)] -= 1
                    if tensor_in_degree[id(inp)] == 0:
                        queue.append(inp)


# ==========================================
# 3. 測試：與你先前的範例一模一樣
# ==========================================
if __name__ == "__main__":
    print("--- 建立計算圖 ---")
    X = Tensor(np.random.normal(0, 1, (3, 3)), requires_grad=False)
    B = Tensor(np.random.normal(0, 1, (3, 3)), requires_grad=True)

    # 這裡的運算會動態呼叫 Add.apply, Mul.apply, MatMul.apply，並串起 grad_fn 指標
    Y = X @ X + 3 * X + B  # (Diamond Pattern)

    print(f"Y 需不需要梯度? {Y.requires_grad}")
    print(f"Y 的 grad_fn (創造者): {type(Y.grad_fn).__name__}") # 應該是 Add

    print("\n--- 執行 Backward ---")
    Y.backward()

    print("✅ 執行成功，完全無遞迴 (Topological Sort queue based)！")
    print("\nB 的梯度 (應全為 1):\n", B.grad)