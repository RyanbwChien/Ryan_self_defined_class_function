# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, defaultdict

class Variable:
    def __init__(self, value, parent=None, grad_fn=None):
        self.value = np.array(value, dtype=float)
        self.parent = parent or []
        self.grad_fn = grad_fn or []
        self.grad = np.zeros_like(self.value)
        
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad = grad

        # =======================================================
        # 步驟 1: 建圖與計算入度 (Indegree)
        # 這裡的 "入度" 指的是在反向傳播圖中，有多少個節點依賴我 (即前向傳播中有多少人用了我)
        # =======================================================
        
        # 記錄每個變數被使用的次數 (Dependency Count)
        grad_counts = defaultdict(int)
        
        # 用一個簡單的堆疊遍歷圖，計算每個 parent 被引用的次數
        # 這一步就像是 TF/PyTorch 在 "編譯/分析" 計算圖
        nodes_to_visit = [self]
        visited_nodes = set()
        
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if id(node) in visited_nodes:
                continue
            visited_nodes.add(id(node))
            
            for p in node.parent:
                grad_counts[id(p)] += 1
                nodes_to_visit.append(p)

        # =======================================================
        # 步驟 2: 基於佇列 (Queue) 的拓撲排序執行 (Kahn's Algorithm)
        # =======================================================
        
        # 佇列中只放 "已經準備好可以計算微分" 的節點
        # 也就是：所有依賴它的下游節點都已經把梯度傳給它了 (grad_counts == 0)
        queue = deque([self])
        
        while queue:
            node = queue.popleft() # 取出一個節點
            
            # 如果這個節點沒有 parent (代表它是葉子節點，如輸入層)，就跳過
            if not node.parent:
                continue

            # 執行 Chain Rule
            for p, gf in zip(node.parent, node.grad_fn):
                # 1. 計算傳遞給 parent 的梯度
                d_p = gf(node.grad)
                
                # --- 加入廣播修正 (Broadcasting Fix) ---
                if d_p.shape != p.grad.shape:
                     while d_p.ndim > p.grad.ndim:
                        d_p = d_p.sum(axis=0)
                     for i, dim in enumerate(p.grad.shape):
                        if dim == 1 and d_p.shape[i] != 1:
                            d_p = d_p.sum(axis=i, keepdims=True)
                # -------------------------------------

                # 2. 累加梯度到 parent
                p.grad += d_p
                
                # 3. 關鍵邏輯：減少 parent 的等待計數
                grad_counts[id(p)] -= 1
                
                # 4. 如果 parent 的所有下游都算完了 (計數歸零)，
                #    代表 parent 的梯度已經累加完畢，可以把它加入佇列去處理它的 parent 了
                if grad_counts[id(p)] == 0:
                    queue.append(p)

    # 運算子重載保持不變
    def __add__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        return Variable(self.value + other.value, parent=[self, other], grad_fn=[lambda g: g, lambda g: g])
    def __radd__(self, other): return self.__add__(other)
    def __matmul__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        return Variable(self.value @ other.value, parent=[self, other], grad_fn=[lambda g: g @ other.value.T, lambda g: self.value.T @ g])
    def __mul__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        return Variable(self.value * other.value, parent=[self, other], grad_fn=[lambda g: g * other.value, lambda g: g * self.value])
    def __rmul__(self, other): return self.__mul__(other)

# 測試
X = Variable(np.random.normal(0,1,(3,3)))     
B = Variable(np.random.normal(0,1,(3,3)))      

# 複雜一點的圖：X 被用了兩次 (Diamond Pattern)
Y = X @ X + 3 * X + B

Y.backward()

print("執行成功，沒有遞迴")
print("X 的梯度:\n", X.grad)