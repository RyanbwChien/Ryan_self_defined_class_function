# -*- coding: utf-8 -*-
"""
為什麼 Optimizer 絕對不能寫在 Tensor 裡？
這不是習慣問題，是責任分離（Separation of Concerns）。
① Tensor 不應該知道「如何被更新」
Tensor 的責任只有三件事：
1️⃣ 存 value
2️⃣ 存 grad
3️⃣ 知道怎麼把 grad 往前傳
"""

"""
在 Autograd 計算圖中：
「inputs」就是「parent」。
「沒有 inputs」=「沒有 parent」=「葉子節點（leaf）」

二、Binary Search Tree vs Autograd Graph：名詞對照表
1️⃣ Binary Tree / BST（資料結構課）
        root
       /    \
   internal  internal
      |
     leaf


root：沒有 parent

leaf：沒有 children

"""

#所以PYTORCH是 如果我有任意一個INPUT 的REQURE_GRAD是TRUE就會讓我的OUTPUT 的REQURE_GRAD也是TRUE嗎
# 所以，為了保證「只要有一個祖先需要梯度，後代就能把梯度傳回去」，Output 必須繼承「需要梯度」的這個屬性。
import numpy as np
from collections import deque, defaultdict
 


class Tensor:
    def __init__(self, value, inputs=[], grad_func=[], require_grad=False):
        self.value = np.array(value, dtype=float)
        self.inputs = inputs
        self.grad_func = grad_func
        self.grad = np.zeros_like(self.value)
        self.require_grad = require_grad

    def detach(self):
        return Tensor(self.value, require_grad=False)    

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value) 
        
        # 1. 建立入度表
        grad_counts = defaultdict(int)    
        node_to_visit = [self]
        visited_nodes = set()
        while node_to_visit:
            current_node = node_to_visit.pop()
            if id(current_node) not in visited_nodes:
                visited_nodes.add(id(current_node))
                for i in current_node.inputs:
                    grad_counts[id(i)] += 1
                    node_to_visit.append(i)
        
        # 2. 梯度累加字典
        grad_dict = defaultdict(lambda: 0) 
        grad_dict[id(self)] = grad          
        
        queue = deque([self])
        
        while queue:
            current_node = queue.popleft()
            
            # 【修正處】使用 pop(key, default)，避免 require_grad=False 的節點導致 KeyError
            # 如果節點不在字典裡，代表它收到的梯度總和是 0
            current_total_grad = grad_dict.pop(id(current_node), 0)
            
            # 如果是葉子節點
            if not current_node.inputs:
                if current_node.require_grad:
                    # 只有 Leaf Node 且需要梯度，才把最終結果寫入 .grad
                    current_node.grad = current_total_grad
                continue
            
            # 如果梯度已經是 0，且不是起點，則不需要再往下傳遞計算（優化效能）
            if np.all(current_total_grad == 0) and current_node is not self:
                # 雖然不計算梯度，但還是要維護拓撲排序的入度
                for i in current_node.inputs:
                    grad_counts[id(i)] -= 1
                    if grad_counts[id(i)] == 0:
                        queue.append(i)
                continue

            for i, f in zip(current_node.inputs, current_node.grad_func):
                if i.require_grad:
                    grad_input = f(current_total_grad) 
                    
                    # Broadcasting 處理
                    if grad_input.shape != i.value.shape:
                        while grad_input.ndim > i.value.ndim:
                            grad_input = grad_input.sum(axis=0)
                        for axis, dim in enumerate(i.value.shape):
                            if dim == 1:
                                grad_input = grad_input.sum(axis=axis, keepdims=True)
                
                    grad_dict[id(i)] += grad_input
                
                # 無論需不需要梯度，都要更新入度以維持拓撲排序順序
                grad_counts[id(i)] -= 1
                if grad_counts[id(i)] == 0:
                    queue.append(i)
# =============================================================================
#     答案是：不會。在正確的自動微分框架中，如果一個節點 i.require_grad 是 False，那麼它的所有祖先路徑都不可能需要梯度。
#     以下是為什麼不需要把梯度放入 grad_dict[id(i)] 的三個核心理由：
#     1. 屬性傳遞的一致性（Propagation Rule）
#     根據我們之前寫的 __add__, __matmul__ 等運算：
#     Output.require_grad = Input_A.require_grad OR Input_B.require_grad
#     這意味著：
#     如果 祖先（Input） 有任何一個人需要梯度，後代（Output） 就絕對會是 True。
#     反過來說（逆否命題）：如果 後代（i） 是 False，則代表 它所有的祖先 必定也全都是 False。
#     結論： 如果 i.require_grad 是 False，這條路徑往前走到底也不會遇到任何需要梯度的 Leaf Node。因此，計算這條路徑的梯度是純粹的浪費。
#     2. 斷路器效應（Stop-Gradient / Detach）
#     在深度學習中，我們有時會手動將某個中間節點設為 require_grad=False（例如 PyTorch 的 .detach()）。
#     目的： 我們就是要截斷梯度的回傳。
#     結果： 當 i.require_grad = False 時，即便它的祖先（例如權重 
#     W
#     W
#     ）原本是 True，但在「這條計算路徑」上，我們不希望梯度傳回去。
#     如果你把梯度依然存入 grad_dict[id(i)] 並繼續往前傳，你就違反了使用者想要「截斷梯度」的意圖。
#     3. 節省計算資源（這是優化的關鍵）
#     自動微分中最耗時的就是矩陣乘法。
# =============================================================================
        
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = self.value @ other.value
        
        return Tensor(out,inputs=[self,other],grad_func=[lambda g:g @ other.value.T,lambda g:self.value.T @ g])
    def __add__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(other)
        # 正統作法
        out_require_grad = self.require_grad or other.require_grad
            
        return Tensor(self.value + other.value, inputs=[self, other], grad_func=[lambda g: g, lambda g: g],require_grad=out_require_grad)
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
B = Tensor(np.random.normal(0,1,(3,3)), require_grad=True)      

# 複雜一點的圖：X 被用了兩次 (Diamond Pattern)
Y = X @ X + 3 * X + B
Y.value
Y.backward()

print("執行成功，沒有遞迴")
print("X 的梯度:\n", B.grad)    
    
    
id(5).__class__    
(5).__hash__()    
    