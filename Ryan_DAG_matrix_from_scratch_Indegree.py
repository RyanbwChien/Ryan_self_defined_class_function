# -*- coding: utf-8 -*-
import numpy as np
from collections import deque, defaultdict

# =============================================================================
# 2️⃣ deque 的設計
# deque = double-ended queue（雙端隊列）
# 可以 從左端 pop / append 或 右端 pop / append，都只要 O(1)
# 沒有搬移元素的問題
# 適合「頭尾頻繁操作」的情況
# 2️⃣ popleft()
# node = queue.popleft()
# 只 把左端指標往右移一格
# 原本元素不搬動，也不留空格
# 看起來 queue 就少了最左邊的元素
# 
# O(1)
# 3️⃣ append()
# queue.append(x)
# 
# 
# 把 x 放在 tail 指向的位置
# tail 指標往右移一格
# 如果當前 block 滿了 → 自動分配新 block，tail 指向新 block
# 整個操作也是 O(1) 平均時間
# 簡單理解：deque 就像「左右可伸縮的陣列 + 分段 block」，append / popleft 都只改指標，不用搬整個陣列
# 
# 4️⃣ 小比喻
# popleft = 滑動門左移，把左邊的人出去
# append = 右邊加一個新元素，指標向右伸長
# 不管隊伍多長，都只動指標，不搬人
# 
# 
# =============================================================================
# =============================================================================
# 假設 DAG 如下（多層共享）：
# 
#       w
#      / \
#    t1   t2
#    |    |
#    x    x
#    |    |
#    y    z
# 
# 
# x 同時被 t1 和 t2 使用
# x 的 parent 上面還有其他節點 (y, z)
# 如果用 DFS 遞迴：
# 先從 w → t1 → x → y 計算梯度
# x 的梯度只累加 t1 的貢獻
# 遞迴返回 t2 → x → z
# ⚠️ 問題：如果在 t2 尚未計算完前，x 的梯度就被用去計算上游（例如 x 又是某個更上層節點的 parent），就可能：
# 梯度不完整
# 導致上層計算時乘上不完整的梯度 → 數值錯誤
# 
# 2️⃣ queue / Kahn 算法怎麼解決
# 核心原理：
# 每個節點記錄 grad_counts
# 表示還有多少個下游節點的梯度沒到達
# 只有 grad_counts = 0 才把節點放入 queue
# 意味著「所有下游梯度都累加完成」
# 從 queue 拿出節點計算，再往 parent 累加
# 保證每個 parent 拿到的梯度都是完整的
# =============================================================================



# =============================================================================
# DAG 方法把每個 parent 的局部貢獻算出來，再累加 → 和手動一次寫完 ∑ ∑ 是一樣的。 
# 如果運算不是加法可加（例如非線性操作裡有交互副作用），那就必須小心。但對純數值計算，這種方法 數學上是正確的。 
# 可是為何他們不直接用全部然後有方程式運算完在傳給後面方式 假設有微分不是靠不同分枝微分相加 而是相乘不就完蛋了
# =============================================================================

# =============================================================================
# 2️⃣ 為什麼需要 grad_counts？
# t2 的 grad_counts = 2，因為有兩個下游節點依賴它（t4, t5）。
# 如果用 DFS 遞迴，可能先走 t2 → t4 → u → w，把 t2 對 y 的梯度算出來
# 但如果 t5 的貢獻還沒算，y.grad 還沒完整 → 若此時梯度就往上傳，會不完整 → 上游節點 u 的梯度也會錯
# 3️⃣ Queue + grad_counts 保證正確
# 初始 queue 放最終輸出 w（grad_counts = 0）
# 反向傳播過程中，每個節點父節點的 grad_counts 都會減 1
# 只有當 t2 的 所有下游梯度都累加完成（t4 和 t5 都算完）時，grad_counts[t2] = 0 → t2 才進入 queue，表示可以安全累加 t2.grad
# =============================================================================


# =============================================================================
# 在拓撲排序（Topological Sort）的機制下，這是一個絕對的鐵律。
# 如果我們把 「放入 Queue 並執行計算」 當作是「點名」的話：
# 「上游的源頭（Input/Parent）」絕對不可能比「下游的結果（Output/Child）」先被點到名。
# 這就是為什麼這個演算法能保證梯度計算正確的核心原因。
# =============================================================================

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
        
        # ***記錄每個變數被使用的次數 (Dependency Count)
        grad_counts = defaultdict(int)
        # grad_counts = dict()
        # 用一個簡單的堆疊遍歷圖，計算每個 parent 被引用的次數
        # 這一步就像是 TF/PyTorch 在 "編譯/分析" 計算圖
        nodes_to_visit = [self]
        visited_nodes = set()
        
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if id(node) in visited_nodes:
                continue
            visited_nodes.add(id(node)) #使用集合保證微一存在
            
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
                # 在數學和程式慣例中：
                # d = Derivative (導數/微分) 或 Delta (變化量)
                # p = Parent (父節點)
                # 所以 d_p 代表的是：「從當前節點 (node) 回傳給父節點 (p) 的那一份梯度貢獻」。
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

# =============================================================================
#     4️⃣ 核心理解
# 
#     DFS 遞迴版：每個 parent 都自己算 → 可能重複或不完整累加
#     
#     Queue 拓撲版：集中管理、按拓撲順序 → 每個 parent 的梯度完整累加後再往上傳
#     
#     所以你理解得沒錯：這種寫法就是 「一次 backward() 就算完整個圖」，不用每個 parent 再呼叫 .backward()。
# =============================================================================
                    
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