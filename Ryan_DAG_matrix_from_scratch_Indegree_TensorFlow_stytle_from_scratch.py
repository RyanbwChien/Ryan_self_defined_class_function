# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 21:13:08 2025

@author: Ryan
"""


# =============================================================================
# MyClass()
# ↓
# PyObject_Call(MyClass)
# ↓
# PyType_Type.tp_call   ← 因為 MyClass 是 type 的 instance
# ↓
# type_call(type=MyClass)
# ↓
# MyClass.tp_new
# ↓
# MyClass.tp_init
# ↓
# instance
# =============================================================================
 
# =============================================================================
# 🧩 slot wrapper 到底是什麼？
# 不是 Python function
# 不是 PyCFunction
# 而是「C 層 dispatcher」
# 
# 簡化長相：
# 
# static PyObject *
# slot_tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
# {
#     PyObject *meth = lookup("__new__", type);
#     return PyObject_Call(meth, args, kwds);
# }
# 
# 
# 👉 它的任務只有一個：
# 
# 把 C slot 呼叫轉成 Python method 呼叫
# 
# 🧩 為什麼不能直接指向 Python function？
# 
# 因為：
# 
# C slot 簽名是固定的（ABI）
# 
# Python function 是動態物件
# 
# 需要：
# 
# argument unpack
# 
# descriptor 綁定
# 
# MRO lookup
# 
# exception translation
# 
# 👉 slot wrapper 是必要的「轉接層」
# =============================================================================

# =============================================================================
# 🧠 最終總結（你現在的位置）
# 
# 你現在的理解可以濃縮成這張表：
# 
# 情境	tp_repr 指向
# 沒覆寫 __repr__	C 實作（base.tp_repr）
# 有覆寫 __repr__	slot_tp_repr
# Python 呼叫 obj.__repr__()	attribute lookup
# C / builtin 呼叫 repr(obj)	tp_repr(obj)
# 
# 👉 slot wrapper 不是多此一舉
# 👉 它是為了補上「C slot 呼叫不做 lookup」這個缺口
# 
# =============================================================================


# watch 決定的是「我要不要結果」
# 不是「你能不能參與計算」
# 沒有 watch 的變數仍然會參與 forward 與 backward 的數值計算，
# 只是其梯度不會被回傳，也不會被用來更新參數。

import numpy as np
from collections import deque, defaultdict

_GRADIENT_REGISTRY = {}

def register_gradient(op_name):
    def wrapper(func):
        _GRADIENT_REGISTRY[op_name] = func
        return func #重點包裝器還是回傳原本自己的函數
    return wrapper


# 即使 OUT 是 Tensor，只要它的祖先有 Variable，它的 ID 就會被記錄，梯度就能傳得回去。 這就是為什麼你的代碼能動的原因！

class Tensor:
    def __init__(self,value, name=None):
        self.value = np.array(value, dtype=float)
        self.name = name
        self.id = id(self)
    def __repr__(self):
        return f"Tesnor(shape = {self.value.shape}, id={self.id})"

class Variable(Tensor):
    def __init__(self,value, name=None):
        super(Variable,self).__init__(value, name=None)
        
class Gradient_Tape:
    def __init__(self):
        self.ops = []
        self.watched_ids = set() # 這裡紀錄所有需要追蹤梯度的 Tensor ID
        self.active = False
    def __enter__(self):
        self.active = True
        global _CURRENT_TAPE
        _CURRENT_TAPE = self 
        return self #當你希望在 with 語句中取得並操作這個上下文管理器實例時。
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        global _CURRENT_TAPE
        _CURRENT_TAPE = None
        # 當離開當次with Gradient_Tape() as tape 呼叫 __exit__ _CURRENT_TAPE將不在記錄運算並且將其淨空
    
    # ✅ 最接近 TensorFlow eager tape 的版本
    # forward：全記
    def record_op(self, op_name, inputs, output):
        for x in inputs:
            if isinstance(x, Variable):
                self.watched_ids.add(x.id)
        self.ops.append({
            "op_name":op_name,
            "inputs":inputs,
            "output":output            
            })
        
    def watch(self,inputs):
        self.watched_ids.add(id(inputs))
    def gradient(self,target, sources):
        """
        全部運算都做完 才會在對loss fcn作微分 tape.gradient(Loss, [X, W, B]) 
        target : loss (Tensor) 最一開始進入結點
        sources : list of params to update (list of Tensors)
        """
        # 哪個 output Tensor 是由哪個 op 產生
        producer_map = {}
        for op in self.ops:
            producer_map[op["output"].id] = op
            
        grad_counts = defaultdict(int)
        for op in self.ops:
            for x in op["inputs"]:
                grad_counts[x.id] += 1
        
        grads = defaultdict(int)
        grads[target.id] = np.ones_like(target.value)
        
        queue = deque([target])
        
        while queue:
            current_tensor = queue.popleft()
            current_grad = grads[current_tensor.id]
            
            if current_tensor.id not in producer_map:
                continue
            
            op_entry = producer_map[current_tensor.id]
            op_name = op_entry["op_name"]
            inputs = op_entry["inputs"]
            
            if op_name not in _GRADIENT_REGISTRY:
                raise ValueError(f"Op {op_name} 沒有註冊微分函數")
            grad_func = _GRADIENT_REGISTRY[op_name]  
            
            input_grads = grad_func(current_grad, inputs)
            
            if not isinstance(input_grads,(list, tuple)):
                input_grads = [input_grads]
            
            for x,g in zip(inputs, input_grads):
                # 廣播修正 (Broadcasting Fix)
                # =============================================================================
                # Broadcasting 的流程是：
                # 先在左邊補 1，使維度數一致，
                # 然後從最後一維開始逐一比對，
                # 若任一維既不相等也不是 1，就報錯。
                # 框架不會先檢查最後一維，也不會推斷語意。
                # =============================================================================


                # =============================================================================
                # 精準描述 broadcasting backward sum
                # 假設 forward：
                # A: (2,3,4)
                # B: (1,3,1)
                # C = A + B
                # broadcast 維度 = 0, 2
                # 非 broadcast 維度 = 1
                # backward：
                # 我們要計算 dL/dB，shape = (1,3,1)
                # 對 broadcast 維度 (0,2) 做 sum
                # 非 broadcast 維度 (1) 保持對應索引
                # =============================================================================

                if g.shape != x.value.shape:
                    while g.ndim > x.value.ndim:
                        g = g.sum(axis=0)
                    for i, dim in enumerate(x.value.shape):
                        if dim == 1 and g.shape[i] != 1:
                            g = g.sum(axis=i, keepdims=True)
                            
                grads[x.id] += g
                grad_counts[x.id] -= 1
                
                if grad_counts[x.id] == 0:
                    queue.append(x)
        return [ grads[s.id]  if s.id in self.watched_ids else None for s in sources]
            
def tf_matmul(a,b):
    val = a.value @ b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("MatMul", [a,b], out)
    return out
        
def tf_add(a,b):
    val = a.value + b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Add", [a,b], out)
    return out

def tf_mul(a,b):
    val = a.value * b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Mul", [a,b], out)
    return out

def tf_sub(a,b):
    val = a.value - b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Sub", [a,b], out)
    return out        

def tf_pow(a,b):
    val = a.value**b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Pow", [a,b], out)
    return out          

def tf_truediv(a,b):
    val = a.value / b.value
    out = Tensor(val)    
    
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("Truediv", [a,b], out)
    return out 

def tf_reduce_sum(a):
    val = np.sum(a.value)
    out = Tensor(val)
    if _CURRENT_TAPE:
        _CURRENT_TAPE.record_op("ReduceSum", [a], out)
    return out

# =============================================================================
# 偏微分不是對「表達式」微分，
# 而是對「函數在座標方向上的變化率」做定義。
# =============================================================================
        
@register_gradient("MatMul")
def grad_matmul(grad, inputs):
    A, B = inputs
    grad_A = grad @ B.value.T
    grad_B = A.value.T @ grad   
    return grad_A, grad_B
        
@register_gradient("Add")
def grad_add(grad, inputs):
    # A, B = inputs
    return grad, grad
                
@register_gradient("Mul")
def grad_mul(grad, inputs):
    A, B = inputs
    grad_A = grad * B.value
    grad_B = grad * A.value 
    return grad_A, grad_B

@register_gradient("Sub")
def grad_sub(grad, inputs):
    # A, B = inputs
    return grad, -grad

@register_gradient("Pow")
def grad_pow(grad, inputs):
    A, B = inputs
    grad_A = grad * (B.value) * (A.value)**(B.value -1)
    grad_B = grad * np.exp(B.value*np.log(A.value))*np.log(A.value)
    return grad_A, grad_B

@register_gradient("Truediv")
def grad_truediv(grad, inputs):
    A, B = inputs
    grad_A = grad * 1/B.value
    grad_B = grad * -A.value *1/(B.value)**2
    return grad_A, grad_B

@register_gradient("ReduceSum")
def grad_reduce_sum(grad, inputs):
    A, = inputs
    return grad * np.ones_like(A.value)
        
X = Tensor(np.random.normal(0,1,(10,3)),"X")
W = Variable(np.random.normal(0,1,(3,3)),"W")
B = Variable(np.random.normal(1.5,1,(1,3)),"B") # Parameter 的 shape 永遠不含 batch 維度
W.value

W_act = np.random.normal(0,1,(3,3))
B_act = np.random.normal(0,1,(1,3))
Y = Tensor(X.value @ W_act + B_act) 
Y.value
_CURRENT_TAPE = None

# =============================================================================
# def loss_fcn(Y,Y_pred):
#     result = tf_truediv(tf_pow(tf_sub(Y,Y_pred),
#                                Tensor(2)),
#                                Tensor(Y.value.shape[0]))
#     return result
# =============================================================================
def loss_fcn(Y, Y_pred):
    diff = tf_sub(Y, Y_pred)
    sq   = tf_pow(diff, Tensor(2))
    total = tf_reduce_sum(sq)
    batch_size = Tensor(np.array([Y.value.shape[0]]))
    mean  = tf_truediv(total,batch_size)
    return mean


def optimize(grads,param,lambdas):
    para_update = []
    for g,p in zip(grads,param):
        p.value = p.value - lambdas*g
        para_update.append(p)
    return para_update
# =============================================================================
# 二、為什麼你直覺會覺得「偏微也該微到 b 裡的 a」？
# 因為你腦中做的是這件事：
# 你已經把「實際路徑」代進去了
# 一旦你寫：
# f(a,b(a))
# 你就已經退出偏微的世界了。
# =============================================================================


epochs = 500

for epoch in range(epochs):
    


    with Gradient_Tape() as tape:
        Y_pred = tf_add(tf_matmul(X,W),B)
        loss  = loss_fcn(Y,Y_pred)
        # record_op 結束
    
    grads = tape.gradient(loss, [W,B])
    
    W,B = optimize(grads,[W,B],0.1)
    
    print(loss.value)


grad_W, grad_B = grads
print("\n--- Backward 完成 ---")
# 訓練到最後本來梯度就是要靠近0
print("Gradient of W:\n", grad_W) 
print("Gradient of B:\n", grad_B)
print("W_act:\n", W_act)
print("B_act:\n", B_act)
print("W:\n", W.value)
print("B:\n", B.value)

# =============================================================================
# # 驗證形狀是否正確
# assert grad_X.shape == X.value.shape
# print("\n形狀檢查通過！")
# =============================================================================
