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
import numpy as np

class Tensor:
    def __init__(self,value, name):
        self.value = np.array(value, dtype=float)
        self.name = name
        self.id = id(self)
    def __repr__(self):
        return f"Tesnor(shape = {self.value.shape}, id={self.id})"
        
        
class Gradient_Tape:
    def __init__(self):
        self.ops = []
        self.active = True
    def __     
        
        
        
        
        
        
        
        
        