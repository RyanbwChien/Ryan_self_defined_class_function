import torch
import torch.nn as nn

d_model = 64
nhead = 8
num_layers = 4

encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
nn.TransformerEncoder(encoder_layer, num_layers)

import tensorflow as tf
# tf.GradientTapegradient(target, sources)


import numpy as np

Grad_func_map = {}


def apply_method(op_name,inputs):
    
    if op_name == "MatMul":
        res =  inputs[0].data @ inputs[1].data
    
    ouput = Tensor(res)
    
    if _Current_Tape is not None:
        _Current_Tape.record(op_name, inputs, ouput)


def _register_method(op_name):
    def decorator(func):        
        Grad_func_map[op_name] = func
        return func
        
    return decorator



def mat_mul(inputs):
    out = inputs[0].data @ inputs[1].data
    return out






class Tensor:
    def __init__(self,x):
        self.data = np.array(x,dtype=np.int32)
        self.id = id(self)
    
    def __matmul__(self,inputs):
        apply_method
        return apply_method("MatMul",inputs)

    

    
class Variable(Tensor):
    def __init__(self):
        super().__init__()
        
_Current_Tape = None

class RyGrandientTape:
    def __enter__(self):
        global _Current_Tape
        if _Current_Tape is None:
            self.history = []
            _Current_Tape = self
        return self
            
    def __exit__(self):
        global _Current_Tape
        _Current_Tape = None
        
    def record(self,op_name,inputs,ouput):
        self.history.append([op_name,inputs,ouput])
        
        
    def gradient(self, target, source):
        pass