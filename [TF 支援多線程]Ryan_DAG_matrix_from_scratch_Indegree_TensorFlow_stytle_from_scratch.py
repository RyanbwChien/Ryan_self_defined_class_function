# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:00:35 2026

@author: USER
"""

import threading
import numpy as np


# ==========================================================
# 1. Gradient function registry
# ==========================================================

_Grad_Map = {}


def register_derivative_ops(op_name):

    def decorator(derivative_func):

        _Grad_Map[op_name] = derivative_func

        return derivative_func

    return decorator



# ==========================================================
# 2. Thread local Autograd Context
# ==========================================================

_context = threading.local()


class AutogradContext:


    @classmethod
    def get_tape_stack(cls):

        # 每個 thread 第一次使用時建立自己的 stack

        if not hasattr(_context, "tape_stack"):
            # stack 解決：同一個 thread 裡面，有沒有多層 tape？

            _context.tape_stack = []

        return _context.tape_stack



    @classmethod
    def current(cls):

        """
        取得目前 active 的 GradientTape

        每個 thread 都有自己的 stack
        """

        stack = cls.get_tape_stack()

        if stack:

            return stack[-1]

        return None



# ==========================================================
# 3. GradientTape
# ==========================================================


class GradientTape:


    def __init__(self):

        self.ops_history = []



    def __enter__(self):

        stack = AutogradContext.get_tape_stack()

        stack.append(self)

        return self



    def __exit__(self, *args):

        stack = AutogradContext.get_tape_stack()

        stack.pop()



    def _record(self, op_name, inputs, output):

        self.ops_history.append(
            (
                op_name,
                inputs,
                output
            )
        )



    def gradient(self, target, sources):


        grads = {

            target.id:
            np.ones_like(target.value)

        }


        for op_name, inputs, output in reversed(self.ops_history):


            # ----------------------------------
            # 取得 output gradient
            # ----------------------------------

            if output.id not in grads:

                continue


            out_grad = grads[output.id]


            # ----------------------------------
            # 找 backward function
            # ----------------------------------

            derivative_func = _Grad_Map[op_name]


            in_grads = derivative_func(
                inputs,
                out_grad
            )



            # ----------------------------------
            # broadcast gradient reduction
            # ----------------------------------

            for x_input, g in zip(inputs, in_grads):


                while g.ndim > x_input.value.ndim:

                    g = g.sum(axis=0)



                for axis, (g_dim, x_dim) in enumerate(
                    zip(
                        g.shape,
                        x_input.value.shape
                    )
                ):

                    if g_dim != x_dim and x_dim == 1:

                        g = g.sum(
                            axis=axis,
                            keepdims=True
                        )



                grads[x_input.id] = (

                    grads.get(
                        x_input.id,
                        0
                    )
                    +
                    g

                )



        return [

            grads.get(
                s.id,
                0
            )

            for s in sources

        ]



# ==========================================================
# 4. derivative functions
# ==========================================================



@register_derivative_ops("__add__")
def add_backward(inputs, grad):

    return (

        grad,
        grad

    )




@register_derivative_ops("__matmul__")
def matmul_backward(inputs, grad):

    return (

        grad @ inputs[1].value.T,

        inputs[0].value.T @ grad

    )