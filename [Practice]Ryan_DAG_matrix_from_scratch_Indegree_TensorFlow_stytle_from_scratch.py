# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:10:22 2026

@author: USER
"""

import numpy as np


# ============================================================
# 1. Tensor
# ============================================================

class Tensor:

    def __init__(self, value):
        self.value = np.array(value, dtype=np.float64)

    def __add__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(
            "add",
            [self, other]
        )

    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return apply_op(
            "matmul",
            [self, other]
        )

    def __repr__(self):
        return f"Tensor({self.value})"


# ============================================================
# 2. Variable
# ============================================================

class Variable(Tensor):

    def assign(self, value):

        self.value = np.array(
            value,
            dtype=np.float64
        )

    def __repr__(self):
        return f"Variable({self.value})"


# ============================================================
# 3. 建立 Tensor / Variable 的 API
# ============================================================

def constant(value):
    return Tensor(value)


def variable(value):
    return Variable(value)


# ============================================================
# 4. Operation Registry
# ============================================================

_OP_REGISTRY = {}


def register_op(op_name):

    def decorator(forward_func):

        _OP_REGISTRY[op_name] = forward_func

        return forward_func

    return decorator


# ============================================================
# 5. Forward Operation
# ============================================================

@register_op("add")
def add_forward(inputs):

    x = inputs[0]
    y = inputs[1]

    return x.value + y.value


@register_op("matmul")
def matmul_forward(inputs):

    x = inputs[0]
    y = inputs[1]

    return x.value @ y.value


# ============================================================
# 6. Gradient Registry
# ============================================================

_GRADIENT_REGISTRY = {}


def register_gradient(op_name):

    def decorator(gradient_func):

        _GRADIENT_REGISTRY[op_name] = gradient_func

        return gradient_func

    return decorator


# ============================================================
# 7. Backward rules
# ============================================================

@register_gradient("add")
def add_gradient(inputs, grad):

    return (
        grad,
        grad
    )


@register_gradient("matmul")
def matmul_gradient(inputs, grad):

    x = inputs[0].value
    y = inputs[1].value

    dx = grad @ y.T
    dy = x.T @ grad

    return (
        dx,
        dy
    )


# ============================================================
# 8. Current GradientTape
# ============================================================

_CURRENT_TAPE = None


def get_current_tape():
    return _CURRENT_TAPE


# ============================================================
# 9. GradientTape
# ============================================================

class GradientTape:

    def __init__(self):

        self.ops_history = []

    def __enter__(self):

        global _CURRENT_TAPE

        self.previous_tape = _CURRENT_TAPE

        _CURRENT_TAPE = self

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        global _CURRENT_TAPE

        _CURRENT_TAPE = self.previous_tape

    def record(
        self,
        op_name,
        inputs,
        output
    ):

        self.ops_history.append(
            (
                op_name,
                inputs,
                output
            )
        )

    def gradient(
        self,
        target,
        sources
    ):

        return backward(
            self,
            target,
            sources
        )


# ============================================================
# 10. Operation 執行核心
# ============================================================

def apply_op(op_name, inputs):

    # 取得 forward function
    forward_func = _OP_REGISTRY[op_name]

    # Forward
    output_value = forward_func(inputs)

    # Forward 結果包成 Tensor
    output = Tensor(output_value)

    # 如果目前有 GradientTape，就記錄這次 Operation
    tape = get_current_tape()

    if tape is not None:

        tape.record(
            op_name,
            inputs,
            output
        )

    return output


# ============================================================
# 11. Broadcasting Gradient
# ============================================================

def reduce_broadcast_gradient(
    grad,
    target_shape
):

    # Gradient 維度比原始 Tensor 多
    while grad.ndim > len(target_shape):

        grad = grad.sum(axis=0)

    # 原始 Tensor 某個維度為 1，
    # Forward 時被 broadcasting
    for axis, (g_dim, t_dim) in enumerate(
        zip(
            grad.shape,
            target_shape
        )
    ):

        if t_dim == 1 and g_dim != 1:

            grad = grad.sum(
                axis=axis,
                keepdims=True
            )

    return grad


# ============================================================
# 12. Backward Engine
# ============================================================

def backward(
    tape,
    target,
    sources
):

    # target 對自己的 gradient = 1
    grads = {
        id(target):
            np.ones_like(target.value)
    }

    # Forward 是正向順序
    # Backward 就反過來
    for (
        op_name,
        inputs,
        output
    ) in reversed(
        tape.ops_history
    ):

        # 如果這個 output 沒有 gradient
        # 表示它不在 target 的反向路徑上
        if id(output) not in grads:
            continue

        # output 的 gradient
        out_grad = grads[id(output)]

        # 找到這個 Operation 對應的 backward
        gradient_func = _GRADIENT_REGISTRY[op_name]

        # 計算每個 input 的 gradient
        input_grads = gradient_func(
            inputs,
            out_grad
        )

        # 把 gradient 傳回 input
        for input_tensor, input_grad in zip(
            inputs,
            input_grads
        ):

            # 處理 broadcasting
            input_grad = reduce_broadcast_gradient(
                input_grad,
                input_tensor.value.shape
            )

            # DAG 中可能有多條路徑
            # 所以 gradient 必須累加
            grads[id(input_tensor)] = (

                grads.get(
                    id(input_tensor),
                    0
                )

                + input_grad
            )

    # 只回傳使用者要求的 source
    return [

        grads.get(
            id(source),
            np.zeros_like(source.value)
        )

        for source in sources

    ]


# ============================================================
# 13. TEST
# ============================================================

if __name__ == "__main__":

    x = variable([
        [1., 2.],
        [3., 4.]
    ])

    w = variable([
        [5., 6.],
        [7., 8.]
    ])

    with GradientTape() as tape:

        y1 = x @ w

        y2 = y1 + x

    gradients = tape.gradient(
        y2,
        [x, w]
    )

    print("x =")
    print(x)

    print()

    print("w =")
    print(w)

    print()

    print("y1 =")
    print(y1)

    print()

    print("y2 =")
    print(y2)

    print()

    print("dx =")
    print(gradients[0])

    print()

    print("dw =")
    print(gradients[1])