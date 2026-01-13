# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:07:45 2026

@author: USER
"""

# =============================================================================
# 1️⃣ Max Pooling 的梯度傳回
# 在 forward 時，你對每個 pooling window 取最大值 
# 𝑦=max(𝑥1,𝑥2,𝑥3,𝑥4)
# y=max(x1,x2​,x3​,x4​)
# 
# backward 時，PyTorch/TensorFlow 都確實是透過 計算圖自動追蹤依賴關係：
# 梯度只會傳回「對輸出有影響的輸入節點」
# 對於 max pooling，就是最大值的那個 pixel會收到梯度，其它 pixel 不會
# 所以從這層「節點」來看，你不需要手動記 mask，框架已經幫你處理好了。
# 
# =============================================================================
import numpy as np
from typing import List

# ----------------------------
# Lightweight Tensor + Autograd
# ----------------------------
class Tensor:
    def __init__(self, value, inputs: List['Tensor']=None, grad_funcs: List=None, require_grad=False):
        self.value = np.array(value, dtype=float)
        self.inputs = inputs or []
        self.grad_funcs = grad_funcs or []
        self.require_grad = require_grad
        self.grad = np.zeros_like(self.value) if self.require_grad else None

    def __repr__(self):
        return f"Tensor(shape={self.value.shape}, require_grad={self.require_grad})"

    # ---------- utility ----------
    def _ensure_tensor(other):
        return other if isinstance(other, Tensor) else Tensor(other, require_grad=False)

    def _set_grad(self):
        if self.require_grad and self.grad is None:
            self.grad = np.zeros_like(self.value)

    # ---------- topo ordering ----------
    def _build_topo(self, visited, topo):
        if id(self) in visited:
            return
        visited.add(id(self))
        for inp in self.inputs:
            inp._build_topo(visited, topo)
        topo.append(self)

    def backward(self, grad=None):
        # build topo
        topo = []
        visited = set()
        self._build_topo(visited, topo)
        # initialize grads
        if grad is None:
            grad = np.ones_like(self.value)
        self._set_grad()
        self.grad[...] = grad  # set root grad

        # iterate reversed topo (from outputs to inputs)
        for node in reversed(topo):
            if not node.inputs:
                continue
            # node may not require_grad but its inputs might
            node_grad = node.grad if node.grad is not None else np.zeros_like(node.value)
            for inp, gfunc in zip(node.inputs, node.grad_funcs):
                if not inp.require_grad:
                    continue
                # compute gradient contribution for this input
                grad_input = gfunc(node_grad)  # numpy array
                # handle broadcasting reduction to match inp.grad.shape
                # sum extra leading dims
                while grad_input.ndim > inp.value.ndim:
                    grad_input = grad_input.sum(axis=0)
                # for dims where inp has size 1 but grad_input has >1, reduce
                for axis, (s_inp, s_g) in enumerate(zip(inp.value.shape, grad_input.shape)):
                    if s_inp == 1 and s_g != 1:
                        grad_input = grad_input.sum(axis=axis, keepdims=True)
                inp._set_grad()
                inp.grad[...] += grad_input

    # ---------- operations ----------
    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out_value = self.value + other.value
        require = self.require_grad or other.require_grad

        def grad_self(g):
            return g  # broadcasting handled later in backward
        def grad_other(g):
            return g

        return Tensor(out_value, inputs=[self, other], grad_funcs=[grad_self, grad_other], require_grad=require)

    __radd__ = __add__

    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        out_value = self.value * other.value
        require = self.require_grad or other.require_grad

        def grad_self(g):
            return g * other.value
        def grad_other(g):
            return g * self.value

        return Tensor(out_value, inputs=[self, other], grad_funcs=[grad_self, grad_other], require_grad=require)

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = Tensor._ensure_tensor(other)
        out_value = self.value @ other.value
        require = self.require_grad or other.require_grad

        def grad_self(g):
            return g @ other.value.T
        def grad_other(g):
            return self.value.T @ g

        return Tensor(out_value, inputs=[self, other], grad_funcs=[grad_self, grad_other], require_grad=require)

    def T(self):
        out_value = self.value.T
        require = self.require_grad

        def grad_T(g):
            return g.T
        return Tensor(out_value, inputs=[self], grad_funcs=[grad_T], require_grad=require)

    def reshape(self, *shape):
        out_value = self.value.reshape(*shape)
        require = self.require_grad

        def grad_reshape(g):
            return g.reshape(self.value.shape)
        return Tensor(out_value, inputs=[self], grad_funcs=[grad_reshape], require_grad=require)

    def sum(self, axis=None, keepdims=False):
        out_value = self.value.sum(axis=axis, keepdims=keepdims)
        require = self.require_grad

        def grad_sum(g):
            # broadcast g back to self.value.shape
            if axis is None:
                return np.ones_like(self.value) * g
            else:
                g_expanded = np.expand_dims(g, axis=axis) if not keepdims else g
                return np.ones_like(self.value) * g_expanded
        return Tensor(out_value, inputs=[self], grad_funcs=[grad_sum], require_grad=require)

    def flatten(self):
        return self.reshape(-1)

# ---------- functional ops ----------
def relu(x: Tensor):
    out_value = np.maximum(0, x.value)
    require = x.require_grad

    def grad_relu(g):
        return g * (x.value > 0)
    return Tensor(out_value, inputs=[x], grad_funcs=[grad_relu], require_grad=require)

# ---------- im2col / col2im helpers (numpy) ----------
def im2col_np(X, kH, kW, stride=1):
    # X: (C, H, W)
    C, H, W = X.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    cols = []
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = X[:, i:i+kH, j:j+kW].reshape(-1)
            cols.append(patch)
    return np.array(cols), out_h, out_w

def col2im_np(cols, X_shape, kH, kW, stride=1):
    C, H, W = X_shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    X_grad = np.zeros(X_shape)
    idx = 0
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = cols[idx].reshape(C, kH, kW)
            X_grad[:, i:i+kH, j:j+kW] += patch
            idx += 1
    return X_grad

# ---------- conv2d as single op (im2col + matmul + bias) ----------
def conv2d(input_t: Tensor, weight_t: Tensor, bias_t: Tensor=None, kernel_size=(3,3), stride=1):
    """
    input_t: Tensor with shape (C_in, H, W)
    weight_t: Tensor with shape (C_out, C_in * kH * kW)
    bias_t: Tensor with shape (C_out,) or None
    returns: Tensor with shape (num_patches, C_out) -- note: we keep 2D [num_slide, C_out]
    """
    C_in = input_t.value.shape[0]
    kH, kW = kernel_size
    cols, out_h, out_w = im2col_np(input_t.value, kH, kW, stride)  # cols: [num_patches, P]
    # conv_out = cols @ W.T + b
    conv_out = cols @ weight_t.value.T
    if bias_t is not None:
        conv_out = conv_out + bias_t.value  # broadcast across rows

    require = input_t.require_grad or weight_t.require_grad or (bias_t.require_grad if bias_t is not None else False)

    # capture local values for grad closure
    cols_local = cols
    in_shape = input_t.value.shape

    def grad_input(g):
        # g: [num_patches, C_out]
        dcols = g @ weight_t.value  # [num_patches, P]
        dX = col2im_np(dcols, in_shape, kH, kW, stride)
        return dX

    def grad_weight(g):
        # grad wrt W: g.T @ cols
        return (g.T @ cols_local)

    def grad_bias(g):
        return g.sum(axis=0)

    grad_funcs = [grad_input, grad_weight]
    if bias_t is not None:
        grad_funcs.append(grad_bias)
        inputs = [input_t, weight_t, bias_t]
    else:
        inputs = [input_t, weight_t]

    return Tensor(conv_out, inputs=inputs, grad_funcs=grad_funcs, require_grad=require)

# ---------- maxpool2d ----------
def maxpool2d(conv_out_t: Tensor, pool_size=2):
    """
    conv_out_t: Tensor shape [num_patches, C_out] OR we can interpret as (C_out, H, W) flattened to [num_patches, C_out].
    For simplicity, we assume conv_out_t.value can be reshaped into (C_out, H, W) with known H,W provided externally.
    To integrate easily, we expect conv_out_t came from conv2d that returned [num_patches, C_out] and user passes out_h,out_w.
    """
    # Here we expect conv_out_t has additional attributes out_h/out_w if created from conv2d.
    # To keep things simple in this demo, we will require the user to provide H and W by attaching them:
    if not hasattr(conv_out_t, 'out_h') or not hasattr(conv_out_t, 'out_w'):
        raise ValueError("conv_out_t must have attributes out_h and out_w (set by caller).")

    out_h = conv_out_t.out_h
    out_w = conv_out_t.out_w
    C_out = conv_out_t.value.shape[1]
    pool_h = pool_w = pool_size

    # reshape per-channel
    pooled_list = []
    masks = []  # store masks per channel per pooling window
    pooled_h = out_h // pool_h
    pooled_w = out_w // pool_w

    conv_reshaped = []
    for c in range(C_out):
        ch = conv_out_t.value[:, c].reshape(out_h, out_w)
        conv_reshaped.append(ch)

    for c in range(C_out):
        ch = conv_reshaped[c]
        pooled = []
        mask_ch = []
        for i in range(0, out_h, pool_h):
            for j in range(0, out_w, pool_w):
                window = ch[i:i+pool_h, j:j+pool_w]
                idx = np.argmax(window)
                max_val = window.flatten()[idx]
                pooled.append(max_val)
                m = np.zeros_like(window, dtype=bool)
                a, b = divmod(idx, pool_w)
                m[a, b] = True
                mask_ch.append(m)
        pooled_list.append(np.array(pooled))
        masks.append(mask_ch)

    pooled_arr = np.array(pooled_list)  # [C_out, pooled_h*pooled_w]

    require = conv_out_t.require_grad

    # capture masks and dims in closure
    def grad_conv_out(g):
        # g: grad of pooled_arr but flattened as same shape
        # Build dconv (out_h, out_w, C_out) then flatten to [num_patches, C_out] shape
        dconv = np.zeros((out_h, out_w, C_out))
        for c in range(C_out):
            idx = 0
            for i in range(0, out_h, pool_h):
                for j in range(0, out_w, pool_w):
                    mask = masks[c][idx]  # shape pool_h x pool_w (bool)
                    d = g[c, idx] if g.ndim == 2 else g.reshape(pooled_arr.shape)[c, idx]
                    # place d into the masked position
                    dconv[i:i+pool_h, j:j+pool_w, c] += mask * d
                    idx += 1
        # convert dconv to same flattened shape as conv_out_t.value (which was [num_patches, C_out])
        # conv_out_t.value[:, c] = ch.reshape(out_h, out_w).flatten()
        num_patches = out_h * out_w
        dconv_flat = np.zeros_like(conv_out_t.value)
        for c in range(C_out):
            dconv_flat[:, c] = dconv[:, :, c].reshape(-1)
        return dconv_flat

    # Return a Tensor whose value is pooled_arr but we also add metadata for dims
    out_tensor = Tensor(pooled_arr, inputs=[conv_out_t], grad_funcs=[grad_conv_out], require_grad=require)
    # attach dims (used by later code)
    out_tensor.pooled_h = pooled_h
    out_tensor.pooled_w = pooled_w
    out_tensor.out_h = out_h
    out_tensor.out_w = out_w
    return out_tensor

# ---------- helper to attach conv out dims when calling conv2d ----------
def conv2d_with_dims(input_t: Tensor, weight_t: Tensor, bias_t: Tensor=None, kernel_size=(3,3), stride=1):
    # run conv2d and attach out_h/out_w so maxpool can use them
    C_in = input_t.value.shape[0]
    kH, kW = kernel_size
    cols, out_h, out_w = im2col_np(input_t.value, kH, kW, stride)
    conv_out = cols @ weight_t.value.T
    if bias_t is not None:
        conv_out = conv_out + bias_t.value
    out_tensor = Tensor(conv_out,
                        inputs=[input_t, weight_t] + ([bias_t] if bias_t is not None else []),
                        grad_funcs=None,  # we will reuse conv2d to set grad_funcs properly below
                        require_grad=(input_t.require_grad or weight_t.require_grad or (bias_t.require_grad if bias_t is not None else False)))
    # set grad_funcs using conv2d() implementation
    # We'll just call conv2d(...) to get a properly constructed tensor and copy grad_funcs
    tmp = conv2d(input_t, weight_t, bias_t, kernel_size=kernel_size, stride=stride)
    out_tensor.grad_funcs = tmp.grad_funcs
    out_tensor.out_h = out_h
    out_tensor.out_w = out_w
    return out_tensor

# ----------------------------
# Demo: small CNN forward + autograd backward
# ----------------------------
if __name__ == "__main__":
    # Dummy input: 3 channels, 5x5
    X = Tensor(np.random.randn(3,5,5), require_grad=True)

    # Conv params
    out_channels = 2
    k = 3
    W = Tensor(np.random.randn(out_channels, 3 * k * k) * 0.1, require_grad=True)
    b = Tensor(np.zeros(out_channels), require_grad=True)

    # conv -> relu -> maxpool -> linear
    conv = conv2d_with_dims(X, W, b, kernel_size=(k,k), stride=1)   # conv.value shape: [num_patches, out_channels]
    conv.out_h = conv.out_h  # already attached
    conv.out_w = conv.out_w

    a = relu(conv)   # requires grad if conv does
    # For maxpool2d we expect conv-shaped dims set (we set in conv2d_with_dims)
    a.out_h = conv.out_h
    a.out_w = conv.out_w
    pooled = maxpool2d(a, pool_size=2)   # pooled.value shape: [C_out, pooled_h*pooled_w]

    # flatten pooled and linear layer (we'll implement linear as W2 @ flat + b2)
    flat = Tensor(pooled.value.flatten(), require_grad=pooled.require_grad)
    # IMPORTANT: To keep autograd chain, create flat as a Tensor that has pooled as input and a grad func:
    def grad_unflatten(g):
        return g.reshape(pooled.value.shape)
    flat = Tensor(flat.value, inputs=[pooled], grad_funcs=[grad_unflatten], require_grad=pooled.require_grad)

    # linear params
    num_classes = 2
    linear_W = Tensor(np.random.randn(num_classes, flat.value.size) * 0.1, require_grad=True)
    linear_b = Tensor(np.zeros(num_classes), require_grad=True)
    logits = linear_W @ flat  # shape [num_classes]
    logits = logits + linear_b

    # Simple scalar loss: sum(logits) (just to test backward)
    loss = logits.sum()

    # Backward: compute grads
    loss.backward()

    # Print some grads
    print("X.grad shape:", X.grad.shape)
    print("W.grad shape:", W.grad.shape)
    print("b.grad shape:", b.grad.shape)
    print("linear_W.grad shape:", linear_W.grad.shape)
    print("linear_b.grad shape:", linear_b.grad.shape)
    # sanity: none of them should be None if require_grad=True
    print("W.grad (sample):\n", W.grad)
