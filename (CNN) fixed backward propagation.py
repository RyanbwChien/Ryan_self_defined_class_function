# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:08:04 2026

@author: USER
"""

import numpy as np

# -----------------------------
# Helper functions
# -----------------------------
def im2col(X, kernel_size, stride=1):
    """把卷積 patch 展開成行"""
    C, H, W = X.shape
    kH, kW = kernel_size
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    cols = []
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = X[:, i:i+kH, j:j+kW].reshape(-1)
            cols.append(patch)
    return np.array(cols), out_h, out_w

def col2im(cols, X_shape, kernel_size, stride=1):
    """把卷積 patch 條回原本形狀，累加梯度"""
    C, H, W = X_shape
    kH, kW = kernel_size
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

# -----------------------------
# CNN class
# -----------------------------
class SimpleCNN:
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, num_classes):
        C = in_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.pool_size = pool_size
        # Initialize weights
        self.W = np.random.randn(out_channels, C*kernel_size*kernel_size) * 0.1
        self.b = np.zeros(out_channels)
        # Linear layer after pooling
        self.num_classes = num_classes
        self.linear_W = None  # will init after knowing pooled size
        self.linear_b = None

    def forward(self, X):
        # Conv
        self.X = X
        self.cols, self.out_h, self.out_w = im2col(X, self.kernel_size)
        self.conv_out = self.cols @ self.W.T + self.b  # shape: [num_slide, out_channels]
        self.conv_out_relu = np.maximum(0, self.conv_out)  # ReLU

        # MaxPooling 2x2
        self.pool_mask = []
        pool_h, pool_w = self.pool_size, self.pool_size
        n_slide, C_out = self.conv_out_relu.shape
        self.pool_out = []
        for c in range(C_out):
            ch_out = self.conv_out_relu[:, c].reshape(self.out_h, self.out_w)
            pooled = []
            mask_ch = []
            for i in range(0, self.out_h, pool_h):
                for j in range(0, self.out_w, pool_w):
                    window = ch_out[i:i+pool_h, j:j+pool_w]
                    max_val = window.max()
                    pooled.append(max_val)
                    mask = (window == max_val)
                    mask_ch.append(mask)
            self.pool_out.append(np.array(pooled))
            self.pool_mask.append(mask_ch)
        self.pool_out = np.array(self.pool_out)  # shape: [C_out, pooled_h*pooled_w]
        
        # Flatten
        self.flat = self.pool_out.flatten()
        # Linear layer
        if self.linear_W is None:
            self.linear_W = np.random.randn(self.num_classes, self.flat.size) * 0.1
            self.linear_b = np.zeros(self.num_classes)
        self.out = self.linear_W @ self.flat + self.linear_b
        return self.out

    def backward(self, dLdy, lr=1e-3):
        # Linear backward
        dW_linear = np.outer(dLdy, self.flat)
        db_linear = dLdy
        dflat = self.linear_W.T @ dLdy

        # MaxPooling backward
        dpool = dflat.reshape(self.pool_out.shape)
        dconv_relu = np.zeros_like(self.conv_out_relu)
        pool_h, pool_w = self.pool_size, self.pool_size
        for c in range(self.conv_out_relu.shape[1]):
            ch_dpool = dpool[c]
            idx = 0
            for i in range(0, self.out_h, pool_h):
                for j in range(0, self.out_w, pool_w):
                    mask = self.pool_mask[c][idx]
                    dconv_relu[i:i+pool_h, j:j+pool_w, c] = mask * ch_dpool[idx]
                    idx += 1

        # ReLU backward
        dconv = dconv_relu * (self.conv_out > 0)

        # Conv backward
        dW = dconv.T @ self.cols
        db = dconv.sum(axis=0)
        dcols = dconv @ self.W
        dX = col2im(dcols, self.X.shape, self.kernel_size)
        
        # Update weights
        self.W -= lr * dW
        self.b -= lr * db
        self.linear_W -= lr * dW_linear
        self.linear_b -= lr * db_linear

        return dX
