# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:11:55 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


# L1 = nn.Linear(6, 3) 
# 建立初始化參數的實體物件，然後有linear DNN forward 函數
# 一旦做了 nn.Linear(6, 3)(x)，因為TORCH的每個變數都是一個獨立的TENSOR，
# 依但做了TORCH物件的"運算"，運算就會將結果包成一個獨立的TENSOR，裡面紀錄有前一次PARENT透過甚麼運算組成，以及對PARENT的微分
# 以LINEAR為例就是 matmul 運算 C=A@B

class torch_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 3)
        self.linear2 = nn.Linear(3, 2)
    def forward(self,x):
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = self.linear2(x)
  
        return x
    
model =  torch_NN()

x = torch.normal(0,1,(100,6))
w1 = torch.arange(0, 3*6, 1, dtype=torch.float32).reshape(3,6)
w2 = torch.arange(0, 2*3, 1, dtype=torch.float32).reshape(2,3)

y = (x@w1.T)@w2.T

model(x)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.3)

# =============================================================================
# 原因： 傳統的清空是把梯度變成全是 0 的 Tensor (也就是 tensor.grad = 0)，這會佔用記憶體並消耗運算資源。加上 set_to_none=True 後，會直接把記憶體釋放掉 (tensor.grad = None)，下次 backward 時再重新分配，這樣訓練速度會稍微快一點點，也會節省一點點記憶體！
# (註：你程式碼中使用的 F.sigmoid(x) 在較新版本的 PyTorch 中已建議改用 torch.sigmoid(x)，除此之外，作為練習程式碼邏輯很清晰！)
# 
# =============================================================================
epochs = 10000
import tqdm
# =============================================================================
# 一般情況下：
# 你每一次迴圈都呼叫 zero_grad 
# →
# →
#  backward 
# →
# →
#  step。
# 硬體不夠，需要跨 Batch 累加時：
# 你可以呼叫 zero_grad 
# →
# →
#  backward 
# →
# →
#  backward 
# →
# →
#  backward 
# →
# →
#  step。
# 你的理解非常透徹，這個設計的確就是為了「跨越單次 backward，讓多次運算的梯度能夠無縫接軌地加在一起」！
# 
# =============================================================================


for e in range(epochs):
    optimizer.zero_grad()
    yhat = model(x)
    loss = criterion(y,yhat)
    loss.backward()
    optimizer.step()
    if e%50==0:
        print(f"current loss {loss}")

model(x)










    