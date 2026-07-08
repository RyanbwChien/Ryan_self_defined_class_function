# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:05:21 2026

@author: USER
"""

# -*- coding: utf-8 -*-
import tensorflow as tf

# ==========================================
# 1. 檢查並確認 GPU 是否有被 TensorFlow 抓到
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 成功找到 GPU: {gpus}")
    # (可選) 設定 GPU 記憶體按需分配，避免 TF 一開始就佔用全部 VRAM
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("❌ 找不到 GPU，將使用 CPU 執行。請檢查 CUDA 驅動程式。")
print("==========================================")

# 準備資料
X = tf.random.normal((1000, 6), 10, 2)
W = tf.reshape(tf.range(1, 19, 1, dtype=tf.float32), (3, 6))
B = tf.reshape(tf.range(1, 4, 1, dtype=tf.float32), (3, 1))
Y = X @ tf.transpose(W) + tf.transpose(B)

class SimpleLinear:
    def __init__(self):
        self.V = tf.Variable(tf.random.normal((3, 6), 0, 1)) 
        self.b = tf.Variable(tf.random.normal((3, 1), 0, 1))
        
    def forward(self, X):
        return X @ tf.transpose(self.V) + tf.transpose(self.b)

epochs = 10000

model = SimpleLinear()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss_fn = tf.keras.losses.MeanSquaredError()

# ==========================================
# 2. 效能加速的靈魂：加入 @tf.function (圖模式)
# ==========================================
# 加上這個裝飾器，TF 會把這個 Python 函數編譯成 C++ 級別的「靜態計算圖」直接送進 GPU 執行
# 這會讓你的 30000 次迴圈速度提升 10 倍以上！
@tf.function 
def train_step(X_data, Y_data):
    with tf.GradientTape() as tape:
        yhat = model.forward(X_data)
        loss_val = loss_fn(Y_data, yhat)
    
    grads = tape.gradient(loss_val, [model.V, model.b])
    optimizer.apply_gradients(zip(grads, [model.V, model.b]))
    return loss_val

# ==========================================
# 3. 指定在 GPU 上執行訓練迴圈 (如果有的話)
# ==========================================
# 使用 tf.device 明確告訴 TF 把運算放在 GPU 上
device_name = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device_name):
    print(f"開始訓練，使用設備: {device_name} ...")
    for e in range(epochs):
        # 呼叫編譯好的計算圖
        current_loss = train_step(X, Y)
        
        # 每 5000 次印出一下 Loss，確認有沒有卡住
        if (e + 1) % 5000 == 0:
            print(f"Epoch {e+1}/{epochs}, Loss: {current_loss.numpy():.4f}")

print("\n--- 訓練完成 ---")
print("預估的 V:\n", model.V.numpy())
print("預估的 b:\n", model.b.numpy())