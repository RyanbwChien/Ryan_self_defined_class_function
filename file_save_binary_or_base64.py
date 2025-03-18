# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 02:46:59 2025

@author: user
"""

import base64

with open(r"C:\Users\user\Desktop\BERT2.png", mode='rb') as f:
    result = f.read()
   
type(result)
# result.decode('utf-8', errors='ignore')

# 2進位資料不需要再編碼
b64str = base64.b64encode(result).decode('utf-8')

base64.b64decode(b64str.encode('utf-8'))
base64.b64decode(b64str)
base64.decodebytes(b64str.encode('utf-8'))

# =============================================================================
# 是的，base64.b64decode(b64str) 和 base64.decodebytes(b64str.encode('utf-8')) 的結果是相同的。
# 
# 這是因為：
# 
# base64.b64decode() 是用來解碼 base64 編碼字串的標準函式，接受的輸入是 bytes 或是 byte-like 物件。
# base64.decodebytes() 是 b64decode() 的別名，並且也接受類似的 bytes 物件作為參數，主要是進行 base64 解碼。
# 所以，b64str.encode('utf-8') 會將 b64str 字串轉換為 bytes，兩個函式 b64decode() 和 decodebytes() 都能正確處理這個 bytes 物件，並返回相同的結果。
# 
# 總結來說，這兩個方法本質上是等價的，會返回相同的解碼結果。
# =============================================================================



# base64.b64encode(result) 會返回一個 bytes 物件（Base64 編碼後的二進位數據）。
# 若要將這個 Base64 編碼結果用作字符串，必須使用 .decode('utf-8') 將其轉換為普通字符串（str 類型）。
# base64.b64decode(b64str) 接受的是一個字符串形式的 Base64 編碼結果，它會自動處理和解碼。

# 但其實，對 b64str 使用 .encode('utf-8') 是多餘的，因為 Base64 字符串本來就是由可以被 UTF-8 解碼的字符組成。因此，你可以直接對 b64str 使用 base64.b64decode()，它會正常工作。

# 將每個 byte 轉換為二進位字串，並用空格分隔
binary_representation = ' '.join(f'{byte:08b}' for byte in "ABC".encode('ascii') )

print(binary_representation)

with open(r"C:\Users\user\Desktop\BERT2_trans.png", mode='wb') as f:
    f.write(result)