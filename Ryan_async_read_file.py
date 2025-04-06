# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 22:52:18 2025

@author: user
"""

import asyncio
import os
import time

# 自定义非同步文件读取函数
async def async_read_file(file_path):
    loop = asyncio.get_event_loop()
    # 使用 run_in_executor 将阻塞的同步文件 I/O 操作放入执行器中
    with open(file_path, mode='rb') as f:
        return await loop.run_in_executor(None, f.read)

async def load_file(file_path):
    content = await async_read_file(file_path)
    return content  # 返回文件内容
    
async def main():
    start_time = time.time()
    tasks = [load_file(i) for i in [r"D:\滙嘉健康生活科技\0. Ryan project R2_(Security C).pptx", r"D:\滙嘉健康生活科技\TIR104_Group1_反詐騙 LINEBOT服務.pdf", r"C:\Users\user\Desktop\Tibame_MySQL\上課資料\99MySQL-附錄.pdf",r"C:\Users\user\ngrok.exe"]]  # 建立所有異步任務
    results = await asyncio.gather(*tasks)  # 正確地等待所有協程完成
    for idx, content in enumerate(results, start=1):
        print(f"File {idx} content (first 100 bytes):\n{content[:100]}")  # 打印每个文件的前100个字节
    all_result = { os.path.basename(i):content for i in [r"D:\滙嘉健康生活科技\0. Ryan project R2_(Security C).pptx", r"D:\滙嘉健康生活科技\TIR104_Group1_反詐騙 LINEBOT服務.pdf", r"C:\Users\user\Desktop\Tibame_MySQL\上課資料\99MySQL-附錄.pdf",r"C:\Users\user\ngrok.exe"]}
    end_time = time.time()
    print(f"proceed time {end_time-start_time}")
    return all_result
if __name__ == "__main__":
    asyncio.run(main())
# proceed time 0.05455160140991211


# =============================================================================
# import time
# start_time = time.time()
# import asyncio
# import os
# import time
# 
# for idx, i in enumerate([r"D:\滙嘉健康生活科技\0. Ryan project R2_(Security C).pptx", r"D:\滙嘉健康生活科技\TIR104_Group1_反詐騙 LINEBOT服務.pdf", r"C:\Users\user\Desktop\Tibame_MySQL\上課資料\99MySQL-附錄.pdf",r"C:\Users\user\ngrok.exe"]):
#     with open(i, mode='rb') as f:
#         content = f.read()
#         print(f"File {idx} content (first 100 bytes):\n{content[:100]}")  # 打印每个文件的前100个字节
#         
#     all_result = { os.path.basename(i):content}
# 
# end_time = time.time()
# print(f"proceed time {end_time-start_time}")
# #proceed time 0.08861327171325684
# =============================================================================
