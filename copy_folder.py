# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:59:40 2025

@author: user
"""

import shutil
import os

src = r"C:\Users\user\Desktop\SRC"
dst = r"C:\Users\user\Desktop\DST"



# 從 Python 3.8 開始，shutil.copytree() 支援 dirs_exist_ok=True 參數，這樣即使目標資料夾已經存在，也能正常執行複製，而不會拋出錯誤。
# shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_folder(src, dst):
    for content in os.listdir(src):
        abspath = os.path.join(src, content)
        if os.path.isdir(abspath):
            if os.path.exists(os.path.join(dst,content)):
                shutil.rmtree(os.path.join(dst,content))
            
            shutil.copytree(abspath, os.path.join(dst,content))
        else:
            shutil.copy2(abspath,dst)
        
        
        
# =============================================================================
# import os
# import shutil
# 
# src = r"C:\Users\user\Desktop\SRC"
# dst = r"C:\Users\user\Desktop\DST"
# 
# # 遍歷來源資料夾中的檔案和資料夾
# for content in os.listdir(src):
#     abspath = os.path.join(src, content)
#     dst_abspath = os.path.join(dst, content)
#     
#     if os.path.isdir(abspath):
#         # 如果是資料夾，且目標資料夾中沒有這個資料夾，則複製
#         if not os.path.exists(dst_abspath):
#             shutil.copytree(abspath, dst_abspath)
#         else:
#             # 如果目標資料夾中已經有相同資料夾，繼續處理裡面的檔案
#             for sub_content in os.listdir(abspath):
#                 sub_src = os.path.join(abspath, sub_content)
#                 sub_dst = os.path.join(dst_abspath, sub_content)
#                 
#                 # 如果檔案存在，則覆蓋
#                 if os.path.exists(sub_dst):
#                     if os.path.isdir(sub_src):
#                         shutil.rmtree(sub_dst)  # 如果是資料夾，先刪除
#                         shutil.copytree(sub_src, sub_dst)
#                     else:
#                         os.remove(sub_dst)  # 如果是檔案，刪除並覆蓋
#                         shutil.copy2(sub_src, sub_dst)
#                 else:
#                     # 若目標資料夾中沒有這個檔案或資料夾，則直接複製
#                     if os.path.isdir(sub_src):
#                         shutil.copytree(sub_src, sub_dst)
#                     else:
#                         shutil.copy2(sub_src, sub_dst)
# 
#     else:
#         # 如果是檔案，檢查目標資料夾中是否已經有這個檔案，若有則覆蓋
#         if os.path.exists(dst_abspath):
#             os.remove(dst_abspath)  # 刪除並覆蓋檔案
#         shutil.copy2(abspath, dst_abspath)
# =============================================================================
