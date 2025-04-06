# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 20:13:39 2025

@author: user
"""

from typing import Literal

def time_gap(time_format:Literal["sec","minute","hour"]):
    def time_gap_inner(fun):
        def warrap(*arg):
            import time
            start = time.time()
            fun(*arg ) # 自己函數本身也會印出值
            end = time.time()
            proc_time = end - start
            match time_format:
                case "sec":
                    print (f"run {proc_time} sec")
                case "minute":
                    print (f"run {proc_time/60:.5f} minute")
                case "hour":
                    print (f"run {proc_time/60/60:.5f} hour")
        return warrap
    return time_gap_inner
        


if __name__ == "__main__":
    @time_gap("minute")
    def anyfunc(x):
        import time
        time.sleep(x)
        print(x)
    
    anyfunc(3)
