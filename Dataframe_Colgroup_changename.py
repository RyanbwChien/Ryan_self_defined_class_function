# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:46:03 2025

@author: user
"""

import pandas as pd
import openpyxl
import string

path = r"D:\Ryan_Personal_WebAPP\pages\VCAR Report_(Security C) - 複製.xls"

data = pd.read_excel(path)
Vcode_group = data["Customer Code"].unique()
label = list(range(len(Vcode_group)))
label_Vcode_group = { i:"Customer_" + str(j) for i,j in zip(Vcode_group,label)}

N_Vendor_Code = data["Vendor Code"].apply(lambda x:  label_Vcode_group[x])

data["Vendor Code"].apply(lambda x:  label_Vcode_group[x])


dmode_group = data["Defect Mode"].unique()
dmode_label = list(range(len(dmode_group)))
dmode_label_dmode_group = { i:"Mode_" + str(j) for i,j in zip(dmode_group,dmode_label )}

N_Defectr_Code = data["Defect Mode"].apply(lambda x:  dmode_label_dmode_group[x])
