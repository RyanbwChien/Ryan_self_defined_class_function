# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:37:38 2025

@author: user
"""

import smtplib

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Hotmail / Outlook SMTP 伺服器資訊
SMTP_SERVER = "smtp.live.com"
SMTP_PORT = 587

# 您的 Hotmail/Outlook 登入資訊
EMAIL_ADDRESS = "saikidd2002@hotmail.com"
EMAIL_PASSWORD = "220721623abc"

# 建立郵件內容
msg = MIMEMultipart()
msg["From"] = EMAIL_ADDRESS
msg["To"] = "saikidd2002@hotmail.com"
msg["Subject"] = "這是測試郵件"

# 郵件正文
body = "你好，這是一封來自 Python 的測試郵件！"
msg.attach(MIMEText(body, "plain"))

try:
    # 連接到 SMTP 伺服器
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # 啟用 TLS 加密
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  # 登入郵件伺服器
    server.sendmail(EMAIL_ADDRESS, "saikidd2002@hotmail.com", msg.as_string())  # 發送郵件
    server.quit()  # 關閉連線
    print("✅ 郵件發送成功！")
except Exception as e:
    print(f"❌ 郵件發送失敗：{e}")



from email.message import EmailMessage
import smtplib

sender = "saikidd2002@hotmail.com"
recipient = "saikidd2002@hotmail.com"
message = "Hello world!"

email = EmailMessage()
email["From"] = sender
email["To"] = recipient
email["Subject"] = "Test Email"
email.set_content(message)

smtp = smtplib.SMTP("smtp-mail.outlook.com", port=587)
smtp.starttls()
smtp.login(sender, "220721623abc")
smtp.sendmail(sender, recipient, email.as_string())
smtp.quit()