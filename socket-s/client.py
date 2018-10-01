# !/usr/bin/env python
# -*- coding:utf-8 -*-

import socket

ip_port = ('127.0.0.1', 9997)
sk = socket.socket()
sk.connect(ip_port)
sk.settimeout(5)
data = sk.recv(1024).decode()
print('服务器:', data)
while True:
    inp = input('你:').strip()
    if not inp:
        continue

    sk.sendall(inp.encode())

    if inp == 'exit':
        print("谢谢使用，再见！")
        break
    data = sk.recv(1024).decode()
    print('服务器:', data)
sk.close()