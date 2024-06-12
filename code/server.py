#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import socket
import struct
# 创建一个socket套接字，该套接字还没有建立连接
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定监听端口，这里必须填本机的ip,后面6688是端口号，自选
server.bind(('192.168.1.116', 6688))
# 开始监听，并设置最大连接数
server.listen(5)
 
print(u'waiting for connect...')
# 等待连接，一旦有客户端连接后，返回一个建立了连接后的套接字和连接的客户端的IP和端口元组
connect, (host, port) = server.accept()
print(u'the client %s:%s has connected.' % (host, port))
 
while True:
    # 接受客户端的数据
    # 先接收保存有数据长度的4个byte
    rec_data_len = connect.recv(4)
    # 得到数据长度
    rec_data_len1=struct.unpack('i',rec_data_len)[0]
    # 接收得到数据长度的数据
    data=connect.recv(rec_data_len1)
    # 如果接受到客户端要quit就结束循环
    if data == b'quit' or data == b'':
        print(b'the client has quit.')
        break
    else:
        # 发送数据给客户端
        connect.send(b'your words has received.')
        print(b'the client say:' + data)
 
# 结束socket
server.close()