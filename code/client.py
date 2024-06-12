#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import socket
import struct
 
# 创建一个socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 主动去连接局域网内IP为192.168.27.238，端口为6666的进程,即服务器端设置的ip和端口
client.connect(('192.168.1.116', 6688))
 
while True:
    # 接受控制台的输入
    data =input()
    # 对数据进行编码格式转换，不然报错
    data = data.encode('utf-8')
    # 如果输入quit则退出连接
    if data == b'quit':
        print(b'connect quit.')
        break
    else:
        #确定要发送数据的长度
        data_len=len(data)
        #将这个长度打包成特定byte的数据，参数'i'表明4个byte
        send_data_len=struct.pack('i',data_len)
        # 发送数据长度
        client.send(send_data_len)
        #发送数据
        client.send(data)
        # 接收服务端的反馈数据
        # 先接收保存有数据长度的4个byte
        rec_data_len = client.recv(4)
        # 得到数据长度
        rec_data_len1=struct.unpack('i',rec_data_len)[0]
        # 接收得到数据长度的数据
        rec_data=client.recv(rec_data_len1)
        print(b'form server receive:' + rec_data)
 
# 发送数据告诉服务器退出连接
client.sendall(b'quit')
client.close()