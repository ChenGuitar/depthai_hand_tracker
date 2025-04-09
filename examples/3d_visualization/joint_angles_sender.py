#!/usr/bin/env python3

import socket
import json
import threading
import time

class JointAnglesSender:
    def __init__(self, server_ip='172.30.83.97', server_port=12346):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        self.reconnect_attempts = 0
        self.MAX_RECONNECT_ATTEMPTS = 3
        self.connected = False
        
        # 在新线程中连接服务器
        threading.Thread(target=self.connect_to_server).start()

    def connect_to_server(self):
        try:
            if self.client_socket:
                self.client_socket.close()
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5)  # 设置超时时间
            self.client_socket.connect((self.server_ip, self.server_port))
            print(f"已连接到关节角度服务器 {self.server_ip}:{self.server_port}")
            self.reconnect_attempts = 0  # 重置重连次数
            self.connected = True
        except Exception as e:
            print(f"连接关节角度服务器失败: {e}")
            self.client_socket = None
            self.connected = False
            if self.reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
                self.reconnect_attempts += 1
                print(f"尝试重新连接 ({self.reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})...")
                time.sleep(2)  # 等待2秒后重试
                self.connect_to_server()

    def send_joint_angles(self, hand_type, joint_angles):
        if not self.connected or not self.client_socket:
            return

        try:
            data = {
                'hand': hand_type,
                'joint_angles': joint_angles,
                'timestamp': time.time()
            }
            json_data = json.dumps(data)
            self.client_socket.send(json_data.encode('utf-8'))
        except Exception as e:
            print(f"发送关节角度数据失败: {e}")
            self.client_socket = None
            self.connected = False
            # 尝试重新连接
            self.connect_to_server()

    def close(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.connected = False 