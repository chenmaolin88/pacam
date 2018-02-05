# coding: utf-8

import SocketServer
import socket
import threading
import base64
import sys
from biz_rule_engine import BizRuleEngine


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):
    ip = ""
    port = 0
    timeOut = 3600  # 设置超时时间变量
    det = None
    recvFlag = True  # 接收数据标识位
    End = '@@END@@'
    SHUTDOWN = '@@SHUTDOWN@@'
    bizRuleEngine = None

    def setup(self):
        self.ip = self.client_address[0].strip()  # 获取客户端的ip
        self.port = self.client_address[1]  # 获取客户端的port
        self.request.settimeout(self.timeOut)  # 对socket设置超时时间
        print(self.ip + ":" + str(self.port) + "连接到服务器！")

        self.bizRuleEngine = BizRuleEngine()  # init bizrule engine

    def recv_end(self):

        total_data = []
        data = ''
        while self.recvFlag:
            data = self.request.recv(8192)

            if (len(data) == 0):  # 说明客户端已断开
                self.recvFlag = False
                break

            if self.End in data:
                total_data.append(data[:data.find(self.End)])
                break

            if self.SHUTDOWN in data:
                self.server.shutdown()
                print("server is shutting down!")
                self.server.server_close()
                break

            total_data.append(data)
            if len(total_data) > 1:
                # check if end_of_data was split
                last_pair = total_data[-2] + total_data[-1]
                if self.End in last_pair:
                    total_data[-2] = last_pair[:last_pair.find(self.End)]
                    total_data.pop()
                    break
        return ''.join(total_data)

    def handle(self):

        while self.recvFlag:  # while循环

            try:
                total_data = self.recv_end()

            except socket.timeout:  # 如果接收超时会抛出socket.timeout异常
                print(self.ip + ":" + str(self.port) + "接收超时！即将断开连接！")
                break  # 记得跳出while循环

            if total_data:  # 判断是否接收到数据
                result = self.bizRuleEngine.detect_water(total_data) # actual detect invoke
                response = bytes("{}".format(result))
                self.request.sendall(response)

    def finish(self):
        print(self.ip + ":" + str(self.port) + "断开连接！")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("参数有误,正确的调用方式:如 python app.py 11.240.98.222 9988")
        exit(1)

    server = SocketServer.ThreadingTCPServer((sys.argv[1], int(sys.argv[2])), ThreadedTCPRequestHandler)
    server.daemon_threads = True  # 若主线程中止，则其他线程也立即停止
    server.serve_forever()
