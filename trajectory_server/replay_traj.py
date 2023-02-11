from socket import *
import threading
import time
import numpy as np
import pandas as pd
from read_json import read_json

class Sendmsg:
    def __init__(self, socket_server):
        self.buffsize = 2048
        self.s = socket_server
        self.conn_list = []
        self.conn_dt = {}
        self.connected_port = []
        self.drone_num = 300
        self.recs()
        # self.t1 = threading.Thread(target=self.recs, args=(), name='rec')
        # self.t2 = threading.Thread(target=self.sds, args=(), name='send')

    def tcplink(self, sock, addr):
        while True:
            try:
                recvdata = sock.recv(self.buffsize).decode('utf-8')
                print(recvdata, addr)
                if not recvdata:
                    break
            except:
                sock.close()
                print(addr, 'offline')
                _index = self.conn_list.index(addr)
                # gui.listBox.delete(_index)
                self.conn_dt.pop(addr)
                self.conn_list.pop(_index)
                break

    def recs(self):
        # while True:
        while len(self.conn_list) < self.drone_num:
            clientsock, clientaddress = self.s.accept()
            if clientaddress[1] not in self.connected_port:
                self.conn_list.append(clientaddress)
                self.conn_dt[clientaddress] = clientsock
                self.connected_port.append(clientaddress[1])
                print('connect from:', clientaddress)






if __name__ == '__main__':
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('127.0.0.1', 9899))
    s.listen(200)  # 最大连接数
    c1 = Sendmsg(s)
    att, defender = read_json("buaa_0.json")
    att_len = len(att)
    defender_len = len(defender)
    time.sleep(3)
    for timestamp in range(att[0].shape[0]):
        cnt = 0
        for index_att in range(att_len):
            x = att[index_att][timestamp][0]*20
            y = att[index_att][timestamp][1]*20
            z = 500
            msg = '_' + str(x * 100) + '_' + str(y * 100) + '_' + str(z * 10) + '_' + str(0) + '_' + str(
                0) + '_' + str(0) + '_' + '(R=0,G=1)'
            c1.conn_dt[c1.conn_list[cnt]].sendall(msg.encode('utf-8'))
            cnt += 1
        for index_def in range(defender_len):
            x = defender[index_def][timestamp][0]*20
            y = defender[index_def][timestamp][1]*20
            z = 500
            msg = '_' + str(x * 100) + '_' + str(y * 100) + '_' + str(z * 10) + '_' + str(0) + '_' + str(
                0) + '_' + str(0) + '_' + '(R=1)'

            c1.conn_dt[c1.conn_list[cnt]].sendall(msg.encode('utf-8'))
            cnt += 1
        time.sleep(0.25)