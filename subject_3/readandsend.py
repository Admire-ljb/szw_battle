from socket import *
import threading
import time
import numpy as np
import pandas as pd

connected_port = []


UAV_Red_Att = pd.read_table('UAV_Red_Att.txt', sep='\t',engine='python')
UAV_Red_pos = pd.read_table('UAV_Red_pos.txt', sep='\t',engine='python')
data = pd.concat([UAV_Red_pos,UAV_Red_Att],axis = 1, join='outer')
np_data = np.array(data)
tmp = np.array(['-218.372 483.365 22.367 ', '0.000 4.811 -232.331 '])
np_data = np.vstack((tmp,np_data))


class Sendmsg:
    def __init__(self, socket_server):
        self.buffsize = 2048
        self.s = socket_server
        self.conn_list = []
        self.conn_dt = {}
        self.drone_num = 40
        self.recs()
        # self.t1 = threading.Thread(target=self.recs, args=(), name='rec')
        self.t2 = threading.Thread(target=self.sds, args=(), name='send')

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
            if clientaddress[1] not in connected_port:
                self.conn_list.append(clientaddress)
                self.conn_dt[clientaddress] = clientsock
                connected_port.append(clientaddress[1])
                print('connect from:', clientaddress)

    def sds(self):
        k = 0
        while True:
            for i in range(len(self.conn_list)):
                pos = np_data[i*2601+k][0].split(" ")
                rpy = np_data[i*2601+k][1].split(" ")
                msg = '_' + str(float(pos[0])*100) + '_' + str(-float(pos[1])*100) \
                      + '_' + str(float(pos[2])*100) + '_' + \
                      rpy[0] + '_' + rpy[1] + '_' + str(float(rpy[2]))
                # print(msg)
                self.conn_dt[self.conn_list[i]].sendall(msg.encode('utf-8'))
                # i += 1
                # if i % 40 == 0:
                #     i = 0
            time.sleep(0.01)
            k += 1


    def run(self):
        self.t2.start()


if __name__ == '__main__':
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('0.0.0.0', 9000))
    s.listen(200)  # 最大连接数
    c1 = Sendmsg(s)
    time.sleep(2)
    c1.run()
