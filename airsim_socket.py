from socket import *
import threading
import airsim
import time
import numpy as np

connected_port = []

class Airsimsocet
class Sendmsg:
    def __init__(self, client_ip, socket_server):
        self.client = airsim.MultirotorClient(client_ip)
        self.buffsize = 2048
        self.s = socket_server
        # self.s = socket(AF_INET, SOCK_STREAM)
        # self.s.bind((self.address, port))
        # self.s.listen(5)  # 最大连接数
        self.conn_list = []
        self.conn_dt = {}
        self.drone_num = len(self.client.listVehicles())
        self.drone_list = self.client.listVehicles()
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
                # gui.listBox.insert(END, clientaddress)
                print('connect from:', clientaddress)
                # 在这里创建线程，就可以每次都将socket进行保持
                # t = threading.Thread(target=self.tcplink, args=(clientsock, clientaddress))
                # t.start()

    def sds(self):
        while True:
            for i in range(len(self.conn_dt)):
                pos = self.client.simGetObjectPose(self.drone_list[i])
                dpos = pos.position
                eular = np.rad2deg(airsim.to_eularian_angles(pos.orientation))
                msg = str(dpos.x_val * 100) + '_' + str(dpos.y_val * 100) + '_' + str(-dpos.z_val * 100) + '_' + \
                      str(eular[0]) + '_' + str(eular[1]) + '_' + str(eular[2])
                # print(msg)
                self.conn_dt[self.conn_list[i]].sendall(msg.encode('utf-8'))
            time.sleep(1)

    def run(self):
        # self.t1.start()
        self.t2.start()


if __name__ == '__main__':
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('0.0.0.0', 9999))
    s.listen(200)  # 最大连接数
    c1 = Sendmsg('192.168.0.107', s)
    # c1.run()
    c2 = Sendmsg('192.168.1.100', s)
    # c1.run()
    c2.run()
