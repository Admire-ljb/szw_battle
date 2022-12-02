import time
from tkinter import *
from socket import *
import threading
import airsim
from multiprocessing import  Process


class Parallel_Airsim:
    def __init__(self, ip_list):
        # veichle init
        self.airsim_client_list = []
        self.airsim_vehicle_dict = {}
        # self.vehicle_data_recv_flag = {}
        self.vehicle_data_recv = {}
        for each in ip_list:
            self.airsim_client_list.append(airsim.MultirotorClient(each))
            for vel in self.airsim_client_list[-1].listVehicles():
                # self.airsim_vehicle_dict[vel] = self.airsim_client_list[-1]
                self.airsim_vehicle_dict[vel] = len(self.airsim_client_list) - 1
                # self.vehicle_data_recv_flag[vel] = False
                self.vehicle_data_recv[vel] = None

        # connect init
        self.address = '127.0.0.1'
        self.port = 9000
        self.buffsize = 1024
        self.s = socket(AF_INET, SOCK_STREAM)
        self.s.bind((self.address, self.port))
        self.s.listen(5)  # 最大连接数
        self.conn_list = []
        self.conn_dt = {}
        self.t1 = threading.Thread(target=self.server, args=(), name='rec')
        self.t1.start()
        self.env_num = len(ip_list)
        self.send_list = []
        self.finish_num = []
        for i in range(self.env_num):
            self.send_list.append(None)
        for i in range(self.env_num):
            self.send_list[i] = socket(AF_INET, SOCK_STREAM)
            self.send_list[i].connect((self.address, self.port))
            # msg = 'fuck'+str(i)
            # self.send_list[i].send(msg.encode())



    def deal_func(self, sock, addr):
        """
        处理airsim env 发送的函数以及参数请求 并存储数据
        """
        while True:
            try:
                recvdata = sock.recv(self.buffsize).decode('utf-8')
                recv_list = recvdata.split('|')
                # print(recv_list, addr)
                id = recv_list[0]
                func = 'self.airsim_client_list[' + id + '].'+ recv_list[1]
                vel_name = recv_list[2]
                args = []
                for i in range(3,len(recv_list)):
                    if recv_list[i][0] == '#':
                        break
                    args.append(recv_list[i])

                # cli_local = self.airsim_client_list[int(id)]

                # for vel in cli_local.listVehicles():
                #     t2 = time.time()
                #
                #     cli_local.simGetObjectPose(vel)
                #     print("循环单个时间", time.time() - t2)

                t1 = time.time()
                data = eval(func)(*args)
                print("并行单个时间",time.time() - t1)
                self.vehicle_data_recv[vel_name] = data
                # print(data)
                # print(vel_name)
                self.finish_num.append(None)
                if not recvdata:
                    break
            except:
                sock.close()
                print(addr, 'offline')
                _index = self.conn_list.index(addr)
                self.conn_dt.pop(addr)
                self.conn_list.pop(_index)
                break


    def server(self):
        """
        监听airsin_env tcp连接
        """
        while True:
            clientsock, clientaddress = self.s.accept()
            if clientaddress not in self.conn_list:
                self.conn_list.append(clientaddress)
                self.conn_dt[clientaddress] = clientsock
            print('connect from:', clientaddress)
            t = threading.Thread(target=self.deal_func, args=(clientsock, clientaddress))
            t.start()


    def simGetObjectPose(self, object_name):

        id = self.airsim_vehicle_dict[object_name]
        #splite by '|'
        args = str(str(id) + '|' + 'simGetObjectPose' + '|' + object_name + '|' + object_name + '|')
        msg = args.encode()
        msg_len = len(msg)
        for i in range(self.buffsize - msg_len):
            msg = msg + '#'.encode()
        self.send_list[id].send(msg)
        return

    def getDistanceSensorData(self, distance_sensor_name='', vehicle_name=''):
        """
        Args:
            distance_sensor_name (str, optional): Name of Distance Sensor to get data from, specified in settings.json
            vehicle_name (str, optional): Name of vehicle to which the sensor corresponds to
        Returns:
            DistanceSensorData:
        """
        id = self.airsim_vehicle_dict[vehicle_name]
        # splite by '|'
        args = str(str(id) + '|' + 'getDistanceSensorData' + '|' + vehicle_name + '|' + distance_sensor_name + '|' + vehicle_name + '|')
        msg = args.encode()
        msg_len = len(msg)
        for i in range(self.buffsize - msg_len):
            msg = msg + '#'.encode()
        self.send_list[id].send(msg)
        return


    def Refresh(self):
        """
        将无人机数据存储list设为none
        """
        for vel in self.vehicle_data_recv:
            self.vehicle_data_recv[vel] = None

    def Is_Done(self):
        """
        是否所有无人机均收到数据
        """
        num = len(self.vehicle_data_recv)
        while True:
            time.sleep(0.2)
            if len(self.finish_num) < num:
                continue
            else:
                break
        return True

        # while True:
        #     flag = True
        #     for vel in self.vehicle_data_recv:
        #         if self.vehicle_data_recv[vel] == None:
        #             flag = False
        #             break
        #     if flag:
        #         break
        # print('Done')
        # return True

p = Parallel_Airsim(['172.19.0.2','172.19.0.3', '172.19.0.4'])
c1 = airsim.MultirotorClient('172.19.0.2')
c2 = airsim.MultirotorClient('172.19.0.3')
c3 = airsim.MultirotorClient('172.19.0.4')
t1 = time.time()
for vel in p.airsim_vehicle_dict:
    p.simGetObjectPose(vel)


p.Is_Done()

print('done1',time.time()-t1)
t2 = time.time()
for vel in c1.listVehicles():
    s = c1.simGetObjectPose(vel)
for vel in c3.listVehicles():
    s = c2.simGetObjectPose(vel)
for vel in c3.listVehicles():
    s = c3.simGetObjectPose(vel)
print('done2', time.time()-t2)

print('fuck')