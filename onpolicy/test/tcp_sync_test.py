import time
from tkinter import *
from socket import *
import threading
import airsim



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
        self.port = 9005
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
        for i in range(self.env_num):
            self.send_list.append(socket(AF_INET, SOCK_STREAM))
            self.send_list[-1].connect((self.address, self.port))
            # msg = 'fuck'+str(i)
            # self.send_list[i].send(msg.encode())



    def deal_func(self, sock, addr):
        """
        处理airsim env 发送的函数以及参数请求 并存储数据
        """
        while True:
            try:
                a = time.time()
                recvdata = sock.recv(self.buffsize).decode('utf-8')
                recv_list = recvdata.split('|')
                # print(recv_list, addr)
                id = recv_list[0]
                func = 'self.airsim_client_list[' + id + '].'+ recv_list[1]
                vel_name = recv_list[2]
                args = []
                for i in range(3, len(recv_list)):
                    if recv_list[i][0] == '#':
                        break
                    args.append(recv_list[i])

                # client.eval(func)(*args)
                b = time.time()
                print("string_time:", b - a)
                data = eval(func)(*args)
                print("API_TIME:", time.time() - b)
                self.vehicle_data_recv[vel_name] = data
                # print(data)
                print(vel_name)
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
        while True:
            flag = True
            for vel in self.vehicle_data_recv:
                if self.vehicle_data_recv[vel] == None:
                    flag = False
                    break
            if flag:
                break
        print('Done')
        return True

p = Parallel_Airsim(['127.0.0.1','10.134.143.20'])

start =time.time()
for vel in p.airsim_vehicle_dict:
    p.simGetObjectPose(vel)

# for i in range(10):
#     name = 'cf'+str(101+i)
#     p.getDistanceSensorData('Distance1', name)
print("total_time:", time.time() - start)
done = p.Is_Done()


a = airsim.MultirotorClient()
b = airsim.MultirotorClient("10.134.143.20")

c = time.time()
for each in a.listVehicles():
    g = time.time()
    s = a.simGetObjectPose(each)
    print("API_TIME_AIRSIM:", time.time()-g)
for each in b.listVehicles():
    s = b.simGetObjectPose(each)
print(time.time() - c)

