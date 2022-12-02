from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from threading import current_thread
import time,\
    random
import airsim

class Parallel_Airsim:
    def __init__(self, poolnum, ip_list):
        self.pool = ThreadPoolExecutor(poolnum)
        self.objs = []
        self.airsim_client_list = []
        self.airsim_vehicle_dict = {}
        self.vehicle_data_recv_flag = {}
        self.vehicle_data_recv = {}
        for each in ip_list:
            self.airsim_client_list.append(airsim.MultirotorClient(each))
            self.airsim_client_list[-1].ip = each
            for vel in self.airsim_client_list[-1].listVehicles():
                self.airsim_vehicle_dict[vel] = self.airsim_client_list[-1]
                self.vehicle_data_recv_flag[vel] = False
                self.vehicle_data_recv[vel] = None


    def task(self, func, args, vehicle_name_list):
        result = []
        for index, arg in enumerate(args):
            t1 = time.time()
            tmp = func(arg)
            result.append(tmp)
            print("并行单个时间",time.time() - t1)
            self.vehicle_data_recv_flag[vehicle_name_list[index]] = True
            self.vehicle_data_recv[vehicle_name_list[index]] = tmp
        return result


    def simGetObjectPose(self, args ,vehicle_name_list):
        obj=self.pool.submit(self.task,  self.airsim_vehicle_dict[vehicle_name_list[0]].simGetObjectPose, args, vehicle_name_list)
        # objs.append(obj)

    def reset_veh_flag(self):
        for i in self.vehicle_data_recv_flag:
            self.vehicle_data_recv_flag[i] = False

a = Parallel_Airsim(2, ['127.0.0.1','10.134.143.20'])
v1l = a.airsim_client_list[0].listVehicles()
v2l = a.airsim_client_list[1].listVehicles()
for vel in v1l:
    a.airsim_vehicle_dict[vel].enableApiControl(True, vehicle_name=vel)
for vel in v2l:
    a.airsim_vehicle_dict[vel].enableApiControl(True, vehicle_name=vel)
t1 = time.time()
a.simGetObjectPose(v1l,v1l)
a.simGetObjectPose(v2l,v2l)
while True:
    flag = True
    for i in a.vehicle_data_recv_flag:
        flag = flag and a.vehicle_data_recv_flag[i]
    if flag:
        a.reset_veh_flag()
        break
print(time.time() - t1)

a = airsim.MultirotorClient()
b = airsim.MultirotorClient("172.19.0.2")

c = time.time()
for each in a.listVehicles():
    g = time.time()
    s = a.simGetObjectPose(each)
    print("API_TIME_AIRSIM:", time.time()-g)
for each in b.listVehicles():
    s = b.simGetObjectPose(each)
print(time.time() - c)
# for i in a.vehicle_data_recv:
#     print(a.vehicle_data_recv[i])
