from onpolicy.envs.airsim_envs.airsim_socket import CustomAirsimClient
from socket import *
import pandas as pd
import time
import airsim
import threading
import numpy as np
find_count_num = 0
class RecurrentList_1(object):
    """循环列表"""
    def __init__(self, data):
        self.data = data
        self.ptr = 0
        self.length = len(self.data)

    def next(self):
        self.ptr += 2
        if find_count_num == 6:
            return self.data[0], self.data[1]
        if self.ptr >= self.length:
            # self.ptr = 0
            return self.data[0], self.data[1]
        return self.data[self.ptr], self.data[self.ptr+1]


class LinkedNode(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_pre = None
        self.x_next = None
        self.y_pre = None
        self.y_next = None



class MtspPolicy:
    def __init__(self, file_name, ip, port):
        self.find_all = False
        _ = pd.read_table(file_name, sep='\t', header=None)
        self.patrol_list = []
        self.name = ['carla_3', 'claudia_2', 'eric_2', 'manuel_2', 'nathan_2', 'sophia_2']
        # self.people_pos = [[-59377.523438, 50322.0], [-138381.0, 84595.0], [28271.0, -106163.0],
        #                    [-122676.9375, -63717.0], [-160053.0, 47145.0], [-100104.0, -53483.0]]
        # self.people_pos = [[-122400, -290100], [-81500, -280300], [-38200, -274300],
        #                    [-3100, -269600], [47800, -256100], [-143300, -226200]]
        self.people_pos = [[-85720.585938, -168809.484375], [-36618.894531, -170253.390625], [-79935.140625, -58067.804688],
                           [-31564.519531, -118898.703125], [-25545.892578, -14972.765625], [30184.84375, -113762.21875]]

        for each in _[0]:
            self.patrol_list.append(each.split(" "))
        for i in range(len(self.name)):
            self.patrol_list[i][2] = self.people_pos[i][0]
            self.patrol_list[i][3] = self.people_pos[i][1]
        self.socket_server = socket(AF_INET, SOCK_STREAM)
        self.socket_server.bind(('127.0.0.1', port))
        self.socket_server.listen(200)  # 最大连接数
        self.client = CustomAirsimClient(ip, self.socket_server, plot_flag=False)
        self.height = -50
        self.velocity = 50
        self.destroy_distance = 40
        time.sleep(2)
        self.get_caught = {}

        self.mission_points = {}
        for mission_point, bp_name in zip(self.patrol_list, self.client.vehicle_dict):
            self.mission_points[bp_name] = RecurrentList_1(np.array(mission_point, dtype=float)/100)
            self.get_caught[bp_name] = 0
        time.sleep(1)
        self.remained_vehicle = self.client.listVehicles()
        self.reset()
        for i in range(len(self.name)):
            pos = self.people_pos[i]
            self.plot_target([pos[0]/100, pos[1]/100])
            p0 = airsim.Vector3r(pos[0]/100, pos[1]/100, 0)
            p = airsim.Pose(position_val=p0)
            self.client.simSetObjectPose(self.name[i], p)

    def reset(self):
        for i in range(5):
            for bp_name in self.mission_points:
                pos = airsim.Pose()
                pos.position.x_val = self.mission_points[bp_name].data[0]
                pos.position.y_val = self.mission_points[bp_name].data[1]
                pos.position.z_val = self.height
                self.client.simSetVehiclePose(pos, ignore_collision=True, vehicle_name=bp_name)
                self.client.enableApiControl(True, vehicle_name=bp_name)
            cnt = 0
            for bp_name in self.name:
                pos = airsim.Pose()
                pos.position.x_val = self.people_pos[cnt][0] / 100
                pos.position.y_val = self.people_pos[cnt][1] / 100
                pos.position.z_val = self.height
                self.client.simSetObjectPose(bp_name, pos)
                cnt += 1
    def fly_patrol(self, bp_name):
        patrol_client = airsim.MultirotorClient(self.client.vehicle_dict[bp_name].client.ip,
                                                self.client.vehicle_dict[bp_name].client.port)
        airsim_name = self.client.vehicle_dict[bp_name].airsim_name
        while True:
            if not self.get_caught[bp_name]:
                x, y = self.mission_points[bp_name].next()

                _ = patrol_client.moveToPositionAsync(x,
                                                      y,
                                                      self.height, self.velocity, vehicle_name=airsim_name,
                                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(is_rate=False))
                _.join()
                global find_count_num
                for pos in self.people_pos:
                    if (x == pos[0]/100) & (y == pos[1]/100):
                        find_count_num = find_count_num + 1
                        self.plot_target([pos[0] / 100, pos[1] / 100], color=[0.0, 1.0, 0.0, 1.0])
                        if find_count_num == 6:
                            self.find_all = True
                        break
            else:
                x, y = self.mission_points[bp_name].data[0], self.mission_points[bp_name].data[1]
                _ = patrol_client.moveToPositionAsync(x,
                                                      y,
                                                      self.height, self.velocity/5, vehicle_name=airsim_name,
                                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(is_rate=False))
                time.sleep(1)

    def fly_detect(self, bp_name):
        cnt = 0
        while True:
            _ = self.client.get_info(bp_name)
            print(_)
            if cnt > 100:
                self.client.enableApiControl(False, bp_name)
            time.sleep(1)
            cnt += 1

    def fly_run(self):
        fly_list = []
        for each in self.mission_points:
            fly_list.append(threading.Thread(target=self.fly_patrol, args=[each]))
            fly_list[-1].start()

        # test_list = []
        # for each in self.start:
        #     test_list.append(threading.Thread(target=self.fly_detect, args=[each]))
        #     test_list[-1].start()

    def get_remain_pose(self):
        pos = []
        for each in self.remained_vehicle:
            pos = self.client.simGetObjectPose(each)
        return np.array(pos)

    def destroy_vehicle(self, bp_name):
        tmp_client = airsim.MultirotorClient(self.client.vehicle_dict[bp_name].client.ip)
        airsim_name = self.client.vehicle_dict[bp_name].airsim_name
        pose = tmp_client.simGetObjectPose(airsim_name)
        pose.position.x_val = 9999
        pose.position.y_val = 9999
        pose.position.z_val = -9999
        for i in range(10):
            tmp_client.simSetVehiclePose(pose, True, vehicle_name=airsim_name)
        self.remained_vehicle.remove(bp_name)

    def plot_target(self, pos, color=None):
        if color is None:
            color = [1.0, 0.0, 0.0, 1.0]
        a = [airsim.Vector3r(pos[0] - 20, pos[1], -50)]
        b = [airsim.Vector3r(pos[0], pos[1] - 20, -50)]
        c = [airsim.Vector3r(pos[0] + 20, pos[1], -50)]
        d = [airsim.Vector3r(pos[0], pos[1] + 20, -50)]
        self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=150.0, duration=0.2,
                               color_rgba=color,
                               is_persistent=True)

if __name__ == "__main__":
    a = MtspPolicy("road_mtsp_50_200.txt", ['127.0.0.1:41451'], 9699)
    time.sleep(1)
    a.fly_run()
    while True:
        if a.find_all:
            print('all people find')
        time.sleep(1)