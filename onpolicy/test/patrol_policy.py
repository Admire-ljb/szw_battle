from onpolicy.envs.airsim_envs.airsim_socket import CustomAirsimClient
from socket import *
import pandas as pd
import time
import airsim
import threading
import numpy as np

class SortedBPname:
    def __init__(self, bp_name, x, y):
        self.x = x
        self.y = y
        self.bp_name = bp_name
        self.index = None
        self.next_x_friend = None
        self.pre_x_friend = None
        self.next_y_friend = None
        self.pre_y_friend = None
        self.next_x_enemy = None
        self.pre_x_enemy = None
        self.next_y_enemy = None
        self.pre_y_enemy = None
        dt = np.dtype([('name', object), ('dis', float)])
        self.distance_matrix = np.zeros(100, dtype=dt)
        for each in self.distance_matrix:
            each[0] = '0'

    def first_sort_friend(self, position_dict, sorted_bp_dict):
        # x = position_dict[self.bp_name][0]
        # y = position_dict[self.bp_name][1]
        pre_x = self.x - 10000
        pre_y = self.y - 10000
        next_x = self.x + 10000
        next_y = self.y + 10000
        for bp_name in position_dict:
            if self.index == sorted_bp_dict[bp_name].index:
                continue
            if self.x == position_dict[bp_name][0]:
                if not self.pre_x_friend or self.pre_x_friend.index < sorted_bp_dict[bp_name].index:
                    if sorted_bp_dict[bp_name].index < self.index:
                        self.pre_x_friend = sorted_bp_dict[bp_name]
                        pre_x = self.x
                if not self.next_x_friend or self.next_x_friend.index > sorted_bp_dict[bp_name].index:
                    if sorted_bp_dict[bp_name].index > self.index:
                        self.next_x_friend = sorted_bp_dict[bp_name]
                        next_x = self.x
            elif self.x < position_dict[bp_name][0] < next_x:
                next_x = position_dict[bp_name][0]
                self.next_x_friend = sorted_bp_dict[bp_name]
            elif pre_x < position_dict[bp_name][0] < self.x:
                pre_x = position_dict[bp_name][0]
                self.pre_x_friend = sorted_bp_dict[bp_name]
            elif pre_x == position_dict[bp_name][0]:
                if self.pre_x_friend.index < sorted_bp_dict[bp_name].index:
                    self.pre_x_friend = sorted_bp_dict[bp_name]
            elif next_x == position_dict[bp_name][0]:
                if self.next_x_friend.index > sorted_bp_dict[bp_name].index:
                    self.next_x_friend = sorted_bp_dict[bp_name]
            if self.y < position_dict[bp_name][1] < next_y:
                next_y = position_dict[bp_name][1]
                self.next_y_friend = sorted_bp_dict[bp_name]
            elif pre_y < position_dict[bp_name][1] < self.y:
                pre_y = position_dict[bp_name][1]
                self.pre_y_friend = sorted_bp_dict[bp_name]

    def first_sort_enemy(self, position_dict, sorted_bp_dict):
        pre_x = self.x - 1000
        pre_y = self.y - 1000
        next_x = self.x + 1000
        next_y = self.y + 1000
        for bp_name in position_dict:
            if self.x < position_dict[bp_name][0] < next_x:
                next_x = position_dict[bp_name][0]
                self.next_x_enemy = sorted_bp_dict[bp_name]
            elif pre_x < position_dict[bp_name][0] < self.x:
                pre_x = position_dict[bp_name][0]
                self.pre_x_enemy = sorted_bp_dict[bp_name]
            if self.y < position_dict[bp_name][1] < next_y:
                next_y = position_dict[bp_name][1]
                self.next_y_enemy = sorted_bp_dict[bp_name]
            elif pre_y < position_dict[bp_name][1] < self.y:
                pre_y = position_dict[bp_name][1]
                self.pre_y_enemy = sorted_bp_dict[bp_name]

    def resort_friend(self):
        if self.pre_x_friend and self.pre_x_friend.x > self.x:
            pre_ = self.pre_x_friend
            if self.pre_x_friend.pre_x_friend:
                self.pre_x_friend = pre_.pre_x_friend
                pre_.pre_x_friend.next_x_friend = self
            else:
                self.pre_x_friend = None
            pre_.pre_x_friend = self
            if self.next_x_friend:
                pre_.next_x_friend = self.next_x_friend
                self.next_x_friend.pre_x_friend = pre_
            else:
                pre_.next_x_friend = None
            self.next_x_friend = pre_

        if self.next_x_friend and self.next_x_friend.x < self.x:
            next_ = self.next_x_friend
            if self.next_x_friend.next_x_friend:
                self.next_x_friend = next_.next_x_friend
                next_.next_x_friend.pre_x_friend = self
            else:
                self.next_x_friend = None
            next_.next_x_friend = self
            if self.pre_x_friend:
                next_.pre_x_friend = self.pre_x_friend
                self.pre_x_friend.next_x_friend = next_
            else:
                next_.next_x_friend = None
            self.pre_x_friend = next_

        if self.pre_y_friend and self.pre_y_friend.y > self.y:
            pre_ = self.pre_y_friend
            if self.pre_y_friend.pre_y_friend:
                self.pre_y_friend = pre_.pre_y_friend
                pre_.pre_y_friend.next_y_friend = self
            else:
                self.pre_y_friend = None
            pre_.pre_y_friend = self
            if self.next_y_friend:
                pre_.next_y_friend = self.next_y_friend
                self.next_y_friend.pre_y_friend = pre_
            else:
                pre_.next_y_friend = None
            self.next_y_friend = pre_

        if self.next_y_friend and self.next_y_friend.y < self.y:
            next_ = self.next_y_friend
            if self.next_y_friend.next_y_friend:
                self.next_y_friend = next_.next_y_friend
                next_.next_y_friend.pre_x_friend = self
            else:
                self.next_y_friend = None
            next_.next_y_friend = self
            if self.pre_y_friend:
                next_.pre_y_friend = self.pre_y_friend
                self.pre_y_friend.next_y_friend = next_
            else:
                next_.next_y_friend = None
            self.pre_x_friend = next_

    def resort_enemy(self):
        if self.next_x_enemy and self.next_x_enemy.x < self.x:
            next_ = self.next_x_enemy.next_x_friend
            self.pre_x_enemy = self.next_x_enemy
            self.next_x_enemy = next_
        if self.next_y_enemy and self.next_y_enemy.y < self.y:
            next_ = self.next_y_enemy.next_y_friend
            self.pre_y_enemy = self.next_y_enemy
            self.next_y_enemy = next_

        if self.pre_x_enemy and self.pre_x_enemy.x > self.x:
            pre_ = self.pre_x_enemy.pre_x_friend
            self.next_x_enemy = self.pre_x_enemy
            self.pre_x_enemy = pre_
        if self.pre_y_enemy and self.pre_y_enemy.y < self.y:
            pre_ = self.pre_y_enemy.pre_y_friend
            self.next_y_enemy = self.pre_y_enemy
            self.pre_y_enemy = pre_

    def update_distance_matrix(self, position_dic):
        self.x = position_dic[self.bp_name][0]
        self.y = position_dic[self.bp_name][1]
        cnt = 0
        for bp_name in position_dic:
            distance = np.sqrt(np.power(position_dic[bp_name][0] - self.x, 2) + np.power(position_dic[bp_name][1] - self.y, 2))
            self.distance_matrix[cnt][0] = bp_name
            self.distance_matrix[cnt][1] = distance
            cnt += 1
        self.distance_matrix.sort(axis=-1, kind='quicksort', order='dis')

    def get_nearest_friend(self, number, position_dic, busy_drones):
        friendly_force = []
        self.update_distance_matrix(position_dic)
        for bp_name, distance in self.distance_matrix:
            if distance == 0 or bp_name in busy_drones:
                continue
            friendly_force.append(bp_name)
            if len(friendly_force) == number:
                break
        return friendly_force

    def get_nearest_enemy(self,  position_dic, pick_list=None):
        min_distance = 50000
        if not pick_list:
            pick_list = []
        nearest_bp_name = None
        for bp_name in position_dic:
            if pick_list and bp_name not in pick_list:
                continue
            dis_tmp = np.sqrt(np.power(position_dic[bp_name][0] - self.x, 2) + np.power(position_dic[bp_name][1] - self.y, 2))
            if dis_tmp < min_distance:
                min_distance = dis_tmp
                nearest_bp_name = bp_name

        return min_distance, nearest_bp_name

    def destroy_self(self):
        if self.pre_x_friend:
            self.pre_x_friend.next_x_friend = self.next_x_friend
        if self.pre_y_friend:
            self.pre_y_friend.next_y_friend = self.next_y_friend
        if self.next_x_friend:
            self.next_x_friend.pre_x_friend = self.pre_x_friend
        if self.next_y_friend:
            self.next_y_friend.pre_y_friend = self.pre_y_friend


class RecurrentList(object):
    """循环列表"""
    def __init__(self, data):
        self.data = data
        self.ptr = 0
        self.length = len(self.data)

    def next(self):
        self.ptr += 2
        if self.ptr >= self.length:
            self.ptr = 0
        return self.data[self.ptr], self.data[self.ptr+1]


class LinkedNode(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_pre = None
        self.x_next = None
        self.y_pre = None
        self.y_next = None


class RecurrentList1(object):
    """循环列表"""
    def __init__(self, data, target_id):
        self.people_No = int(target_id)
        self.data = data
        self.ptr = 0
        self.length = len(self.data)


class PatrolPolicy:
    def __init__(self, file_name, ip, port):
        _ = pd.read_table(file_name, sep='\t', header=None)
        self.patrol_list = []
        self.protect_people_list = []

        self.name = []
        self.people_pos = [[-86039.335938, -168649.078125], [-31225.988281, -173175.546875], [30029.246094, -113497.929688],
                           [-80276.257812, -57764.664062], [-26463.941406, -13750.384766]]
        self.get_rescued = [False, False, False, False, False]
        self.summoned_vehicle = [0, 0, 0, 0, 0]
        for each in _[0]:
            #TODO
            #添加字符串处理
            tmp = each.split('|')
            self.protect_people_list.append(tmp[0])
            self.patrol_list.append(tmp[1].split(" "))

        self.socket_server = socket(AF_INET, SOCK_STREAM)
        self.socket_server.bind(('127.0.0.1', port))
        self.socket_server.listen(200)  # 最大连接数
        self.client = CustomAirsimClient(ip, self.socket_server, plot_flag=False, plot_color="100")
        self.SceneObjects = self.client.client.simListSceneObjects()
        self.goal_num = 0
        for each in self.SceneObjects:
            if 'people' in each:
                self.name.append(each)
                self.goal_num += 1
        self.height = -40
        self.velocity = 10
        self.destroy_distance = 40
        time.sleep(2)
        self.get_caught = {}

        self.mission_points = {}
        self.sorted_bp_dict = {}
        for mission_point, bp_name, target_id in zip(self.patrol_list, self.client.vehicle_dict, self.protect_people_list):
            self.mission_points[bp_name] = RecurrentList1(np.array(mission_point, dtype=float)/100, target_id)
            self.get_caught[bp_name] = 0

        time.sleep(1)

        self.remained_vehicle = self.client.listVehicles()
        self.position_dict = {}
        self.reset()
        cnt = 0
        for bp_name in self.position_dict:
            self.sorted_bp_dict[bp_name] = SortedBPname(bp_name, self.position_dict[bp_name][0],
                                                        self.position_dict[bp_name][1])
            self.sorted_bp_dict[bp_name].index = cnt
            cnt += 1
        for bp_name in self.sorted_bp_dict:
            self.sorted_bp_dict[bp_name].first_sort_friend(self.position_dict, self.sorted_bp_dict)

        time.sleep(1)
        for i in range(len(self.name)):
            pos = self.people_pos[i]
            self.plot_target([pos[0]/100, pos[1]/100])
            p0 = airsim.Vector3r(pos[0]/100, pos[1]/100, 0)
            p = airsim.Pose(position_val=p0)
            self.client.simSetObjectPose(self.name[i], p)

    def next(self, Recu):
        Recu.ptr += 2
        if self.get_rescued[Recu.people_No]:
            return self.people_pos[Recu.people_No][0] / 100, self.people_pos[Recu.people_No][1] / 100

        if Recu.ptr >= Recu.length:
            Recu.ptr = 0
        return Recu.data[Recu.ptr], Recu.data[Recu.ptr + 1]

    def countdown_attack(self, delay_time, people_id):
        time.sleep(delay_time)
        self.get_rescued[people_id] = True
    def reset(self):
        for i in range(5):
            for bp_name in self.mission_points:
                pos = airsim.Pose()
                pos.position.x_val = self.mission_points[bp_name].data[0]
                pos.position.y_val = self.mission_points[bp_name].data[1]
                pos.position.z_val = self.height
                # self.client.simSetVehiclePose(pos, ignore_collision=True, vehicle_name=bp_name)
                # self.client.enableApiControl(True, vehicle_name=bp_name)
                self.client.simSetVehiclePose(pos, ignore_collision=True, vehicle_name=bp_name)
                self.position_dict[bp_name] = np.array([pos.position.x_val, pos.position.y_val])
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
                x, y = self.next(self.mission_points[bp_name])

                _ = patrol_client.moveToPositionAsync(x,
                                                      y,
                                                      self.height, self.velocity, vehicle_name=airsim_name,
                                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(is_rate=False))
                # _.join()
            else:
                x, y = self.mission_points[bp_name].data[0], self.mission_points[bp_name].data[1]
                _ = patrol_client.moveToPositionAsync(x,
                                                      y,
                                                      self.height, self.velocity/5, vehicle_name=airsim_name,
                                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(is_rate=False))
                # time.sleep(1)
            #
            for each in range(30):
                time.sleep(0.3)
                # print('get_pose')
                pose = patrol_client.simGetObjectPose(airsim_name).position
                self.position_dict[bp_name] = np.array([pose.x_val, pose.y_val])

                # pose = patrol_client.simGetObjectPose(bp_name).position
                self.sorted_bp_dict[bp_name].x = pose.x_val
                self.sorted_bp_dict[bp_name].y = pose.y_val
            # self.sorted_bp_dict[bp_name].resort_friend()
            # self.position_dict[bp_name] = np.array([pose.x_val, pose.y_val])
            # _.join()

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
        self.sorted_bp_dict[bp_name].destroy_self()
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
    a = PatrolPolicy("road_50_5.txt", ['10.134.142.129:41451'], 9699)
    time.sleep(1)
    a.fly_run()
