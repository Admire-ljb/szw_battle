from onpolicy.envs.airsim_envs.airsim_socket import CustomAirsimClient
from socket import *
import pandas as pd
import time
import airsim
import threading
import numpy as np
# from traind_policy import *


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

    def get_nearest_friend(self, ban_name=False):

        dis = dis2 = dis3 = dis4 = 100000
        x_pre_index = 0
        y_pre_index = 0
        x_next_index = 0
        y_next_index = 0
        if self.pre_x_friend:
            dis = np.sqrt(np.power(self.pre_x_friend.x - self.x, 2) + np.power(self.pre_x_friend.y - self.y, 2))
            if self.pre_x_friend.pre_x_friend:
                dis_temp = np.sqrt(np.power(self.pre_x_friend.pre_x_friend.x - self.x, 2) + np.power(self.pre_x_friend.pre_x_friend.y - self.y, 2))
                if dis_temp < dis:
                    dis = dis_temp
                    x_pre_index = 1
            if self.pre_x_friend.pre_x_friend.pre_x_friend:
                dis_temp = np.sqrt(np.power(self.pre_x_friend.pre_x_friend.pre_x_friend.x - self.x, 2) + np.power(
                    self.pre_x_friend.pre_x_friend.pre_x_friend.y - self.y, 2))
                if dis_temp < dis:
                    dis = dis_temp
                    x_pre_index = 2
        if self.next_x_friend:
            dis2 = np.sqrt(np.power(self.next_x_friend.x - self.x, 2) + np.power(self.next_x_friend.y - self.y, 2))
            if self.next_x_friend.next_x_friend:
                dis_temp = np.sqrt(np.power(self.next_x_friend.next_x_friend.x - self.x, 2) + np.power(
                    self.next_x_friend.next_x_friend.y - self.y, 2))
                if dis_temp < dis2:
                    dis2 = dis_temp
                    x_next_index = 1
            if self.next_x_friend.next_x_friend.next_x_friend:
                dis_temp = np.sqrt(np.power(self.next_x_friend.next_x_friend.next_x_friend.x - self.x, 2) + np.power(
                    self.next_x_friend.next_x_friend.next_x_friend.y - self.y, 2))
                if dis_temp < dis2:
                    dis2 = dis_temp
                    x_next_index = 2
        if self.pre_y_friend:
            dis3 = np.sqrt(np.power(self.pre_y_friend.x - self.x, 2) + np.power(self.pre_y_friend.y - self.y, 2))
            if self.pre_y_friend.pre_y_friend:
                dis_temp = np.sqrt(np.power(self.pre_y_friend.pre_y_friend.x - self.x, 2) + np.power(self.pre_y_friend.pre_y_friend.y - self.y, 2))
                if dis_temp < dis3:
                    dis3 = dis_temp
                    y_pre_index = 1
            if self.pre_y_friend.pre_y_friend.pre_y_friend:
                dis_temp = np.sqrt(np.power(self.pre_y_friend.pre_y_friend.pre_y_friend.x - self.x, 2) + np.power(
                    self.pre_y_friend.pre_y_friend.pre_y_friend.y - self.y, 2))
                if dis_temp < dis3:
                    dis3 = dis_temp
                    y_pre_index = 2
        if self.next_y_friend:
            dis4 = np.sqrt(np.power(self.next_y_friend.x - self.x, 2) + np.power(self.next_y_friend.y - self.y, 2))
            if self.next_y_friend.next_y_friend:
                dis_temp = np.sqrt(np.power(self.next_y_friend.next_y_friend.x - self.x, 2) + np.power(
                    self.next_y_friend.next_y_friend.y - self.y, 2))
                if dis_temp < dis4:
                    dis4 = dis_temp
                    y_next_index = 1
            if self.next_y_friend.next_y_friend.next_y_friend:
                dis_temp = np.sqrt(np.power(self.next_y_friend.next_y_friend.next_y_friend.x - self.x, 2) + np.power(
                    self.next_y_friend.next_y_friend.next_y_friend.y - self.y, 2))
                if dis_temp < dis4:
                    dis4 = dis_temp
                    y_next_index = 2
        index = np.argmin([dis, dis2, dis3, dis4])

        if index == 0 and self.pre_x_friend.bp_name != ban_name:
            return dis, self.pre_x_friend
        if index == 1 and self.next_x_friend.bp_name != ban_name:
            return dis2, self.next_x_friend
        if index == 2 and self.pre_y_friend.bp_name != ban_name:
            return dis3, self.pre_y_friend
        if index == 3 and self.next_y_friend != ban_name:
            return dis4, self.next_y_friend
        else:
            return dis, np.random.choice([self.pre_x_friend, self.next_x_friend, self.pre_y_friend, self.next_y_friend])

    def get_nearest_enemy(self, ban_list):
        dis = dis2 = dis3 = dis4 = 1000000
        if self.pre_x_enemy in ban_list:
            self.pre_x_enemy = self.pre_x_enemy.get_nearest_friend()
        if self.pre_y_enemy in ban_list:
            self.pre_y_enemy = self.pre_y_enemy.get_nearest_friend()
        if self.next_x_enemy in ban_list:
            self.next_x_enemy = self.next_x_enemy.get_nearest_friend()
        if self.next_y_enemy in ban_list:
            self.next_y_enemy = self.next_y_enemy.get_nearest_friend()
        if self.pre_x_enemy:
            dis = np.sqrt(np.power(self.pre_x_enemy.x - self.x, 2) + np.power(self.pre_x_enemy.y - self.y, 2))
        if self.next_x_enemy:
            dis2 = np.sqrt(np.power(self.next_x_enemy.x - self.x, 2) + np.power(self.next_x_enemy.y - self.y, 2))
        if self.pre_y_enemy:
            dis3 = np.sqrt(np.power(self.pre_y_enemy.x - self.x, 2) + np.power(self.pre_y_enemy.y - self.y, 2))
        if self.next_y_enemy:
            dis4 = np.sqrt(np.power(self.next_y_enemy.x - self.x, 2) + np.power(self.next_y_enemy.y - self.y, 2))
        index = np.argmin([dis, dis2, dis3, dis4])
        if index == 0:
            return dis, self.pre_x_enemy
        if index == 1:
            return dis2, self.next_x_enemy
        if index == 2:
            return dis3, self.pre_y_enemy
        else:
            return dis4, self.next_y_enemy

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


class FixedPolicy:
    def __init__(self, file_name, ip, port):
        _ = pd.read_table(file_name, sep='\t', header=None)
        self.patrol_list = []
        for each in _[0]:
            self.patrol_list.append(each.split(" "))
        self.socket_server = socket(AF_INET, SOCK_STREAM)
        self.socket_server.bind(('127.0.0.1', port))
        self.socket_server.listen(200)  # 最大连接数
        self.client = CustomAirsimClient(ip, self.socket_server, plot_flag=False)
        self.height = -40
        self.velocity = 10
        self.destroy_distance = 40
        time.sleep(2)
        self.get_caught = {}
        self.mission_points = {}
        self.sorted_bp_dict = {}
        for mission_point, bp_name in zip(self.patrol_list, self.client.vehicle_dict):
            self.mission_points[bp_name] = RecurrentList(np.array(mission_point, dtype=float)/100)
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

    def reset(self):
        for i in range(5):
            for bp_name in self.mission_points:
                pos = airsim.Pose()
                pos.position.x_val = self.mission_points[bp_name].data[0]
                pos.position.y_val = self.mission_points[bp_name].data[1]
                pos.position.z_val = self.height
                self.client.simSetVehiclePose(pos, ignore_collision=True, vehicle_name=bp_name)
                self.position_dict[bp_name] = np.array([pos.position.x_val, pos.position.y_val])
                self.client.enableApiControl(True, vehicle_name=bp_name)

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

            else:
                # x, y = self.mission_points[bp_name].data[0], self.mission_points[bp_name].data[1]
                x, y = self.mission_points[bp_name].next()
                _ = patrol_client.moveToPositionAsync(x,
                                                      y,
                                                      self.height, self.velocity/5, vehicle_name=airsim_name,
                                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(is_rate=False))
                # time.sleep(1)
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


if __name__ == "__main__":
    a = FixedPolicy("patrol_100.txt", ['127.0.0.1:41451', '10.134.142.129:41451'], 9699)
    time.sleep(1)
    a.fly_run()
