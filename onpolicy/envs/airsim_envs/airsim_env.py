import time
from tkinter import W

import torch
from gym.spaces import MultiDiscrete

from .drone_dynamics import DroneDynamicsAirsim
import airsim
import gym
from gym import spaces
import numpy as np
from pathlib import Path
from os import remove
import math
import cv2

# 可视化
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

# 继承gym的环境。用gym的框架套住airsim


def judge_dis(point, agent, dis):
    if math.sqrt(pow(point[0]-agent.x, 2) + pow(point[1] - agent.y, 2)) < dis:
        return True
    else:
        return False


def covered_by_id(cur_n, target_id):
    cnt = 1
    while cur_n > 0:
        a = cur_n % np.power(2, cnt)
        if a > 0:
            if a == np.power(2, target_id - 1):
                return True
            cur_n -= a
        if cur_n == np.power(2, target_id - 1):
            return True
        cnt += 1
    return False

class AirSimDroneEnv(gym.Env, QtCore.QThread):
    # action_signal = pyqtSignal(int, np.ndarray) # action: [v_xy, v_z, yaw_rate])
    # state_signal = pyqtSignal(int, np.ndarray) # state_raw: [dist_xy, dist_z, relative_yaw, v_xy, v_z, yaw_rate]
    # attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray) # attitude (pitch, roll, yaw  current and command)
    # reward_signal = pyqtSignal(int, float, float) # step_reward, total_reward
    # pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # goal_pose start_pose current_pose trajectory
    def __init__(self, cfg) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        torch.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        #conig
        self.current_step = 0
        # self.max_episode_steps = 1000
        self.model = None
        self.data_path = None
        self.shared_reward = True
        self.cfg = cfg
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.velocity_step = cfg.getfloat('options', 'velocity_step')
        self.acc_step = cfg.getfloat('options', 'acc_step')
        self.yaw_step = cfg.getfloat('options', 'yaw_step')
        self.ayaw_step = cfg.getfloat('options', 'ayaw_step')
        self.vz_step = cfg.getfloat('options', 'vz_step')
        self.az_step = cfg.getfloat('options', 'az_step')
        self.depth_image_request = airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, True, False)
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.init_yaw_degree = np.array(cfg.get('options', 'init_yaw_degree').split()).astype(int)

        self.max_vertical_difference = 5
        self.dt = cfg.getfloat('multirotor', 'dt')
        self.env_name = cfg.get('options', 'env_name')
        self.num_of_drones = cfg.getint('options', 'num_of_drone')
        if self.navigation_3d:
            # self.state_feature_length = 5 * self.num_of_drones
            self.state_feature_length = 9

        else:
            self.state_feature_length = 7
            # self.state_feature_length = 3 * self.num_of_drones
        self.axy_n = np.zeros(self.num_of_drones)
        self.vxy_n = np.zeros(self.num_of_drones)
        self.yaw_acc_n = np.zeros(self.num_of_drones)
        self.yaw_sp_n = np.zeros(self.num_of_drones)
        self.vx_n = self.vy_n = self.v_z_n = np.zeros(self.num_of_drones)
        self.ax_n = self.ay_n = self.a_z_n = np.zeros(self.num_of_drones)
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.world_clock = cfg.getint('options', 'world_clock')
        self.discrete_grid_x = cfg.getint('options', 'discrete_grid_x')
        self.discrete_grid_y = cfg.getint('options', 'discrete_grid_y')
        print('Environment: ', self.env_name, "num_of_drones: ", self.num_of_drones)
        self.agents = []
        self.client = airsim.MultirotorClient(ip=cfg.get('options', 'ip'))
        self.trajectory_list = []
        self.coverage_area = np.zeros(self.discrete_grid_x * self.discrete_grid_x)
        self.cur_state = None
        self.init_pose = []
        self.client.reset()
        for i in range(self.num_of_drones):
            self.agents.append(DroneDynamicsAirsim(self.cfg, self.client, i + 1))
            self.trajectory_list.append([])
            self.agents[i].reset(self.init_yaw_degree[i], 0)
            self.init_pose.append(self.client.simGetObjectPose(self.agents[i].name).position)

        # TODO cfg
        self.work_space_x = [-15, 77]
        self.work_space_y = [-47, 30]
        self.work_space_z = [-100, 100]
        self.work_space_x_length = self.work_space_x[1] - self.work_space_x[0]
        self.work_space_y_length = self.work_space_y[1] - self.work_space_y[0]
        self.work_space_z_length = self.work_space_z[1] - self.work_space_z[0]
        self.max_episode_steps = 2000
        self.center = [sum(self.work_space_x) / 2, sum(self.work_space_y)/2 , sum(self.work_space_z)/ 2]
        # trainning state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = np.zeros(self.num_of_drones)
        self.previous_distance_from_des_point = 0

        # other settings
        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')

        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')
        self.observation_space = []
        self.share_observation_space = []

        self.action_space = []
        for each in range(self.num_of_drones):
            self.observation_space.append(spaces.Box(
                low=0, high=1, shape=(self.state_feature_length + 38 + self.discrete_grid_x * self.discrete_grid_y, ), dtype=np.float32))
            self.action_space.append(self.agents[0].action_space)
        self.share_observation_space = [spaces.Box(
            low=0, high=1, shape=((self.state_feature_length + 38 + self.discrete_grid_x * self.discrete_grid_y) * self.num_of_drones, ),
            dtype=np.float32) for _ in range(self.num_of_drones)]
        self.wait_step = 10
        self.first_assign = 1

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        # set action for each agent
        self.compute_velocity(action_n)
        # self.client.simPause(False)
        self.client.simPause(False)
        fi = []
        for i, agent in enumerate(self.agents):
            if not agent.is_crash:
                if i == 0 and self.wait_step > 0:
                    self.wait_step -= 1
                    continue
                fi.append(self._set_action(i, agent))
        for f in fi:
            f.join()
        self.client.simPause(True)
            # if i % int(1 // self.dt) == 0:
            #     self.trajectory_list[i].append(agent.get_position())

        # time.sleep(0.1 / self.world_clock)
        # self.client.simPause(True)
        # advance world state
        # self.world.step()  # core.step()
        # record observation for each agent
        self.cur_state = self.get_all_state()
        for i, agent in enumerate(self.agents):
            agent.x = self.cur_state[i].position.x_val
            agent.y = self.cur_state[i].position.y_val
            agent.z = self.cur_state[i].position.z_val
            _ = self.is_crashed(agent)
            obs_n.append(self._get_obs(agent))
            # print(obs_n[i])
            done_n.append(self._get_done(agent))
            reward_n.append(self._get_reward(obs_n[i], agent, action_n[i]))
            info = self._get_info(agent, reward_n[i])
            if done_n[i]:
                print(info)
            info_n.append(info)

        # for flag in done_n:
        #     if flag:
        #         for each in range(len(done_n)):
        #             done_n[each] = True
        self.cumulated_episode_reward += np.array(reward_n)

        """all agents get total reward in cooperative case, if shared reward,
        all agents have the same reward, and reward is sum"""
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.num_of_drones
        return obs_n, reward_n, done_n, info_n

    def get_all_state(self):
        cur_state = []
        for each in self.agents:
            tmp = self.client.getMultirotorState(each.name).kinematics_estimated
            # tmp.position += self.init_pose[each.id - 1]
            pose = self.client.simGetObjectPose(each.name)
            tmp.position = pose.position
            cur_state.append(tmp)
        return cur_state

    def compute_velocity(self, action_n):
        # yaw_rate_sp_n = action_n[:, -1]
        # if self.navigation_3d:
        #     self.v_z_n = action_n[:, 1].astype('float')
        # else:
        #     self. v_z_n = np.zeros(self.num_of_drones)
        # yaw_n_list = []
        # for each in self.agents:
        #     yaw_n_list.append(each.get_attitude()[2])
        # yaw_n = np.array(yaw_n_list)
        # v_xy_sp_n = self.vxy_n + action_n[:, 0] * self.dt
        # self.yaw_sp_n = yaw_n + yaw_rate_sp_n * self.dt
        # self.yaw_sp_n[self.yaw_sp_n > math.radians(180)] -= math.pi * 2
        # self.yaw_sp_n[self.yaw_sp_n < math.radians(-180)] += math.pi * 2
        # self.vx_n = v_xy_sp_n * np.cos(self.yaw_sp_n)
        # self.vy_n = v_xy_sp_n * np.sin(self.yaw_sp_n)
        action_n = np.array(action_n)
        action_n -= 5
        # action_n[:, 0] += 3
        if self.navigation_3d:
            self.a_z_n += action_n[:, 0] * self.az_step
            # self.v_z_n += action_n[:, 1] * self.vz_step
        else:
            self.a_z_n = np.zeros(self.num_of_drones)
        self.v_z_n += self.a_z_n * self.dt
        self.v_z_n[self.v_z_n > self.v_z_max] = self.v_z_max
        self.v_z_n[self.v_z_n < -self.v_z_max] = -self.v_z_max
        self.axy_n = np.array([self.v_xy_max] * self.num_of_drones)
        # self.vxy_n += action_n[:, 0] * self.velocity_step
        self.vxy_n += self.axy_n * self.dt
        self.vxy_n[self.vxy_n > self.v_xy_max] = self.v_xy_max
        # self.vxy_n[self.vxy_n < self.v_xy_min] = self.v_xy_min
        # self.axy_n += 5 * self.acc_step
        # self.vxy_n += action_n[:, 0] * self.velocity_step
        # self.vxy_n[self.vxy_n < self.v_xy_min] = self.v_xy_min
        self.yaw_acc_n = np.radians(action_n[:, -1] * self.yaw_step)
        yaw_n = []
        for each in range(0, self.num_of_drones):
            yaw_n.append(self.agents[each].get_attitude()[2])
        yaw_n = np.array(yaw_n)
        self.yaw_sp_n = yaw_n + self.yaw_acc_n * self.dt
        self.yaw_sp_n[self.yaw_sp_n > math.radians(180)] -= math.pi * 2
        self.yaw_sp_n[self.yaw_sp_n < math.radians(-180)] += math.pi * 2
        # self.yaw_sp_n[self.yaw_sp_n > self.yaw_rate_max_rad] = self.yaw_rate_max_rad
        # self.yaw_sp_n[:] = 0
        self.vx_n = self.vxy_n * np.cos(self.yaw_sp_n)
        self.vy_n = self.vxy_n * np.sin(self.yaw_sp_n)

    # set env action for a particular agent
    def _set_action(self, i, agent):
        if self.navigation_3d:
            return self.client.moveByVelocityAsync(self.vx_n[i], self.vy_n[i], -self.v_z_n[i], self.dt,
                                            vehicle_name=agent.name,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(
                                               self.yaw_sp_n[i])))
        else:
            return self.client.moveByVelocityZAsync(self.vx_n[i], self.vy_n[i], - agent.start_position[2], self.dt,
                                             vehicle_name=agent.name,
                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(
                                                 self.yaw_sp_n[i])))
            # print(self.vx_n, self.vy_n, math.degrees(self.yaw_sp_n[0]))

    def _get_obs_with_state(self, agent):
        image = self.get_depth_image(agent)
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        image_scaled = image_resize * 100
        self.min_distance_to_obstacles = image_scaled.min()
        image_scaled = -np.clip(image_scaled, 0, self.max_depth_meters) / self.max_depth_meters * 255 + 255
        image_uint8 = image_scaled.astype(np.uint8)

        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[
            1] == self.screen_width, 'image size not match'

        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = agent._get_state_feature()

        assert (self.state_feature_length == state_feature.shape[
            0]), 'state_length {0} is not equal to state_feature_length {1}' \
            .format(self.state_feature_length, state_feature.shape[0])
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        # image_with_state = image_with_state.swapaxes(0, 2)
        # image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state.reshape(-1,)

    def _get_obs(self, agent):
        if agent.is_crash:
            return np.zeros(self.state_feature_length + 38 + self.discrete_grid_x * self.discrete_grid_y)
        state_feature_other = np.array([])
        state_feature_self = np.array([])
        for each in range(self.num_of_drones):
            # velocity_xy = math.sqrt(pow(self.cur_state[each].linear_velocity.x_val, 2) + pow(self.cur_state[each].linear_velocity.y_val, 2))
            # velocity_xy_norm = (velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min)
            velocity_z = self.cur_state[each].linear_velocity.z_val
            velocity_z_norm = (velocity_z / self.v_z_max / 2 + 0.5)
            yaw_rate_sp = self.cur_state[each].angular_velocity.z_val
            yaw_rate_norm = (yaw_rate_sp / self.yaw_rate_max_rad / 2 + 0.5)

            x_norm = (self.cur_state[each].position.x_val - self.work_space_x[0]) / (self.work_space_x[1] - self.work_space_x[0])
            y_norm = (self.cur_state[each].position.y_val - self.work_space_y[0]) / (self.work_space_y[1] - self.work_space_y[0])
            z_norm = (self.cur_state[each].position.z_val - self.work_space_z[0]) / (self.work_space_z[1] - self.work_space_z[0])
            if self.navigation_3d:
                tmp = [x_norm, y_norm, z_norm,
                       velocity_z_norm, yaw_rate_norm]
            else:

                tmp = [x_norm, y_norm, yaw_rate_norm]
            if each + 1 != agent.id:
                state_feature_other = np.append(state_feature_other, tmp)
            else:
                state_feature_self = np.append(state_feature_self, tmp)
        # state_feature = np.append(state_feature_self, state_feature_other)
        state_feature = state_feature_self
        state_feature = np.append(state_feature,
                                  [(agent.work_space_x[0] - self.work_space_x[0])/self.work_space_x_length,
                                   (agent.work_space_x[1] - self.work_space_x[0])/self.work_space_x_length,
                                   (agent.work_space_y[0] - self.work_space_y[0])/self.work_space_y_length,
                                   (agent.work_space_y[1] - self.work_space_y[0])/ self.work_space_y_length])
        distance_sensors_norm = []
        for each in range(36):
            distance_sensors_norm.append(self.client.getDistanceSensorData("Distance"+str(each), agent.name).distance / 40)
        # min_dis = 1000
        # max_dis = 0
        # effective_direction_min = 0
        # effective_direction_max = 0
        # forward_dis = self.client.getDistanceSensorData("Distance0", agent.name).distance
        # for each in range(36):
        #     tmp = self.client.getDistanceSensorData("Distance"+str(each), agent.name)
        #     if tmp.distance < min_dis:
        #         min_dis = tmp.distance
        #         effective_direction_min = each * 10
        #     if tmp.distance > max_dis:
        #         max_dis = tmp.max_distance
        #         effective_direction_max = each
        # min_dis_norm = min_dis / 40
        # forward_dis_norm = forward_dis / 40
        yaw = self.cur_state[agent.id-1].orientation.z_val
        yaw_norm = yaw / math.pi / 2 + 0.5
        # effective_direction_norm_min = effective_direction_min / 360
        # effective_direction_norm_max = effective_direction_max / 360
        return np.append(state_feature,
                         [yaw_norm, self.init_yaw_degree[agent.id - 1] / 360] + distance_sensors_norm + list(self.coverage_area/self.num_of_drones)).clip(0, 1)

    def is_duplicate(self, agent, index):
        if not self.coverage_area[index] or self.coverage_area[index] == agent.id:
            return False
        else:
            return True

    def is_new_area(self, agent, index):
        if not self.coverage_area[index]:
            self.coverage_area[index] = agent.id
            return True
        else:
            return False

    def _get_reward(self, obs, agent, action):
        if agent.is_crash:
            return 0
        reward = 1
        reward_outside = -10
        min_distance = obs[9:45].min()
        if min_distance < 0.03:
            reward -= np.power((0.05-min_distance) * 20, 2) * 1000
        x_index = (agent.x - self.work_space_x[0]) * self.discrete_grid_x // self.work_space_x_length
        y_index = (agent.y - self.work_space_y[0]) * self.discrete_grid_x // self.work_space_y_length
        index = x_index * self.discrete_grid_y + y_index
        index = int(np.clip(index, 0, self.discrete_grid_y * self.discrete_grid_x))
        # if self.is_duplicate(agent, index):
        #     reward -= 3
        if self.is_new_area(agent, index):
            reward += 30
            covered_x = [self.work_space_x[0] + x_index * self.work_space_x_length / self.discrete_grid_x,
                         self.work_space_x[0] + (x_index+1) * self.work_space_x_length / self.discrete_grid_x]
            covered_y = [self.work_space_y[0] + y_index * self.work_space_y_length / self.discrete_grid_y,
                         self.work_space_y[0] + (y_index+1) * self.work_space_y_length / self.discrete_grid_y]
            a = [airsim.Vector3r(covered_x[0], covered_y[0], -agent.z)]
            b = [airsim.Vector3r(covered_x[0], covered_y[1], -agent.z)]
            c = [airsim.Vector3r(covered_x[1], covered_y[1], -agent.z)]
            d = [airsim.Vector3r(covered_x[1], covered_y[0], -agent.z)]
            if agent.id == 1:
                self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, duration=5,
                                            color_rgba=[1.0, 0.0, 0.0, 1.0],
                                            is_persistent=False)
            elif agent.id == 2:
                self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, duration=5,
                                            color_rgba=[0.0, 1.0, 0.0, 1.0],
                                            is_persistent=False)
            elif agent.id == 3:
                self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, duration=5,
                                            color_rgba=[0.0, 0.0, 1.0, 1.0],
                                            is_persistent=False)
            else:
                self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, duration=5, color_rgba=[1.0, 1.0, 1.0, 1.0],
                                            is_persistent=False)
            # normalized to 100 according to goal_distance
            # reward_obs = 0
            # action_cost = 0
            # add yaw_rate cost
            # yaw_speed_cost = 0 * abs(action[-1]) / agent.yaw_rate_max_rad

            # if agent.navigation_3d:
            #     v_z_cost = 0 * abs(action[1]) / agent.v_z_max
            #     action_cost += v_z_cost

            # action_cost += yaw_speed_cost

        self.is_crashed(agent)
            # reward += reward_crash
        if self.is_not_inside_workspace(agent):
            reward += reward_outside * np.power(0.75, int(np.log2(self.current_step)))
        print(reward)
        return reward

    def _get_done(self, agent):
        is_not_inside_workspace_now = self.is_not_inside_workspace(agent)
        # has_reached_des_pose = agent.is_in_desired_pose()

        # We see if we are outside the Learning Space
        episode_done = agent.is_crash or \
                       self.current_step >= self.max_episode_steps
        return episode_done

    def is_crashed(self, agent):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo(vehicle_name=agent.name)
        if collision_info.has_collided:
            is_crashed = True
            agent.is_crash = True
        return is_crashed

    def _get_info(self, agent, reward):
        info = {
            'id': agent.id,
            # 'is_success': agent.is_in_desired_pose(),
            'is_crash': agent.is_crash,
            'is_not_in_workspace': self.is_not_inside_workspace(agent),
            'step_num': self.current_step,
            'individual_reward': reward
        }
        return info

    def assign_area(self):
        x_left = 0
        y_left = 0

        self.agents[0].x, self.agents[0].y, self.agents[0].z = self.agents[0].get_position()
        self.agents[0].work_space_x = [self.work_space_x[0], self.agents[0].x+7]
        self.agents[0].work_space_y = [self.work_space_y[0], self.agents[0].y+2]
        a = [airsim.Vector3r(self.agents[0].work_space_x[0], self.agents[0].work_space_y[0], -self.agents[0].z)]
        b = [airsim.Vector3r(self.agents[0].work_space_x[0], self.agents[0].work_space_y[1], -self.agents[0].z)]
        c = [airsim.Vector3r(self.agents[0].work_space_x[1], self.agents[0].work_space_y[1], -self.agents[0].z)]
        d = [airsim.Vector3r(self.agents[0].work_space_x[1], self.agents[0].work_space_y[0], -self.agents[0].z)]

        self.client.simPlotLineList(a+b+b+c+c+d+d+a, thickness=15.0, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

        self.agents[1].x, self.agents[1].y, self.agents[1].z = self.agents[1].get_position()

        self.agents[1].work_space_x = [self.agents[1].x-1, self.work_space_x[1]]
        self.agents[1].work_space_y = [self.work_space_y[0], self.agents[1].y+5]
        a = [airsim.Vector3r(self.agents[1].work_space_x[0], self.agents[1].work_space_y[0], -self.agents[1].z)]
        b = [airsim.Vector3r(self.agents[1].work_space_x[0], self.agents[1].work_space_y[1], -self.agents[1].z)]
        c = [airsim.Vector3r(self.agents[1].work_space_x[1], self.agents[1].work_space_y[1], -self.agents[1].z)]
        d = [airsim.Vector3r(self.agents[1].work_space_x[1], self.agents[1].work_space_y[0], -self.agents[1].z)]
        self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, color_rgba=[0.0, 1.0, 0.0, 1.0], is_persistent=True)
        self.agents[2].x, self.agents[2].y, self.agents[2].z = self.agents[2].get_position()

        self.agents[2].work_space_x = [self.agents[2].x-1, self.work_space_x[1]]
        self.agents[2].work_space_y = [self.agents[2].y-5, self.work_space_y[1]]

        a = [airsim.Vector3r(self.agents[2].work_space_x[0], self.agents[2].work_space_y[0], -self.agents[2].z)]
        b = [airsim.Vector3r(self.agents[2].work_space_x[0], self.agents[2].work_space_y[1], -self.agents[2].z)]
        c = [airsim.Vector3r(self.agents[2].work_space_x[1], self.agents[2].work_space_y[1], -self.agents[2].z)]
        d = [airsim.Vector3r(self.agents[2].work_space_x[1], self.agents[2].work_space_y[0], -self.agents[2].z)]
        self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, color_rgba=[0.0, 0.0, 1.0, 1.0], is_persistent=True)

        self.agents[3].x, self.agents[3].y, self.agents[3].z = self.agents[3].get_position()
        self.agents[3].work_space_x = [self.work_space_x[0], self.agents[3].x+1]
        self.agents[3].work_space_y = [self.agents[3].y-5, self.work_space_y[1]]
        a = [airsim.Vector3r(self.agents[3].work_space_x[0], self.agents[3].work_space_y[0], -self.agents[3].z)]
        b = [airsim.Vector3r(self.agents[3].work_space_x[0], self.agents[3].work_space_y[1], -self.agents[3].z)]
        c = [airsim.Vector3r(self.agents[3].work_space_x[1], self.agents[3].work_space_y[1], -self.agents[3].z)]
        d = [airsim.Vector3r(self.agents[3].work_space_x[1], self.agents[3].work_space_y[0], -self.agents[3].z)]
        self.client.simPlotLineList(a + b + b + c + c + d + d + a, thickness=15.0, color_rgba=[1.0, 1.0, 1.0, 1.0], is_persistent=True)

        # i = 0
        # x_blocks = self.num_of_drones // 2
        # x_length = self.work_space_x_length / x_blocks
        # x_start = self.work_space_x[0]
        # for each in range(x_blocks):
        #     x_end = x_start + x_length
        #     self.agents[i].work_space_x = [x_start, x_end]
        #     self.agents[i].work_space_y = [self.work_space_y[0], self.center[1]]
        #     x_start = x_end
        #     i += 1
        # x_start = self.work_space_x[0]
        # x_blocks_2 = self.num_of_drones - x_blocks
        # x_length = self.work_space_x_length / x_blocks_2
        # for each in range(x_blocks, self.num_of_drones):
        #     x_end = x_start + x_length
        #     self.agents[i].work_space_x = [x_start, x_end]
        #     self.agents[i].work_space_y = [self.center[1], self.work_space_y[1]]
        #     x_start = x_end
        #     i += 1

    def reset(self):

        # reset state
        self.wait_step = 10
        self.client.reset()
        # self.init_pose = []
        fi = []

        # sample_aera = np.random.randint(8)
        self.coverage_area = np.zeros(self.discrete_grid_y * self.discrete_grid_x)

        sample_aera = 4
        self.client.simPause(False)
        for agent in self.agents:
            fi.append(agent.reset(self.init_yaw_degree[agent.id-1], sample_aera))

        for f in fi:
            f.join()

        # self.agents[0].is_crash = True
        # self.agents[1].is_crash = True
        # self.agents[2].is_crash = True
        self.client.simPause(True)

        # time.sleep(2/self.world_clock)
        self.cur_state = self.get_all_state()
        self.episode_num += 1
        self.cumulated_episode_reward = np.zeros(self.num_of_drones)
        self.current_step = 0
        self.trajectory_list = []
        if self.first_assign:
            self.assign_area()
            self.first_assign = 0
        # time.sleep(2)
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            self.trajectory_list.append([])
        return obs_n

    # def render(self):
    #     return self.get_obs()

    def is_not_inside_workspace(self, agent):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = [agent.x, agent.y, agent.z]
        if current_position[0] < agent.work_space_x[0] or current_position[0] > agent.work_space_x[1] or \
                current_position[1] < agent.work_space_y[0] or current_position[1] > agent.work_space_y[1]:
                # current_position[2] < agent.work_space_z[0] or current_position[2] > agent.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    # #被捕
    # def is_captured(self):
    #     try:
    #         my_file = Path(hp.env_communicate_path + "swapFile")
    #         if my_file.is_file():
    #             self.active_bomb = True
    #             file = open(hp.env_communicate_path +'swapReading','w')
    #             file.write("reading")
    #             remove("./swapFile")
    #             print("niSiLe")
    #             file.close()
    #             remove("./swapReading")
    #             return True
    #     finally:
    #         pass
    #     return False

    def get_depth_image(self, agent):
        # rgb_image_request = airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)
        # print(agent.name +':getting image!。。。。。')
        responses = self.client.simGetImages([self.depth_image_request], vehicle_name=agent.name)
        # responses = self.client.simGetImages([depth_image_request], vehicle_name=agent.name)
        # print('done!')

        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        return depth_img

    def bomb(self):
        if self.active_bomb:
            self.active_bomb = False
            pass

