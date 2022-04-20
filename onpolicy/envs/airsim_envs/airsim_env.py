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
        self.max_episode_steps = 1000
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
            self.state_feature_length = 6 * self.num_of_drones
        else:
            self.state_feature_length = 4 * self.num_of_drones
        self.axy_n = np.zeros(self.num_of_drones)
        self.vxy_n = np.zeros(self.num_of_drones)
        self.yaw_acc_n = np.zeros(self.num_of_drones)
        self.yaw_sp_n = np.zeros(self.num_of_drones)
        self.vx_n = self.vy_n = self.v_z_n = np.zeros(self.num_of_drones)
        self.ax_n = self.ay_n = self.a_z_n = np.zeros(self.num_of_drones)
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.world_clock = cfg.getint('options', 'world_clock')
        print('Environment: ', self.env_name, "num_of_drones: ", self.num_of_drones)
        self.agents = []
        self.client = airsim.MultirotorClient(ip=cfg.get('options', 'ip'))
        self.trajectory_list = []
        self.cur_state = None
        self.init_pose = []
        for i in range(self.num_of_drones):
            self.agents.append(DroneDynamicsAirsim(self.cfg, self.client, i + 1))
            self.trajectory_list.append([])


        # TODO cfg
        self.work_space_x = [-15, 80]
        self.work_space_y = [-40, 25]
        self.work_space_z = [-10, 10]
        self.max_episode_steps = 500

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
            self.observation_space.append(spaces.Box(low=0, high=100, shape=(self.state_feature_length + 3, ), dtype=np.uint8))
            self.action_space.append(self.agents[0].action_space)
        self.share_observation_space = [spaces.Box(
            low=0, high=100, shape=((self.state_feature_length + 3) * self.num_of_drones, ),
            dtype=np.uint8) for _ in range(self.num_of_drones)]

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
        for i, agent in enumerate(self.agents):
            self._set_action(i, agent)
            # if i % int(1 // self.dt) == 0:
            #     self.trajectory_list[i].append(agent.get_position())

        time.sleep(0.1 / self.world_clock)
        # self.client.simPause(True)
        # advance world state
        # self.world.step()  # core.step()
        # record observation for each agent
        self.cur_state = self.get_all_state()
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            reward_n.append(self._get_reward(done_n[i], agent, action_n[i]))
            info = self._get_info(agent, reward_n[i])
            if done_n[i]:
                print(info)
            info_n.append(info)

        for flag in done_n:
            if flag:
                for each in range(len(done_n)):
                    done_n[each] = True
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
            tmp.position += self.init_pose[each.id - 1]
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
        action_n[:, 0] += 3
        if self.navigation_3d:
            self.a_z_n += action_n[:, 1] * self.az_step
            # self.v_z_n += action_n[:, 1] * self.vz_step

        else:
            self.a_z_n = np.zeros(self.num_of_drones)
        self.v_z_n += self.a_z_n * self.dt
        self.v_z_n[self.v_z_n > self.v_z_max] = self.v_z_max
        self.v_z_n[self.v_z_n < -self.v_z_max] = -self.v_z_max
        self.axy_n += action_n[:, 0] * self.acc_step
        # self.vxy_n += action_n[:, 0] * self.velocity_step
        self.vxy_n += self.axy_n * self.dt
        self.vxy_n[self.vxy_n > self.v_xy_max] = self.v_xy_max
        self.vxy_n[self.vxy_n < self.v_xy_min] = self.v_xy_min

        # self.vxy_n[self.vxy_n < self.v_xy_min] = self.v_xy_min
        self.yaw_acc_n = np.radians(action_n[:, -1] * self.yaw_step)
        yaw_n = []
        for each in range(0, self.num_of_drones):
            yaw_n.append(self.agents[each].get_attitude()[2])
        yaw_n = np.array(yaw_n)
        self.yaw_sp_n = yaw_n + self.yaw_acc_n * self.dt
        self.yaw_sp_n[self.yaw_sp_n > math.radians(180)] -= math.pi * 2
        self.yaw_sp_n[self.yaw_sp_n < math.radians(-180)] += math.pi * 2
        self.vx_n = self.vxy_n * np.cos(self.yaw_sp_n)
        self.vy_n = self.vxy_n * np.sin(self.yaw_sp_n)

    # set env action for a particular agent
    def _set_action(self, i, agent):
        if self.navigation_3d:
            self.client.moveByVelocityAsync(self.vx_n[i], self.vy_n[i], -self.v_z_n[i], self.dt * 2.5,
                                            vehicle_name=agent.name,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(
                                               self.yaw_sp_n[i])))
        else:
            self.client.moveByVelocityZAsync(self.vx_n[i], self.vy_n[i], - agent.start_position[2], self.dt * 2.5,
                                             vehicle_name=agent.name,
                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                             yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(
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
        state_feature_other = np.array([])
        state_feature_self = np.array([])
        for each in range(self.num_of_drones):
            velocity_xy = math.sqrt(pow(self.cur_state[each].linear_velocity.x_val, 2) + pow(self.cur_state[each].linear_velocity.y_val, 2))
            velocity_xy_norm = (velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 100
            velocity_z = self.cur_state[each].linear_velocity.z_val
            velocity_z_norm = (velocity_z / self.v_z_max / 2 + 0.5) * 100
            yaw_rate = self.cur_state[each].angular_velocity.z_val
            yaw_rate_norm = (yaw_rate / self.yaw_rate_max_rad / 2 + 0.5) * 100

            x_norm = (self.cur_state[each].position.x_val - self.work_space_x[0]) / (self.work_space_x[1] - self.work_space_x[0]) * 100
            y_norm = (self.cur_state[each].position.y_val - self.work_space_y[0]) / (self.work_space_y[1] - self.work_space_y[0]) * 100
            z_norm = (self.cur_state[each].position.z_val - self.work_space_z[0]) / (self.work_space_z[1] - self.work_space_z[0]) * 100
            if self.navigation_3d:
                tmp = [x_norm, y_norm, z_norm,
                       velocity_xy_norm, velocity_z_norm, yaw_rate_norm]
            else:

                tmp = [x_norm, y_norm, velocity_xy_norm, yaw_rate_norm]
            if each + 1 != agent.id:
                state_feature_other = np.append(state_feature_other, tmp)
            else:
                state_feature_self = np.append(state_feature_self, tmp)
        state_feature = np.append(state_feature_self, state_feature_other)
        min_dis = 1000
        effective_direction = 0
        for each in range(36):
            tmp = self.client.getDistanceSensorData("Distance"+str(each), agent.name)
            if tmp.distance < min_dis:
                min_dis = tmp.distance
                effective_direction = each * 10
        min_dis_norm = min_dis / 40 * 100
        effective_direction_norm = effective_direction / 360 * 100
        return np.append(state_feature, [min_dis_norm, effective_direction_norm, self.init_yaw_degree[agent.id-1]])

    def is_duplicate(self, agent):
        for each, points in enumerate(self.trajectory_list):
            if each != agent.id - 1:
                for point in points:
                    if judge_dis(point, agent, 3):
                        return True
        return False

    def is_new_area(self, agent):
        for point in self.trajectory_list[agent.id-1]:
            if judge_dis(point, agent, 1):
                return False
        else:
            self.trajectory_list[agent.id-1].append(agent.get_position())
            return True

    def _get_reward(self, done, agent, action):
        reward = 1
        reward_crash = -300
        reward_outside = -300
        if not done:
            if self.is_duplicate(agent):
                reward -= 2
            elif self.is_new_area(agent):
                reward += 5
            # normalized to 100 according to goal_distance
            # reward_obs = 0
            # action_cost = 0
            # add yaw_rate cost
            # yaw_speed_cost = 0 * abs(action[-1]) / agent.yaw_rate_max_rad

            # if agent.navigation_3d:
            #     v_z_cost = 0 * abs(action[1]) / agent.v_z_max
            #     action_cost += v_z_cost

            # action_cost += yaw_speed_cost
        else:
            if agent.is_crashed():
                reward = reward_crash
            elif self.is_not_inside_workspace(agent):
                reward = reward_outside
        return reward

    def _get_done(self, agent):
        is_not_inside_workspace_now = self.is_not_inside_workspace(agent)
        # has_reached_des_pose = agent.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed(agent)

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or \
                       too_close_to_obstable or \
                       self.current_step >= self.max_episode_steps
        return episode_done

    def is_crashed(self, agent):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo(vehicle_name=agent.name)
        if collision_info.has_collided:
            is_crashed = True
        return is_crashed

    def _get_info(self, agent, reward):
        info = {
            'id': agent.id,
            # 'is_success': agent.is_in_desired_pose(),
            'is_crash': agent.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(agent),
            'step_num': self.current_step
            # 'individual_reward': reward
        }
        return info

    def reset(self):

        # reset state
        self.client.reset()
        self.init_pose = []

        for agent in self.agents:
            self.init_pose.append(self.client.simGetObjectPose(agent.name).position)
            agent.reset(self.init_yaw_degree[agent.id-1])
        time.sleep(2/self.world_clock)
        self.cur_state = self.get_all_state()
        self.episode_num += 1
        # self.current_step = 0
        self.cumulated_episode_reward = np.zeros(self.num_of_drones)
        self.current_step = 0
        self.trajectory_list = []
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
        current_position = agent.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
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

