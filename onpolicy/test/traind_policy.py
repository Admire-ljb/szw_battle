import airsim

from onpolicy.envs.airsim_envs.airsim_socket import CustomAirsimClient
from socket import *
import pandas as pd
import time
import threading
import numpy as np
from fixed_policy import *
import torch
import configparser
from onpolicy.envs.airsim_envs.airsim_env import AirSimDroneEnv
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic


class Myconf(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=None)

    def optionxform(self, optionstr: str) -> str:
        return optionstr

def get_distance(x, y):
    return np.sqrt(np.power(x[0]-y[0] , 2) + np.power(x[1] - y[1], 2))

def _t2n(x):
    return x.detach().cpu().numpy()

def mission_points_init(center, number_of_drones):
    r = 800
    line_list = []
    for i in range(number_of_drones):
        pt = [0, 0]
        pt[0] = center[0] + r * np.cos(2 * np.pi / number_of_drones * i)
        pt[1] = center[1] + r * np.sin(2 * np.pi / number_of_drones * i)
        line_list.append(pt)
    return line_list

class TrainedPolicy:
    def __init__(self,   actor, airsim_env: AirSimDroneEnv):
        # _ = pd.read_table(file_name, sep='\t', header=None)
        # mission_list = []
        # for points in _[0]:
        #     mission_list.append(points.split(" "))
        # self.plot_client = airsim.MultirotorClient('127.0.0.1')
        self.actor = actor
        self.env = airsim_env
        self.height = -40
        self.destroy_distance = 40
        self.remained_vehicle = self.env.client.assigned_blueprint.copy()
        # self.attack_flag = np.zeros(len(self.remained_vehicle), dtype=object)
        self.attack_flag = {}
        self.attacked_enemy = {}
        self.enemy_position = {}
        self.mission_points = {}
        time.sleep(2)
        # for mission_point, bp_name in zip(mission_list, self.env.client.vehicle_dict):
        #     self.mission_points[bp_name] = RecurrentList(np.array(mission_point, dtype=float) / 100)
        self.position_dict = {}

        self.done_n = None
        self.destroyed_enemy = []
        self.agents = {}
        self.sorted_bp_dict = {}
        self.bp_names_list = []
        self.obs_n = self.reset()
        for agent in self.env.agents:
            self.position_dict[agent.name] = np.array([agent.x, agent.y])
            self.sorted_bp_dict[agent.name] = SortedBPname(agent.name, agent.x, agent.y)
        for bp_name in self.sorted_bp_dict:
            self.sorted_bp_dict[bp_name].first_sort_friend(self.position_dict, self.sorted_bp_dict)
            self.bp_names_list.append(bp_name)
            self.sorted_bp_dict[bp_name].index = len(self.bp_names_list) - 1
            self.attack_flag[bp_name] = 0
        self.task_cancel = False
        time.sleep(1)
        self.obs_n = self.reset()

    def reset(self):
        s = self.env.reset()
        cnt = 0
        # bp_name = self.env.agents[0].name
        line_list = mission_points_init([self.env.agents[0].x, self.env.agents[0].y], len(self.env.agents))
        for agent in self.env.agents:
            self.agents[agent.name] = agent
        for bp_name in self.remained_vehicle:
            self.position_dict[bp_name] = np.array([self.agents[bp_name].x, self.agents[bp_name].y])
            self.agents[bp_name].goal_position = line_list[cnt]
            cnt += 1
        return s

    def fly_run(self):
        _ = threading.Thread(target=self.fly_to_target, args=[])
        _.start()

    def fly_to_target(self):
        rnn_states_actor_n = []
        rnn_states_actor = np.zeros(
            (1, 1, 64),
            dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        for i in range(len(self.remained_vehicle)):
            rnn_states_actor_n.append(rnn_states_actor)

        action_n = np.zeros(len(self.remained_vehicle))
        while True:
            for i in range(len(self.remained_vehicle)):
                obs_tmp = self.obs_n[i].copy()
                obs_tmp = np.reshape(obs_tmp, (1, 5))
                tmp, _, rnn_states_actor_n[i] = actor1(obs_tmp, rnn_states_actor_n[i], masks)
                action_n[i] = int(tmp[0][0])
                rnn_states_actor_n[i] = np.array(np.split(_t2n(rnn_states_actor_n[i]), 1))
                rnn_states_actor_n[i] = np.reshape(rnn_states_actor_n[i], (1, 1, 64))

                # position_list.append(self.env.client.simGetObjectPose(self.remained_vehicle[i]).position)
            # self.position_list = position_list
            action_n = np.reshape(action_n, (num_agents, 1))
            self.obs_n, _, self.done_n, _ = env.step(action_n)
            for bp_name in self.remained_vehicle:
                self.position_dict[bp_name] = np.array([self.agents[bp_name].x, self.agents[bp_name].y])
                self.sorted_bp_dict[bp_name].x = self.agents[bp_name].x
                self.sorted_bp_dict[bp_name].y = self.agents[bp_name].y

    def find_first_enemy(self, enemy_class: FixedPolicy):
        find = 0

        # for bp_name in enemy_class.remained_vehicle:
        #     self.enemy_position[bp_name] = enemy_class.position_dict[bp_name]
            # goal.append(self.enemy_position[bp_name])
        # bp_name = self.sorted_bp_dict[self.bp_names_list[0]].get_nearest_enemy()[1].bp_name
        # self.agents[self.bp_names_list[0]].goal_position = enemy_class.position_dict[bp_name]

        while not find:
            time.sleep(0.2)
            # for i in agent_id_list:
            # enemy_name = enemy_class.remained_vehicle[0]
            for bp_name in self.sorted_bp_dict:
                dis, bp_sorted = self.sorted_bp_dict[bp_name].get_nearest_enemy()
                if dis < self.destroy_distance + 80:
                    self.attack_flag[bp_name] = bp_sorted.bp_name
                    self.agents[bp_name].goal_name = bp_sorted.bp_name

                    find = 1
                    self.attacked_enemy[bp_sorted.bp_name].append(bp_name)

        for bp_name in self.remained_vehicle:
            if not self.attack_flag[bp_name]:
                enemy_bp_name = enemy_class.remained_vehicle[np.random.randint(0, len(enemy_class.remained_vehicle))]
                pose = enemy_class.position_dict[enemy_bp_name]
                self.agents[bp_name].goal_position[0] = pose[0]
                self.agents[bp_name].goal_position[1] = pose[1]
                self.agents[bp_name].goal_name = enemy_bp_name

    def assign_goal(self, bp_name, goal):
        self.agents[bp_name].goal_position = goal

    def assign_enemy(self, enemy_class: FixedPolicy):
        # 分配没有任务的无人机随机指派点
        for i, bp_name in enumerate(self.sorted_bp_dict):
            if not self.attack_flag[bp_name] and self.done_n[i]:
                enemy_bp_name = enemy_class.remained_vehicle[np.random.randint(0, len(enemy_class.remained_vehicle))]
                pose = self.enemy_position[enemy_bp_name]
                self.agents[bp_name].goal_position = pose
                # self.env.agents[i].goal_position[1] = pose.y_val
                self.env.agents[i].goal_name = enemy_bp_name

    def assign_attack(self, drone_num=3):
        for bp_name in self.attack_flag:
            if self.attack_flag[bp_name]:
                cur_drone = drone_num - len(self.attacked_enemy[self.attack_flag[bp_name]])
                for i in range(cur_drone):
                    self.find_nearest_unassigned_agent(bp_name)

    def find_nearest_unassigned_agent(self, bp_name):
        # 找到最近的友方无人机进行协同打击
        # bp_name = self.bp_names_list[i]
        dis, friend = self.sorted_bp_dict[bp_name].get_nearest_friend()
        if not self.attack_flag[friend.bp_name]:
            self.attack_flag[friend.bp_name] = self.attack_flag[bp_name]
            self.agents[friend.bp_name].goal_name = self.agents[bp_name].goal_name
            self.attacked_enemy[self.agents[bp_name].goal_name].append(friend.bp_name)

        # else:
        #     if self.sorted_bp_dict[bp_name].pre_x_friend and not self.attack_flag[self.sorted_bp_dict[bp_name].pre_x_friend.bp_name] :
        #         self.attack_flag[self.sorted_bp_dict[bp_name].pre_x_friend.bp_name] = self.attack_flag[bp_name]
        #         self.agents[self.sorted_bp_dict[bp_name].pre_x_friend.bp_name].goal_name = self.agents[bp_name].goal_name
        #     if self.sorted_bp_dict[bp_name].pre_y_friend and not self.attack_flag[self.sorted_bp_dict[bp_name].pre_y_friend.bp_name]:
        #         self.attack_flag[self.sorted_bp_dict[bp_name].pre_y_friend.bp_name] = self.attack_flag[bp_name]
        #         self.agents[self.sorted_bp_dict[bp_name].pre_y_friend.bp_name].goal_name = self.agents[bp_name].goal_name
        #     if self.sorted_bp_dict[bp_name].next_x_friend and not self.attack_flag[self.sorted_bp_dict[bp_name].next_x_friend.bp_name]:
        #         self.attack_flag[self.sorted_bp_dict[bp_name].next_x_friend.bp_name] = self.attack_flag[bp_name]
        #         self.agents[self.sorted_bp_dict[bp_name].next_x_friend.bp_name].goal_name = self.agents[bp_name].goal_name
        #     if self.sorted_bp_dict[bp_name].next_y_friend and not self.attack_flag[self.sorted_bp_dict[bp_name].next_y_friend.bp_name]:
        #         self.attack_flag[self.sorted_bp_dict[bp_name].next_y_friend.bp_name] = self.attack_flag[bp_name]
        #         self.agents[self.sorted_bp_dict[bp_name].next_y_friend.bp_name].goal_name = self.agents[bp_name].goal_name
        #     else:
        #         self.find_nearest_unassigned_agent(friend.bp_name)


    def attack_enemy(self, enemy_class: FixedPolicy):
        # 持续追踪打击
        for self_bp_name in self.attack_flag:
            if self.attack_flag[self_bp_name]:
                try:
                    pose = self.enemy_position[self.attack_flag[self_bp_name]]
                except KeyError:
                    self.attack_flag[self_bp_name] = 0
                    continue
                self.agents[self_bp_name].goal_position = pose
                # self.env.agents[self_id].goal_position[1] = pose[1]
                enemy_class.get_caught[self.attack_flag[self_bp_name]] = 1

                self.plot_attack([self_bp_name], enemy_class.position_dict[self.attack_flag[self_bp_name]], '127.0.0.1', color=[0.0, 0.0, 1.0, 1.0])

    def get_nearest_enemy(self, bp_name):
        # bp_name = self.bp_names_list[i]
        dis, enemy = self.sorted_bp_dict[bp_name].get_nearest_enemy()
        return enemy.bp_name, dis

    def detect_destroy_distance(self,  enemy_class: FixedPolicy):
        attacked_enemy = {}
        for bp_name in self.attack_flag:
            if not self.attack_flag[bp_name] and not self.agents[bp_name].wait_step:
                enemy_name, dis = self.get_nearest_enemy(bp_name)
                if enemy_name not in self.destroyed_enemy and dis < self.destroy_distance + 40:
                    self.attack_flag[bp_name] = enemy_name
                    self.agents[bp_name].goal_name = enemy_name
                    self.attacked_enemy[enemy_name].append(bp_name)
            if self.attack_flag[bp_name] and not self.agents[bp_name].wait_step:

                try:
                    attacked_enemy[self.attack_flag[bp_name]]
                except KeyError:
                    attacked_enemy[self.attack_flag[bp_name]] = []
                if get_distance(self.position_dict[bp_name], enemy_class.position_dict[self.attack_flag[bp_name]]) < self.destroy_distance:
                    attacked_enemy[self.attack_flag[bp_name]].append(bp_name)
        # 满足条件后触发导弹攻击
        for enemy_bp_name in attacked_enemy:
            if len(attacked_enemy[enemy_bp_name]) >= 2:
                # for self_bp_name in attacked_enemy[enemy_bp_name]:
                #     self.attack_flag[self_bp_name] = 0
                for bp_name in self.attack_flag:
                    if self.attack_flag[bp_name] == enemy_bp_name:
                        self.attack_flag[bp_name] = 0
                _ = threading.Thread(target=self.destroy_enemy, args=[attacked_enemy[enemy_bp_name], enemy_class, enemy_bp_name])
                _.start()


        # for _, enemy in enumerate(self.attack_flag):
        #     if enemy:
        #         friendly_force = np.where(self.attack_flag == enemy)[0]
        #         target_enemy_pose = self.enemy_position[enemy]
        #         cnt = 0
        #         for agent in friendly_force:
        #             if np.sqrt(np.power(self.position_dict[agent][0]-target_enemy_pose[0], 2) + np.power(self.position_dict[agent][1]-target_enemy_pose[1], 2)) < self.destroy_distance:
        #                 cnt += 1
        #         if cnt >= 2:
        #             self.attack_flag[friendly_force] = 0
        #             _ = threading.Thread(target=self.destroy_enemy, args=[friendly_force, enemy_class, enemy])
        #             _.start()

    def destroy_enemy(self, friendly_force, enemy_class: FixedPolicy, enemy_bp_name):
        # enemy_class.client.simSetVehiclePose(self.enemy_position[enemy], True, enemy)
        self.destroyed_enemy.append(enemy_bp_name)
        enemy_class.client.enableApiControl(False, enemy_bp_name)
        for self_bp_name in friendly_force:
            self.agents[self_bp_name].wait_step = 10
        while self.agents[friendly_force[0]].wait_step > 0:
            time.sleep(0.5)
            self.plot_attack(friendly_force, enemy_class.position_dict[enemy_bp_name])
        enemy_class.destroy_vehicle(enemy_bp_name)
        # self.enemy_position.pop(enemy_bp_name)
    # def update_enemy_pose(self, enemy_class: FixedPolicy):
    #     while True:
    #         for bp_name in enemy_class.remained_vehicle:
    #             self.enemy_position[bp_name] = enemy_class.client.simGetObjectPose(bp_name)

    def cancel_task(self):
        self.task_cancel = True
        for agent in self.env.agents:
            agent.goal_position = [agent.x, agent.y]

    def attack_run(self, enemy_class: FixedPolicy):
        for bp_name in self.sorted_bp_dict:
            self.sorted_bp_dict[bp_name].first_sort_enemy(enemy_class.position_dict, enemy_class.sorted_bp_dict)
        for enemy_bp_name in enemy_class.position_dict:
            self.attacked_enemy[enemy_bp_name] = []
        self.find_first_enemy(enemy_class)

        while len(enemy_class.remained_vehicle):
            if self.task_cancel:
                self.task_cancel = False
                print("attack mission canceled")
                break
            for bp_name in enemy_class.remained_vehicle:
                self.enemy_position[bp_name] = enemy_class.position_dict[bp_name]
            for bp_name in self.sorted_bp_dict:
                self.sorted_bp_dict[bp_name].resort_friend()
                self.sorted_bp_dict[bp_name].resort_enemy()
            # 发现目标的无人机组织协同打击
            self.assign_attack()
            # 对召集到的无人机分配进攻目标
            self.assign_enemy(enemy_class)
            # 开启线程进行持续追踪打击，并且防守方人机反方向逃离
            self.attack_enemy(enemy_class)
            # 判断是否存在摧毁距离内的无人机，若有，发射导弹
            self.detect_destroy_distance(enemy_class)
            time.sleep(1)
        if not enemy_class.remained_vehicle:
            print("attack mission done")

    def plot_attack(self, friendly_force, enemy_pose, ip='127.0.0.1', color=None):
        if color is None:
            color = [1.0, 1.0, 0.0, 1.0]
        plot_client = airsim.MultirotorClient(ip)

        for name in friendly_force:
            plot_client.simPlotLineList([airsim.Vector3r(self.position_dict[name][0], self.position_dict[name][1], -40)
                                        , airsim.Vector3r(enemy_pose[0], enemy_pose[1], -40)], thickness=50.0, duration=0.2,
                                        color_rgba=color,
                                        is_persistent=False)
            # time.sleep(0.1)



if __name__ == "__main__":
    default_cfg = 'D:/crazyflie-simulation/airsim_mappo/onpolicy/envs/airsim_envs/cfg/default.cfg'
    cfg = Myconf()
    cfg.read(default_cfg)
    for each in cfg.items("algorithm"):
        cfg.__dict__[each[0]] = each[1]
    if cfg.getboolean('algorithm', 'cuda') and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))
        if cfg.getboolean('algorithm', 'cuda_deterministic'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))

    # seed
    torch.manual_seed(cfg.getint('algorithm', 'seed'))
    torch.cuda.manual_seed_all(cfg.getint('algorithm', 'seed'))
    np.random.seed(cfg.getint('algorithm', 'seed'))

    # env init
    env = AirSimDroneEnv(cfg)
    num_agents = cfg.getint('options', 'num_of_drone')

    config = {
        "cfg": cfg,
        "envs": env,
        "num_agents": num_agents,
        "device": device
    }

    # load model
    policy_actor_state_dict = torch.load(str(cfg.get("algorithm", 'model_dir')) + '/actor.pt')
    actor1 = R_Actor(config['cfg'], config['envs'].observation_space[0], config['envs'].action_space[0], config['device'])
    actor1.load_state_dict(policy_actor_state_dict)

    patrol_drones = FixedPolicy("patrol_50.txt", ["127.0.0.1:41451", '10.134.142.129:41451'], 9699)
    attack_drones = TrainedPolicy(actor1, env)
    patrol_drones.fly_run()
    attack_drones.fly_run()
    while True:
        attack_drones.attack_run(patrol_drones)

    # a.fly_run()
