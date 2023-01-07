import argparse
import os
from socket import *
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2
import time
from MADDPG import MADDPG
import threading

def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


class Sendmsg:
    def __init__(self, socket_server):
        self.buffsize = 2048
        self.s = socket_server
        self.conn_list = []
        self.conn_dt = {}
        self.connected_port = []
        self.drone_num = 4
        self.recs()
        # self.t1 = threading.Thread(target=self.recs, args=(), name='rec')
        # self.t2 = threading.Thread(target=self.sds, args=(), name='send')

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
            if clientaddress[1] not in self.connected_port:
                self.conn_list.append(clientaddress)
                self.conn_dt[clientaddress] = clientsock
                self.connected_port.append(clientaddress[1])
                print('connect from:', clientaddress)

    # def sds(self):
    #     k = 0
    #     time.sleep(10)
    #     while True:
    #         for i in range(len(self.conn_list)):
    #             if i < 40:
    #                 pos = np_data_red[i*100001+k][0].split(" ")
    #                 rpy = np_data_red[i*100001+k][1].split(" ")
    #             else:
    #                 pos = np_data_blue[(i-40) * 100001 + k][0].split(" ")
    #                 rpy = np_data_blue[(i-40) * 100001 + k][1].split(" ")
    #             msg = '_' + str(float(pos[0])*100) + '_' + str(-float(pos[1])*100) \
    #                   + '_' + str(float(pos[2])*100) + '_' + \
    #                   rpy[0] + '_' + rpy[1] + '_' + str(float(rpy[2])+180)
    #             # print(msg)
    #             self.conn_dt[self.conn_list[i]].sendall(msg.encode('utf-8'))
    #             # i += 1
    #             # if i % 40 == 0:
    #             #     i = 0
    #         time.sleep(0.001)
    #         k += 1

    def run(self):
        self.t2.start()


if __name__ == '__main__':
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('127.0.0.1', 9899))
    s.listen(200)  # 最大连接数
    c1 = Sendmsg(s)


    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_tag_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)
            # print(action)
            a = env.step(action)
            # print(a)
            next_obs, reward, done, _, info = a
            for index, each in enumerate(next_obs):
                if each[0:2] == 'ag':
                    color = '(R=0,G=1)'
                else:
                    color = '(R=1)'
                x = next_obs[each][2]
                y = next_obs[each][3]
                z = 100
                msg = '_' + str(x * 10000) + '_' + str(y * 10000) + '_' + str(z*100) + '_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + color
                c1.conn_dt[c1.conn_list[index]].sendall(msg.encode('utf-8'))
            env.render()
            maddpg.add(obs, action, reward, next_obs, done)
            time.sleep(0.15)
            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
