import pickle
from matplotlib import pyplot as plt
import numpy as np

f = open('./results/simple_tag_v2/7/rewards.pkl','rb')
data = pickle.load(f)
adv_0 = data['rewards']['adversary_0']
adv_1 = data['rewards']['adversary_1']
adv_2 = data['rewards']['adversary_2']
good = data['rewards']['agent_0']
rew_adv_0 = [0]
rew_adv_1 = [0]
rew_adv_2 = [0]
rew_good = [0]
x = [0]
cnt_adv_0 = 0
cnt_adv_1 = 0
cnt_adv_2 = 0
cnt_good = 0
for i in range(30000):
    if (i+1) % 100 == 0:
        x.append(i+1)
        rew_adv_0.append(cnt_adv_0 / 100)
        rew_adv_1.append(cnt_adv_1 / 100)
        rew_adv_2.append(cnt_adv_2 / 100)
        rew_good.append(cnt_good / 100)
        cnt_adv_0 = 0
        cnt_adv_1 = 0
        cnt_adv_2 = 0
        cnt_good = 0
    cnt_adv_0 = cnt_adv_0 + adv_0[i]
    cnt_adv_1 = cnt_adv_1 + adv_1[i]
    cnt_adv_2 = cnt_adv_2 + adv_2[i]
    cnt_good = cnt_good + good[i]
plt.plot(x, rew_adv_0)
plt.plot(x, rew_adv_1)
plt.plot(x, rew_adv_2)
plt.plot(x, rew_good)
plt.xlabel('episode num')
plt.ylabel('rewards')
plt.legend(['adv_0','adv_1','adv_2','good'])
plt.show()