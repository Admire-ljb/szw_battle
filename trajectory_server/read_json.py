import json
import numpy as np


def read_json(traj_name):
    att = []
    defender = []
    atk_flag = 0
    def_flag = 0
    with open(traj_name, 'r') as f:
        cnt = 1
        data_read = json.load(f)
        while not atk_flag or not def_flag:
            try:
                # att.append(np.asarray(data_read['att']['predator' + str(cnt)]))
                att.append(np.asarray(data_read['att']['agent' + str(cnt)]))
            except KeyError:
                atk_flag = 1
            try:
                # defender.append(np.asarray(data_read['def']['prey' + str(cnt)]))
                defender.append(np.asarray(data_read['def']['agent' + str(cnt)]))
            except KeyError:
                def_flag = 1
            cnt += 1
    return att, defender


