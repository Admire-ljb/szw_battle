from mult_definedes import *
from mtsp_policy_50_6_find import MtspPolicy
import time

if __name__ == "__main__":
    drone_swarm = MtspPolicy("road.txt", ['10.134.142.129:41451', '10.134.143.20:41453'], 9699)
    time.sleep(1)
    drone_swarm.fly_run()
    # drone_swarm.find_all = True
    # for debug
    while True:
        if drone_swarm.find_all:
            print('all people find')
            break
        time.sleep(1)

    start_search([(-880, -1700), (-341, -1714), (-825, 591), (-336, -1159), (-292, -153), (284, -1159)])

