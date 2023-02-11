import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import math
from socket import *


def findAllPath(graph,start,end):
    path=[]
    stack=[]
    stack.append(start)
    visited=set()
    visited.add(start)
    seen_path={}
    #seen_node=[]
    while (len(stack)>0):
        start=stack[-1]
        nodes=graph[start]
        if start not in seen_path.keys():
            seen_path[start]=[]
        g=0
        for w in nodes:
            if w not in visited and w not in seen_path[start]:
                g=g+1
                stack.append(w)
                visited.add(w)
                seen_path[start].append(w)
                if w==end:
                    path.append(list(stack))
                    old_pop=stack.pop()
                    visited.remove(old_pop)
                break
        if g==0:
            old_pop=stack.pop()
            del seen_path[old_pop]
            visited.remove(old_pop)
    return path


def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


def all_arrive():
    sum = 0
    for i in finish:
        sum += i
    if sum == cars_num:
        return 0
    else:
        return 1


def control_choice(pos,witchcar,index,angel):
    angel_to = (math.atan2(des[witchcar][index][1] - pos[1],des[witchcar][index][0] - pos[0]))*180/3.14
    delta = angel_to - angel
    if delta > 180:
        delta = delta - 360
    elif delta < -180:
        delta = delta + 360
    angel_offset = 3
    if abs(pos[0] - des[witchcar][index][0]) < 10 and abs (pos[1] - des[witchcar][index][1]) < 10:
            #print("arrive")
        return 1,index+1
    elif abs(pos[0] - des[witchcar][index][0]) < 60 and abs (pos[1] - des[witchcar][index][1]) < 60:
        return 5,index
    elif  abs(delta) > angel_offset :
        #print("delta:",delta)
        if delta > 0:
            return 2,index
        elif delta < 0:
            return 3,index
    else:
        return 4,index


def control(controller):
    if controller == 1:
        return 0,0,1
    elif controller == 2:
        return 0.5,0.3,0 #right
    elif controller == 3:
        return 0.5,-0.3,0 #left
    elif controller == 5:
        return 0,0,1
    else:
        return 1.5,0,0


class Sendmsg:
    def __init__(self, socket_server):
        self.buffsize = 2048
        self.s = socket_server
        self.conn_list = []
        self.conn_dt = {}
        self.connected_port = []
        self.drone_num = 36
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

def getpoint(destination_people,dic):
    destination_point = []
    for i in destination_people:
        distance = -1
        for j in dic:
            nowdis = (dic[j]["pos"][0] - i[0])**2 +(dic[j]["pos"][1] - i[1])**2
            if distance == -1:
                distance = nowdis
                short = j
            elif distance > nowdis:
                distance = nowdis
                short = j

        destination_point.append(short)
    return destination_point


def start_search(destination_people_pose):
    global dic

    txt_tables = []
    f = open("map.txt", "r",encoding='utf-8')
    line = f.readline()

    dic = {}
    count = 0

    s = socket(AF_INET, SOCK_STREAM)
    s.bind(('127.0.0.1', 9499))
    s.listen(200)  # 最大连接数
    c1 = Sendmsg(s)
    time.sleep(1)
    while line:
        #print(count)
        if(line == " " or line == "\n"):
            break
        data = eval(line)
        dic[count] = {}
        dic[count]["pos"] = (data[0], data[1])
        dic[count]["link"] = []
        for i in range(2,len(data)):
            dic[count]["link"].append(data[i])
        count += 1
        line = f.readline()
    #print(dic)
    maps = {}
    for i in dic:
        maps[i] = dic[i]["link"]


    #print(map)


    global cars_num



    with open('node.txt', 'r', encoding='utf-8') as f3:
        data = f3.readlines()
        line = data[0]
        start_pos = list(map(int, line.split()))
    f3.close()

    start = start_pos.copy()
    print(start)
    cars_num = len(start)
    print(cars_num)
    global finish
    finish = [0 for i in range(cars_num)]

    global des

    destination_people = getpoint(destination_people_pose, dic)
    destination = destination_people  #可以返回营救人员位置 可以改成无人机提供的终点坐标，destination里面装的是6个目的地组成的数组，从外面传入
    i = 0
    #[2,2,2,6,6,12,12,16,16,18,18,18,8]  #起点
    des = []
    fpath = []
    des_choice = {}

    for i in destination:
        des_choice[i] = 0


    each_descar = cars_num/len(destination)

    for i in range(len(start)):
        path_long = 100
        path_now = []
        for r in range(len(destination)):
            if des_choice[destination[r]] >= each_descar:
                continue
            path_now = find_shortest_path(maps,start[i],destination[r],path=[])
            if len(path_now) < path_long:
                fpath = path_now
                path_long = len(fpath)

        des_choice[fpath[-1]] += 1
        print(fpath)
        for j in range(1,len(fpath)):
            fpath[j] = dic[fpath[j]]["pos"]
        des.append(fpath[1:])

    print(des_choice)
    print(des)


    client = airsim.CarClient()
    client.reset()
    client.confirmConnection()
    for i in range(1,cars_num+1):
        strs = "Car" + str(i)
        client.enableApiControl(True, strs)

    # client.enableApiControl(True, "Car1")
    # client.enableApiControl(True, "Car2")
    # client.enableApiControl(True, "Car3")
    # client.enableApiControl(True, "Car4")
    # client.enableApiControl(True, "Car5")
    # client.enableApiControl(True, "Car6")
    # client.enableApiControl(True, "Car7")

    Carctrl = []

    for i in range(1,cars_num+1):
        controller = airsim.CarControls()
        Carctrl.append(controller)
    # car_controls1 = airsim.CarControls()
    # car_controls2 = airsim.CarControls()
    # car_controls3 = airsim.CarControls()
    # car_controls4 = airsim.CarControls()
    # car_controls5 = airsim.CarControls()
    # car_controls6 = airsim.CarControls()
    # car_controls7 = airsim.CarControls()

    index = [0 for i in range(cars_num+1)]

    episode = 0
    while all_arrive():
        print(cars_num)
        for cnum in range(1, cars_num+1):
            # get state of the car
            strs = "Car" + str(cnum)
            print("finish:",finish)
            print(cnum,strs)
            car_state = client.getCarState(strs)
            oz = car_state.kinematics_estimated.orientation.z_val
            angel = math.asin(oz) * 360 / 3.14
            pose = client.simGetObjectPose(strs)
            simget_pos = pose.position
            euler = np.rad2deg(airsim.to_eularian_angles(pose.orientation))
            i = index[cnum - 1]
            car_controls = Carctrl[cnum -1]

            if finish[cnum - 1]:
                continue
            if i == len(des[cnum-1]):
                car_controls.brake = 1
                client.setCarControls(car_controls,strs)
                finish[cnum -1] = 1
                continue
            x = simget_pos.x_val
            y = simget_pos.y_val
            z = simget_pos.z_val
            controller, i = control_choice([x, y], cnum-1,i, angel)
            index[cnum - 1] = i
            car_controls.throttle, car_controls.steering, car_controls.brake = control(controller)
            vx = car_state.kinematics_estimated.linear_velocity.x_val
            vy = car_state.kinematics_estimated.linear_velocity.y_val
            if vx ** 2 + vy ** 2 < 100 and controller == 5:
                car_controls.brake = 0
                car_controls.throttle = 0.5
            elif vx ** 2 + vy ** 2 > 400:
                car_controls.throttle = 0
            client.setCarControls(car_controls,strs)

            msg = '_' + str(x * 100) + '_' + str(y * 100) + '_' + str(z*100) + '_' + str(euler[0]) + '_' + str(euler[1]
                ) + '_' + str(euler[2]) + '_' + '(R=0,G=1)'
            c1.conn_dt[c1.conn_list[cnum-1]].sendall(msg.encode('utf-8'))
            if cnum == 3:
                print(car_controls.brake)
            # print(strs)
            # print(angel)
            # print(i)
            # if controller == 1 or controller == 5:
            #     print(i, controller)
            #     print("vel:")
            #     print(car_state.kinematics_estimated.linear_velocity)
            #     print("pos:")
            #     print(simget_pos)
            # if i == 3:
            #     print(i, controller)
            #     print("vel:")
            #     print(car_state.kinematics_estimated.linear_velocity)
            #     print("pos:")
            #     print(simget_pos)

            # x = car_state.kinematics_estimated.position.x_val
            # y = car_state.kinematics_estimated.position.y_val
            # ox = car_state.kinematics_estimated.orientation.x_val
            # oy = car_state.kinematics_estimated.orientation.y_val
            # if controller == 5:
            #     time.sleep(1)
        episode += 1
        print(episode)
        time.sleep(0.05)

if __name__ == '__main__':
    start_search([13,14,16,17,10,18])