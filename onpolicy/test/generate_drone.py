import numpy as np
import random

point = [
    [[-128863.24, -114447.57], [-91591.03, -114447.57], [-62305.74, -114743.38], [-33464.17, -115039.19],[1145.71, -115039.19]],
    [[-124869.83, -81020.90], [-86858.10, -82647.84], [None, None], [-32576.76, -86493.40], [-6545.39, -86493.40]],
    [[-121763.81, -56468.54], [-82273.03, -59130.85], [-61122.55, -60314.10], [-33020.50, -61645.26],[-9207.71, -62384.79]],
    [[None, None], [-78279.58, -38424.10], [-64672.23, -33691.10], [-30358.14, -37684.57], [-10834.62, -37684.57]],
    [[-112889.42, -1299.81], [-73546.49, -5145.36], [-61714.03, -7955.56], [-29470.64, -15942.45], [-10920.83, -21404.53]]
]

lineRow = [
    [[0, 0], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 3]], [[0, 3], [0, 4]],
    [[1, 0], [1, 1]], [[1, 3], [1, 4]],
    [[2, 0], [2, 1]], [[2, 1], [2, 2]], [[2, 2], [2, 3]], [[2, 3], [2, 4]],
    [[3, 3], [3, 4]],
    [[4, 0], [4, 1]], [[4, 1], [4, 2]], [[4, 2], [4, 3]], [[4, 3], [4, 4]]
]

lineColumn = [
    [[0, 0], [1, 0]], [[1, 0], [2, 0]],
    [[0, 1], [1, 1]], [[1, 1], [2, 1]], [[2, 1], [3, 1]], [[3, 1], [4, 1]],
    [[0, 2], [2, 2]], [[2, 2], [3, 2]], [[3, 2], [4, 2]],
    [[0, 3], [1, 3]], [[1, 3], [2, 3]], [[2, 3], [3, 3]], [[3, 3], [4, 3]],
    [[0, 4], [1, 4]], [[1, 4], [2, 4]], [[2, 4], [3, 4]], [[3, 4], [4, 4]]
]

minDis = 6080.796281976566
dronenum = 132


def write(way):
    # w打开只写文件，若文件存在则文件长度清为0，即该文件内容会消失。若文件不存在则建立该文件
    file = open("out.txt", "w")
    li = [[1, 2, 3, 4], [5, 6, 7, 8]]
    for i in range(len(li)):
        # 将输出语句的内容写入到文件中, 使用end=?来控制是否换行
        print(li[i], file=file)
    file.close()
    file = open("out2.txt", "w")
    # 将26个字母写入到文件中
    for i in range(26):
        # chr函数可以将ascii码转换为具体的字符写入到文件中
        print(chr(i + 97), file=file)
    file.close()


def euclidean(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def cal_splite_num(lineList):
    splite_num = []
    splite_way = []
    for line in lineList:
        sp_line = []
        p1 = point[line[0][0]][line[0][1]]
        p2 = point[line[1][0]][line[1][1]]
        ##随机偏移
        p1[0] = p1[0] + random.randint(-100, 100)
        p1[1] = p1[1] + random.randint(-100, 100)
        p2[0] = p2[0] + random.randint(-100, 100)
        p2[1] = p2[1] + random.randint(-100, 100)

        vec = [p2[0] - p1[0], p2[1] - p1[1]]
        eul = euclidean(p1, p2)
        n_t = int(eul / minDis)
        pp = p1
        for i in range(n_t):
            pt = [
                pp[0] + vec[0] / n_t,
                pp[1] + vec[1] / n_t
            ]
            sp_line.append([pp, pt])
            pp = pt
        splite_num.append(len(sp_line))
        for way in sp_line:
            splite_way.append(way)
    return splite_num, splite_way

n1,way1 = cal_splite_num(lineRow)
n2,way2 = cal_splite_num(lineColumn)

totole_num = 0
for t in n1:
    totole_num = totole_num + t
for t in n2:
    totole_num = totole_num + t

print(totole_num)
print(way1 + way2)

waytotle = way1 + way2
file = open("patrol_100.txt", "w")
for i in range(100):
    p1 = waytotle[i][0]
    p2 = waytotle[i][1]
    if (random.randint(0, 1)):
        print(p1[0], p1[1], p2[0], p2[1], file=file)
    else:
        print(p2[0], p2[1], p1[0], p1[1], file=file)

file.close()