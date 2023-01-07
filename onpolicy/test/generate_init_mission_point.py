import numpy as np

center = [-1332.0, -16.0]
R = 80000
Num = 20
line_List = []
for i in range(Num):
    pt = [0, 0]
    pt[0] = center[0] + R * np.cos(2 * np.pi / Num * i)
    pt[1] = center[1] + R * np.sin(2 * np.pi / Num * i)
    line_List.append(pt)