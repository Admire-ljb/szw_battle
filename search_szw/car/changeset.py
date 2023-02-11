import json

def changeset():
    docpath = r'C:\Users\Administrator\Documents\AirSim'

    a = {
      "SettingsVersion": 1.2,
      "SimMode": "Car",

      "Vehicles": {

      }
    }

    f = open("map.txt", "r",encoding='utf-8')
    line = f.readline()
    global dic
    dic = {}
    count = 0

    while line:
        #print(count)
        if(line == " " or line == "\n"):
            break
        data = eval(line)
        dic[count] = {}
        dic[count]["pos"] = (data[0],data[1])
        dic[count]["link"] = []
        for i in range(2,len(data)):
            dic[count]["link"].append(data[i])
        count += 1
        line = f.readline()


    start_pos = [1,1,1,6,6,6,3,3,3,7,7,7,18,18,18,19,19,19,12,12,12,13,13,13,22,22,22,23,23,23,11,11,11,21,21,21]
    file = open('node.txt', 'w')
    mid = str(start_pos).replace('[', '').replace(']', '')
    mid = mid.replace("'", '').replace(',', '')
    file.write(mid)
    file.close()


    print(len(start_pos))
    countings = {}
    offset = [0,-1,1,-2,2]

    new = a["Vehicles"]
    for i in range(len(start_pos)):
        if start_pos[i] not in countings:
            countings[start_pos[i]] = 0
        else:
            countings[start_pos[i]] += 1

        strs = "Car" + str(i+1)
        new[strs] = {}
        new[strs]["VehicleType"] = "PhysXCar"
        new[strs]["X"] = dic[start_pos[i]]["pos"][0]
        new[strs]["Y"] = dic[start_pos[i]]["pos"][1] + offset[countings[start_pos[i]]] * 10
        new[strs]["Z"] = -4
        new[strs]["Yaw"] = 0


    print(a)
    b = json.dumps(a,indent=2)
    f2 = open(docpath, 'w')
    f2.write(b)
    f2.close()


if __name__ == '__main__':
    changeset()