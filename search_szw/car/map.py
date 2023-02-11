txt_tables = []
f = open("map.txt", "r",encoding='utf-8')
line = f.readline()
global dic
dic = {}
count = 0

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
#print(dic)
map = {}
for i in dic:
    map[i] = dic[i]["link"]


#print(map)

a = 14
b = 16

path = find_shortest_path(map,6,14)
print(map)
