改变6个人物位置只需要调整self.people_pos，其与self.name一一对应 不用修改txt和ue4
```
self.name = ['carla_3', 'claudia_2', 'eric_2', 'manuel_2', 'nathan_2', 'sophia_2']
self.people_pos = [[-59377.523438, 50322.0], [-138381.0, 84595.0], [28271.0, -106163.0],
                   [-122676.9375, -63717.0], [-160053.0, 47145.0], [-100104.0, -53483.0]]
```

self.find_all == True时，6个人物均找到

人物框的绘制使用的是airsim里的函数 其中is_persistent=True

车辆port 9499
