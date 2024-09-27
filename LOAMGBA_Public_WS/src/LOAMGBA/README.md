# LOAMGBA

## Prerequisites (dependencies)
- Ubuntu 18.04

    ceres-solver: 2.0.0

     eigen:3.3.5

    gtsam:4.1.0

    

## How to use? 
- First, install the above mentioned dependencies, and follow below lines. 
```
	cd ~/LOAMGBA_Public_WS/
    catkin_make
    source devel/setup.bash
    roslaunch faloamBA faloam.launch 
    rosbag play kitii00part.bag
```

NOTICE:

​	1.如果要可视化回环效果，则需要打开回环节点alaserPGO，并将rviz文件替换成

​	2.如果需要播放80线数据的包(80.bag)，将scan_line改成80即可

​	3.如果是Ubuntu 20.04系统，需要把#include<cv.hpp> 改成 #include<opencv2/opencv.hpp>，

并且把话题中的"/"去掉。
