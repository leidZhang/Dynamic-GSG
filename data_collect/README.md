# Deployment with Realsense D455

- We made some modifications based on vins-fusion so that it can publish timestamp-aligned RGB-Depth-Pose data frames in the ros1 environment.

- We provide the modified vins-fusion docker image and the code for data collection using Realsense D455.

- We developed the deployment code on ubuntu 20.04 using ROS noetic and nvidia-docker. Make sure you installed them correctly.

## Installation

- Compile data collection workspace
```
cd ~/data_collection
catkin_make
source devel/setup.bash
```

- Download the docker image from <a href="https://drive.google.com/drive/folders/1tRXL-aOD62UBWlYrZRSofQiMRl7PEOLX?usp=sharing">Google Driver</a> and load the image.
- Create a container.
```
# load the image
docker load -i vins_RGBDP.tar vins_RGBDP:1.0

# create a container.
docker run -it --privileged --name vins_RGBDP  --net=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=unix$DISPLAY \
-e GDK_SCALE \
-e GDK_DPI_SCALE \
-v /dev:/dev \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e QT_X11_NO_MITSHM=1 \
--gpus all \
vins_RGBDP:1.0 /bin/bash
```

## Usage

**Start Vins-Fusion in Docker**

- Open two terminals in docker, one for visualization and one for starting the vins node

```
# terminal one
cd /root/catkin_ws
source devel/setup.bash
roslaunch vins vins_rviz.launch

# terminal two
cd /root/catkin_ws
source devel/setup.bash
rosrun vins vins_node /root/catkin_ws/src/VINS-Fusion/config/realsense_d455/realsense_stereo_imu_config.yaml
```

**Start Realsense Camera and Data Collection**

- Download the <a href="https://drive.google.com/drive/folders/1tRXL-aOD62UBWlYrZRSofQiMRl7PEOLX?usp=sharing">realsense launch file</a> we tested.
- Modify the file save path in src/vins_saver/scripts/saver.py

```
# start Realsense D455
roslaunch realsense2_camera rs_camera_d455.launch

# Data Collection
cd ~/data_collection
source devel/setup.bash
rosrun vins_saver saver.py
```






