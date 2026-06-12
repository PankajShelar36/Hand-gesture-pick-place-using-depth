# 🤖 Vision-Guided Pick and Place System using UR10, ROS 2 & Intel RealSense

## 📌 Overview

This project implements a vision-guided robotic manipulation system using a Universal Robots UR10 manipulator, Intel RealSense D435i camera, and ROS 2 Humble.

The system performs real-time object detection, pose estimation, pick-and-place operations, voice-controlled picking, gesture-based handover, and collision monitoring using point cloud data.

The project supports both:

* Simulation using UR ROS 2 Driver
* Real UR10 Robot Deployment

---

# 🛠️ Hardware Used

* Universal Robots UR10
* Intel RealSense D435i
* Robotiq Gripper
* Ubuntu 22.04
* ROS 2 Humble
* Industrial Ethernet Network

---

# 💻 Software Stack

* ROS 2 Humble
* MoveIt 2
* UR ROS 2 Driver
* Intel RealSense ROS Driver
* OpenCV
* TF2
* Point Cloud Processing
* Python
* RViz

---

# 🚀 Features

✅ Vision Guided Pick and Place

✅ Real-Time Object Detection

✅ Voice Command Based Picking

✅ Any-Pose Object Detection

✅ Gesture-Based Human Handover

✅ Point Cloud Collision Monitoring

✅ UR10 Motion Planning using MoveIt 2

✅ Real Robot and Simulation Support

---

# 📂 System Architecture

```text
RealSense Camera
        │
        ▼
Object Detection Node
        │
        ▼
Object Pose Estimation
        │
        ▼
Pick Node
        │
        ▼
MoveIt Motion Planning
        │
        ▼
UR10 Robot + Robotiq Gripper
```

# ⚙️ Installation

```bash
mkdir -p ~/ur_camera_ws/src

cd ~/ur_camera_ws

colcon build

source install/setup.bash
```

---

# ▶️ Running the System

## Reset ROS 2 Daemon

```bash
ros2 daemon stop
ros2 daemon start
```

---

## Terminal 1 – UR Driver

### Simulation

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
ur_type:=ur10 \
robot_ip:=192.168.1.1 \
use_fake_hardware:=true \
launch_rviz:=true
```

### Real Robot

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
ur_type:=ur10 \
robot_ip:=192.168.1.102 \
launch_rviz:=true
```

---

## Terminal 2 – Intel RealSense D435i

```bash
source /opt/ros/humble/setup.bash

ros2 launch realsense2_camera rs_launch.py \
align_depth.enable:=true
```

---

## Terminal 3 – MoveIt 2

### Simulation

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
ur_type:=ur10 \
use_fake_hardware:=true \
launch_rviz:=false
```

### Real Robot

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
ur_type:=ur10 \
launch_rviz:=true
```

---

## Terminal 4 – Camera TF Publisher

```bash
ros2 run tf2_ros static_transform_publisher \
--x 0.0 \
--y -1.18 \
--z 0.793 \
--roll 0.0 \
--pitch 0.935 \
--yaw 1.5708 \
--frame-id base \
--child-frame-id camera_link
```

---

## Terminal 5 – Detection Nodes

### Voice Command Detection

```bash
cd ~/ur_camera_ws

source install/setup.bash

export DISPLAY=:1

ros2 run pick_place voice_color_detector_rs
```

### Any-Pose Detection

```bash
cd ~/ur_camera_ws

source install/setup.bash

export DISPLAY=:1

ros2 run pick_place any_pose_detector_rs
```

---

## Terminal 6 – Robotiq Gripper

### Start Gripper

```bash
cd ~/ur_camera_ws/src

python3 updatedcode2.py
```

### Gripper Testing

```bash
cd ~/ur_camera_ws/src

python3 gripper_test.py
```

---

## Terminal 7 – Pick Nodes

### Voice Controlled Pick

```bash
cd ~/ur_camera_ws

source install/setup.bash

ros2 run pick_place voice_pick_node
```

### Any-Pose Pick

```bash
cd ~/ur_camera_ws

source install/setup.bash

ros2 run pick_place pick_node_any_pose
```

### Gesture Handover

```bash
cd ~/ur_camera_ws

source install/setup.bash

ros2 run pick_place gesture_handover_pick_node
```

### Point Cloud Collision Monitoring

```bash
cd ~/ur_camera_ws

source install/setup.bash

ros2 run pick_place pointcloud_collision_monitor
```

---

# 🔄 Code Backup / Branch Recovery

### Revert to Red-Only Detection

```bash
git checkout working-baseline -- src/pick_place/pick_place/
```

### Revert to Any-Pose Version

```bash
git checkout 03605e2 -- src/pick_place/pick_place/
```

---

# 📊 Applications

* Industrial Pick and Place
* Vision-Guided Manipulation
* Smart Manufacturing
* Human Robot Collaboration
* AI-Based Robotic Automation
* Warehouse Automation

