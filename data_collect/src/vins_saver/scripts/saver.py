#!/usr/bin/env python3

import os
from dataclasses import dataclass
import rospy
import time
import numpy as np
import threading
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R
from vins_saver.msg import RGBDPoseMsg

@dataclass
class RGBDPose:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray

class Saver:
    def __init__(self):
        self.bridge = CvBridge()
        self.msgs = []
        self.save_dir = "/home/woosh/DynamicGSG/realsense_data/"
        self.save_dir += time.strftime("/%Y-%m-%d-%H-%M-%S", time.localtime())
        self.rgb_dir = self.save_dir + "/rgb"
        self.depth_dir = self.save_dir + "/depth"
        self.pose_dir = self.save_dir + "/poses"
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        os.makedirs(self.pose_dir)
        self.i = [0, 0]
        rospy.Subscriber('/vins_estimator/RGBDPose', RGBDPoseMsg, self.callback, queue_size=1000)
        self.running = True
        self.thread = threading.Thread(target=self.writer)
        self.thread.start()
        input("Press Enter to stop\n")
        rospy.signal_shutdown("User stopped")
        self.close()

    def callback(self, msg):
        print(f"r{self.i[0]} w{self.i[1]}", end="\r")
        
        msg_data = RGBDPose(self.bridge.imgmsg_to_cv2(msg.rgb_image, "rgb8"),
                            self.bridge.imgmsg_to_cv2(msg.depth_image, "mono16"),
                            np.array([msg.pose.pose.position.x,
                                      msg.pose.pose.position.y,
                                      msg.pose.pose.position.z,
                                      msg.pose.pose.orientation.x,
                                      msg.pose.pose.orientation.y,
                                      msg.pose.pose.orientation.z,
                                      msg.pose.pose.orientation.w]))
        self.msgs.append(msg_data)
        self.i[0] += 1

    def quaternion_to_c2w(self, q):
        r = R.from_quat(q[3:]).as_matrix() # xyzw for 1.10
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = q[:3]
        return c2w
        
    def writer(self):
        while self.running:
            if len(self.msgs) > 0:
                print(f"r{self.i[0]} w{self.i[1]}", end="\r")
                num_str = f"{self.i[1]:06d}"
                data = self.msgs.pop(0)
                cv2.imwrite(f"{self.rgb_dir}/{num_str}.jpg", data.rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(f"{self.depth_dir}/{num_str}.png", data.depth)
                c2w = self.quaternion_to_c2w(data.pose)
                np.save(f"{self.pose_dir}/{num_str}.npy", c2w)
                self.i[1] += 1
    
    def close(self):
        print("Closing")
        while len(self.msgs) > 0:
            time.sleep(0.1)
        self.running = False
        self.thread.join()

if __name__ == '__main__':
    rospy.init_node('saver')
    s = Saver()
