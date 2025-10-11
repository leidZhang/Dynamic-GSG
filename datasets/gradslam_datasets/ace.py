import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from scipy.spatial.transform import Rotation as R # 用于四元数到旋转矩阵的转换

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ACEDataset(GradSLAMDataset):
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "ace_poses.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")))
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, "depth", "*.png")))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
            
        for i in range(self.num_imgs):
            line = lines[i]
            data = list(map(float, line.split()))
            quaternion = np.array(data[1:5])
            translation = np.array(data[5:8]) 
            rotation_matrix = R.from_quat(quaternion).as_matrix() # qx, qy, qz, qw

            c2w = np.eye(4)              
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = translation   

            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)