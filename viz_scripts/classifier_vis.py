import copy
from pathlib import Path
import pickle
import gzip
from tqdm import tqdm
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import matplotlib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open_clip
import cv2
import faiss
from ultralytics import SAM, YOLO
from scipy.spatial import ConvexHull, Delaunay
from collections import Counter

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from scripts.dynamic_gsg_real_ssim import (
    get_dataset, 
    add_new_gaussians, 
    initialize_optimizer,
    get_loss,
    )
from utils.keyframe_selection import keyframe_selection_overlap
from datasets.gradslam_datasets import load_dataset_config
from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation, prune_gaussians, densify
from utils.slam_classes import MapObjectList
from utils.object_helpers import ObjectClasses
from utils.map_objects_utils_up_with_groupv3 import (
    read_color_book,
    process_this_frame_detection,
    render_curr_frame_with_idx,
    compute_similarities_and_merge,
)

from utils.vlm import get_obj_captions_from_image_gpt4v
import numpy as np
import open3d as o3d

from vlm_utils.vlm import consolidate_captions, query_with_llm, descriptive_query_with_llm


def load_variables(file_path: Path) -> dict:
    variables = dict(np.load(file_path, allow_pickle=True))
    variables = {k: torch.tensor(variables[k]).cuda().float() for k in variables.keys()}

    return variables

def load_keyframe_list(file_path: Path) -> list:
    with gzip.open(file_path, 'rb') as f:
        keyframe_list = pickle.load(f)
    print(f"Keyframe list loaded from {file_path}")
    return keyframe_list


def load_data(file_path: Path, lf_cfg: dict) -> MapObjectList:
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    objects = MapObjectList()
    objects.load_serializable(data)

    return objects


def highlight_idx_object(scene_data : dict, text_query_idx, objects_idx, whether_show_bg=True):
        
        scene_data_copy = copy.deepcopy(scene_data)

        RED = torch.tensor([1, 0, 0], dtype=torch.float32, device='cuda')
        BLUE = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda')

        if isinstance(text_query_idx, str):
            # run color_by_idx
            indices = np.where(objects_idx == int(text_query_idx))[0]
            color = RED
        elif isinstance(text_query_idx, list): 
            # run color_by_attached_objects
            indices = np.where(np.isin(objects_idx, text_query_idx))[0]
            color = BLUE
        elif isinstance(text_query_idx, int):
            # run color_by_relationships
            indices = np.where(objects_idx == int(text_query_idx))[0]
            color = RED


        if indices.shape[0] == 0:
            print(f"Index {text_query_idx} do not be used")
        
        scene_data_copy['colors_precomp'][indices, :] = color
    
        if not whether_show_bg:
            indices = np.where(objects_idx != 0)[0]
            scene_data_copy['means3D'] = scene_data_copy['means3D'][indices, :]
            scene_data_copy['colors_precomp'] = scene_data_copy['colors_precomp'][indices, :]
            scene_data_copy['rotations'] = scene_data_copy['rotations'][indices, :]
            scene_data_copy['opacities'] = scene_data_copy['opacities'][indices, :]
            scene_data_copy['scales'] = scene_data_copy['scales'][indices, :]
            scene_data_copy['means2D'] = scene_data_copy['means2D'][indices, :]
    
        return scene_data_copy


def highlight_most_similar_object(scene_data : dict, objects, objects_idx, similarity_colors : np.ndarray, whether_show_bg=True):
    
    scene_data_copy = copy.deepcopy(scene_data)
    
    for i, obj in enumerate(objects):
        indices = np.where(objects_idx == obj['idx'])[0]
        color = torch.from_numpy(similarity_colors[i, :]).type(torch.float32).to('cuda')
        scene_data_copy['colors_precomp'][indices, :] = color

    if not whether_show_bg:
        indices = np.where(objects_idx != 0)[0]
        scene_data_copy['means3D'] = scene_data_copy['means3D'][indices, :]
        scene_data_copy['colors_precomp'] = scene_data_copy['colors_precomp'][indices, :]
        scene_data_copy['rotations'] = scene_data_copy['rotations'][indices, :]
        scene_data_copy['opacities'] = scene_data_copy['opacities'][indices, :]
        scene_data_copy['scales'] = scene_data_copy['scales'][indices, :]
        scene_data_copy['means2D'] = scene_data_copy['means2D'][indices, :]

    return scene_data_copy

def points_inside_convex_hull(point_cloud, remove_outliers=True, outlier_factor=1.2):
    
    # Extract the masked points from the point cloud
    masked_points = point_cloud.cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    objects_idx = all_params['object_idx']
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys() if k != 'keyframe_list'}
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    # Check if Gaussians are Isotropic or Anisotropic
    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    featurevar = {
        'means3D': params['means3D'],
        'colors_precomp': params['features'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], first_frame_w2c),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    params['object_idx'] = params['object_idx'].detach().cpu().numpy()
    return rendervar, featurevar, depth_rendervar, objects_idx, all_w2cs, params


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil

# Run RANSAC
def ransac_plane_fitting(pcd, distance_threshold=0.04, ransac_n=3, num_iterations=1000):
    # get plane_model and points' inliers
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

    [a, b, c, d] = plane_model  
    normal_vector = np.array([a, b, c])
    
    return normal_vector, inliers


def transformed_points(pcd_np, normal_vector, inliers):

    if not isinstance(pcd_np, np.ndarray):
        pcd_np = np.asarray(pcd_np)

    floor_inlier_points = pcd_np[inliers]
    mask = np.ones(len(pcd_np), dtype=bool)
    mask[inliers] = False
    non_floor_inlier_points = pcd_np[mask]

    z_axis = np.array([0, 0, -1])
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    # 计算将平面旋转到z=0平面所需的旋转轴和旋转角度
    rotation_axis = np.cross(normal_vector_normalized, z_axis)
    # 防止数值误差导致的arccos参数超过[-1, 1]
    rotation_angle = np.arccos(np.clip(np.dot(normal_vector_normalized, z_axis), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) > 1e-6:
        # 旋转轴加旋转角可以转换为旋转矩阵——罗德里格斯公式
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        rotation_matrix = np.eye(3)
    
    # 将floor旋转到与z平面对齐
    rotated_floor_inlier_points = floor_inlier_points.dot(rotation_matrix.T)
    # 添加平移向量
    rotated_floor_inlier_points = rotated_floor_inlier_points + np.array([0, 0, 3.0])

    non_floor_inlier_points = non_floor_inlier_points.dot(rotation_matrix.T)
    # 添加同样的平移向量保持相对位置
    non_floor_inlier_points = non_floor_inlier_points + np.array([0, 0, 3.0])

    pcd_np[inliers] = rotated_floor_inlier_points
    pcd_np[mask] = non_floor_inlier_points

    z_max = np.max(rotated_floor_inlier_points[:, 2])
    z_min = np.min(rotated_floor_inlier_points[:, 2])
    z_threshold = (z_max - z_min) / 10

    mask_ceiling = [i for i in range(len(pcd_np)) if pcd_np[i, 2] > (z_max - z_threshold)]
    mask = np.ones(len(pcd_np), dtype=bool)
    mask[mask_ceiling] = False

    return mask, pcd_np, z_min


def is_point_in_plane(obj, extent):
    if isinstance(obj, dict):
        p = obj['center'][:2]
    else:
        p = obj
    A, B, C, D = extent
    polygon = [A, B, C, D]
    n = len(polygon)
    inside = False
    x, y = p
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
    return inside

def init_scene_graph(objects, attached_room):
    scene_graph = []    # 存储obj-attached的idx对应关系
    center_graph = []   # 存储obj-attached的center对应关系
    for obj in objects:
        if obj['caption'] == "Standalone":
            continue
        if obj['attached'] is None:
            # obj_z_min = obj['extent'][1]
            # obj_z_max = obj['extent'][2]
            for candidate in attached_room:
                # candidate_z_min = candidate['extent'][1]
                # candidate_z_max = candidate['extent'][2]
                # 对于那些正中心落在candidate_object范围内的物体，会存在两种情况：1.在其上方（桌子） 2.在其中间（沙发）
                if is_point_in_plane(obj, candidate['extent'][0]):
                    # if abs(obj_z_min - candidate_z_max) < 0.20 or (obj_z_min >= candidate_z_min and obj_z_max <= candidate_z_max):
                    obj['attached'] = [candidate['idx'], candidate['center']]
                # else:
                #     # 对于中心不在candidate_object范围内的物体（可能识别不准造成），判断其边界四个角是否在candidate_object范围内
                #     if abs(obj_z_min - candidate_z_max) < 0.15 and (is_point_in_plane(obj['extent'][0][0], candidate['extent'][0]) or 
                #                                                     is_point_in_plane(obj['extent'][0][1], candidate['extent'][0]) or 
                #                                                     is_point_in_plane(obj['extent'][0][2], candidate['extent'][0]) or 
                #                                                     is_point_in_plane(obj['extent'][0][3], candidate['extent'][0])):
                #         obj['attached'] = [candidate['idx'], candidate['center']]

    for obj in objects:
        if obj['attached'] is not None and obj['attached'] != 'room':
            scene_graph.append((obj['idx'], obj['attached'][0]))
            center_graph.append((obj['center'], obj['attached'][1]))

    return scene_graph, center_graph


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def process_pcd(pcd, downsample_voxel_size=0.01, dbscan_remove_noise=True, dbscan_eps=0.1, dbscan_min_points=10, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pass
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        
        pcd = largest_cluster_pcd
        
    return pcd

def transform_coordinate(scene_data, scene_depth_data, objects_idx, viz_cfg, k, save_path):

    non_object_pcd = scene_data['means3D'][(objects_idx == 0).flatten()]
    non_object = o3d.geometry.PointCloud()
    non_object.points = o3d.utility.Vector3dVector(non_object_pcd.contiguous().double().cpu().numpy())

    non_object = process_pcd(non_object)

    mask, inliers = iter_data(non_object, scene_data, scene_depth_data, viz_cfg, k, save_path)

    while get_obj_captions_from_image_gpt4v(save_path) == "no":
        mask_non_floor = np.ones(len(non_object_pcd), dtype=bool)
        mask_non_floor[inliers] = False
        non_object_pcd = non_object_pcd[mask_non_floor]
        non_object.points = o3d.utility.Vector3dVector(non_object_pcd.contiguous().double().cpu().numpy())
        mask, inliers = iter_data(non_object, scene_data, scene_depth_data, viz_cfg, k, save_path)
    else:
        return mask


def iter_data(non_object, scene_data, scene_depth_data, viz_cfg, k, save_path):
    normal_vector, inliers = ransac_plane_fitting(non_object)
    mask, transformed_pcd, _ = transformed_points(
        scene_data['means3D'].detach().cpu().numpy(),
        normal_vector,
        inliers,
    )
    scene_data['means3D'] = torch.tensor(transformed_pcd).cuda()

    transformed_scene_data = {k: scene_data[k][mask] for k in scene_data.keys() if k != 'opacities'}
    transformed_scene_data['opacities'] = scene_data['opacities'][mask.flatten()]

    transformed_scene_depth_data = {k: scene_depth_data[k][mask] for k in scene_depth_data.keys() if k != 'opacities'}
    transformed_scene_depth_data['opacities'] = scene_depth_data['opacities'][mask.flatten()]

    w2c = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    image, _, _ = render(w2c, k, transformed_scene_data, transformed_scene_depth_data, viz_cfg)
    image = (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return mask, inliers


def visualize(scene_path, objects_path, config, viz_cfg, lf_cfg, data_cfg):

    if not viz_cfg["no_clip"]:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", viz_cfg['clip_model_path']
        )
        clip_model = clip_model.to('cuda')
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")
    
    # Load Scene Data
    w2c, k = load_camera(viz_cfg, scene_path)

    color_book = read_color_book(lf_cfg['color_book_path'])

    scene_data, scene_features_data, scene_depth_data, objects_idx, all_w2cs, all_params = load_scene_data(scene_path, w2c, k)

    objects = load_data(objects_path, lf_cfg)

    # vis.create_window()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=int(viz_cfg['viz_w'] * viz_cfg['view_scale']), 
                      height=int(viz_cfg['viz_h'] * viz_cfg['view_scale']),
                      visible=True)
    
    im, depth, sil = render(w2c, k, scene_data, scene_depth_data, viz_cfg)

    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    # pcd.transform(tf)
    vis.add_geometry(pcd)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    save_path = "./transformed_scene.png"
    mask = transform_coordinate(scene_data, scene_depth_data, objects_idx, viz_cfg, k, save_path)

    # 将ceiling的点云mask掉，方便查看
    transformed_scene_data = {k: scene_data[k][mask] for k in scene_data.keys() if k != 'opacities'}
    transformed_scene_data['opacities'] = scene_data['opacities'][mask.flatten()]
    scene_data = transformed_scene_data

    transformed_scene_features_data = {k: scene_features_data[k][mask] for k in scene_features_data.keys() if k != 'opacities'}
    transformed_scene_features_data['opacities'] = scene_features_data['opacities'][mask.flatten()]
    scene_features_data = transformed_scene_features_data

    transformed_scene_depth_data = {k: scene_depth_data[k][mask] for k in scene_depth_data.keys() if k != 'opacities'}
    transformed_scene_depth_data['opacities'] = scene_depth_data['opacities'][mask.flatten()]
    scene_depth_data = transformed_scene_depth_data

    objects_idx = objects_idx[mask]

    to_remove = []
    attached_room = []
    obj_count = 0
    for obj in tqdm(objects):
        indices = np.where(objects_idx == obj['idx'])[0]

        if indices.shape[0] < 150:
            to_remove.append(obj['idx'])
            continue

        print(f"Object {obj['idx']} has {indices.shape[0]} points")
        
        obj_count += 1
        object_pcd_tensor = scene_data['means3D'][indices].detach()
    
        # convex = points_inside_convex_hull(object_pcd_tensor, remove_outliers=True, outlier_factor=1.0)
        # object_pcd_np = object_pcd_tensor[convex].detach().cpu().numpy()
        
        object_pcd_np = object_pcd_tensor.detach().cpu().numpy()
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)

        cl, _ = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.7)
        cl1, _ = cl.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
        object_pcd = cl1.voxel_down_sample(voxel_size=0.1)

        pcd_clusters = object_pcd.cluster_dbscan(eps=1, min_points=10,)
        
        # Convert to numpy arrays
        obj_points = np.asarray(object_pcd.points)
        pcd_clusters = np.array(pcd_clusters)

        # Count all labels in the cluster
        counter = Counter(pcd_clusters)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            
            # Create mask for points in the largest cluster
            largest_mask = pcd_clusters == most_common_label

            # Apply mask
            largest_cluster_points = obj_points[largest_mask]

            # Create a new PointCloud object
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

            # bbox = largest_cluster_pcd.get_oriented_bounding_box(robust=True)
        else:
            to_remove.append(obj['idx'])
            continue

        print(f"After filter Object {obj['idx']} has {len(largest_cluster_points)} points")

        pcd_xy = largest_cluster_points[:, :2]

        centroid = np.mean(pcd_xy, axis=0)
        centered_points = pcd_xy - centroid

        # Run PCA
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        principal_axes = eigenvectors[:, idx]

        transformed_point = centered_points @ principal_axes
        min_bounds = np.min(transformed_point, axis=0)
        max_bounds = np.max(transformed_point, axis=0)

        corners = np.array([
            [min_bounds[0], min_bounds[1]],
            [max_bounds[0], min_bounds[1]],
            [max_bounds[0], max_bounds[1]],
            [min_bounds[0], max_bounds[1]]
        ])

        rectangle_corners = (corners @ principal_axes.T) + centroid

        if counter:
            obj['extent'] = [rectangle_corners, min(largest_cluster_points[:, 2]), max(largest_cluster_points[:, 2])]
        else:
            obj['extent'] = [rectangle_corners, min(object_pcd_np[:, 2]), max(object_pcd_np[:, 2])]
        obj['attached'] = None

        # if (min(largest_cluster_points[:, 2]) < threshold + 0.30):
        if (obj['caption'] == "Asset"):
            obj['attached'] = 'room'
            attached_room.append(obj)

        bbox = object_pcd.get_oriented_bounding_box()

        if bbox is not None:
            obj['bbox'] = bbox
            obj['center'] = bbox.center

    objects = [obj for obj in objects if obj['idx'] not in to_remove]

    scene_graph, _ = init_scene_graph(objects, attached_room)
    
    merged_groups = {}
    for child, parent in scene_graph:
        # Add child to parent's group
        if parent in merged_groups:
            merged_groups[parent].append(child)
        else:
            # Create new group with parent as key
            merged_groups[parent] = [child]

    objects = MapObjectList(objects)

    w = viz_cfg['viz_w']
    h = viz_cfg['viz_h']

    if viz_cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
        
        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)


    # Initialize View Control
    view_k = k * viz_cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if viz_cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(viz_cfg['viz_h'] * viz_cfg['view_scale'])
    cparams.intrinsic.width = int(viz_cfg['viz_w'] * viz_cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = viz_cfg['view_scale']
    render_options.light_on = False


    def color_by_rgb(vis):
        print("Now color by rgb")
        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, viz_cfg)

            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()


    def color_by_clip_sim(vis):
        # Interactive Rendering
        print("Now color by language feature simlarity")
        text_query = input("Enter what you are interested in to query: ")
        text_queries = [text_query]

        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        max = text_query_ft.max()
        min = text_query_ft.min()

        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")

        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )

        cmap = matplotlib.colormaps.get_cmap("turbo")
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        max_prob_object_idx = objects[max_prob_idx]['idx']

        select_scene_data = highlight_most_similar_object(scene_data, objects, objects_idx, similarity_colors, viz_cfg['show_bg'])
        print(f"{text_query}'s idx: {objects[max_prob_idx]['idx']}")
        print(f"{text_query}'s position: {objects[max_prob_idx]['center']}")


        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, select_scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()


    def color_by_idx(vis):
        # Interactive Rendering
        print("Now color by idx")
        text_query_idx = input("Enter which idx you are interested in to query: ")

        select_scene_data = highlight_idx_object(scene_data, text_query_idx, objects_idx, viz_cfg['show_bg'])

        # print(f"{objects[max_prob_idx]['class_id']}'s position: {objects[max_prob_idx]['center']}")

        # print(f"{text_query}'s position: {objects[max_prob_idx]['center']}")

        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, select_scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

        
    def color_by_attached_objects(vis):
        print(f"There are {len(merged_groups)} furnitures placed on the floor in the scene, their idx are: {list(merged_groups.keys())}")
        text_query = input("Enter the edge's index in the scene_graph that you want to show: ")

        select_scene_data = highlight_idx_object(scene_data, text_query, objects_idx, viz_cfg['show_bg'])
        select_scene_data = highlight_idx_object(select_scene_data, merged_groups[int(text_query)], objects_idx, viz_cfg['show_bg'])

        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, select_scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

    import re
    def extract_on_phrases(sentence):
        pattern = r'''
            \b(a|the|an|some)\s+          # a/the/an/some
            ([\w\s]+)                     # object1
            \b                            
            .*                            
            \b(on\s+(?:top\s+of\s+)?)\b   # preposition
            \s*
            (a\s+[\w\s]+|the\s+[\w\s]+|[\w\s]+)  # object2
            \b
        '''
        matches = re.findall(pattern, sentence, flags=re.IGNORECASE | re.VERBOSE)
        
        results = []
        for match in matches:
            obj1 = f"{match[0]} {match[1].strip()}"  # object1 (for example, "a book")
            preposition = match[2].strip()           # preposition (for example "on")
            obj2 = match[3].strip()                  # object2 (for example, "the end table")
            results.append((obj1, preposition, obj2))
        
        return results

    def color_by_relationships(vis):
        text_query = input("Now you can query based on the scene graph, please follow the format 'X on Y' to query: ")

        print("*************************LLM + scene_graph*************************")
        top3_object = descriptive_query_with_llm(objects, text_query, merged_groups, True)

        print("*************************LLM*************************")
        top3_object = descriptive_query_with_llm(objects, text_query, False)

        print("*************************CLIP + scene_graph*************************")

        print(extract_on_phrases(text_query))

        if extract_on_phrases(text_query) == []:
            text_queries = [text_query]

            text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
            text_query_ft = clip_model.encode_text(text_queries_tokenized)
            text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
            text_query_ft = text_query_ft.squeeze()

            objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
            objects_clip_fts = objects_clip_fts.to("cuda")

            similarities = F.cosine_similarity(
                text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
            )
            probs = F.softmax(similarities, dim=0)

            sorted_indices = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            top3_idx = [objects[x[0]]['idx'] for x in sorted_indices]
            top3_objects = [(idx, similarities[sorted_indices[i][0]]) for i, idx in enumerate(top3_idx)]

            print("Top 3 candidate objects:")
            for i, (idx, sim) in enumerate(top3_objects[:3], 1):
                print(f"{i}. Object {idx} (similarity: {sim:.6f})")
            
        else:
            target, _, guide = extract_on_phrases(text_query)[0]
            guide = [guide]
            target = [target]

            guide_text_queries_tokenized = clip_tokenizer(guide).to("cuda")
            guide_text_query_ft = clip_model.encode_text(guide_text_queries_tokenized)
            guide_text_query_ft = guide_text_query_ft / guide_text_query_ft.norm(dim=-1, keepdim=True)
            guide_text_query_ft = guide_text_query_ft.squeeze()

            target_text_queries_tokenized = clip_tokenizer(target).to("cuda")
            target_text_query_ft = clip_model.encode_text(target_text_queries_tokenized)
            target_text_query_ft = target_text_query_ft / target_text_query_ft.norm(dim=-1, keepdim=True)
            target_text_query_ft = target_text_query_ft.squeeze()

            if 'wall' in guide[0].lower() or 'floor' in guide[0].lower() or 'room' in guide[0].lower() or 'ceiling' in guide[0].lower():

                objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
                objects_clip_fts = objects_clip_fts.to("cuda")

                similarities = F.cosine_similarity(
                    target_text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
                )
                probs = F.softmax(similarities, dim=0)
                
                sorted_indices = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                top3_idx = [objects[x[0]]['idx'] for x in sorted_indices]
                top3_objects = [(idx, similarities[sorted_indices[i][0]]) for i, idx in enumerate(top3_idx)]

                print("Top 3 candidate objects:")
                for i, (idx, sim) in enumerate(top3_objects[:3], 1):
                    print(f"{i}. Object {idx} (similarity: {sim:.6f})")
            
            else:

                support_objects = [obj for obj in objects if obj['idx'] in merged_groups.keys()]

                mismatched_support_objects = []
                target_object_idx = None
                top3_objects = []

                while True:
                    support_objects = [obj for obj in support_objects if obj['idx'] not in mismatched_support_objects]
                    if support_objects == []:
                        top3_objects = sorted(top3_objects, key=lambda x: x[1], reverse=True)
                        print("Top 3 candidate objects:")
                        for i, (idx, sim) in enumerate(top3_objects[:3], 1):
                            print(f"{i}. Object {idx} (similarity: {sim:.6f})")
                        break
                    support_objects = MapObjectList(support_objects)
                    support_objects_clip_fts = support_objects.get_stacked_values_torch("clip_ft")
                    support_objects_clip_fts = support_objects_clip_fts.to("cuda")

                    similarities = F.cosine_similarity(
                        guide_text_query_ft.unsqueeze(0), support_objects_clip_fts, dim=-1
                    )
                    probs = F.softmax(similarities, dim=0)
                    max_prob_idx = torch.argmax(probs)
                    max_prob_object_idx = support_objects[max_prob_idx]['idx']
                    max_prob = similarities[max_prob_idx]

                    # print(f"support_object's idx: {max_prob_object_idx} similarity: {max_prob}")

                    if max_prob < 0.20:
                        if top3_objects != []:
                            top3_objects = sorted(top3_objects, key=lambda x: x[1], reverse=True)
                            print("Top 3 candidate objects:")
                            for i, (idx, sim) in enumerate(top3_objects[:3], 1):
                                print(f"{i}. Object {idx} (similarity: {sim:.6f})")
                        else:
                            print(f"Can't find an object in your query which {target} is placed on the {guide}!")
                        break

                    candidates = merged_groups[max_prob_object_idx]
                    candidate_objects = [obj for obj in objects if obj['idx'] in candidates]
                    candidate_objects = MapObjectList(candidate_objects)
                    candidate_objects_clip_fts = candidate_objects.get_stacked_values_torch("clip_ft")
                    candidate_objects_clip_fts = candidate_objects_clip_fts.to("cuda")

                    candidate_similarities = F.cosine_similarity(
                        target_text_query_ft.unsqueeze(0), candidate_objects_clip_fts, dim=-1
                    )
                    candidate_probs = F.softmax(candidate_similarities, dim=0)

                    sorted_indices = sorted(enumerate(candidate_probs), key=lambda x: x[1], reverse=True)[:3]
                    top3_idx = [candidate_objects[x[0]]['idx'] for x in sorted_indices]
                    top3_similarities = [candidate_similarities[x[0]] for x in sorted_indices]
                    for i in range(len(top3_idx)):
                        top3_objects.append((top3_idx[i], top3_similarities[i]))

                    candidate_max_prob_idx = torch.argmax(candidate_probs)
                    candidate_max_prob_object_idx = candidate_objects[candidate_max_prob_idx]['idx']
                    candidate_max_prob = candidate_similarities[candidate_max_prob_idx]

                    mismatched_support_objects.append(max_prob_object_idx)

                    # if candidate_max_prob < 0.25:
                    #     print(f"candidate_object's idx: {candidate_max_prob_object_idx} similarity: {candidate_max_prob} Try to search another location!")
                    #     mismatched_support_objects.append(max_prob_object_idx)
                    # else:
                    #     target_object_idx = candidate_max_prob_object_idx
                    #     mismatched_support_objects.append(max_prob_object_idx)
                    #     print("**********Find the target object!**********")
                    #     print(f"{guide}'s idx: {max_prob_object_idx}\n{target}'s idx: {candidate_max_prob_object_idx} similarity: {candidate_max_prob}")
                    #     break

        target_object_idx = top3_objects[0][0]
        if target_object_idx is not None:
            select_scene_data = highlight_idx_object(scene_data, target_object_idx, objects_idx, viz_cfg['show_bg'])
        else:
            select_scene_data = scene_data

        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, select_scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

    def color_by_llm(vis):
        # Interactive Rendering
        print("Now you are querying with the gpt-4o-mini.")
        text_query = input("Enter what you are interested in to query: ")
        
        top3_object = query_with_llm(objects, text_query)
        top3_object_idx = [int(obj['idx']) for obj in top3_object]

        select_scene_data = highlight_idx_object(scene_data, top3_object_idx, objects_idx, viz_cfg['show_bg'])

        print(f"{text_query}'s idx: {top3_object_idx[0]}")
        print(f"top3 objects' idx: {top3_object_idx}")

        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / viz_cfg['view_scale']
            k[2, 2] = 1
            w2c = cam_params.extrinsic

            im, depth, sil = render(w2c, k, select_scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
            
            # Update Gaussians
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()
        

    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("C"), color_by_clip_sim)
    vis.register_key_callback(ord("P"), color_by_idx)
    vis.register_key_callback(ord("A"), color_by_attached_objects)
    vis.register_key_callback(ord("O"), color_by_relationships)
    vis.register_key_callback(ord("L"), color_by_llm)

    print(f"Object count: {obj_count}")

    scene_centers_data = copy.deepcopy(scene_data)
    
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / viz_cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if viz_cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_centers_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_centers_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, viz_cfg)
            if viz_cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, viz_cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        # pcd.transform(tf)

        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params_with_idx.npz")
        objects_path = os.path.join(results_dir, "objects.pkl.gz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]
    lf_cfg = experiment.config["lang"]
    data_cfg = experiment.config["data"]
    config = experiment.config

    # Visualize Final Reconstruction
    visualize(scene_path, objects_path, config, viz_cfg, lf_cfg, data_cfg)
