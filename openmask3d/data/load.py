import numpy as np
from PIL import Image
import open3d as o3d
import imageio
import torch
import math
import os

def get_number_of_images(poses_path):
    i = 0
    while(os.path.isfile(os.path.join(poses_path, str(i) + '.txt'))): i += 1
    return i

class Camera:
    def __init__(self, 
                 intrinsic_path, 
                 intrinsic_resolution, 
                 poses_path, 
                 depths_path, 
                 extension_depth, 
                 depth_scale):
        self.intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        self.intrinsic_original_resolution = intrinsic_resolution
        self.poses_path = poses_path
        self.depths_path = depths_path
        self.extension_depth = extension_depth
        self.depth_scale = depth_scale
    
    def get_adapted_intrinsic(self, desired_resolution):
        '''Get adjusted camera intrinsics.'''
        if self.intrinsic_original_resolution == desired_resolution:
            return self.intrinsic
        
        resize_width = int(math.floor(desired_resolution[1] * float(
                        self.intrinsic_original_resolution[0]) / float(self.intrinsic_original_resolution[1])))
        
        adapted_intrinsic = self.intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(self.intrinsic_original_resolution[0])
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(self.intrinsic_original_resolution[1])
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(self.intrinsic_original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(self.intrinsic_original_resolution[1] - 1)
        return adapted_intrinsic
    
    def load_poses(self, indices):
        path = os.path.join(self.poses_path, str(0) + '.txt')
        shape = np.linalg.inv(np.loadtxt(path))[:3, :].shape
        poses = np.zeros((len(indices), shape[0], shape[1]))
        for i, idx in enumerate(indices):
            path = os.path.join(self.poses_path, str(idx) + '.txt')
            poses[i] = np.linalg.inv(np.loadtxt(path))[:3, :]
        return poses
    
    def load_depth(self, idx, depth_scale):
        depth_path = os.path.join(self.depths_path, str(idx) + self.extension_depth)
        sensor_depth = imageio.v2.imread(depth_path) / depth_scale
        return sensor_depth


class Images:
    def __init__(self, 
                 images_path, 
                 extension, 
                 indices):
        self.images_path = images_path
        self.extension = extension
        self.indices = indices
        self.images = self.load_images(indices)
    
    def load_images(self, indices):
        images = []
        for idx in indices:
            img_path = os.path.join(self.images_path, str(idx) + self.extension)
            images.append(Image.open(img_path).convert("RGB"))
        return images
    def get_as_np_list(self):
        images = []
        for i in range(len(self.images)):
            images.append(np.asarray(self.images[i]))
        return images
    
class InstanceMasks3D:
    def __init__(self, masks_path):
        self.masks = torch.load(masks_path)
        self.num_masks = self.masks.shape[1]
    
    
class PointCloud:
    def __init__(self, 
                 point_cloud_path):
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        self.points = np.asarray(pcd.points)
        self.num_points = self.points.shape[0]
    
    def get_homogeneous_coordinates(self):
        return np.append(self.points, np.ones((self.num_points,1)), axis = -1)
    