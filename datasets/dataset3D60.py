import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path, PureWindowsPath

import socket
import random

localname = str(socket.getfqdn()).split('.')[0]


def flip_y(ang):
    return np.array([-np.cos(ang), 0, np.sin(ang), 0, 1, 0, -np.sin(ang), 0, np.cos(ang)]).reshape(3,3)

def rot_x(ang):
    return np.array([1, 0, 0, 0, np.cos(ang), -np.sin(ang), 0, np.sin(ang), np.cos(ang)]).reshape(3,3)

class Dataset3D60(Dataset):
    def __init__(self, filenames_filepath, height, width, color_augmentation=True,
                 rotate=True,
                 flip=True,
                 is_training=False):
        
        self.rotate = rotate
        self.flip = flip
        self.is_training = is_training
        self.length = 0
        self.height = height
        self.width = width
        self.filenames_filepath = filenames_filepath
        self.delim = " "
        self.color_augmentation = color_augmentation
        self.data_paths = {}
        self.init_data_dict()
        self.gather_filepaths()
        self.max_depth = 10.0
        self.min_depth = 0.1
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(
                    brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug= transforms.ColorJitter(
                    brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
                

    def init_data_dict(self):
        self.data_paths = {
            "rgb": [],
            "surface": [],
            "depth": []
        }
    
    def gather_filepaths(self):
        if localname == "singapura":
            local = '/local/scratch3/Datasets/3D60_extract/Center(Left_Down)/'
        else:
            local = str(os.path.abspath(os.getcwd())) + '/../3D60_extract/Center(Left_Down)/'
        fd = open(self.filenames_filepath, 'r')
        lines = fd.readlines()
        for line in lines:
            splits = line.split(self.delim)
            rgb_path = splits[0]
            rgb_path = Path(PureWindowsPath(rgb_path))
            self.data_paths["rgb"].append(local+str(rgb_path))
            depth_path = splits[3]
            depth_path = Path(PureWindowsPath(depth_path))
            self.data_paths["depth"].append(local+str(depth_path))
            norm_path = splits[6]
            norm_path = Path(PureWindowsPath(norm_path))
            self.data_paths["surface"].append(local+str(norm_path))
        fd.close()
        assert len(self.data_paths["rgb"]) == len(self.data_paths["surface"]) == len(self.data_paths["depth"])
        self.length = len(self.data_paths["rgb"])

    def load_rgb(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 3), dtype = np.float32)
        rgb_np = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        return rgb_np

    def load_depth(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 1), dtype = np.float32), np.zeros((self.height, self.width, 1), dtype = np.float32)
        depth_np = cv2.imread(filepath,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,:1]
        return depth_np
    
    def load_float(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 3), dtype = np.float32), np.zeros((self.height, self.width, 3), dtype = np.float32)
        surface_np = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        surface_np[np.isnan(surface_np)] = 0
        return surface_np

    def clean_normal(self, normal):
        shape = np.array(normal).shape
        vecs = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0 ,1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]
        for vec in vecs:
            vec_mat = np.asarray(vec, dtype = np.float32)
            vec_mat = np.expand_dims(vec_mat, 0)
            vec_mat = np.expand_dims(vec_mat, 1)
            vec_mat = vec_mat.repeat(shape[0], 0)
            vec_mat = vec_mat.repeat(shape[1], 1)
            inds = np.isclose(normal, vec_mat, 0.0001, 0.1)
            inds = inds[:, :, 0] & inds[:, :, 1] & inds[:, :, 2]
            normal[inds, 0] = vec[0]
            normal[inds, 1] = vec[1]
            normal[inds, 2] = vec[2]
        return normal

    def make_tensor(self, np_array):
        np_array = np_array.transpose(2, 0, 1)
        tensor = torch.from_numpy(np_array)
        return torch.as_tensor(tensor, dtype = torch.float32)
    
    def flip_surface_normal(self, normal):
        normal = np.flip(normal, axis=1)
        H,W,C = normal.shape
        normal = normal.reshape(H*W,C).T
        rot_x_mat = flip_y(np.pi)
        normal = rot_x_mat @ normal
        normal = normal.T.reshape(H,W,C)
        return normal

    def rotate_surface_normal(self, normal, roll_idx):
        normal = np.roll(normal, roll_idx, 1)
        H,W,C = normal.shape
        normal = normal.reshape(H*W,C).T
        rot_ag = roll_idx/W*np.pi*2
        rot_x_mat = rot_x(rot_ag)
        normal = rot_x_mat @ normal
        normal = normal.T.reshape(H,W,C)
        return normal

    def load_item(self, idx):
        item = { }
        if (idx >= self.length):
            print("Index out of range.")
        else:
            rgb_np = self.load_rgb(self.data_paths["rgb"][idx])
            depth_np = self.load_depth(self.data_paths["depth"][idx])
            surface_np = self.load_float(self.data_paths["surface"][idx])
            surface_np = self.clean_normal(surface_np)

            # Random flip
            if self.is_training and self.flip and random.random() > 0.5:
                rgb_np = np.flip(rgb_np, axis=1)
                depth_np = np.flip(depth_np, axis=1)
                surface_np = self.flip_surface_normal(surface_np)

            # Random horizontal rotate
            if self.is_training and self.rotate and random.random() > 0.5:
                roll_idx = random.randint(0, self.width)
                rgb_np = np.roll(rgb_np, roll_idx, 1)
                depth_np = np.roll(depth_np, roll_idx, 1)
                surface_np = self.rotate_surface_normal(surface_np, roll_idx)

            if self.is_training and self.color_augmentation and random.random() > 0.5:
                aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb_np)))
            else:
                aug_rgb = rgb_np

            depth_mask_np = ((depth_np <= self.max_depth) & (depth_np > self.min_depth))
            normal_mask_np = np.ones_like(surface_np)
            normal_mask_np[np.sum(surface_np, 2) == 0] = 0
            mask_np = depth_mask_np * normal_mask_np[:,:,:1]

            mask = self.make_tensor(mask_np)

            depth = self.make_tensor(depth_np.copy()) * mask
            surface = self.make_tensor(surface_np.copy()) * mask

            rgb = self.to_tensor(rgb_np.copy())
            aug_rgb = self.to_tensor(aug_rgb.copy())
            item["normalized_rgb"] = self.normalize(aug_rgb)

            item["aug_rgb"] = self.normalize(aug_rgb)
            item['gt_surface'] = surface
            item['gt_depth'] = depth

            item['mask'] = mask
            item['ori_rgb'] = self.normalize(rgb)

            item['surface_filename'] = os.path.splitext(os.path.basename(self.data_paths["surface"][idx]))[0]
            item['depth_filename'] = os.path.splitext(os.path.basename(self.data_paths["depth"][idx]))[0]
            return item
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.load_item(idx)