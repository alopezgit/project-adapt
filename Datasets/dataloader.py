import cv2
import numpy as np
import numpy.matlib 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read, depth_read_carla, Projector
import math

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from Utils.utils import return_poses, sample_mask


def get_loader(args, dataset, only_train=False, is_carla=False):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args.crop_h, args.crop_w)

    train_dataset = Dataset_loader(
            args.data_path_source, args.data_path_target, dataset.train_paths, args.input_type, resize=None,
            crop=crop_size, 
            max_depth=args.max_depth, sparse_val=args.sparse_val,
            is_carla=is_carla, args=args)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.nworkers,
        pin_memory=False, drop_last=True)
    return train_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, source_path, target_path, dataset_type, input_type, resize,
                 crop, max_depth, sparse_val=0.0, 
                 train=False,
                 is_carla=False, args=None):

        # Constants
        self.use_rgb = input_type == 'rgb'
        self.dataset_type = dataset_type
        self.resize = resize
        self.crop = crop
        self.max_depth = max_depth
        self.sparse_val = sparse_val
        self.projector = Projector(source_path, target_path)
        # Transformations
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)
        # Names
        self.img_name = 'img'
        self.lidar_name = 'lidar_in' 
        self.gt_name = 'gt' 
        # Define random sampler
        self.carla = is_carla

    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['lidar_in'])

        
    def define_transforms(self, input, gt, img=None):
        input, gt = self.center_crop(input), self.center_crop(gt)
        if self.use_rgb:
            img = self.center_crop(img)
        if self.carla:
            input, gt = depth_read_carla(input), depth_read_carla(gt)
        else:
            input, gt = depth_read(input, self.sparse_val), depth_read(gt, self.sparse_val)
    
        return input, gt, img

    def carla_transform_img(self, img):
        img = F.center_crop(img, (352, 1216))
        w, h = img.size
        img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)
        return img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        sparse_depth_name = os.path.join(self.dataset_type[self.lidar_name][idx])
        gt_name = os.path.join(self.dataset_type[self.gt_name][idx])
        right = 0
        if self.carla:
            right = random.random() > 0.5
            if right:
                gt_name = gt_name.replace('Central', 'Right')
        with open(sparse_depth_name, 'rb') as f:
            sparse_depth = Image.open(f)
            w, h = sparse_depth.size
            if not self.carla:
                sparse_depth = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w)
            else:
                sparse_depth = self.carla_transform_img(sparse_depth)
        with open(gt_name, 'rb') as f:
            gt = Image.open(f)
            if not self.carla:
                gt = F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
            else:
                gt = self.carla_transform_img(gt)

        img = None
        if self.use_rgb:
            img_name = self.dataset_type[self.img_name][idx]
            if self.carla:
                if right:
                    img_name = img_name.replace('Central', 'Right')
            with open(img_name, 'rb') as f:
                img = (Image.open(f).convert('RGB'))
            if not self.carla:
                img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)
            else:
                img = self.carla_transform_img(img)

        sparse_depth_np, gt_np, img_pil = self.define_transforms(sparse_depth, gt, img)
        input, gt = self.totensor(sparse_depth_np).float(), self.totensor(gt_np).float()

        if self.use_rgb:
            img_tensor = self.totensor(img_pil).float()
            img_tensor = img_tensor*255.0
            input = torch.cat((input, img_tensor), dim=0)
        else:
            img_name = gt_name
        if self.carla:
            lidar_in, mask_lidar = self.projector.project(input[0], right)
            input[0] = lidar_in

        return input, gt, img_name

