from PIL import Image
import numpy as np
import argparse
import os
import torch.optim
from torch.optim import lr_scheduler
import errno
import sys
from torchvision import transforms
import torchvision
import torch.nn.init as init
import torch.distributed as dist
import torch.nn.functional as F
import random 
import numpy.matlib 
import math
import pykitti


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def first_run(save_path):
    txt_file = os.path.join(save_path, 'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return ''
        return saved_epoch
    return ''


def depth_read(img, sparse_val):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(img, dtype=int)
    depth_png = np.expand_dims(depth_png, axis=2)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = sparse_val
    return depth

def depth_read_carla(img):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = np.array(img)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    grayscale = np.dot(array[:, :, :3], [1.0, 256.0, 256.0 * 256.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)
    return 1000 * grayscale



# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



def sample_mask(data_path, enlarge=1):
    data_path = os.path.join(data_path, 'train')
    folders = os.listdir(data_path)
    folder = random.sample(folders, 1)[0]
    proj_data = 'proj_depth/velodyne_raw/image_02/'
    all_images_path = os.path.join(data_path, folder, proj_data)
    image_path = random.sample(os.listdir(all_images_path), 1)[0]
    f = os.path.join(all_images_path, image_path)
    sparse_depth_small = (torch.from_numpy(np.asarray(Image.open(f))) > 0).float()
    h, w = sparse_depth_small.shape[0], sparse_depth_small.shape[1]
    if h % 2 == 1:
        odd_h = 1
    else:
        odd_h = 0
    if w % 2 == 1:
        odd_w = 1
    else:
        odd_w = 0
    sparse_depth_small = sparse_depth_small[int((h-352)/2):-int((h-352)/2)-odd_h, int((w-1216)/2):-int((w-1216)/2)-odd_w]
    offset_h = int((1392-352)/2)
    offset_w = int((1392-1216)/2)
    sparse_depth = np.zeros((1392, 1392))
    if enlarge:
        sparse_depth[offset_h:offset_h+352, offset_w:offset_w+1216] = sparse_depth_small
    else:
        sparse_depth = sparse_depth_small[96:, :].numpy()
    sparse_depth = torch.from_numpy(sparse_depth).float()
    return sparse_depth

def filter_data(input_d, gt, max_depth, kernel_filt=41):
    input_d[input_d>max_depth] = 0
    gt[gt>1.1*max_depth] = 0
    if kernel_filt % 2 == 0:
        kernel_filt = kernel_filt + 1
    weights = torch.FloatTensor(kernel_filt, kernel_filt).fill_(1).unsqueeze(0).unsqueeze(0).cuda()
    weights.requires_grad = False
    conv = torch.nn.Conv2d(1,1,kernel_size=kernel_filt,stride=1,padding=(kernel_filt//2), bias=False).cuda()
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights)
    mask = conv(input_d) == 0
    gt[mask] = 0
    return input_d, gt

def filter_sparse_guidance(sparse_depth, f_window, f_thresh):
    with torch.no_grad():
        mask = sparse_depth == 0
        sparse_depth_min = -F.max_pool2d(-sparse_depth -100*mask.float(), f_window)
        sparse_depth_min = sparse_depth * (sparse_depth <= F.interpolate(sparse_depth_min, size=sparse_depth.shape[2:]) + f_thresh).float()
        mask_1 = (sparse_depth>0).float() * (sparse_depth_min == sparse_depth).float()
    return mask_1 * sparse_depth

def return_poses(poses_path):
    central_to_car_transform = np.load(os.path.join(poses_path, 'central_to_car_transform.npy'))
    right_to_car_transform = np.load(os.path.join(poses_path, 'right_to_car_transform.npy'))
    lidar_to_car_transform = np.load(os.path.join(poses_path, 'lidar_to_car_transform.npy'))   
    return torch.from_numpy(central_to_car_transform), torch.from_numpy(right_to_car_transform), torch.from_numpy(lidar_to_car_transform)

class Projector():

    def __init__(self, poses_path, masks_path, cuda=False, shape_image=[1392, 1392]):
        self.cuda = cuda
        self.pose_left, self.pose_right, self.pose_lidar = return_poses(poses_path)
        self.masks_path = masks_path
        self.image_width = shape_image[1]
        self.image_height = shape_image[0]
        # 2d pixel coordinates
        self.pixel_length = self.image_width * self.image_height
        u_coord = np.matlib.repmat(numpy.r_[self.image_width-1:-1:-1],
                         self.image_height, 1).reshape(self.pixel_length)
        v_coord = np.matlib.repmat(numpy.c_[self.image_height-1:-1:-1],
                         1, self.image_width).reshape(self.pixel_length)

        # We build the intrinsics from carla, where fx=fy 
        # and fx = width / (2*atan(FOV (in º)*π/360)
        CameraFOV = 90
        K = np.eye(3)
        f = self.image_width/(2.0 * math.tan(CameraFOV * math.pi / 360.0))
        Cu = self.image_width/2.0
        Cv = self.image_height/2.0
        K[0,0] = K[1,1] = f
        K[0,2] = Cu
        K[1,2] = Cv  
        self.K = K
        K_inv = numpy.linalg.inv(K)
        p2d = numpy.array([u_coord, v_coord, numpy.ones_like(u_coord)])
        p3d = numpy.dot(K_inv, p2d)
        self.p3d = torch.from_numpy(p3d).float()
        self.ones = torch.FloatTensor(self.pixel_length).fill_(1).float()
        self.K = torch.from_numpy(self.K).float()
        if self.cuda:
            self.p3d = self.p3d.cuda()
            self.ones = self.ones.cuda()
            self.pose_left = self.pose_left.cuda()
            self.pose_right = self.pose_right.cuda()
            self.pose_lidar = self.pose_lidar.cuda()
            self.K = self.K.cuda()


    def project(self, lidar_in, right=0, use_mask=1, lidar_mask=None):
        lidar_in_big = torch.FloatTensor(1, self.image_height, self.image_width).fill_(0)
        lidar_out_big = torch.FloatTensor(self.image_height, self.image_width).fill_(0)
        if self.cuda:
            lidar_in_big = lidar_in_big.cuda()
            lidar_out_big = lidar_out_big.cuda()

        offset_h = int((self.image_height-352)/2)
        offset_w = int((self.image_width-1216)/2)
        lidar_in_big[:, offset_h+96:offset_h+352, offset_w:offset_w+1216] = lidar_in
        
        # Sparsify data using random masks from kitti
        if lidar_mask is None:
            lidar_mask = sample_mask(self.masks_path)
        if self.cuda:
            lidar_mask = lidar_mask.cuda()
        if use_mask:
            lidar_in_big = lidar_mask.unsqueeze(0) * lidar_in_big

        reshaped_lidar = lidar_in_big.reshape(self.pixel_length)
        p3d = self.p3d * reshaped_lidar
        # Compute transformation matrix
        if right:
            pose = torch.mm(self.pose_right.inverse(), self.pose_lidar).float()
        else:
            pose = torch.mm(self.pose_left.inverse(), self.pose_lidar).float()
        # Transform 3d points and project
        p3d = torch.mm(pose, torch.cat([p3d, self.ones.unsqueeze(0)]))
        p2d = torch.mm(self.K, p3d[:3])
        x = self.image_width - p2d[0, :] / p2d[2, :] - 1
        y = self.image_height - p2d[1, :] / p2d[2, :] - 1
        z = p2d[2,:]
        mask = (x >= 0) & (y >= 0) & (x.round() < self.image_width) & (y.round() < self.image_height) & (z > 0)
        x = x[mask].cpu()
        y = y[mask].cpu()
        z = z[mask].cpu()
        z, ind_sort = z.sort(0, descending=True)
        x = x[ind_sort].cpu()
        y = y[ind_sort].cpu()
        lidar_out_big = lidar_out_big.cpu()
        ind = torch.stack([y.round().long(), x.round().long()])
        lidar_out_big[ind[0], ind[1]] = z
        lidar_out = lidar_out_big[offset_h+96:offset_h+352, offset_w:offset_w+1216].unsqueeze(0)
        if self.cuda:
            lidar_out = lidar_out.cuda()
        return lidar_out, lidar_mask