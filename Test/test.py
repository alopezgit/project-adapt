import argparse
import torch
import torchvision.transforms as transforms
import os, sys
from PIL import Image
import glob
import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
cwd = os.getcwd()
print(cwd)
import numpy as np
from Utils.utils import str2bool, AverageMeter, depth_read
from Loss.benchmark_metrics import Metrics, allowed_metrics

import Models
import Datasets
from PIL import ImageOps
import matplotlib.pyplot as plt
import time
import cv2  
from torch.utils.data import Dataset, DataLoader

#Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task TEST')
parser.add_argument('--dataset', type=str, default='kitti', choices = Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--mod', type=str, default='mod', choices = Models.allowed_models(), help='Model for use')
parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True, help='Use GPUs')
parser.add_argument('--input_type', type=str, default='rgb', help='use rgb for rgbdepth')
# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')

# Paths settings
parser.add_argument('--save_path', type= str, default='../Saved/best', help='save path')
parser.add_argument('--model_path', type= str, default='../Saved/best', help='path to the model weights in save path')
parser.add_argument('--data_path', type=str, default='/media/adrian/SSD/Kitti/depth_completion/', help='path to desired datasets')

# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
parser.add_argument('--multi', type=str2bool, nargs='?', const=True, default=False, help="use multiple gpus")
parser.add_argument('--max_depth', type=float, default=85.0, help="maximum depth of input")
parser.add_argument('--sparse_val', type=float, default=0.0, help="encode sparse values with 0")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')

parser.add_argument('--upload_web', type=str2bool, nargs='?', const=True, default=False, help="compute results for online test set")

class KittiDataset(Dataset):
    def __init__(self, lidar_files, rgb_files):
        self.to_tensor = transforms.ToTensor()
        self.lidar_files = lidar_files
        self.rgb_files = rgb_files

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, idx):
        lidar = self.lidar_files[idx]
        raw_path = os.path.join(lidar)
        raw_pil = Image.open(raw_path)

        assert raw_pil.size == (1216, 352)

        crop = 352-args.crop_h
        raw_pil_crop = raw_pil.crop((0, crop, 1216, 352))

        raw = depth_read(raw_pil_crop, args.sparse_val)
        raw = self.to_tensor(raw).float()

        rgb = self.rgb_files[idx]
        rgb_path = os.path.join(rgb)
        rgb_pil = Image.open(rgb_path)
        assert rgb_pil.size == (1216, 352)
        rgb_pil_crop = rgb_pil.crop((0, crop, 1216, 352))
        rgb = self.to_tensor(rgb_pil_crop).float()
        rgb = rgb*255.0

        return raw, rgb, os.path.basename(lidar), crop

def main():
    global args
    global dataset
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn
 
    model_path = os.path.join(args.save_path, args.model_path)
    if not args.upload_web:
        save_root = os.path.join(os.path.dirname(model_path), 'results')
    else:
        save_root = os.path.join(os.path.dirname(model_path), 'online_test')

    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    print("==========\nArgs:{}\n==========".format(args))
    # INIT
    print("Init model: '{}'".format(args.mod))
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in)
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        target_state = model.state_dict()
        for name, val in checkpoint['state_dict'].items():
            if name not in target_state or target_state[name].shape != val.shape:
                continue
            try:
                target_state[name].copy_(val)
            except RuntimeError:

                continue
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        return

    if args.cuda:
        model = model.cuda()
    print("Initializing dataset {}".format(args.dataset))
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)
    dataset.prepare_dataset()
    to_pil = transforms.ToPILImage()
    model.eval()
    print("===> Start testing")
    if not args.upload_web:
        lidar_paths = dataset.selected_paths['lidar_in']
        imgs_paths = dataset.selected_paths['img']
    else:
        lidar_paths = dataset.test_files['lidar_in']
        imgs_paths = dataset.test_files['img']
        
    dataset = KittiDataset(lidar_paths, imgs_paths)

    test_loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=4,
        pin_memory=False, drop_last=True)

    with torch.no_grad():
        for i, (raw, rgb, basename, crop) in tqdm.tqdm(enumerate(test_loader)):
            crop = crop[0].item()
            basename = basename[0]
            valid_mask = (raw > 0).detach().float()

            input = raw.cuda()

            if args.input_type == 'rgb':
                input = torch.cat((input, rgb.cuda()), 1)

            output = model(input)[0]
            output = torch.clamp(output, min=0, max=85)


            output = output * 256.
            raw = raw * 256.
            output = output[0][0:1].cpu()
            data = output[0].numpy()

            if crop != 0:
                padding = (0, 0, crop, 0)
                output = torch.nn.functional.pad(output, padding, "constant", 0)
                output[:, 0:crop] = output[:, crop].repeat(crop, 1)

            pil_img = to_pil(output.int())
            assert pil_img.size == (1216, 352)
            pil_img.save(os.path.join(save_root, basename))
    print('num imgs: ', i + 1)


if __name__ == '__main__':
    main()
