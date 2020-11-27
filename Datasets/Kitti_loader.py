import os
import sys
import re

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class Kitti_preprocessing(object):
    def __init__(self, dataset_path, input_type='depth'):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.test_files = {'img': [], 'lidar_in': []}
        self.dataset_path = dataset_path
        self.left_side_selection = 'image_02'
        self.right_side_selection = 'image_03'

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                if 'velodyne_raw' in root.split(os.sep):
                    for file in sorted(files):
                        if 'train' in root.split(os.sep):
                            self.train_paths['lidar_in'].append(os.path.join(root, file))
                            rgb_root = root.replace('proj_depth/velodyne_raw/', '')
                            rgb_root = os.path.join(rgb_root, 'data')
                            date = re.findall(r'[0-9]+_[0-9]+_[0-9]+', rgb_root)[0]
                            rgb_root = rgb_root.replace('train', date)
                            self.train_paths['img'].append(os.path.join(rgb_root, file))
                            self.train_paths['gt'].append(os.path.join(root.replace('velodyne_raw', 'groundtruth'), file))
                            # assert os.path.exists(os.path.join(rgb_root, file))
                            assert os.path.exists(os.path.join(root.replace('velodyne_raw', 'groundtruth'), file))
                        elif 'val' in root.split(os.sep):
                            self.val_paths['lidar_in'].append(os.path.join(root, file))
                            rgb_root = root.replace('proj_depth/velodyne_raw/', '')
                            rgb_root = os.path.join(rgb_root, 'data')
                            date = re.findall(r'[0-9]+_[0-9]+_[0-9]+', rgb_root)[0]
                            rgb_root = rgb_root.replace('val', date)
                            self.val_paths['img'].append(os.path.join(rgb_root, file))
                            self.val_paths['gt'].append(os.path.join(root.replace('velodyne_raw', 'groundtruth'), file))
                            # assert os.path.exists(os.path.join(rgb_root, file))
                            assert os.path.exists(os.path.join(root.replace('velodyne_raw', 'groundtruth'), file))


    def get_selected_paths(self, selection):
        files = []
        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):
            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))
        return files

    def prepare_dataset(self):
        path_to_val_sel = 'depth_selection/val_selection_cropped'
        path_to_test = 'depth_selection/test_depth_completion_anonymous'
        self.get_paths()
        self.selected_paths['lidar_in'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'velodyne_raw'))
        self.selected_paths['gt'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'groundtruth_depth'))
        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
        self.test_files['lidar_in'] = self.get_selected_paths(os.path.join(path_to_test, 'velodyne_raw'))
        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
        self.test_files['img'] = self.get_selected_paths(os.path.join(path_to_test, 'image'))

