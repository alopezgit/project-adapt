import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

def check_files_exists(full_path):
    assert os.path.exists(full_path.replace('Depth', 'RGB'))
    assert os.path.exists(full_path.replace('Central', 'Right'))
    assert os.path.exists(full_path.replace('CentralDepth', 'RightRGB'))
    assert os.path.exists(full_path.replace('Central', 'Lid'))

class Carla_preprocessing(object):
    def __init__(self, dataset_path, input_type='depth'):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': [], 'semantic':[]}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': [], 'semantic':[]}
        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': [], 'semantic':[]}
        self.test_files = {'img': [], 'lidar_in': []}
        self.dataset_path = dataset_path
        self.left_side_selection = 'image_02'
        self.right_side_selection = 'image_03'
        self.depth_keyword = 'episode'

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set), followlinks=True):
                if self.depth_keyword in root:
                    if 'train' in root:
                        for file in sorted(files):
                            if 'CentralDepth' not in file:
                                continue
                            full_path = os.path.join(root, file)
                            check_files_exists(full_path)
                            self.train_paths['lidar_in'].append(full_path.replace('Central', 'Lid'))
                            if os.path.exists(full_path.replace('Depth', 'SemanticSeg')):
                                self.train_paths['semantic'].append(full_path.replace('Depth', 'SemanticSeg'))
                            self.train_paths['gt'].append(full_path)
                            self.train_paths['img'].append(full_path.replace('Depth', 'RGB'))

                    if 'validation' in root:
                        for file in sorted(files):
                            if 'CentralDepth' not in file:
                                continue
                            full_path = os.path.join(root, file)
                            check_files_exists(full_path)
                            self.val_paths['lidar_in'].append(full_path.replace('Central', 'Lid'))
                            self.val_paths['gt'].append(full_path)
                            self.val_paths['semantic'].append(full_path.replace('Depth', 'SemanticSeg'))
                            self.val_paths['img'].extend(full_path.replace('Depth', 'RGB'))


    def prepare_dataset(self):
        self.get_paths()