# Project To Adapt
In this repository you can find the official PyTorch implementation of [Project To Adapt](https://arxiv.org/abs/2008.01034) (ACCV20, Oral).
## Environment
We tested the code using PyTorch 1.2 and Ubuntu 18.04. To download the extra required packages to run the code, you can use `pip install -r requirements.txt`

## Datasets
We use two datasets in our code. In our project, we collect a synthetic dataset using CARLA 0.84. The synthetic data can be downloaded running the following code:
```
cd Data/Carla
./download_carla_data.sh
```
In `Data/Carla` you will also find the camera poses used for the data generation in three different files.

You also need to download and extract in `Data/Kitti` the relevant KITTI data as we use KITTI as our target dataset. To do so, you need to go to the [KITTI depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) benchmark, and then request access and download: the annotated depth maps data set (14 GB), projected raw LiDaR scans data set (2 GB), and the manually selected validation and test data sets (5 GB). Additionally, you also need to download the raw KITTI dataset to have access to the RGB images. You can use the raw data development kit in [the KITTI raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) website to download the raw data. After downloading and extracting the data in a folder, you should have the following data structure:
```
|--Data/Kitti
  |--train
  	|--2011_09_26_drive_0001_sync
  	|--...
  |--val
  	|--2011_09_26_drive_0002_sync
  	|--...
  |--depth_selection
  	|--test_depth_completion_anonymous
  	|--val_selection_cropped
  |--2011_09_26
  |--2011_09_28
  |--2011_09_29
  |--2011_09_30
  |--2011_10_03
```

## Training
We perform two steps of training as explained in the paper. To run the first training step, you can run:
```
./train_1st_step.sh [carla_data_path] [kitti_data_path] [batch_size]
```
where the data path are by default `./Data/Carla` and `./Data/Kitti`, and the default batch size is 4.
```
./train_2nd_step.sh [carla_data_path] [kitti_data_path] [batch_size]
```
where in this case the default batch size is 2. In this case batch size=2 refers to using 2 source dataset and 2 target dataset images, amounting to a total of 4 images per batch. In this second step, we load the model trained during the first step.

## Testing
To evaluate your models, we use the oficial evaluation code from the Kitti devkit.
```
./Test/test.sh [save_name] [checkpoint_path] [online_test]
```
The last argument, `[online_test]` is a boolean argument set to `False` by default. When set to `False` it runs the evaluation in the selected validation set (1000 images) where the ground-truth is publicly available, so you automatically obtain the RMSE, MAE and other error metrics. When set to `True`, it saves the predicted depth for the online test images where no ground-truth is given. If you want to obtain the results for the online test set you need to upload the predicted depth to the [KITTI website](http://www.cvlibs.net/datasets/kitti/user_login.php).

## Citation
If you use Project To Adapt for your research, you can cite the paper using the following Bibtex entry:
```
@inproceedings{lopez2020p2a,
  title={Project to Adapt: Domain Adaptation for Depth Completion from Noisy and Sparse Sensor Data},
  author={Lopez-Rodriguez, Adrian and Busam, Benjamin and Mikolajczyk, Krystian},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2020}
}
```
## Acknowledgements
We adapted the code from [FusionNet](https://github.com/wvangansbeke/Sparse-Depth-Completion).
