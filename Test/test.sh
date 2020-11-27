#!/bin/bash

echo 'Save path is: 'Saved/$1/$2
echo 'Data path is: '${3-./Data/Kitti}
if [ ${4-False} = False ] || [ ${4-False} = false ]
then
	echo 'Computing results for selected validation set'
else
	echo 'Saving predictions for online test set'
fi
python Test/test.py --data_path ${3-./Data/Kitti} --save_path Saved/$1 --model_path $2 --upload_web ${4-False}

# Arguments for evaluate_depth file: 
# - ground truth directory
# - results directory
if [ ${4-False} = False ] || [ ${4-False} = false ]
then
	Test/devkit/cpp/evaluate_depth ${3-./Data/Kitti}/depth_selection/val_selection_cropped/groundtruth_depth Saved/$1/results 
fi
