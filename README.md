## Introduction

This is the code for Multiple Instance Active Learning for Object Detection, ID 453 for CVPR 2021. Thanks for your review.

## Environment Installation
```
conda create -n mmdet python=3.7 -y
conda activate mmdet
pip install -r requirements.txt
```
Please refer to [MMDetection v2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) if you encounter any problems.

## Modification in mmcv packages
```
cp -v epoch_based_runner.py ~/anaconda3/envs/mmdet/lib/python3.7/site-packages/mmcv/runner/
```
The purpose to modify this python file is to train two dataloaders at the same time.

## Datasets preparation
Please download VOC2007 datasets (trainval+test) and VOC2012 datasets (trainval) from:
VOC2007 (trainval): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
VOC2007 (test): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
VOC2012 (trainval): http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
And after that, please ensure the file directory tree is as below:
```
├── VOCdevkit
│   ├── VOC2007
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   ├── VOC2012
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
```
You may also use the following commands directly:
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
```
