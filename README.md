## Introduction

This is the code for Multiple Instance Active Learning for Object Detection, ID 453 for CVPR 2021.

Thanks for your review.

## Environment Installation

A Linux platform and [anaconda3](https://www.anaconda.com/) is recommended, since they can install and manage environments and packages conveniently and efficiently.

After anaconda3 installation, you can create a conda environment and install the required packages as below:

```
conda create -n mmdet python=3.7 -y
conda activate mmdet
pip install -r requirements.txt
```

You may change the name of the conda environment if you like, but you will need to pay attention to the following steps correspondingly if you do so.

Please refer to [MMDetection v2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) if you encounter any problems.

## Modification in mmcv Packages

To train two dataloaders (i.e., the labeled set dataloader and the unlabeled set dataloader mentioned in the paper) at the same time, you will need to modify the ``` epoch_based_runner.py ``` in the mmcv package.

Considering that this will affect all code that uses this environment, so we suggest you set up a separate environment for MIAL (i.e., the "mmdet" environment created above).

```
cp -v epoch_based_runner.py ~/anaconda3/envs/mmdet/lib/python3.7/site-packages/mmcv/runner/
```

## Datasets Preparation

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
cd $YOUR_DATASET_PATH
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
```
After that, please modify the corresponding dataset directory in this repository, they are located in:
```
Line 1 in configs/debug.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
Line 1 in configs/_base_/voc0712.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
```
Please change the "$YOUR_DATASET_PATH"s above to your actual dataset directory (i.e., the directory where you intend to put the downloaded VOC tar file).

And please use the absolute path (i.e., start with "~/" or "/") but not a relative path (i.e., start with "./" or "../"）.

## Training and Test

We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

And we also recommend you to use only 1 GPU to do, because there may be some problems on arranging the end of two dataloaders when you use several GPUs. The more GPUs you use, the more likely the training or test iteration will be interrupted suddenly.

If you use only 1 GPU, you can use the ``` script.sh ``` file directly as below:
