## Introduction

This is the code for *Multiple Instance Active Learning for Object Detection*, anonymous ID 453 for CVPR 2021.

## Environment Installation

A Linux platform (Ours are Ubuntu 18.04 LTS) and [anaconda3](https://www.anaconda.com/) is recommended, since they can install and manage environments and packages conveniently and efficiently.

A TITAN V GPU and [CUDA 10.2](https://developer.nvidia.com/cuda-toolkit-archive) with [CuDNN 7.6.5](https://developer.nvidia.com/cudnn) is recommended, since they can speed up model training.

After anaconda3 installation, you can create a conda environment and install the required packages as below:

```
conda create -n mmdet python=3.7 -y
conda activate mmdet
pip install numpy
pip install -r requirements.txt
```

You may change the name of the conda environment if you like, but you will need to pay attention to the following steps correspondingly if you do so.

Please refer to [MMDetection v2.3.0](https://github.com/open-mmlab/mmdetection/tree/v2.3.0) if you encounter any problems.

## Modification in the mmcv Package

To train with two dataloaders (i.e., the labeled set dataloader and the unlabeled set dataloader mentioned in the paper) at the same time, you will need to modify the ` epoch_based_runner.py ` in the mmcv package.

Considering that this will affect all code that uses this environment, so we suggest you set up a separate environment for MIAL (i.e., the ` mmdet `environment created above).

```
cp -v epoch_based_runner.py ~/anaconda3/envs/mmdet/lib/python3.7/site-packages/mmcv/runner/
```

## Datasets Preparation

Please download VOC2007 datasets (*trainval*+*test*) and VOC2012 datasets (*trainval*) from:

VOC2007 (*trainval*): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

VOC2007 (*test*): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

VOC2012 (*trainval*): http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

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
Line 1 of configs/MIAL.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
Line 1 of configs/_base_/voc0712.py: data_root='$YOUR_DATASET_PATH/VOCdevkit/'
```
Please change the ` $YOUR_DATASET_PATH `s above to your actual dataset directory (i.e., the directory where you intend to put the downloaded VOC tar file).

And please use the absolute path (i.e., start with ` / `) but not a relative path (i.e., start with ` ./ ` or ` ../ `）.

## Training and Test

We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

And we also recommend you to use a single GPU, because the usage of multi-GPU may result in errors caused by the multi-processing of the dataloader.

If you use only a single GPU, you can use the ` script.sh ` file directly as below:
```
chmod 777 ./script.sh
./script.sh $YOUR_GPU_ID
```
Please change the ` $YOUR_GPU_ID ` above to your actual GPU ID number (usually a non-negative number).

Please ignore the error ` rm: cannot remove './log_nohup/nohup_$YOUR_GPU_ID.log': No such file or directory ` if you run the ` script.sh ` file for the first time.

The ` script.sh ` file will use the GPU with the ID number ` $YOUR_GPU_ID ` and PORT `(30000+$YOUR_GPU_ID*100)` to train and test.

The log file will not flush in the terminal, but will be saved and updated in the file `./log_nohup/nohup_$YOUR_GPU_ID.log` and ` ./work_dirs/retina/$TIMESTAMP.log ` . These two logs are the same. You can change the directories and names of the latter log files in Line 48 of `./configs/MIAL.py` .

You can also use other files in the directory ` './work_dirs/retina/ ` if you like, they are as follows:

- **JSON file `$TIMESTAMP.log.json`**

  You can load the losses and mAPs during training and test from it more conveniently than from the `./work_dirs/retina/$TIMESTAMP.log` file.

- **npy file `X_L_$CYCLE.npy` and `X_U_$CYCLE.npy`**

  The `$CYCLE` is an integer from 0 to 6, which are the active learning cycles.

  You can load the indexes of the labeled set and unlabeled set for each cycle from them.

  The indexes are the integers from 0 to 16550 for PASCAL VOC datasets, where 0 to 5010 is for PASCAL VOC 2007 *trainval* set and 5011 to 16550 for PASCAL VOC 2012 *trainval* set.

  An example code for loading these files is the Line 108-114 in the `./tools/train.py` file (which are in comments now).

- **pth file `epoch_$EPOCH.pth` and `latest.pth`**

  The `$EPOCH` is an integer from 0 to 2, which are the epochs of the last label set training.

  You can load the model state dictionary from them.

  An example code for loading these files is the Line 109, 143-145 in the `./tools/train.py` file (which are in comments now).

- **txt file `trainval_L_07.txt`, `trainval_U_07.txt`, `trainval_L_12.txt` and `trainval_U_12.txt` in each `cycle$CYCLE` directory**

  The `$CYCLE` is the same as above.

  You can load the names of JPEG images of the labeled set and unlabeled set for each cycle from them.

  "L" is for the labeled set and "U" is for the unlabeled set. "07" is for the PASCAL VOC 2007 *trainval* set and "12" is for the PASCAL VOC 2012 *trainval* set.

## Code Structure
```
├── $YOUR_ANACONDA_TORY
│   ├── anaconda3
│   │   ├── envs
│   │   │   ├── $YOUR_ENVIRONMENT_NAME
│   │   │   │   ├── lib
│   │   │   │   │   ├── python3.7
│   │   │   │   │   │   ├── site-packages
│   │   │   │   │   │   │   ├── mmcv
│   │   │   │   │   │   │   │   ├── runner
│   │   │   │   │   │   │   │   │   ├── epoch_based_runner.py
│
├── ...
│
├── configs
│   ├── _base_
│   │   ├── default_runtime.py
│   │   ├── retinanet_r50_fpn.py
│   │   ├── voc0712.py
│   ├── MIAL.py
│── log_nohup
├── mmdet
│   ├── apis
│   │   ├── __init__.py
│   │   ├── test.py
│   │   ├── train.py
│   ├── models
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── MIAL_head.py
│   │   │   ├── MIAL_retina_head.py
│   │   │   ├── base_dense_head.py 
│   │   ├── detectors
│   │   │   ├── base.py
│   │   │   ├── single_stage.py
│   ├── utils
│   │   ├── active_datasets.py
├── tools
│   ├── train.py
├── work_dirs
│   ├── retina
├── script.sh
```

The code files and folders shown above are the main part of MIAL, while other code files and folders are created following MMDetection to avoid potential problems.

The explanation of each code file or folder is as follows:

- **epoch_based_runner.py**: Code for training and test in each epoch, which can be called from `./apis/train.py`
- **configs**: Configuration folder, including runnning settings, model settings, dataset settings and other custom settings for active learning and MIAL.
  - **_base_**: 
    - **default_runtime.py**:
    - **retinanet_r50_fpn.py**:
    - **voc0712.py**:
  - **MIAL.py**: Configuration code in general including most custom settings, containing active learning dataset settings, model training and test parameter settings, custom hyper-parameter settings, log file and model saving settings, which can be mainly called from `./tools/train.py`. The more detailed introduction of each parameter is in the comments of this file.
- **log_nohup**: Log folder for storing log output on each GPU temporarily.
- **mmdet**: The core codes for MIAL, including intermidiate training code, object detectors and detection heads and active learning dataset establishment.
- **tools**: Outer training and test code for MIAL.
  - **train.py**: Outer code for training and test for MIAL, including generating PASCAL VOC datasets for active learning, loading image sets and models, Instance Uncertainty Re-weighting and Informative Image Selection in general, which can be called from `./script.sh`.
- **work_dirs**: Work directory of the index and image name of the labeled set and unlabeled set for each cycle, all log and json outputs and the model state dictionary for the last 3 cycle, which are introduced in the **Training and Test** part above.
- **script.sh**: The script to run MIAL on a single GPU. You can run it to train and test MIAL simply and directly mentioned in the **Training and Test** part above as long as you have prepared the conda environment and PASCAL VOC 2007+2012 datasets.
