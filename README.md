

# A Circular Window-based Cascade Transformer for Online Action Detection

## Introduction

This is a PyTorch implementation for our "A Circular Window-based Cascade Transformer for Online Action Detection".

![network](assert/network.png?raw=true)

## Environment



## Data Preparation

1. Download the [`THUMOS'14`](https://www.crcv.ucf.edu/THUMOS14/) and [`TVSeries`](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) datasets.

2. Extract video frames at 24 FPS;

3. For constructing the target files, we follow the method used in [LSTR](https://github.com/amazon-science/long-short-term-transformer).

4. The data should be organized according to the following structure. Please modify the root path of the dataset in the configuration file of the corresponding dataset.

    ```
    $DATASET_ROOT
    ├── frames/
    |   ├── video_test_0000004/ (6L images)
    |   |   ├── img_00000.jpg
    |   |   ├── ...
    │   ├── ...
    ├── targets/
    |   ├── video_test_0000004.npy (of size L x 22)
    |   ├──...
    ```

## Training

* X

The commands are as follows.

XX

## Online Inference

There are *two kinds* of evaluation methods in our code.

X

## Acknowledge

The project is built upon [MViTv2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) and [LSTR](https://github.com/amazon-science/long-short-term-transformer).
