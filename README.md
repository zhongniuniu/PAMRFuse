# Unsupervised fusion of misaligned PAT and MRI images via mutually reinforcing cross-modality image generation and registration


## Contents

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Instructions for Running Code](#instructions-for-running-code)


# 1. Overview

This repository provides the code for our papar [Unsupervised fusion of misaligned PAT and MRI images via mutually reinforcing cross-modality image generation and registration].

by Yutian Zhong, Shuangyang Zhang, Zhenyang Liu, Xiaoming Zhang, Zongxin Mo, Yizhe Zhang, Haoyu Hu, Wufan Chen, and Li Qi.

Photoacoustic tomography (PAT) and magnetic resonance imaging (MRI) are two advanced imaging techniques widely used in pre-clinical research. PAT has high optical contrast and deep imaging range but poor soft tissue contrast, whereas MRI provides excellent soft tissue information but poor temporal resolution. Despite recent advances in medical image fusion with pre-aligned multimodal data, PAT-MRI image fusion remains challenging due to misaligned images and spatial distortion. To address these issues, we propose an unsupervised multi-stage deep learning framework called PAMRFuse for misaligned PAT and MRI image fusion. PAMRFuse comprises a multimodal to unimodal registration network to accurately align the input PAT-MRI image pairs and a self-attentive fusion network that selects information-rich features for fusion. We employ an end-to-end mutually reinforcing mode in our registration network, which enables joint optimization of cross-modality image generation and registration. To the best of our knowledge, this is the first attempt at information fusion for misaligned PAT and MRI. Qualitative and quantitative experimental results show the excellent performance of our method in fusing PAT-MRI images of small animals captured from commercial imaging systems.

# 2. Installation Guide

Before running this package, users should have `Python`, `PyTorch`, and several python packages installed (`numpy`, `skimage`, `yaml`, `opencv`, `odl`) .


## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:

```
Python        (3.6.6)
torch         (1.9.0+cu111)
visdom        (0.1.8.9)
numpy         (1.19.2)
skimage       (0.15.0)
Yaml          (5.4.1)
cv2           (3.4.2)
PIL          (8.3.2)
```

## Package Installment

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install PyTorch, please refer to their official [website](https://pytorch.org). 

```
pip install package-name
```

# 3. Instructions for Running Code

We provide PAMRFuse trained weights under [weights](./weights) folder.

## Synthesis and Registration Test:

```
python reg_test.py
```
Modify the parameters in the .yaml file (./reg_Yaml) as needed. If other port parameters are used, you need to modify the port in yaml. The registered and fused test images are in folders [reg_data](./reg_data) and [fusion_data](./fusion_data), respectively.

## Fusion Test:

```
python fusion_test.py
```

## Synthesis and Registration Train

The training dataset is too large to be uploaded and downloaded. It may be more convenient to create your own dataset. The code to create your own training dataset can be found under [data_process](./data_process) folder.

1. Create dataset
   -  train path/A/
   -  train path/B/
   -  val path/A/
   -  val path/B/ 
2. The default data file form is .npy and normalized to [-1,1].

```
python reg_train.py
```

## Fusion Train

```
python fusion_main.py
```

If you have any question, please email to me (ytzhong.smu@qq.com).
