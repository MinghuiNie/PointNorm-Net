# PointNorm-Net: Self-Supervised Normal Prediction of 3D Point Clouds via Multi-Modal Distribution Estimation (TPAMI 2025)
[[Paper](https://arxiv.org/abs/2304.04884)] [[Code](https://github.com/MinghuiNie/PointNorm-Net)] [[Project](https://minghuinie.github.io/PointNorm-Net/)] 

This paper will published at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Please refer to our paper and project for more detail.
### Introduction
This is the code for self-supervised normal estimation using PointNorm-Net.
It allows to train, test and evaluate our self-supervised normal estimation model.
We provide the code for train a model or use a pretrained model on your own data.

Please follow the installation instructions below.
 
### Instructions

##### 1. Requirements

Install [PyTorch](https://pytorch.org/).

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.
For a full list of requirements see `requirements.txt`.

#####  2. Estimate normal vectors for your data:

To test DeepFit on your own data. Run the `compute_normals.py` in the `./tutorial` directory.
It allows you to specify the input file path (`.xyz` file), output path for the estimated normals, jet order (1-4), and a mode (use pretrained DeepFit or our pytorch implementation of the classic jet fitting).

To help you get started, we provide a step by step tutorial `./tutorial/DeepFit_tutorial.ipynb` with extended explenations, interactive visualizations and example files.

 ##### 3.Reproduce the results in the paper:
Run `get_data.py` to download PCPNet data.

Alternatively, Download the PCPNet data from this [link](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) and place it in  `./data/pcpnet/` directory.

To test the model and output all normal estimations for the dataset run `test_n_est.py`. This will export the normal estimations for each file in the provided file list as a `.normals` file.  

To evaluate the results and output a report run `evaluate.py`

To get all of the method's outputs exported (`beta, weights, normals, curvatures`) run `test_c_est.py`.

To evaluate curvature estimation performance run `evaluate_curvatures.py` (after exporting the results).

##### 4.Train your own model:
To train a model run `train_n_est.py`.

To train, test and evaluate run `run_DeepFit_single_experiment.py`.
Alternatively you can run individual train, test and evaluation.

#### Visualization
Click on the link for details on [how to visialize normal vectors on 3D point clouds](http://www.itzikbs.com/how-to-visualize-normal-vectors-on-3d-point-clouds).

For a quick visualization of a single 3D point cloud with the normal vector overlay run the `visualize_normals.m` script provided MATLAB code in `./MATLAB`.

For visualizing all of the PCPNet dataset results and exporting images use `export_visualizations.m`.

### Citation

Please cite our paper if you use this code in your own work:

## PointNorm
PointNorm adopts the traditional optimization method, uses the Adam optimizer, does not use the deep neural network training, and is the original traditional unsupervised method.

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.

PointNorm is just a network-less version of PointNorm-Net. 

```
@article{zhang2025pointnorm,
  title={PointNorm-Net: Self-Supervised Normal Prediction of 3D Point Clouds via Multi-Modal Distribution Estimation},
  author={Jie Zhang, Minghui Nie, Changqing Zou, Jian Liu, Ligang Liu and Junjie Cao},
  booktitle={arXiv preprint arXiv:2304.04884},
  year={2025},
  month={April}
}
```

 ### License
See LICENSE file.
