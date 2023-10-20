## PointNorm-Net: Unsupervised Normal Estimation of 3D Point Clouds via Local Multi-Sampling Consensus

[[Paper](https://arxiv.org/abs/2304.04884)] [[Code](https://github.com/MinghuiNie/PointNorm-Net)]

### Introduction
This is the code for unsupervised normal estimation using PointNorm-Net.
It allows to train, test and evaluate our unsupervised normal estimation model.
We provide the code for train a model or use a pretrained model on your own data.

Please follow the installation instructions below.

Abstract:

Supervised deep normal estimators have made great strides on synthetic benchmarks recently, while their performances
drop dramatically when applied to real-world scenarios. Building high-quality real training data to boost those supervied methods is not
trivial because point-wise annotation of normals for varying-scale real-world 3D scenes is a tedious and expensive task. This paper
proposes PointNorm-Net, the first unsupervised deep framework, to address the challenge. The key novelty of PointNorm-Net is a
normal consensus (MSC) paradigm built on local consensus of points, which can be integrated into either deep or traditional
optimization-based normal estimation frameworks. Comprehensive experiments demonstrate better generalization ability and even
superior performance over the SOTA conventional and deep methods on three public real datasets. It also shows noticeably superiority
over existing supervised deep normal estimators on the most popular synthetic datasets


### Citation

Please cite our paper if you use this code in your own work:

```
@article{Zhang2023PointNorm,
  title={PointNorm-Net: Unsupervised Normal Estimation of 3D Point Clouds via Local Multi-Sampling Consensus},
  author={Jie Zhang, Minghui Nie, Junjie Cao, Jian Liu, Changqing Zou and Ligang Liu},
  booktitle={arXiv preprint arXiv:2304.04884},
  year={2023},
  month={April}
}
```

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

 ### License
See LICENSE file.
