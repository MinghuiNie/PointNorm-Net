# PointNorm-Net: Self-Supervised Normal Prediction of 3D Point Clouds via Multi-Modal Distribution Estimation (TPAMI 2025)
[[Project](https://minghuinie.github.io/PointNorm-Net/)] [[Paper](https://arxiv.org/abs/2304.04884)] [[Supplementary](https://drive.google.com/file/d/1j6GIqZHthQc_zU0ifTn5sFsggC5MLxlI/view?usp=drive_link)]
![图片描述](/docs/teaser14.jpg)
This paper will published at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Please refer to our paper and project for more detail.
### Introduction
This is the code for self-supervised normal estimation using PointNorm-Net. It allows to train, test and evaluate our self-supervised normal estimation model. We provide the code for train a model or use a pretrained model on your own data.

Technical Note: The implementation framework of this code resembles DeepFit in architecture and shares similar usage patterns. For specific implementation details, we direct readers to the original [DeepFit](https://github.com/sitzikbs/DeepFit) reference.

Please follow the installation instructions below.
 
### Instructions

##### 1. Requirements

Install [PyTorch](https://pytorch.org/).

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.
For a full list of requirements see `requirements.txt`.

#####  2. Estimate normal vectors for your data:

To test PointNorm-Net on your own data. Run the `test_n_est.py`.
It allows you to specify the input file path (`.xyz` file), output path for the estimated normals.


##### 4.Train your own model:
To train a model, the user should first use the`compute_multi_normal_v4.m`script located in the`weights&features`folder to calculate both the`.features`and`.weights`files required for network training. These calculations only require the positions of the point clouds. Then, the user can run`train_n_est.py`. 

To test run `test_n_esy.py`. Note that the pre-trained model should be replaced with the user-trained model. 

To evaluate run `evaluate.py`. 

### PointNorm
PointNorm adopts the traditional optimization method, uses the Adam optimizer, does not use the deep neural network training, and is the original traditional self-supervised method. To put it crudely (though not quite correctly), PointNorm is just a network-less version of PointNorm-Net. 

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.

### Citation

Please cite our paper if you use this code in your own work:

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
