# PointNorm

[[Paper](https://arxiv.org/abs/2304.04884)] [[Code](https://github.com/MinghuiNie/PointNorm-Net)]

PointNorm adopts the traditional optimization method, uses the Adam optimizer, does not use the deep neural network training, and is the original traditional unsupervised method.

The code was tested with Python 3.7.3, torch 1.4.0, torchvision 0.5.0, CUDA 10.1.243, and cuDNN 7605 on Ubuntu 18.04.

PointNorm is just a network-less version of PointNorm-Net. 


## Citation

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

## License
See LICENSE file.