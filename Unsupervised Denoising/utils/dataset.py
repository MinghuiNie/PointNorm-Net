import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

class H5Dataset(Dataset):

    def __init__(self, h5py_filename, dataset_name, normal_name='normal', batch_size=1, transform=None):
        super().__init__()
        h5file = h5py.File(h5py_filename, mode='r')
        self.pointclouds = h5file[dataset_name]
        self.normals = h5file[normal_name] if normal_name is not None else None
        self.transform = transform
        self.batch_size = batch_size
              
    def __len__(self):
        return (self.pointclouds.shape[0] // self.batch_size) * self.batch_size

    def __getitem__(self, idx:int):
        item = {
            'pos' : torch.FloatTensor(self.pointclouds[idx]),
        }
        if self.normals is not None:
            item['normal'] = torch.FloatTensor(self.normals[idx])

        if self.transform is not None:
            item = self.transform(item)

        return item


class MultipleH5Dataset(Dataset):

    def __init__(self, files, dataset_name, guess_name, normal_name=None, batch_size=1, transform=None, random_get=False, subset_size=-1):
        super().__init__()
        pointclouds = []
        normals = []
        guess_value = []
        for filename in files:
            h5file = h5py.File(filename, mode='r')
            pointclouds.append(h5file[dataset_name])
            guess_value.append(h5file[guess_name])
            if normal_name is not None:
                normals.append(h5file[normal_name])
        self.pointclouds = np.concatenate(pointclouds, axis=0)
        self.guess_value = np.concatenate(guess_value, axis=0)
        self.normals = np.concatenate(normals, axis=0) if normal_name is not None else None
        self.transform = transform
        self.batch_size = batch_size
        self.random_get = random_get
        self.subset_size = subset_size

    def __len__(self):
        if self.subset_size >= self.batch_size:
            return (self.subset_size // self.batch_size) * self.batch_size
        else:
            return (self.pointclouds.shape[0] // self.batch_size) * self.batch_size

    def __getitem__(self, idx:int):
        if self.random_get:
            idx = np.random.randint(0, self.pointclouds.shape[0] - 1)

        item = {
            'pos' : torch.FloatTensor(self.pointclouds[idx]),
            'vaz' : torch.FloatTensor(self.guess_value[idx]),
        }
        if self.normals is not None:
            item['normal'] = torch.FloatTensor(self.normals[idx])

        if self.transform is not None:
            item = self.transform(item)

        return item


if __name__ == '__main__':
    dataset = MultipleH5Dataset([
        './data/10kh5py.h5', 
        './data/20kh5py.h5', 
        './data/30kh5py.h5',
        './data/40kh5py.h5',
        './data/50kh5py.h5'], dataset_name='train')
    print(dataset.pointclouds.shape)