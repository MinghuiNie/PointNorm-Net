import torch
from torch.nn import Module, Sequential, ModuleList

from .blocks import *
from .utils import *
from .loss import *

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


class DenoiseNet(Module):

    def __init__(
        self, 
        loss_rec='unsupervised', 
        loss_ds='cd', 
        activation='relu', 
        dynamic_graph=True, 
        conv_knns=[8, 16, 24], 
        conv_channels=32, 
        conv_layer_out_dim=24, 
        gpool_use_mlp=False, 
        use_random_mesh=False,
        use_random_pool=False,
        no_prefilter=False,
    ):
        super().__init__()
        self.feats = ModuleList()
        self.feat_dim = 0
        for knn in conv_knns:
            feat_unit = FeatureExtraction(dynamic_graph=dynamic_graph, conv_knn=knn, conv_channels=conv_channels, conv_layer_out_dim=conv_layer_out_dim, activation=activation)
            self.feats.append(feat_unit)
            self.feat_dim += feat_unit.out_channels

        self.downsample = DownsampleAdjust(feature_dim=self.feat_dim, ratio=0.5, use_mlp=gpool_use_mlp, activation=activation, random_pool=use_random_pool, pre_filter=not no_prefilter)

        if use_random_mesh:
            self.upsample = Upsampling(feature_dim=self.feat_dim, mesh_dim=2, mesh_steps=2, use_random_mesh=True, activation=activation)
        else:
            self.upsample = Upsampling(feature_dim=self.feat_dim, mesh_dim=1, mesh_steps=2, use_random_mesh=False, activation=activation)
        
        self.loss_ds = get_loss_layer(loss_ds)
        self.loss_rec = get_loss_layer(loss_rec)

        # print('Loss: ')
        # print(self.loss_ds)
        # print(self.loss_rec)
        
        self.epoch = 0

    def forward(self, pos):
        self.epoch += 1

        feats = []
        for feat_unit in self.feats:
            feats.append(feat_unit(pos))
        feat = torch.cat(feats, dim=-1)

        #ax = plt.subplot(projection = '3d')
        #ax.scatter(pos.cpu().detach().numpy()[0,:,0], pos.cpu().detach().numpy()[0,:,1], pos.cpu().detach().numpy()[0,:,2],c = 'g')

        idx, pos, feat, score = self.downsample(pos, feat)
        
        #ax.scatter(pos.cpu().detach().numpy()[0,:,0], pos.cpu().detach().numpy()[0,:,1], pos.cpu().detach().numpy()[0,:,2],c = 'r')

        self.adjusted = pos
        pos = self.upsample(pos, feat)
        
        #ax.scatter(pos.cpu().detach().numpy()[0,:,0], pos.cpu().detach().numpy()[0,:,1], pos.cpu().detach().numpy()[0,:,2],c = 'b')
        #plt.show()
        self.preds = pos
        return pos, score


    def get_loss(self, inputs, guess,score,  **kwargs):
        loss = self.loss_rec(preds=self.preds, inputs=inputs, guess = guess, score = score, epoch=self.epoch)
        #loss = self.loss_rec(preds=self.preds, inputs=inputs, epoch=self.epoch)
        loss = loss.requires_grad_(True)
        
        if self.loss_ds is None:
            loss = loss + self.loss_ds(preds=self.preds, gts=gts, normals=normals, inputs=inputs, epoch=self.epoch)
        return loss

