from ctypes import pointer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.spatial as spatial
import os
import os.path
import torch.optim.lr_scheduler as lr_scheduler

class Normals(torch.nn.Module):
    def __init__(self):
        super(Normals, self).__init__()
        self.n = torch.nn.parameter.Parameter(torch.tensor([1.0,0,0]), requires_grad=True)

    def forward(self, mcn):
        tem = torch.Tensor([0,0,0]).cuda()
        if (self.n== tem)[0] and  (self.n== tem)[1] and  (self.n== tem)[2]:
            pred = None
            e = 0
        else:
            pred = self.n/torch.sqrt(torch.sum(self.n**2))
            e = -0.5 * torch.exp(-torch.norm(torch.cross(pred.expand(mcn.size()), mcn, dim=-1), p=2, dim=1)**2/0.25).mean()
        return e, pred

# Input from this
indirname = '../data/pcv_data_v1_v2/v1_test.txt'
output_dir = os.path.join('/home/zhangjie/unsupervised-normal-estimation/MUSNE_pcv_200/')

list_model = []
with open (indirname, 'r') as f:
    for line in f:
        list_model.append(line.rstrip('\n'))

for i in range(len(list_model)):
    point = np.loadtxt('../data/pcv_data_v1_v2/'+list_model[i]+'.xyz')
    #idx = np.loadtxt('../data/pcv_data_v1_v2/'+list_model[i]+'.pidx')
    weight = np.load('../data/pcv_data_v1_v2/'+list_model[i]+'.weights.npy')
    point =  torch.FloatTensor(point)
    #idx =  torch.FloatTensor(idx)
    weight =  torch.FloatTensor(weight)

    kdtree = spatial.cKDTree(point, 10)
    
    pca_knn = 0
    mean_weights = torch.mean(weight, 0)[0]
    if mean_weights < 0.02:
            pca_knn = 32
    elif  mean_weights < 0.14:
            pca_knn = 128
    elif  mean_weights < 0.16:
            pca_knn = 256
    else:
            pca_knn = 450
    
    normals_idx = torch.zeros(len(point), 3, dtype=torch.float)
    for ii in range(len(point)):
        point_distances_pca, pca_point_inds = kdtree.query(point[ii,:], k=pca_knn)
        rad_pca = max(point_distances_pca)
        pca_pts = point[pca_point_inds, :]
        patch_normal_pca_sel = torch.zeros(200, 3, dtype=torch.float)

        patch_normal_pca = torch.zeros(200, 3, dtype=torch.float)
        pca_pts_center_point = pca_pts - point[ii,:]
        pca_pts_center_point = pca_pts_center_point / rad_pca
        for iii in range(200):
            point_num = 4
            id_pca  = np.random.choice(pca_point_inds,point_num)
            point_pca = point[id_pca, :]
            point_pca_mean = point_pca.mean(0)
            point_pca = point_pca - point_pca_mean
            trans, _, _ = torch.svd(torch.t(point_pca))
            patch_normal_pca[iii,:] = trans[:,2]
        
        sigm = 0.01
        select_num = -181          
        if pca_knn == 450:
            select_num = -201
        if pca_knn == 256:
            select_num = -201
        patch_points_dis = torch.mm(patch_normal_pca, pca_pts_center_point.transpose(1, 0))**2
        patch_dis = torch.sum(torch.exp(-patch_points_dis/sigm), 1)
        [v, id] = torch.sort(patch_dis)
        if select_num == -181:
            patch_normal_pca_sel[0:180] = patch_normal_pca[id[select_num:-1],:]
            patch_normal_pca_sel = patch_normal_pca_sel[0:180]
        else:
            patch_normal_pca_sel = patch_normal_pca

        mcn = patch_normal_pca_sel.cuda()
        mcn_mean = patch_normal_pca_sel.cuda()
        for jj in range(mcn_mean.shape[0]):
            if mcn_mean[jj,0]*mcn_mean[1,0]+mcn_mean[jj,1]*mcn_mean[1,1]+mcn_mean[jj,2]*mcn_mean[1,2] < 0:
                mcn_mean[jj,:] = -mcn_mean[jj,:]

        model =Normals().cuda()
        optimizer = optim.Adam(model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0000001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.1)
        nn.init.constant_(model.n[0], mcn_mean.mean(0)[0])
        nn.init.constant_(model.n[1], mcn_mean.mean(0)[1])
        nn.init.constant_(model.n[2], mcn_mean.mean(0)[2])

        for epoch in range(100):
            optimizer. zero_grad() 
            e,pred= model(mcn)
            if pred == None:
                for jj in range(mcn.shape[0]):
                    if mcn[jj,0]*mcn[1,0]+mcn[jj,1]*mcn[1,1]+mcn[jj,2]*mcn[1,2] < 0:
                        mcn[jj,:] = -mcn[jj,:]
                pred = mcn.mean(0)
                pred = pred/torch.sqrt(torch.sum(pred**2))
                break
            e.backward() 
            optimizer. step()
            if (ii % 100) == 0:
                print( f'shape={i} point={ii} epoch={epoch}  loglik={-e .item():.3}')      
        normals_idx[ii,:] = pred
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, list_model[i] + '.normals'),normals_idx.detach())
    print('saved normals for ' + list_model[i])
