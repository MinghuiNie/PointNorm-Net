import torch
import  os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
from tqdm import tqdm

def cd_loss(preds, gts):
    def batch_pairwise_dist(x, y):
        bs, num_points_x, points_dim = x.size()
        _,num_points_y, _ = y.size()
        x = x.squeeze()
        y = y.squeeze()

        loss_1 = 0
        x_line = x.size()[0]
        for i in range(0 , x_line):
            current_x = x[i,:]
            current_x_norm = current_x - y
            minxs = torch.min(torch.sqrt(torch.sum(input = current_x_norm**2 , dim = 1) ))
            loss_1 = (loss_1 + minxs)
        loss_1 = loss_1/x_line  #CD later

        loss_2 = 0
        y_line = y.size()[0]
        for i in range(0,y_line):
            current_y = y[i,:]
            current_y_norm = current_y - x
            minys = torch.min(torch.sqrt(torch.sum(input = current_y_norm**2,dim = 1)))
            loss_2 = loss_2 +minys
        loss_2 = loss_2 / y_line #CD former

        return loss_1 , loss_2

    loss_1 , loss_2= batch_pairwise_dist(gts, preds)
    loss_3 = loss_1 + loss_2 
    print(loss_3)
    print(loss_1)
    return loss_3, loss_2  #loss_3 is CD, loss_2 is P2S.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--predicts', type=str, default='./mhn_predict_12loss_sel_69epoch/test-025/our-010.TXT')
    parser.add_argument('--grounds', type=str, default='./data/gts-50k')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    device = torch.device("cpu" if args.gpu_idx < 0 else "cuda:%d" % 0)

    preds_filename = []
    with open(args.predicts, 'r') as DSFile:        
        for line in DSFile:
            modelName = line.rstrip()
            preds_filename.append(modelName)

    total_loss_cd = 0
    total_loss_p2s =  0
    count = 0
    for current_preds_file in preds_filename:
        print('[INFO] Loading: %s' % current_preds_file)
        preds = np.loadtxt(os.path.join('./mhn_predict_12loss_sel_69epoch/test-025', current_preds_file)).astype(np.float32)
        gts = np.loadtxt(os.path.join('./gts-50k', current_preds_file)).astype(np.float32)
        preds = torch.FloatTensor(preds).unsqueeze(0).to(device)
        gts = torch.FloatTensor(gts).unsqueeze(0).to(device)
        [current_loss_cd, current_loss_p2s] = cd_loss(preds, gts)
        total_loss_cd = total_loss_cd+current_loss_cd
        total_loss_p2s = total_loss_p2s+current_loss_p2s
        count = count + 1

    total_loss_cd = total_loss_cd/(count*2)
    total_loss_p2s = total_loss_p2s/count
    print(total_loss_cd)
    print(total_loss_p2s)
