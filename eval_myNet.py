#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:50:15 2021
eval script 
@author: pingping
"""

import torch

import matplotlib.pylab  as plt
from scipy.io import savemat, loadmat
from chamfer_distance import ChamferDistance
from DatasetUofSC_Mem import DatasetUofSC,collate_wrap
from Models import MyNet
import configs
import numpy as np

def save_pcd(path,pred_pcd):
    '''
    pred_pcd: N by 3
    '''
    np.savetxt(path + '.xyz', pred_pcd, fmt='%.6f')
#%matplotlib inline
    
def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=0.8, figsize=(15, 7), elev=20, azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig

cfg = configs.get_mynet_config()
valid_dataset = DatasetUofSC(cfg.data.test_path)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=1,
                                                num_workers=1,
                                                #collate_fn=collate_wrap,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True)  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNet(cfg).to(device)
model.load_state_dict(torch.load(cfg.ckpt_path,map_location=device))
model.eval()

criterion = ChamferDistance()
total_chd = []
each_loss = []
for i,data in enumerate(valid_data_loader):
    data = [ele.to(device) for ele in data]
    #gtcloud, ptcloud, patchindicator, patch_pos
            
    with torch.no_grad():
        decoded, local_r, normal = model(data[1:])
        #pred_result, patch_fold, pd_normal = model(data[2:])
        idx = torch.randint((decoded.size(1)),(20000,))
        pred = decoded[0,idx,:]
        dist1, dist2, idx1, idx2 = criterion(pred.reshape(1,-1,3), data[0].reshape(1,-1,3))
        loss_pred = (torch.mean(dist1)) + (torch.mean(dist2)) 
        dist3, dist4, idx1, idx2 = criterion(data[1].reshape(1,-1,3), data[0].reshape(1,-1,3))
        loss_incomp = (torch.mean(dist3)) + (torch.mean(dist4)) 
        total_chd.append(loss_pred.cpu().numpy())
        each_loss.append(loss_pred.item())
        gen_data = {
        'pred': decoded.cpu().numpy().reshape((-1,3)),
        'gtcomplete': data[0].cpu().numpy().reshape((-1,3)),
        
        'incomplete': data[1].cpu().numpy().reshape((-1,3)),
        
        'cd_loss_pred': loss_pred.cpu().numpy(),
        'cd_loss_incomp': loss_incomp.cpu().numpy(),

        }

        savemat("output/initial_result"+str(i)+".mat", gen_data)
        save_pcd("result/pred"+str(i)+".xyz",decoded.cpu().numpy().reshape((-1,3)))
        save_pcd("incomplete/in"+str(i)+".xyz",data[1].cpu().numpy().reshape((-1,3)))
        save_pcd("gt/gt"+str(i)+".xyz",data[0].cpu().numpy().reshape((-1,3)))
        print(i)
        #break
save_pcd("result/chd_loss.txt",np.array(each_loss))
print(np.mean(total_chd))
#        if i > 10:
#            exit(0)



#savemat("initial_result.mat", gen_data)
