#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:07:17 2021
UofSC indoor point cloud Data set processing
@author: pingping
"""
import os
from torch_geometric.data import (Data, InMemoryDataset)

import torch
import open3d
import numpy as np

    
class DatasetUofSC(torch.utils.data.dataset.Dataset):
    
    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.scene_list = read_list(txt_path+'/datalist.txt')
        
    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        path = self.scene_list[idx]
        inpath = path[0].lstrip('..')
        gtpath = path[1].lstrip('..')
        # get all patches inside each scene folder
        
        partial_pcd_path = self.txt_path + inpath
        gt_pcd_path = self.txt_path + gtpath
        
        ptcloud = []
        ptnormal = []
        patchindcator = []
        patch_pos = []
        
        #load ground truth PCD
        pc = open3d.io.read_point_cloud(gt_pcd_path)
        gtpatch = np.array(pc.points).astype(np.float32)
        gtcloud = torch.from_numpy(gtpatch)

        gtnormal = np.array(pc.normals).astype(np.float32)
        gtnormal = torch.from_numpy(gtnormal)
        
        #load incomplete PCD
        pc = open3d.io.read_point_cloud(partial_pcd_path)

        # crop pcd into small patches and concate/stack
        #bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-30, 0, -10), max_bound=(10, 20, 10))
        patch_num = 4
        voxel_size = 1/patch_num
        patch_i = 0
        for idx in range(patch_num):
            for idy in range(patch_num):
                low_x = idx*voxel_size
                hig_x = (idx+1)*voxel_size
                low_y = idy*voxel_size
                hig_y = (idy+1)*voxel_size
                bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(low_x,low_y, 0), max_bound=(hig_x, hig_y, 1))
                patch_pcd = pc.crop(bbox)
                
                #check number of points in patch_pcd
                ptpatch = np.array(patch_pcd.points).astype(np.float32)
                if ptpatch.shape[0] <= 200:
                    continue
                
                # resample number of points to 1024
                index = np.random.choice(ptpatch.shape[0],1024)
                new_patch_pt = ptpatch[index,:]
                ptpatch = torch.from_numpy(new_patch_pt)
                patch_normal = np.array(patch_pcd.normals).astype(np.float32)
                patch_normal = torch.from_numpy(patch_normal[index,:])
                
                ptcloud.append(ptpatch)
                ptnormal.append(patch_normal)
                
                patch_pos.append([idx,idy])
                
                patch_vec = torch.zeros(ptpatch.size(0),dtype=torch.int64) + patch_i
                patch_i = patch_i +1
                patchindcator.append(patch_vec)
                
        patch_pos = torch.tensor(patch_pos,dtype=torch.long)
        ptcloud = torch.stack(ptcloud, dim=0)
        ptnormal = torch.stack(ptnormal, dim=0)
        patchindicator = torch.stack(patchindcator, dim=0)

        return gtcloud,gtnormal, ptcloud, ptnormal, patchindicator, patch_pos


def read_list(processed_path):
    '''
    token shape: 'incomplete' \t 'groundtruth'
    '''
    file_list = []
    with open(processed_path, 'r') as f:
        for line in f.readlines():
            token = line.strip('\n').split('\t')
            tt = token[0]
            file_list.append(token)
            #if tt==split:
            #    file_list.append(token[1:])
    return file_list
    
def collate_wrap(batch):
    '''
    costom collate wrapper
    '''
    data = []
    batch_index = 0
    for pt, gt,pn in batch:
        ## gen batch indicator for each points
        patch_vec = torch.zeros(pt.size(0),dtype=torch.int64) + batch_index
        data += [Data(pos=pt, y=gt, patch=patch_vec, normal = pn)]
        batch_index = batch_index + 1
    return InMemoryDataset.collate(data)

