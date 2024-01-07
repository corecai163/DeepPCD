#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:07:17 2021
UofSC indoor point cloud Data set processing
@author: pingping
"""
import os
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)

import torch
import open3d
import numpy as np

from dataclasses import dataclass

    
class DatasetUofSC(torch.utils.data.dataset.Dataset):
    
    def __init__(self, txt_path):
        self.scene_list = read_list(txt_path+'/datalist.txt')
        self.data_list = get_all_item(self.scene_list,txt_path)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):    
        data = self.data_list[idx]
        gtcloud = data.y
        #gtnormal = data.gt_normal
        ptcloud = data.pos
        #ptnormal = data.pt_normal
        patchindicator = data.pi
        patch_pos = data.patch_pos
        return gtcloud, ptcloud, patchindicator, patch_pos

def read_list(processed_path):
    '''
    token shape: 'incomplete_pcd' \t 'gt_pcd'
    '''
    file_list = []
    with open(processed_path, 'r') as f:
        count = 0
        for line in f.readlines():
#            count = count + 1
#            if count > 10:
#               break
            token = line.strip('\n').split('\t')
            #tt = token[0]
            file_list.append(token)
            #if tt==split:
            #    file_list.append(token[1:])
    return file_list

def get_all_item(scene_list,txt_path):
    '''
    '''
    data = []
    for idx in range(len(scene_list)):
        path = scene_list[idx]
        inpath = path[0].lstrip('..')
        gtpath = path[1].lstrip('..')
        # get all patches inside each scene folder
        
        partial_pcd_path = txt_path + inpath
        gt_pcd_path = txt_path + gtpath
        
        ptcloud = []
        #ptnormal = []
        patchindcator = []
        patch_pos = []
        
        #load ground truth PCD
        pc = open3d.io.read_point_cloud(gt_pcd_path)
        #down sample
        gtpatch = np.array(pc.points).astype(np.float32)
        index = np.random.choice(gtpatch.shape[0],20000)
        new_patch_gt = gtpatch[index,:]
        
        #gtcloud = torch.from_numpy(new_patch_gt)
        gtcloud = torch.from_numpy(gtpatch)

        #gtnormal = np.array(pc.normals).astype(np.float32)
        #gtnormal = torch.from_numpy(gtnormal)
        
        #load incomplete PCD
        pcd = open3d.io.read_point_cloud(partial_pcd_path)
        # down sample
        in_pcd = np.array(pcd.points).astype(np.float32)
        index = np.random.choice(in_pcd.shape[0],20000)
        new_in = in_pcd[index,:]
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(new_in)

        # crop pcd into small patches and concate/stack
        #bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-30, 0, -10), max_bound=(10, 20, 10))
        patch_num = 5
        padding = 0.01/patch_num
        voxel_size = 1/patch_num
        patch_i = 0
        for idx in range(patch_num):
            for idy in range(patch_num):
                low_x = idx*voxel_size -padding
                hig_x = (idx+1)*voxel_size +padding
                low_y = idy*voxel_size -padding
                hig_y = (idy+1)*voxel_size +padding
                bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(low_x,low_y, 0), max_bound=(hig_x, hig_y, 1))
                patch_pcd = pc.crop(bbox)
                
                #check number of points in patch_pcd
                ptpatch = np.array(patch_pcd.points).astype(np.float32)
                if ptpatch.shape[0] <= 100:
                    continue
                
                # resample number of points to 1024
                index = np.random.choice(ptpatch.shape[0],1024)
                new_patch_pt = ptpatch[index,:]
                ptpatch = torch.from_numpy(new_patch_pt)
                #patch_normal = np.array(patch_pcd.normals).astype(np.float32)
                #patch_normal = torch.from_numpy(patch_normal[index,:])
                
                ptcloud.append(ptpatch)
                #ptnormal.append(patch_normal)
                
                patch_pos.append([idx,idy])
                
                patch_vec = torch.zeros(ptpatch.size(0),dtype=torch.int64) + patch_i
                patch_i = patch_i +1
                patchindcator.append(patch_vec)
                
        patch_pos = torch.tensor(patch_pos,dtype=torch.long)
        ptcloud = torch.stack(ptcloud, dim=0)
        #ptnormal = torch.stack(ptnormal, dim=0)
        patchindicator = torch.stack(patchindcator, dim=0)
        data.append(Data(pos=ptcloud, y=gtcloud, pi=patchindicator, 
                      pt_normal=None, gt_normal=None, patch_pos=patch_pos))
    return data
    
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

