#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:32:15 2021
My model
@author: pingping
"""

import torch
import random
from torch.nn import Sequential as Seq, ReLU, GELU, Tanh, BatchNorm1d as BN
from torch.nn import Dropout, Softmax, Linear, LayerNorm
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
from ViT import Transformer

#u,a batch*3
#out batch*3
def proj_u_a(u,a):
    batch=u.shape[0]
    top = u[:,0]*a[:,0] + u[:,1]*a[:,1]+u[:,2]*a[:,2]
    bottom = u[:,0]*u[:,0] + u[:,1]*u[:,1]+u[:,2]*u[:,2]
    bottom = torch.max(torch.autograd.Variable(torch.zeros(batch).cuda())+1e-7, bottom)
    factor = (top/bottom).view(batch,1).expand(batch,3)
    out = factor* u
    return out

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-7]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

#matrices batch*3*3
def compute_rotation_matrix_from_matrix(matrices):
    b = matrices.shape[0]
    a1 = matrices[:,:,0]#batch*3
    a2 = matrices[:,:,1]
    a3 = matrices[:,:,2]
    
    u1 = a1
    u2 = a2 - proj_u_a(u1,a2)
    u3 = a3 - proj_u_a(u1,a3) - proj_u_a(u2,a3)
    
    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)
    
    rmat = torch.cat((e1.view(b, 3,1), e2.view(b,3,1),e3.view(b,3,1)), 2)
    
    return rmat

def MLP(channels):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])
    
    
class PointNetPP(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(PointNetPP, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch, num_samples=32):
        idx = fps(pos, batch, ratio=self.ratio)
        
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalPool(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalPool, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)

        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch



class PlanePriorNet(torch.nn.Module):
    '''
    Predict Plane Coefficient and 
    Generate 3D initial Points for Folding and
    Generate Transformed feature for Encoder
    '''
    # create voxel grids
    voxgrid = [[-0.2, 0.2, 4], [-0.2, 0.2, 4]]
    x = np.linspace(*voxgrid[0])
    y = np.linspace(*voxgrid[1])
    xy = np.meshgrid(x, y)
    points = torch.tensor(np.array(xy), dtype=torch.float32).to('cuda')
    points = points.view(2,-1).T
    supp = torch.cuda.FloatTensor(points.size(0), 1).fill_(0).to('cuda')
    new_points = torch.cat((points,supp),1)
    
    def __init__(self):
        super(PlanePriorNet, self).__init__()
        self.shift_mlp = MLP([768+3,256,128,3])
        #self.point_shift = Seq(Linear(256, 64), Linear(64,  3), Tanh())
        self.mlp = MLP([768+128,512,256,3*3])
        self.tanh = torch.nn.Tanh()
        self.conv = PointConv(MLP([3+3,64,128]))
#        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, pos, batch, diff, local_fea, num_points = 512):
        '''
        x: features, N by 768
        pos : point positon (x,y,z), N by 3
        normal : point normal (a,b,c)
        batch : 1D batch indicator for x
        num_points : number of points to generate for folding module
        num_samples : max num of neighbors for radius
        '''
        point_num = 64  # 2048*fpx_ratio1*fpx_ratio2*
        patch_num = torch.max(batch).item()+1
        # local_fea : patch by parent_pos.size(1) by 1 by 512
        local_fea = torch.repeat_interleave(local_fea.view(patch_num, 768), point_num, dim=0)       
        
        n_pos = self.tanh(self.shift_mlp(torch.cat((local_fea, pos), -1)))
        n_pos = n_pos + pos
        pos_c = torch.cat((pos,n_pos),dim=0)
        batch_c = torch.cat((batch,batch),dim=0)
        row, col = radius(pos_c, pos_c, 0.3, batch_c, batch_c,
                          max_num_neighbors=9)
        edge_index = torch.stack([col, row], dim=0)
        pos_fea = self.conv(pos_c, (pos_c, pos_c), edge_index)
        # predict rotation
        local_fea = local_fea.repeat(2,1)
        rot_fea = torch.cat((local_fea, pos_fea), -1)
        normal_c = self.mlp(rot_fea)
        rot_matrix = normal_c.view(patch_num*point_num*2,3,3)
        #rot_matrix = compute_rotation_matrix_from_matrix(rot_matrix)
        p = self.new_points.repeat(patch_num*point_num*2,1)
        #Rotate
        rot_constrain = torch.bmm(rot_matrix.transpose(2,1),rot_matrix)
        rot_matrix = torch.repeat_interleave(rot_matrix, 16, dim=0)
        delta_rot_xyz = torch.bmm(rot_matrix,p.view(-1,3,1))
        
        #Transform
        center_pos = pos_c.view(patch_num*point_num*2,3,1).repeat(16,1,1)
        p_rotate = delta_rot_xyz + center_pos
        
        plane_init = p_rotate.view(patch_num,-1,3)
        
        #plane_init = torch.cat()
        # initial pos shift
        #pos_c = pos.view(patch_num*point_num,3)
        
        return plane_init, rot_constrain
    


class PointExtractor(torch.nn.Module):
    '''
    Feature extractor in Vision Transformer
    '''
    def __init__(self,dim=768):
        super(PointExtractor,self).__init__()
        # pointNet++ module  
        self.pn1_module = PointNetPP(0.25, 0.15, MLP([3 + 3, 64, 128]))
        self.pn2_module = PointNetPP(0.25, 0.3, MLP([128 + 3, 256, 384]))
        #self.pn3_module = PointNetPP(0.5, 0.4, MLP([384 + 3, 512]))
        #self.pn3_module = PointNetPP(0.5,0.3, MLP([256 + 3, 256, 128]))
        # patch feature module
        self.gn1_module = GlobalPool(MLP([384 + 3, 512, dim]))
        
    def forward(self, pos, pi):
        '''
        inputs : ptcloud, ptnormal, patch_indicator
        pos: batch(1) by patch by N by 3
        '''
        ## scale each patch
        #print(pos.size())
        pos = pos.squeeze()
        v_min,i = torch.min(pos,1)
        v_max,i = torch.max(pos,1)
        # scale xyz for each patch
        diff,i = torch.max(v_max - v_min,1)
        vv_min = v_min.view(pos.size(0),1,pos.size(2)).repeat(1,pos.size(1),1)
        v_diff = diff.view(pos.size(0),1,1).repeat(1,pos.size(1),pos.size(2))
        
        pos = (pos-vv_min)/v_diff
        scale = [v_min, diff]
        #flatten input
        pos = pos.view(-1,3)
        
        pi = pi.view(-1)

        pn1_out = self.pn1_module(pos, pos, pi)
        pn2_out = self.pn2_module(*pn1_out)
        #pn3_out = self.pn3_module(*pn2_out)
        gn1_out = self.gn1_module(*pn2_out)
        #pn3_out = self.pn3_module(*pn2_out)
        return gn1_out, pn2_out, scale


class MyNet(torch.nn.Module):
    '''
    Main Structure for MyNet
    '''
    def __init__(self,config):
        super(MyNet, self).__init__()
        self.transformer = Transformer(PointExtractor(dim=config.hidden_size))
        # plane Feature
        self.plane_point_generator = PlanePriorNet()
        self.mlp_patch = Seq(Linear(768+3, 512), Linear(512,  256), ReLU())
        self.lin_patch = Seq(Linear(256, 64), Linear(64,  3), Tanh())
        self.mlp_global = Seq(Linear(768+3, 512), Linear(512,  256), ReLU())
        self.lin_global = Seq(Linear(256, 64), Linear(64,  3), Tanh())
        self.dropout = Dropout(0.7)
        self.l_norm = LayerNorm(256, eps=1e-6)
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def patch_shifting(self, pt,patch_encoded,global_fea):
        '''
        patch_encoded: local_feature for each patch
            size: batch(1) by patch by 512
        global_encoded: global_feature for each patch
            size: batch(1) by patch by 512
        
        # return
            xyz: patch completed
            pt*v_diff + vv_min  : init points
        '''
        # local_fea: batch by 512
        local_fea = patch_encoded[0][0]
        patch_num = local_fea.size(0)
        [v_min, diff] = patch_encoded[2]
        #print(patch_encoded[1].size())
        pt = pt.view(patch_num,-1,3)
        local_fea = local_fea.view(patch_num, 1, local_fea.size(-1)).repeat(1,pt.size(1), 1)
        displacement_fea = torch.cat((local_fea, pt), -1)
        
        out = self.mlp_patch(displacement_fea)
        out = self.l_norm(out)
        out = self.lin_patch(out)

        out = out+pt
        # rescale xyz for each patch
        v_diff = diff.view(out.size(0),1,1).repeat(1,out.size(1),3)
        vv_min = v_min.view(out.size(0),1,3).repeat(1,out.size(1),1)
        xyz = out*v_diff + vv_min
        #torch.sigmoid(out)
        pt = pt.view(patch_num,-1,3)
        
        return xyz, pt*v_diff + vv_min
    
    def global_shifting(self, global_encoded, xyz):
        '''
        global_encoded: global_feature
            size: batch by patch by 512

        '''
        #print(global_encoded.size())
        global_fea = torch.max(global_encoded,dim=1)[0]
        #print(global_fea.size())
        batch_num = 1
        # pt : initial folding points from patch folding: batch(1) by patch * N by 1 by 3
        pt = xyz.view(1, -1,3)
        # global_fea : batch by patch * N  by 512
        global_fea = global_fea.repeat(1, pt.size(0)*pt.size(1), 1)
        out = torch.cat((global_fea,pt), -1)
        # point shifting
        out = self.mlp_global(out)
        out = self.l_norm(out)
        out = self.lin_global(out)
        pos = pt + out
        return pos
    
    def forward(self, data):
        
        # Vision Transformer
        patch_fea, global_fea, attn_weights = self.transformer(data)
        
        # plane point generator
        # pt : initial folding points : batch by N by 3
        local_fea = patch_fea[0][0]
        patch_num = local_fea.size(0)
        [v_min, diff] = patch_fea[2]
        pt,pred_normal = self.plane_point_generator(*patch_fea[1],diff,local_fea)
        partial = data[0].view(-1,3)
        
        # PATCH FOLDING
        patch_fold = self.patch_shifting(pt,patch_fea,global_fea)
        patch_rec = patch_fold[0].view(-1,3)
        #print(patch_rec.size())
        #print(partial.size())
        #merged = torch.cat([partial,patch_rec],0)
        #num_points = merged.size(0)
#        indice = random.sample(range(num_points), 20000)
#        indice = torch.tensor(indice)
#        pos = merged[indice]
        #idx = fps(merged,ratio=0.5)
        #pos = merged[idx]
        # GLOBAL FOLDING
        global_fold = self.global_shifting(global_fea, patch_rec)
        return global_fold, patch_fold, pred_normal
