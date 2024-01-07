#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 9:06:11 2021
train script
@author: pingping
"""
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from chamfer_distance import ChamferDistance
from DatasetUofSC_Mem import DatasetUofSC
from Models import MyNet
import configs
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def valid(with_structure=False):
    model.eval()
    total_loss = 0
    with tqdm(valid_data_loader) as t:
        for step, data in enumerate(t):
        
            data = [ele.to(device) for ele in data]
            #return gtcloud,gtnormal, ptcloud, ptnormal, patchindicator, patch_pos
            decoded, local_r, normal = model(data[1:])
            dist1, dist2,idx1, idx2 = criterion(decoded.reshape(1,-1,3), data[0].reshape(1,-1,3))
            if step==250-1:
                writer.add_mesh('decoded', vertices=decoded.reshape(1,-1,3))
                writer.add_mesh('local_r', vertices=local_r[0].reshape(1,-1,3))
                writer.add_mesh('initial', vertices=local_r[1].reshape(1,-1,3))
                writer.add_mesh('gt', vertices=data[0].reshape(1,-1,3))
            loss = ((torch.mean(dist1)) + (torch.mean(dist2)) )
            total_loss += 1e3* loss.item()# * data.num_graphs

    return total_loss / len(valid_dataset)
    
def train(with_structure=True):
    model.train()
    total_loss = 0
    total_global_loss = 0
    total_patch_loss=0
    with tqdm(train_data_loader) as t:
        for step, data in enumerate(t):

            data = [ele.to(device) for ele in data]
            #return gtcloud, ptcloud, patchindicator, patch_pos
            gt_rot = torch.eye(3).to(device)
            optimizer.zero_grad()
            decoded, local_r, pd_normal = model(data[1:])
            dist1, dist2,idx1, idx2 = criterion(decoded.reshape(1,-1,3), data[0].reshape(1,-1,3))
            dist3, dist4,idx3, idx4 = criterion(local_r[0].reshape(1,-1,3), data[0].reshape(1,-1,3))
            #dist5, dist6,idx5, idx6 = criterion(local_r[1].reshape(1,-1,3), data[0].reshape(1,-1,3))
            
            rot_loss = rot_criterion(gt_rot.repeat(pd_normal.size(0),1,1),pd_normal)
            global_loss = (1*torch.mean(dist1) + 1*torch.mean(dist2))
            patch_loss = torch.mean(dist3) +  torch.mean(dist4)
            loss = global_loss + 0.1*patch_loss + cfg.lambda_rot *rot_loss
            #+ 5*torch.mean(dist5) + 5* torch.mean(dist6)) + cfg.lambda_rot *rot_loss
                     #1*torch.mean(dist5) + 1* torch.mean(dist6)) #+0.0001*normal_loss
            loss.backward()
            total_loss += loss.item()# * data.num_graphs
            total_global_loss += global_loss.item()
            total_patch_loss += patch_loss.item()
            optimizer.step()
            writer.add_scalar('Loss/global_loss:', global_loss.item(), epoch)
            writer.add_scalar('Loss/patch_loss:', patch_loss.item(), epoch)
    return total_loss / len(train_dataset)

def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    
    cfg = configs.get_mynet_config()
    output_dir = os.path.join('./trained', datetime.now().isoformat())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    savefile = output_dir+'/Models.py'
    shutil.copyfile('./Models.py', savefile)
    shutil.copyfile('./configs.py', output_dir+'/configs.py')
    #tensorborad writer
    writer = SummaryWriter(comment="patchNum_4_lambda_0.001_in30")
                     
    #dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    train_dataset = DatasetUofSC(cfg.data.train_path)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    ## batch size must be 1
                                                    batch_size=1,
                                                    num_workers=1,
                                                    #collate_fn=collate_wrap,
                                                    pin_memory=True,
                                                    shuffle=False,
                                                    drop_last=True)
    valid_dataset = DatasetUofSC(cfg.data.test_path)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=1,
                                                    num_workers=1,
                                                    #collate_fn=collate_wrap,
                                                    pin_memory=True,
                                                    shuffle=False,
                                                    drop_last=True)  
    device = torch.device(cfg.gpu if torch.cuda.is_available() else 'cpu')
    model = MyNet(cfg).to(device)
    
    if cfg.load_model:
        model.load_state_dict(torch.load(cfg.ckpt_path,map_location=device))
    else:
        model.apply(init_weights)
    
    print(model)
    #writer.add_graph(model, input_to_model=None, verbose=False)
    print('Training started:')
    criterion = ChamferDistance()
    rot_criterion = torch.nn.MSELoss()

    lrate = cfg.learning_rate
    
    #writer.add_hparams({'lr': lrate, 'lambda_rot': cfg.lambda_rot})
    
    #SGD with Momentum
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(foldingnet.parameters(),lr = 0.0001,weight_decay=1e-6)
    best_loss = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10)
    
    for epoch in range(0, 10):
        loss = train()
        print('Training Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        #scheduler_steplr.step()
        writer.add_scalar('Loss/train', loss, epoch)
        if epoch % 5 == 0:
            loss = valid()
            print('Validate Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
            writer.add_scalar('Loss/test', loss, epoch)
        if epoch % 100 == 0:
            torch.save(model.state_dict(),'./trained/My_net_Ch'+'{}'.format(epoch) +'.pt')
        
    #optimizer = torch.optim.Adam(model.parameters(), lr=lrate/10)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10)
    for epoch in range(10, 501):
        print('epoch: ', epoch, 'optimizer: ', optimizer.param_groups[0]['lr'])
        loss = train()
        print('Training Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        #scheduler.step(loss)
        scheduler_steplr.step()
        writer.add_scalar('Loss/train', loss, epoch)
        if epoch % 5 == 0:
            loss = valid()
            if loss < best_loss:
                torch.save(model.state_dict(),output_dir+'/My_net_ChBest.pt')
            print('Validate Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
            writer.add_scalar('Loss/test', loss, epoch)
            
        if epoch % 100 == 0:
            torch.save(model.state_dict(),output_dir+'/My_net_Ch'+'{}'.format(epoch) +'.pt')
            
    writer.close()
