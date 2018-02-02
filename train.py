#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:41:03 2017

@author: Damily
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from data import loadview_data, loadsubject_data
from model_LSTM import ActNet,GenNet
import time
from torch.optim.lr_scheduler import StepLR
import os
####################################### 
import argparse
parser = argparse.ArgumentParser(description="Compress Num")
parser.add_argument('--Num',type=int,default=75)
args = parser.parse_args()
Num = args.Num
dir_file = 'model/subject/compress_{}'.format(Num)
if not(os.path.exists(dir_file)):
   os.mkdir(dir_file)
######################################
print("*********")
batch_size = 256
#train_data, train_label, test_data, test_label  = loadview_data()
train_data, train_label, test_data, test_label  = loadsubject_data()
dsets = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
######################################################################
allloss = []
def train_model(model_ft, model_gen, criterion, MSEdis, optimizer, scheduler, num_epochs=60):
    since = time.time()
    model_ft.train(True) 
    model_gen.train(True) 
    dset_sizes = len(dsets)
    for epoch in range(num_epochs):
        print('Data Size',dset_sizes)           
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dset_loaders =  DataLoader(dataset=dsets, num_workers=4,batch_size= batch_size, shuffle=True)
        # Each epoch has a training and validation phase
        # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        count = 0
        scheduler.step()
        # Iterate over data.
        for data in dset_loaders:
            # get the inputs
            inputs, labels = data
            labels,inputs = Variable(labels.cuda()),Variable(inputs.cuda())  
            # zero the parameter gradients
            optimizer.zero_grad()           
            #train 
            Deinputs  = model_gen(inputs)
            De_class  = model_ft(Deinputs)
            out_class = model_ft(inputs)
            
            loss_De_class = criterion(De_class,labels)
            
            loss_out_class = criterion(out_class,labels)
            
            loss_mse = MSEdis(Deinputs, inputs)
            
            loss  = loss_De_class + loss_out_class + loss_mse
            
            loss.backward()
            optimizer.step() 
            
            _, preds = torch.max(De_class.data, 1) 
            # backward + optimize only if in training phase
            count +=1
            if count%10==0 or inputs.size()[0]<batch_size:
                print('Epoch:{}:loss_De_class:{:.3f} loss_out_class:{:.3f}'.format(epoch,loss_De_class.data[0], loss_out_class.data[0])) 
                allloss.append(loss.data[0]) 
				
            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects / (dset_sizes*1.0)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        if (epoch+1)%50==0 :
            model_out_path = dir_file + "/ClassLSTM_epoch_{}.pth".format(epoch)
            torch.save(model_ft, model_out_path)
            model_out_path = dir_file + "/GenLSTM_epoch_{}.pth".format(epoch)
            torch.save(model_gen, model_out_path)
                
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return allloss
######################################################################
model_ft = ActNet().cuda()
model_gen = GenNet(Num).cuda()

criterion = nn.CrossEntropyLoss().cuda()
MSEdis = nn.MSELoss().cuda()
optimizer = optim.Adam(list(model_ft.parameters())+list(model_gen.parameters()),lr=0.001)
scheduler = StepLR(optimizer,step_size=400,gamma=0.1)
######################################################################
# Train
allloss = train_model(model_ft, model_gen, criterion, optimizer,scheduler,num_epochs=500)
######################################
