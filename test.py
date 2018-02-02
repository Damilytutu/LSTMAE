# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:43:23 2017

@author: Damily
"""


from __future__ import print_function, division
import torch
import torch.nn as nn
import scipy.io as scio
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from data import loadview_data, loadsubject_data
from model import ActNet,GenNet
import numpy as np

#######################################   
batch_size = 16
#train_data, train_label, test_data, test_label  = loadview_data()
train_data, train_label, test_data, test_label  = loadsubject_data()

dsets = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
dset_loaders =  DataLoader(dataset=dsets, num_workers=4,batch_size= batch_size, shuffle=False)
dset_sizes = len(dsets)
######################################################################
def test_model(model_ft, model_gen, criterion):
    model_ft.eval()
    running_loss = 0.0
    running_corrects = 0
    running_errors = 0
    cont = 0
    
    Deinputs_array_total = []
    De_class_array_total = []
    out_class_array_total = []
    # Iterate over data.
    for data in dset_loaders:
        # get the inputs
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        # forward
        Deinputs = model_gen(inputs)
        De_class = model_ft(Deinputs)
        out_class = model_ft(inputs)
        
        De_class_array = De_class.data.cpu().numpy()
        out_class_array = out_class.data.cpu().numpy()
        
        # the generated data from autoencoder
        #Deinputs_array = Deinputs.data.cpu().numpy()
        
        #if Deinputs_array_total == []:
        #    Deinputs_array_total = Deinputs_array
        #    print(Deinputs_array_total.shape)
        #else:
        #    Deinputs_array_total = np.concatenate((Deinputs_array_total, Deinputs_array), axis=0)
        #    print('********')
        #    print(Deinputs_array_total.shape)

             
        if De_class_array_total == []:
            De_class_array_total = np.vstack((De_class_array))
        else:
            De_class_array_total = np.vstack((De_class_array_total, De_class_array))
        
        if out_class_array_total == []:
            out_class_array_total = np.vstack((out_class_array))
        else:
            out_class_array_total = np.vstack((out_class_array_total, out_class_array))
        
        # three test methods
        #_, preds = torch.max(De_class.data, 1)
        #_, preds = torch.max(out_class.data, 1)
        _, preds = torch.max((De_class.data + out_class.data), 1)
        
        
        loss = criterion(De_class, labels)   
        if cont==0:
            outPre = De_class.data.cpu()
        else:
            outPre = torch.cat((outPre,De_class.data.cpu()),0)
        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
        running_errors += torch.sum(preds != labels.data)
        print('Num:',cont)
        cont +=1
    #print(Deinputs_array_total.shape)
    #scio.savemat('ende_100.mat', {'pred': Deinputs_array_total})

    print('Loss: {:.4f} Acc: {:.4f} Err: {:.4f}'.format(running_loss/dset_sizes,
                    running_corrects/(1.0*dset_sizes), running_errors/(1.0*dset_sizes)))
       
    return outPre
    
######################################################################
modelft_dir = 'model/subject/compress_100/' 
modelft_file1 = modelft_dir + 'ClassLSTM_epoch_499.pth'
modelft_file2 = modelft_dir + 'GenLSTM_epoch_499.pth'

model_ft = torch.load(modelft_file1).cuda()
model_gen = torch.load(modelft_file2).cuda()

criterion = nn.CrossEntropyLoss().cuda()
outPre = test_model(model_ft, model_gen, criterion)

    
    

