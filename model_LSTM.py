import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from torch.legacy.nn import CMul


class ActNet(nn.Module):
    def __init__(self):
        super(ActNet, self).__init__()
        self.lstm = nn.LSTM(150, 100, 3, batch_first=True,dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(100, 60),nn.ELU())
   
    def forward(self, inputs):
        features,_ = self.lstm(inputs)
        out = self.linear(features[:,-1,:])
        return out  
    
class GenNet(nn.Module):
    def __init__(self,Num):
        super(GenNet, self).__init__() 
        
        self.Enlstm = nn.LSTM(150, Num, 2, batch_first=True,dropout=0.5)
        self.Delstm = nn.LSTM(Num, 150, 2, batch_first=True,dropout=0.5)
   
    def forward(self, inputs):
        encoder,_ = self.Enlstm(inputs)
        decoder,_ = self.Delstm(encoder) 
        return decoder
 
