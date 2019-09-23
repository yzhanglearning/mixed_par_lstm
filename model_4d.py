## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

import numpy as np
import scipy.misc as misc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t 


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1,10, kernel_size=3)
        self.conv2 = nn.Conv3d(10,20, kernel_size=3)
        self.conv3 = nn.Conv3d(20,40, kernel_size=2)
#        self.conv4 = nn.Conv3d(40,80, kernel_size=3)   
        self.drop = nn.Dropout3d()
        
        self.fc1 = nn.Linear(10*10*2*80, 1280)   # 4x4x4x80
        self.fc2 = nn.Linear(1280, 320)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=None, padding=0)
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)

    def forward(self,x):
        x = self.batchnorm(x)
        x = self.relu(self.pool(self.conv1(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv2(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv3(x)))
        x = self.drop(x)
#        x = self.relu(self.pool(self.conv4(x)))
#        x = self.drop(x)
        x = x.view(-1, 10*10*2*80)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        print(x.shape)
        return x



class CombineRNN(nn.Module):
    def __init__(self):
        super(CombineRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size = 320,
            hidden_size = 128,
            num_layers = 1,
            batch_first = True)
        self.linear1 = nn.Linear(128,64)
        self.linear2 = nn.Linear(64,1)

    def forward(self, x):
        batch_size, C, H, W, Z, time_steps = x.size()  # batch, channel, height, width, depth, time
#        print(batch_size, H, W, Z, time_steps)
        
        c_in = x.view(batch_size*time_steps, C, H, W, Z)  # batch_size*time_steps becomes dummy_batch
#        c_in = c_in.unsqueeze(1)
        c_out = self.cnn(c_in)
#        c_out = c_out.squeeze(1)
        r_in = c_out.view(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear1(r_out[:,-1,:])
        r_output = self.linear2(r_out2)

        return r_output








class ModelParallelCombineRNN(CombineRNN):
    def __init__(self, devices):
        super(ModelParallelCombineRNN, self).__init__()

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices

        self.batchnorm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        
        self.seq1 = nn.Sequential(
            self.batchnorm,
            self.cnn).to(self.devices[0])

        self.seq2 = nn.Sequential(
            self.rnn).to(self.devices[0])

        self.seq3 = nn.Sequential(
            self.linear1,
            self.linear2).to(self.devices[1])

    def forward(self, x):
        
        batch_size, C, H, W, Z, time_steps = x.size()  # batch, channel, height, width, depth, time
#        print(batch_size, H, W, Z, time_steps)
        
        c_in = x.view(batch_size*time_steps, C, H, W, Z)  # batch_size*time_steps becomes dummy_batch
#        c_in = c_in.unsqueeze(1)
        c_out = self.seq1(c_in)
#        c_out = c_out.squeeze(1)
        r_in = c_out.view(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
        r_out, (h_n, h_c) = self.seq2(r_in)
        r_out = r_out.to(devices[1])
        #r_out2 = self.linear1(r_out[:,-1,:])
        #r_output = self.linear2(r_out2)
        r_output = self.seq3(r_out[:,-1,:])
                                   
        return r_output








        
