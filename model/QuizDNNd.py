# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 07:18:07 2020

@author: divva
"""

import torch.nn as nn


class QuizDNN(nn.Module):

    def __init__(self):
        """ This function instantiates all the model layers """

        super(QuizDNN, self).__init__()

        # Input Block
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # output_size = 222

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) # output_size = 222
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )#outputsize = 220

        # POOLING BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 110

        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ) # output_size = 108

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ) # output_size = 8
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # output_size = 6
        
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) # output_size = 6
        
        #x8 = self.pool1(x5+x6+x7)
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ) # output_size = 6
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # output_size = 6

        # OUTPUT BLOCK
        #avgpool
        self.gap = nn.Sequential(nn.AvgPool2d(1))

        self.x13 = nn.Sequential(
            nn.Linear(16, 10)
        )


    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x1+x2)
        x4 = self.pool1(x1+x2+x3)
        
        x5 = self.convblock4(x4)
        x6 = self.convblock5(x4+x5)
        x7 = self.convblock6(x4+x5+x6)
        
        x8 = self.pool1(x5+x6+x7)
        x9 = self.convblock7(x8)
        x10 = self.convblock8(x8+x9)
        x11 = self.convblock9(x8+x9+x10)
        x12 = self.gap(x11)
        x12 = self.view(-1,16)
        x13 = self.x13(x12)
        
        return x13