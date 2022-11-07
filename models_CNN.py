import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,input_channels, num_classes, density_level,density_level_fc, density_level_last_layer, selected_nodes_count, num_conv, l_reuse, benchmark):
        super(CNN, self).__init__()
        #model param
        self.l_reuse = l_reuse
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.num_conv = num_conv
        self.kernal_size = 3
        self.conv2_W = 6 
        self.conv2_H = 6
        self.featuremap_layers_size=[input_channels, 64, 128, 256, 2048, 2048, self.num_classes] 
        drop1_value = 0.2
        drop2_value = 0.2
        # BN is used for sim-CIFAR100 benchmark instead of dropout
        if benchmark == 'CIFAR100':
            drop1_value = 0.0
            drop2_value = 0.0
        self.drop1 = nn.Dropout(drop1_value)
        self.drop2 = nn.Dropout(drop2_value)

        #model arch
        self.feature_1 = nn.Sequential(
            nn.Conv2d(self.featuremap_layers_size[0], self.featuremap_layers_size[1], kernel_size=self.kernal_size, stride=1,bias=True),
            nn.ReLU(inplace=True),
        )
        self.feature_2 = nn.Sequential(
            nn.Conv2d(self.featuremap_layers_size[1], self.featuremap_layers_size[2], kernel_size=self.kernal_size, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )

        if benchmark == 'CIFAR100':
            self.maxpool_BN_2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(self.featuremap_layers_size[2]),
            )
        else:
            self.maxpool_BN_2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.feature_3 = nn.Sequential(
            nn.Conv2d(self.featuremap_layers_size[2], self.featuremap_layers_size[3], kernel_size=self.kernal_size, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        if benchmark == 'CIFAR100':
            self.maxpool_BN_4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(self.featuremap_layers_size[3]),            
            )
        else:
            self.maxpool_BN_4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.linear_layer_1 = nn.Sequential(
            nn.Linear(self.conv2_W*self.conv2_H*self.featuremap_layers_size[3], self.featuremap_layers_size[4]),
            nn.ReLU(inplace=True),
        )
        self.linear_layer_2 = nn.Sequential(
            nn.Linear(self.featuremap_layers_size[4], self.featuremap_layers_size[5]),
            nn.ReLU(inplace=True),
        )
        self.classifier =  nn.Linear(self.featuremap_layers_size[5], self.num_classes)

        self.layers_names = []
        for name, param in self.named_parameters():
            if self.take_layer(name,param):
                self.layers_names.append(name)
        self.layers_names.append(name)

        #### sparsity level 
        self.density_level = density_level
        self.density_level_fc = density_level_fc
        self.density_level_last_layer = density_level_last_layer
        self.noParams = []
        for i in range(len(selected_nodes_count)-1):
            if i < len(selected_nodes_count)-3:
                print("alloc_nodes_count[i]",selected_nodes_count[i])
                print("alloc_nodes_count[i+1]",selected_nodes_count[i+1])
                self.noParams.append(int(self.density_level*selected_nodes_count[i]*selected_nodes_count[i+1]))
            elif i == len(selected_nodes_count)-3:
                print("alloc_nodes_count[i]",selected_nodes_count[i])
                print("alloc_nodes_count[i+1]",selected_nodes_count[i+1])
                self.noParams.append(int(self.density_level_fc*selected_nodes_count[i]*selected_nodes_count[i+1]))
            else:
                self.noParams.append(int(self.density_level_last_layer*selected_nodes_count[i]*selected_nodes_count[i+1]))
        print("param for task 0", self.noParams)

    def forward(self, x):
        f_x1 = self.feature_1(x)
        f_x2 = self.feature_2(f_x1)      
        M_x2 = self.maxpool_BN_2(f_x2)
        f_x3 = self.feature_3(M_x2)
        f_x4 = self.drop1(f_x3)
        M_x4 = self.maxpool_BN_4(f_x4)
        flatten = M_x4.view(M_x4.size(0), -1)
        L_x1 = self.linear_layer_1(flatten)
        L_x2 = self.linear_layer_2(self.drop2(L_x1))
        x = self.classifier(self.drop2(L_x2))
        return x, f_x1, f_x2, f_x4, L_x1, L_x2    

    def take_layer(self, name,param):
        if len(param.shape)>1:
            return True
        else:
            return False

    def last_layer(self,name):
        if ((name in self.layers_names[-1]) or (name in self.layers_names[-2])):
            return True
        else:
            return False
    def flatten_layer(self,name):
        if("linear_layer_1.0" in name):
            return True
        return False