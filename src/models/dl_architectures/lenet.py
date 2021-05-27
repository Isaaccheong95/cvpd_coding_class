import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic
ic.configureOutput(includeContext=True)



class Lenet(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        
        num_channels = kwargs["num_channels"]
        num_cls = kwargs["num_cls"]
        
        if "fc_size" in kwargs:
            fc_size = kwargs["fc_size"]
        else:
            fc_size = 256
        
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(fc_size, 120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84, num_cls)
        self.relu5 = nn.ReLU()

    def forward(self, x):
                
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.relu4(x)
        
        x = self.fc3(x)
        x = self.relu5(x)
                
        return x
