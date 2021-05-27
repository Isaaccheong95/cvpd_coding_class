import random
import os
from os.path import join
import numpy as np
import idx2numpy
from .cifar_utils import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data_augmentation import *

from icecream import ic
ic.configureOutput(includeContext=True)


class CIFARDataloader(Dataset):
    
    def __init__(self, args, augs=None, train=True):
        
        if train:
            self.train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                *augs,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        else:
            self.test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        self.args = args
        self.train = train
        self.root = args["dataset"]["cifar"]["root"]
        
        self.ims, self.lbls = load_CIFAR10(self.root, train=train)
        
        
    def __getitem__(self, idx):
        
        img = np.array(self.ims[idx])
        lbl = np.array(self.lbls[idx])
        
        # Perform data augmentation
        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)
        
        return img, lbl
        
        
    def __len__(self):

        return self.ims.shape[0]