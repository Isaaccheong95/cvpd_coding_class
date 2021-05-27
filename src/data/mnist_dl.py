import random
import os
from os.path import join
import numpy as np
import idx2numpy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data_augmentation import *

from icecream import ic
ic.configureOutput(includeContext=True)


class MNISTDataloader(Dataset):
    
    def __init__(self, args, augs=None, train=True):
                
        if train:
            self.train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                *augs,
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            self.test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.args = args
        self.train = train
        self.paths = args["dataset"][args["dataset"]["name"]]
                
        if train:
            self.images_path = join(self.paths["train_set_path"]["root"], self.paths["train_set_path"]["images"])
            self.labels_path = join(self.paths["train_set_path"]["root"], self.paths["train_set_path"]["labels"])
        else:
            self.images_path = join(self.paths["test_set_path"]["root"], self.paths["test_set_path"]["images"])
            self.labels_path = join(self.paths["test_set_path"]["root"], self.paths["test_set_path"]["labels"])
        
        self.ims = idx2numpy.convert_from_file(self.images_path)
        self.lbls = idx2numpy.convert_from_file(self.labels_path)
        
        
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
        