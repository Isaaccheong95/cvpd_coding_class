import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from icecream import ic

ic.configureOutput(includeContext=True)


class v_flip:
    """
    Randomly performs vertical flipping on the image.

    """

    def __init__(self, prob_flip=0.5):

        self.prob_flip = prob_flip

    def __call__(self, sample):

        if random.random() < self.prob_flip:
            #             ic("v flip")
            sample = TF.vflip(sample)

        return sample


class h_flip:
    """
    Randomly performs horizontal flipping on the image.
    """

    def __init__(self, prob_flip=0.5):

        self.prob_flip = prob_flip

    def __call__(self, sample):

        if random.random() < self.prob_flip:
            #             ic("horizontal flip")
            sample = TF.hflip(sample)

        return sample