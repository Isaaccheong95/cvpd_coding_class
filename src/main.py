import argparse
import json
from collections import namedtuple
import random
import numpy as np
from icecream import ic
import os
from os.path import join
import ruamel.yaml as yaml
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.optim as optim
from torch import optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import *

sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), "../"))

# Import dataloaders
from data import *

# Import data augmentation functions
from data.data_augmentation import *

# Import DL models
from models.dl_architectures import *

# Import training codes
from models.training_regimes import *

# Import validation codes
from models.validation import *

ic.configureOutput(includeContext=True)


def set_seed(seed):
    """
    Sets the random seed for PRNGs for all the libraries being used in this repo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Command line argument for reading in the path to the configuration file being used for model training.
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration file being used for model training.",
    )
    args_config_path = parser.parse_args()

    # Read in the contents of the config file.
    with open(args_config_path.config_path) as f:
        args = yaml.load(f, Loader=yaml.Loader)

    # Set the random seeds.
    set_seed(args["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = {"mnist": MNISTDataloader, "cifar": CIFARDataloader}[args["dataset"]["name"]]
    model = {"lenet": Lenet, "resnet18": ResNet18}[args["architecture"]["name"]]
    train = {"lenet": train_lenet, "resnet18": train_resnet}[
        args["architecture"]["name"]
    ]
    test = {"lenet": validate_lenet, "resnet18": validate_resnet}[
        args["architecture"]["name"]
    ]
    opt = {"Adam": optim.Adam, "SGD": optim.SGD}[args["optimizer"]["name"]]
    lr_sched = {"StepLR": StepLR, "ExponentialLR": ExponentialLR}[
        args["lr_scheduler"]["name"]
    ]

    augs_dict = {"h_flip": h_flip(), "v_flip": v_flip()}
    augs = [augs_dict.get(key) for key in args["dataset"]["aug"]]

    train_set = dl(
        args=args,
        augs=augs,
    )

    test_set = dl(
        args=args,
        train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args["batch_size"], num_workers=1, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args["test_batch_size"],
        num_workers=1,
        shuffle=False,
    )

    model_args = {}
    model_args["num_cls"] = args["dataset"][args["dataset"]["name"]]["num_cls"]
    model_args["num_channels"] = args["dataset"][args["dataset"]["name"]][
        "num_channels"
    ]

    if args["architecture"]["name"] in args["architecture"]:
        model_args.update(
            args["architecture"][args["architecture"]["name"]][args["dataset"]["name"]]
        )

    net = model(**model_args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = opt(
        net.parameters(), lr=args["lr"], **args["optimizer"][args["optimizer"]["name"]]
    )
    scheduler = lr_sched(
        optimizer, **args["lr_scheduler"][args["lr_scheduler"]["name"]]
    )

    print("Starting training...")

    # Model training begins here
    for epoch in range(1, args["epochs"] + 1):
        print(f'Epoch {epoch} out of {args["epochs"]}...')

        # Train for 1 epoch
        train(args, net, device, train_loader, optimizer, criterion, epoch)
        # Validate model
        test(net, device, test_loader, criterion)

        # Update LR
        scheduler.step()

    # Save model weights
    torch.save(
        net.state_dict(), f'{args["dataset"]["name"]}_{args["architecture"]["name"]}.pt'
    )


if __name__ == "__main__":
    main()