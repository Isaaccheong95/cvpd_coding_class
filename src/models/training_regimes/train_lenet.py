import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic
ic.configureOutput(includeContext=True)


def train_lenet(args, model, device, train_loader, optimizer, criterion, epoch):
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        data = data.to(device)
        target = target.type(torch.LongTensor).to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args["log_interval"] == 0:
            
            print('Train Epoch: {}\t {:.0f}%\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
            
            if args["dry_run"]:
                break