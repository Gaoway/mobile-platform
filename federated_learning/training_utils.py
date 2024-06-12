import torch.nn as nn
import torch
import math
from datasets import DataLoaderHelper
from typing import Tuple

def train(model:nn.Module, 
          data_loader:DataLoaderHelper, 
          optimizer:torch.optim.SGD, 
          device:torch.device, 
          local_iters:int=None) -> Tuple[float, int, int]:

    model.train()
    model.to(device)

    # 如果未指明本地更新的次数，那就默认把所有的本地数据过一遍
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)

    train_loss = 0.0
    correct_num = 0
    samples_num = 0
    loss_func = nn.CrossEntropyLoss()

    for _ in range(local_iters):
        data : torch.Tensor
        target : torch.Tensor
        output : torch.Tensor
        loss : torch. Tensor

        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        output = model(data)

        optimizer.zero_grad()

        loss = loss_func(output, target.long())
        pred = output.argmax(1, keepdim=True)
        correct_num += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item() * data.size(0)
        samples_num += data.size(0)

        loss.backward()
        optimizer.step()
        
        data, target, output, loss, pred = None, None, None, None, None

    if samples_num != 0:
        train_loss /= samples_num
    
    model.to('cpu')

    return train_loss, correct_num, samples_num

def test(model:nn.Module, 
         data_loader:DataLoaderHelper, 
         device:torch.device) -> Tuple[float, int, int]:

    model.eval()
    model.to(device)
    data_loader = data_loader.loader
    test_loss = 0.0
    correct_num = 0
    samples_num = 0
    loss_func = nn.CrossEntropyLoss() 

    with torch.no_grad():
        for data, target in data_loader:

            data : torch.Tensor
            target : torch.Tensor
            output : torch.Tensor
            loss : torch. Tensor

            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_func(output, target.long())
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(1, keepdim=True)
            correct_num += pred.eq(target.view_as(pred)).sum().item()
            samples_num += data.size(0)

            data, target, output, loss, pred = None, None, None, None, None

    if samples_num != 0:        
        test_loss /= samples_num

    model.to('cpu')

    return test_loss, correct_num, samples_num