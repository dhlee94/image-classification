import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
import time
import pickle
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.utils import AverageMeter, classification_accruracy_multi
from .criterion import *
from .optimizer import *
import copy

def train(model=None, write_iter_num=5, train_dataset=None, optimizer=None, device=None, criterion=torch.nn.BCELoss(), epoch=None, file=None):
    #scaler = torch.cuda.amp.GradScaler()
    assert train_dataset is not None, print("train_dataset is none")
    model.train()        
    ave_accuracy = AverageMeter()
    #scaler = torch.cuda.amp.GradScaler()
    for idx, (Image, Label) in enumerate(tqdm(train_dataset)):
        #model input data
        Input = Image.to(device, non_blocking=True)
        label = Label.to(device, non_blocking=True)
        Output = model(Input)
        loss = criterion(Output, label)            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = classification_accruracy_multi(Output, label)   
        ave_accuracy.update(accuracy)
        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                       f'Loss : {loss :.4f} '
                       f'Accuracy : {accuracy :.2f} ')
        if idx % (2*write_iter_num) == 0:
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                    f'Loss : {loss :.4f} '
                    f'Accuracy : {accuracy :.2f} ', file=file)
    tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n')
    tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n', file=file)
    
def valid(model=None, write_iter_num=5, valid_dataset=None, criterion=torch.nn.BCELoss(), device=None, epoch=None, file=None):
    ave_accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (Image, Label) in enumerate(tqdm(valid_dataset)):
            #model input data
            Input = Image.to(device, non_blocking=True)
            label = Label.to(device, non_blocking=True)
            Output = model(Input)
            loss = criterion(Output, label)
            accuracy = classification_accruracy_multi(Output, label)
            ave_accuracy.update(accuracy)
            if idx % (write_iter_num) == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(valid_dataset)} '
                        f'Validation Loss : {loss :.2f} '
                        f'Validation Accuracy : {accuracy :.2f} ')
            if idx % (2*write_iter_num) == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(valid_dataset)} '
                            f'Validation Loss : {loss :.2f} '
                            f'Validation Accuracy : {accuracy :.2f} ', file=file)
        tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} \n', file=file)
    return ave_accuracy.average()  