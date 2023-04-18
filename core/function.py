import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
import time
import wandb
import pickle
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.utils import AverageMeter, classification_accruracy, specificity_and_sensitivity, classification_accruracy_multi, classification_f1_multi
from .criterion import *
from .optimizer import *
import copy

def train(model=None, write_iter_num=5, train_dataset=None, optimizer=None, device=None, criterion=torch.nn.BCELoss(), epoch=None, file=None):
    best_loss = 0
    scaler = torch.cuda.amp.GradScaler()
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
        accuracy, num_true, num_target = classification_accruracy(Output, label, thresh=0.5, sigmoid=True)   
        specificity, sensitivity, TP, TN, FP, FN = specificity_and_sensitivity(Output, label, thresh=0.5)
        ave_accuracy.update(accuracy)
        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                       f'Loss : {loss :.4f} '
                       f'Accuracy : {accuracy :.2f} '
                       f'Number of True in Predict : {num_true}, '
                       f'Number of True in Target : {num_target} '
                       f'Specificity : {specificity :.2f} '
                       f'Sensitivity : {sensitivity :.2f} '
                       f'TP : {TP} TN : {TN}  FP : {FP}  FN : {FN} ')
        if idx % (2*write_iter_num) == 0:
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                    f'Loss : {loss :.4f} '
                    f'Accuracy : {accuracy :.2f} '
                    f'Number of True in Predict : {num_true}, '
                    f'Number of True in Target : {num_target} '
                    f'Specificity : {specificity :.2f} '
                    f'Sensitivity : {sensitivity :.2f} '
                    f'TP : {TP} TN : {TN}  FP : {FP}  FN : {FN} ', file=file)
    tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n')
    tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n', file=file)
    
def valid(model=None, write_iter_num=5, valid_dataset=None, criterion=torch.nn.BCELoss(), device=None, epoch=None, file=None):
    ave_accuracy = AverageMeter()
    ave_sensitivity = AverageMeter()
    ave_specificity = AverageMeter()
    ave_tp = AverageMeter()
    ave_tn = AverageMeter()
    ave_fp = AverageMeter()
    ave_fn = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (Image, Label) in enumerate(tqdm(valid_dataset)):
            #model input data
            Input = Image.to(device, non_blocking=True)
            label = Label.to(device, non_blocking=True)
            Output = model(Input)
            loss = criterion(Output, label)
            accuracy, num_true, num_target = classification_accruracy(Output, label, thresh=0.5,  sigmoid=True)
            specificity, sensitivity, TP, TN, FP, FN = specificity_and_sensitivity(Output, label, thresh=0.5)
            ave_sensitivity.update(sensitivity)
            ave_specificity.update(specificity)
            ave_tp.update(TP)
            ave_tn.update(TN)
            ave_fp.update(FP)
            ave_fn.update(FN)
            ave_accuracy.update(accuracy)
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(valid_dataset)} '
                       f'Validation Loss : {loss :.2f} '
                       f'Validation Accuracy : {accuracy :.2f} '
                       f'Number of True in Predict : {num_true}, '
                       f'Number of True in Target : {num_target} '
                       f'Specificity : {specificity :.2f} '
                       f'Sensitivity : {sensitivity :.2f} '
                       f'TP : {TP} TN : {TN}  FP : {FP}  FN : {FN} ')
        tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} '
                f'Average Sensitivity : {ave_sensitivity.average() :.2f} '
                f'Average Specificity : {ave_specificity.average() :.2f} '
                f'Average TP : {ave_tp.average() :.2f} '
                f'Average TN : {ave_tn.average() :.2f} '
                f'Average FP : {ave_fp.average() :.2f} '
                f'Average FN : {ave_fn.average() :.2f} \n\n')
        tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} '
                f'Average Sensitivity : {ave_sensitivity.average() :.2f} '
                f'Average Specificity : {ave_specificity.average() :.2f} '
                f'Average TP : {ave_tp.average() :.2f} '
                f'Average TN : {ave_tn.average() :.2f} '
                f'Average FP : {ave_fp.average() :.2f} '
                f'Average FN : {ave_fn.average() :.2f} \n\n', file=file)
    return ave_accuracy.average()  