from utils.utils import AddParserManager, seed_everything
import argparse
from core.criterion import *
import numpy as np
from core.function import train_epoch
from core.optimizer import CosineAnnealingWarmUpRestarts
from data.dataset import ImageDataset_Classification
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.ConvNext import ConvNeXt
from models.CoatNet import CoatNet
from models.EfficientNetV2 import EfficientNet
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch
import torch.optim as optim
import pandas as pd
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='path to CSV file')
parser.add_argument('--world_size', type=int, default=1, help='Number of Process for multi-GPU')
parser.add_argument('--gpu', default="0", type=str, help='GPU id to use.')
parser.add_argument('--log-path', default="./log", type=str, help='Write Log Path')
parser.add_argument('--num_class', type=int, default=8, help='Number of Class')
parser.add_argument('--batch_size', type=int, default=4, help='Number of Batch Size')
parser.add_argument('--img_size', type=int, default=512, help='Image Size')
parser.add_argument('--epoch', default=100, type=int, help='Number of Epoch')
parser.add_argument('--workers', type=int, default=1, help='Number of Workers')
parser.add_argument('--input_channel', type=int, default=3, help='Number of Input Channel')
parser.add_argument('--optim', default="SGD", type=str, help='type of optimizer')
parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
parser.add_argument('--lr', default=1e-4, type=float, help='Train Learning Rate')
parser.add_argument('--optimizer_eps', default=1e-8, type=float, help='AdamW optimizer eps')
parser.add_argument('--optimizer_betas', default=(0.9, 0.999), help='AdamW optimizer betas')
parser.add_argument('--weight_decay', default=0.95, type=float, help='AdamW optimizer weight decay')
parser.add_argument('--scheduler', default="LambdaLR", type=str, help='type of Scheduler')
parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
parser.add_argument('--t_scheduler', default=80, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
parser.add_argument('--eta_scheduler', default=1.25e-3, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
parser.add_argument('--up_scheduler', default=8, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')
parser.add_argument('--model_path', default="./weight/best_model.pth", type=str, help='Model Path')
parser.add_argument('--model_save_path', default="./weight", type=str, help='Model Save Path')
parser.add_argument('--retrain', default=False, type=bool, help='Model Save Path')
parser.add_argument('--write_iter_num', default=10, type=int, help='Write iter num')

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_csv_path = pd.read_csv(os.path.join(args.csv_path, 'Train_data.csv'))
    valid_csv_path = pd.read_csv(os.path.join(args.csv_path, 'Valid_data.csv'))
    
    image_shape = to_2tuple(args.img_size)
    height, width = image_shape
    train_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.ShiftScaleRotate(p=1, rotate_limit=90),
                    albumentations.VerticalFlip(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.GaussNoise(p=1)                    
                ],p=1)
            ]
        )
    valid_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.VerticalFlip(p=1)
                ],p=1)
            ]
        )
    model = ConvNeXt(in_chans=args.input_channel, num_classes=args.num_class)
    #model = CoatNet(in_channels=args.input_channel, out_class=args.num_class, img_size=args.img_size)
    #model = EfficientNet(in_channels=args.input_channel, out_channels=args.num_class, pre_train=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), eps=args.optimizer_eps, betas=args.optimizer_betas,
                                        lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler=='LambdaLR':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:args.lambda_weight**epoch)
    else:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_scheduler, T_mult=args.trigger_scheduler, 
                                                  eta_max=args.eta_scheduler, T_up=args.up_scheduler, gamma=args.gamma_scheduler)
        
    
    train_dataset = ImageDataset_Classification(csv_file=train_csv_path, transform=train_transform, type='others', num_class=args.num_class)
    valid_dataset = ImageDataset_Classification(csv_file=valid_csv_path, transform=valid_transform, type='others', num_class=args.num_class)
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=True)   
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=False)   
    start_epoch = 0
    best_acc = 0
    if args.retrain:
        checkpoint = torch.load(args.model_path, map_location={'cuda:0':'cpu'})
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    train_epoch(model=model, write_iter_num=args.write_iter_num, trainloader=trainloader, validloader=validloader, optimizer=optimizer, scheduler=scheduler, device=device, 
                criterion=criterion, start_epoch=start_epoch, end_epoch=args.epoch, log_path=args.log_path, model_path=args.model_save_path, best_loss=best_acc)
if __name__ == '__main__':
    main()