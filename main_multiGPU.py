from utils.utils import AddParserManager, seed_everything, select_idx_for_train
import argparse
from core.function import train_multi, valid_multi
from core.criterion import MultiClassDiceCELoss
import pickle
import numpy as np
from core.optimizer import CosineAnnealingWarmUpRestarts
from data.dataset import ImageDataset_Classification
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.EfficientNet import EfficientNet
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
import cv2
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.utils import save_checkpoint, save_checkpoint_remove_module

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='path to CSV file')
parser.add_argument('--world_size', type=int, default=1, help='Number of Process for multi-GPU')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--log-path', default='/data02/scripts/dh/backup/log/Lung_256_classification', type=str, help='Write Log Path')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
parser.add_argument("opts",
                        help="Modify config options by adding 'KEY VALUE' pairs. ",
                        default=None,
                        nargs=argparse.REMAINDER)
parser.add_argument('--num_class', type=int, default=1, help='Number of Class')
parser.add_argument('--batch_size', type=int, default=1, help='Number of Batch Size')
parser.add_argument('--epoch', default=100, type=int, help='Number of Epoch')
parser.add_argument('--workers', type=int, default=1, help='Number of Workers')
parser.add_argument('--input_channel', type=int, default=3, help='Number of Input Channel')
parser.add_argument('--optim', default='SGD', type=str, help='type of optimizer')
parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
parser.add_argument('--lr', default=1e-4, type=float, help='Train Learning Rate')
parser.add_argument('--optimizer_eps', default=1e-8, type=float, help='AdamW optimizer eps')
parser.add_argument('--optimizer_betas', default=(0.9, 0.999), help='AdamW optimizer betas')
parser.add_argument('--weight_decay', default=0.95, type=float, help='AdamW optimizer weight decay')
parser.add_argument('--scheduler', default='LambdaLR', type=str, help='type of Scheduler')
parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
parser.add_argument('--t_scheduler', default=80, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
parser.add_argument('--eta_scheduler', default=1.25e-3, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
parser.add_argument('--up_scheduler', default=8, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')
parser.add_argument('--model_path', default='./Weight/best_model.pth', type=str, help='Model Path')
parser.add_argument('--model_save_path', default='./Weight', type=str, help='Model Save Path')
parser.add_argument('--retrain', default=False, type=bool, help='Model Save Path')
parser.add_argument('--write_iter_num', default=10, type=int, help='Write iter num')

def main():    
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    log_path = args.log_path
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    print("Reading Data")
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
        
    model = EfficientNet(in_channels=args.input_channel, out_channels=args.num_class)
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    if args.optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=args.optimizer_eps, betas=args.optimizer_betas,
                                        lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler=='LambdaLR':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:args.lambda_weight**epoch)
    else:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_scheduler, T_mult=args.trigger_scheduler, 
                                                  eta_max=args.eta_scheduler, T_up=args.up_scheduler, gamma=args.gamma_scheduler)
    
    train_dataset = ImageDataset_Classification(Data_path=train_csv_path, transform=train_transform, type='others')
    valid_dataset = ImageDataset_Classification(Data_path=valid_csv_path, transform=valid_transform, type='others')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=(train_sampler is None), sampler=train_sampler)   
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=False, sampler=val_sampler)   
    start_epoch = 0
    best_loss = 0
    if args.retrain:
        if args.gpu is None:
            checkpoint = torch.load(args.model_path)
        elif torch.cuda.is_available():
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.model_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    for epoch in range(start_epoch, args.epoch):
        is_best = False
        file = open(os.path.join(log_path, f'{epoch}_log.txt'), 'a')
        if args.distributed:
            train_sampler.set_epoch(epoch)    
        train_multi(model=model, write_iter_num=args.write_iter_num, train_dataset=trainloader, optimizer=optimizer, 
                    device=device, criterion=criterion, epoch=epoch, file=file)
        accuracy = valid_multi(model=model, write_iter_num=args.write_iter_num, valid_dataset=validloader, criterion=criterion, 
                               device=device, epoch=epoch, file=file)
        scheduler.step()
        is_best = accuracy > best_loss
        best_loss = max(best_loss, accuracy)            
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_loss,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best=is_best, path=args.model_save_path)
        file.close()

if __name__ == '__main__':
    main()