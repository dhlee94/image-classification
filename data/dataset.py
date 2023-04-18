from configparser import Interpolation
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import torchvision
import cv2
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class ImageDataset_Classification(Dataset):
    def __init__(self, csv_file, type='ZSCORE', transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self.type = type
        
    def __len__(self):
        return len(self.csv_path['image'])

    def _normalization(self, data, type='ZSCORE'):
        if type=='ROBUST':
            robustScaler = RobustScaler()
            robustScaler.fit(data)
            data = robustScaler.transform(data)
            return torch.from_numpy(data)
        else:
            if type=='ZSCORE':
                data = torch.Tensor.float(torch.from_numpy(data))
                data = data.permute(2, 0, 1)
                return (data-torch.mean(data))/torch.std(data)
            elif type=='MINMAX':
                data = torch.from_numpy(data)
                data = data.permute(2, 0, 1)
                min_value = torch.min(data.reshape(data.shape[0],-1))
                max_value = torch.max(data.reshape(data.shape[0],-1))
                return (data-min_value)/(max_value - min_value)
            else:
                data = torch.from_numpy(data)
                data = data.permute(2, 0, 1)
                return (data)/255
                
    def __getitem__(self, idx):
        input = cv2.imread(self.csv_file['image'][idx])
        target = self.csv_file['label'][idx]
        if self.transform:
            input = self.transform(image=input)
        image = self._normalization(input['image'], type=self.type)
        return image, target