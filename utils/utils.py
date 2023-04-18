import os
import yaml
from easydict import EasyDict
import random
import cv2
import numpy as np
import torch
import time
import logging
from pathlib import Path
from torch.nn import functional as F
import torch
import shutil
from collections import OrderedDict
from sklearn import metrics

class AddParserManager:
    def __init__(self, cfg_file_path, cfg_name, is_json=False):
        super().__init__()
        self.values = EasyDict()
        if cfg_file_path:
            self.config_file_path = cfg_file_path
            self.config_name = cfg_name
            self.reload()

    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()

    def update(self, in_dict):
        for (key, value) in in_dict.item():
            if isinstance(value, dict):
                for (key2, value2) in value.item():
                    if isinstance(value2, dict):
                        for (key3, value3) in value2.item():
                            self.values[key][key2][key3] = value3

                    else:
                        self.values[key][key2] = value2

            else:
                self.values[key] = value

    def export(self, save_cfg_path):
        if save_cfg_path:
            with open(save_cfg_path, 'w') as f:
                yaml.dump(dict(self.values), f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def classification_accruracy(predict, target, thresh=0.5, sigmoid=False):
    predict = torch.sigmoid(predict).detach().cpu().numpy() if sigmoid else predict
    target = target.detach().cpu().numpy()
    predict[predict>=thresh] = 1
    predict[predict<thresh] = 0
    accuracy = metrics.accuracy_score(target, predict)
    #precision = metrics.precision_score(target, predict)
    #recall = metrics.recall_score(target, predict)
    #f1 = metrics.f1_score(target, predict)    
    return accuracy, np.sum(predict, axis=0), np.sum(target, axis=0)

def classification_accruracy_multi(predict, target, softmax=True):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy() if softmax else predict
    target = target.detach().cpu().numpy()
    predict = np.array(predict.argmax(axis=-1))
    target = np.array(target.argmax(axis=-1))
    accuracy = metrics.accuracy_score(target, predict)
    #precision = metrics.precision_score(target, predict)
    #recall = metrics.recall_score(target, predict)
    #f1 = metrics.f1_score(target, predict)   
    return accuracy

def classification_f1_multi(predict, target, softmax=True):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy() if softmax else predict
    target = target.detach().cpu().numpy()
    predict = np.array(predict.argmax(axis=-1))
    target = np.array(target.argmax(axis=-1))
    #accuracy = metrics.accuracy_score(target, predict)
    precision = metrics.precision_score(target, predict)
    recall = metrics.recall_score(target, predict)
    f1 = metrics.f1_score(target, predict)   
    return f1, precision, recall

def specificity_and_sensitivity(predict, target, thresh, softmax=False, sigmoid=False):
    if sigmoid:
        predict = torch.sigmoid(predict).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict[predict>=thresh] = 1
        predict[predict<thresh] = 0
    elif softmax:
        predict = torch.softmax(predict, dim=-1).detach().cpu().numpy()
        predict = np.array(predict.argmax(axis=-1))
        target = np.array(target.detach().cpu().numpy().argmax(axis=-1))
    cfx = metrics.confusion_matrix(target, predict)
    sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
    specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
    return specificity, sensitivity, cfx[0,0], cfx[1,1], cfx[1,0], cfx[0,1]

def specificity_and_sensitivity_per_class(predict, target, num_class):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    predict = predict.argmax(axis=-1)
    target = target.argmax(axis=-1)
    predict = np.where(predict=num_class)
    target = np.where(target=num_class)
    cfx = metrics.confusion_matrix(target, predict)
    sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
    specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
    return specificity, sensitivity, cfx[0,0], cfx[1,1], cfx[1,0], cfx[0,1]

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))