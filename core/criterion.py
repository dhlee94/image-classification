import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import logging
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(target.size(0), -1)
            target = target.view(-1).contiguous()            
        #target = target.view(-1,1)
        #logpt = F.log_softmax(input, dim=-1)
        logpt = F.logsigmoid(input)
        target = target.type(torch.long)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def forward(self, score, target):
        loss = self.criterion(score, target[:].long())

        return loss

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class DistanceLoss(nn.Module):
    def __init__(self, n_classes, softmax=False, loss_function=nn.BCELoss, sigmoid=False, weight=[1., 1., 1.]):
        super(DistanceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax
        self.sigmoid = sigmoid
        weight = torch.Tensor(weight)
        self.loss = loss_function(weight=weight)

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        elif self.sigmoid:
            inputs = torch.sigmoid(inputs)
        target = target.long()
        loss = self.loss(inputs, target)
        return loss

class MultiChannelLoss(nn.Module):
    def __init__(self):
        super(MultiChannelLoss, self).__init__()
        self.CrossEntropy = nn.BCEWithLogitsLoss()
    def forward(self, scores, targets):
        loss = 0
        for score, target in zip(scores.chunk(scores.size(1), dim=1), targets.chunk(targets.size(1), dim=1)):
            loss += self.CrossEntropy(score, target)
        return loss / scores.size(1)

class Weighted_L1_Loss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(Weighted_L1_Loss, self).__init__()
        self.weights = weights
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, input, target):
        loss = 0
        for i in range(target.shape[1]):
            loss += self.criterion(input[:,i,:,:], target[:,i,:,:]) * self.weights[i]

        return loss        

class BCEWithMSELoss(nn.Module):
    def __init__(self, weights, bce_weight, reduction='mean'):
        super(BCEWithMSELoss, self).__init__()
        self.weights = weights
        self.regression_loss = nn.L1Loss(reduction=reduction)
        self.cross_entropy = nn.BCEWithLogitsLoss(weight=bce_weight)
    def forward(self, input, target):
        entropy_loss = self.cross_entropy(input, target)
        input = torch.sigmoid(input)
        regression_loss = self.regression_loss(input, target)
        loss = self.weights*(entropy_loss) + (1-self.weights)*regression_loss
        return loss   

class EntropyWithDistanceLoss(nn.Module):
    def __init__(self, weights, entropy_weight, reduction='mean', sigmoid=False):
        super(EntropyWithDistanceLoss, self).__init__()
        self.weights = weights
        self.sigmoid = sigmoid
        self.regression_loss = nn.L1Loss(reduction=reduction)
        self.cross_entropy = nn.CrossEntropyLoss(weight=entropy_weight)
    def forward(self, input, target):
        entropy_loss = self.cross_entropy(input, target)
        input = torch.sigmoid(input) if self.sigmoid else torch.softmax(input, dim=-1)
        regression_loss = self.regression_loss(input, target)
        loss = self.weights*(entropy_loss) + (1-self.weights)*regression_loss
        return loss   

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma*BCE_loss
        if self.reduction=='mean':
            return F_loss.mean()
        elif self.reduction=='sum':
            return F_loss.sum()