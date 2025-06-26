import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from scipy.optimize import minimize
import math
from typing import Union
from .utils import *

# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=0,):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class CrossEntropyLoss2dnew(nn.Module):
    def __init__(self, weight=None, ignore_index=0,):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)




# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
    
def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    
    if weight_pos is not None:        
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos
        
    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None] = None,
              #weight: Optional[Tensor] = None,
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.
    """
    target = target.unsqueeze(1)
    target = target.expand(-1, 6, -1, -1)
    if ignore_index is not None:
        num_classes = pred.shape[1]
        #print('num_classes:', num_classes)
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        #print('target.shape:',target.shape)
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0  # if the ignored index is the only class
    input = pred.flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def balanced_f1_loss(predictions, targets, smooth=1e-6):
    predictions = predictions.round()  # 将概率值转换为0或1
    TP = (predictions * targets).sum()  # 真正例
    FP = ((predictions - targets) * (predictions == 1)).sum()  # 假正例
    FN = ((targets - predictions) * (targets == 1)).sum()  # 假负例
    precision = TP / (TP + FP + smooth)  # 精确度
    recall = TP / (TP + FN + smooth)  # 召回率
    f1 = 2 * (precision * recall) / (precision + recall + smooth)  # F1分数
    return 1 - f1  # 返回1-F1值作为损失

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

