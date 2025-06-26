import os
import random
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
import functools

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.fileio import load

def read_idtxt(path):
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label):
    valid = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)
    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
        #TP, bins = numClass, range = (0, numClass))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))

    # (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(0, numClass))
    # (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(0, numClass))
    
    union_hist = pred_hist + lab_hist - TP_hist
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)

    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist, union_hist)

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    one_hot_label = torch.eye(
        n_classes, device='cuda', requires_grad=requires_grad)[label]

    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

class ObjectLoss(nn.Module):
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        num_object = int(torch.max(gt)) + 1
        num_object = min(num_object, self.max_object)
        # if num_object == 1:
        #     num_object=2
        total_object_loss = 0

        for object_index in range(1, num_object):
            # ObjectLoss gt shape: torch.Size([10, 256, 256])
            # print('ObjectLoss gt shape:',gt.shape)
            # 创建掩码图：根据条件判断 gt即 （SAM得到的object） 是否与 object_index 相等，如果相等则将对应位置的值设为 1，否则设为 0。
            # 然后使用 unsqueeze(1) 方法在维度 1 上进行扩展，将 mask 张量的形状变为 [10, 1, 256, 256]。
            # 最后使用 .to('cuda') 将 mask 张量移动到 GPU 上。
            mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
            # ObjectLoss mask: torch.Size([10, 1, 256, 256])
            # print('ObjectLoss mask:',mask.shape)
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
            # ObjectLoss num_point: torch.Size([10, 1, 1, 1])
            # print('ObjectLoss num_point:',num_point.shape)
            avg_pool = mask / (num_point + 1)
            # ObjectLoss avg_pool: torch.Size([10, 1, 256, 256])
            # print('ObjectLoss avg_pool:',avg_pool.shape)
            # ObjectLoss pred: torch.Size([10, 6, 256, 256])
            # print('ObjectLoss pred:',pred.shape)

            object_feature = pred.mul(avg_pool)
            # ObjectLoss object_feature: torch.Size([10, 6, 256, 256])
            # print('ObjectLoss object_feature:',object_feature.shape)

            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, gt.shape[1], gt.shape[2])
            avg_feature = avg_feature.mul(mask)

            object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss

        return total_object_loss


# class BoundaryLoss(nn.Module):
#     def __init__(self, theta0=3, theta=5):
#         super().__init__()
#
#         self.theta0 = theta0
#         self.theta = theta
#
#     def forward(self, pred, gt):
#         """
#         Input:
#             - pred: the output from model (before softmax)
#                     shape (N, C, H, W)
#             - gt: ground truth map
#                     shape (N, H, w)
#         Return:
#             - boundary loss, averaged over mini-bathc
#         """
#
#         n, c, _, _ = pred.shape
#         #BoundaryLoss pred: torch.Size([16, 6, 256, 256])
#         #print('BoundaryLoss pred1:',pred.shape)
#         # softmax so that predicted map can be distributed in [0, 1], torch.Size([16, 6, 256, 256])
#         pred = torch.softmax(pred, dim=1)
#
#
#
#         # one-hot vector of ground truth
#         #BoundaryLoss gt: torch.Size([16, 256, 256])
#         #print('BoundaryLoss gt:',gt.shape)
#
#
#         one_hot_gt = one_hot(gt, c)
#         #one_hot_gt: torch.Size([10, 6, 256, 256])
#         #print('one_hot_gt:',one_hot_gt.shape)
#         #1-one_hot_gt: torch.Size([10, 6, 256, 256])
#         #print('1-one_hot_gt:',(1-one_hot_gt).shape)
#
#         # boundary map
#         gt_b = F.max_pool2d(
#             1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
#         gt_b -= 1 - one_hot_gt
#
#         pred_b = F.max_pool2d(
#             1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
#         pred_b -= 1 - pred
#
#         # extended boundary map
#         gt_b_ext = F.max_pool2d(
#             gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
#
#         pred_b_ext = F.max_pool2d(
#             pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
#
#         # reshape
#         gt_b = gt_b.view(n, c, -1)
#         pred_b = pred_b.view(n, c, -1)
#         gt_b_ext = gt_b_ext.view(n, c, -1)
#         pred_b_ext = pred_b_ext.view(n, c, -1)
#
#         #self add
#         # print('gt_b device:',gt_b.device)  #cuda:0
#         # print('pred_b device:',pred_b.device)  #cuda:0
#         # print('gt_b_ext device:',gt_b_ext.device)  #cuda:0
#         # print('pred_b_ext device:',pred_b_ext.device)  #cuda:0
#
#
#         # Precision, Recall
#         P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
#         R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
#
#         # Boundary F1 Score
#         BF1 = 2 * P * R / (P + R + 1e-7)
#
#         # summing BF1 Score for each class and average over mini-batch
#         loss = torch.mean(1 - BF1)
#
#         return loss

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = load(class_weight)

    return class_weight


def reduce_loss(loss, reduction) -> torch.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
