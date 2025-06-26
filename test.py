import os
import sys

sys.path.append('..')
import cv2
import time
import datetime
import numpy as np
import torch.autograd
from skimage import io
from scipy import stats
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, intersectionAndUnion, AverageMeter, CaclTP
from skimage.morphology import binary_dilation, disk
# Choose model and data
##################################################
import data_loader.GID as GID
from models.CT_Net import CT_Net as Net
import multiprocessing
from segment_anything import SamPredictor, sam_model_registry

NET_NAME = 'TCNET'
DATA_NAME = 'GID'
##################################################

# Change testing parameters here
working_path = os.path.abspath('.')
args = {
    'gpu': True,
    's_class': 0,
    'val_batch_size': 1,
    'size_local': 256,
    'size_context': 256 * 3,
    'data_dir': 'save',
    'data1_dir':  'dataset',
    'load_path': '.pth'


}
GID_COLORMAP = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255],
                         [255, 255, 0], [0, 255, 0], [255, 0, 0]])
GID_CLASSES = ['Impervious surfaces', 'Building', 'Low vegetation', 'Car', 'Tree', 'background']

def norm_gray(x, out_range=(0, 255)):
    # x=x*(x>0)
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0] + 1e-10)
    y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return y.astype('uint8')


def draw_rectangle(img, pos='boundary', color=(0, 255, 0), thick=2, text='context window'):
    h, w, c = img.shape
    if pos == 'boundary':
        start_pos = (0, 0)
        end_pos = (h - 1, w - 1)
    elif pos == 'center':
        start_pos = (h // 2 - 64, w // 2 - 64)
        end_pos = (h // 2 + 64, w // 2 + 64)
    cv2.rectangle(img, start_pos, end_pos, color, thick)
    if pos == 'boundary':
        cv2.putText(img, text, (start_pos[0] + 15, start_pos[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.2, color)
    elif pos == 'center':
        cv2.putText(img, text, (start_pos[0] - 24, start_pos[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.2, color)
    return img


def compute_multiclass_boundary_f1(
        pred_mask: np.ndarray,
        true_mask: np.ndarray,
        dist_threshold: int = 2,
        num_classes: int = 6
) -> Tuple[float, List[float]]:
    """
    计算多类别掩码的边界F1分数

    参数:
        pred_mask : 预测的分割掩码 (H, W), 值0-5
        true_mask : 真实的分割掩码 (H, W), 值0-5
        dist_threshold : 边界匹配距离容差阈值(像素), 默认2px
        num_classes : 类别数量, 默认6类

    返回:
        overall_f1 : 所有类别的平均边界F1分数
        class_f1s : 每个类别的边界F1分数列表
    """
    # 初始化每个类别的统计结果
    class_tp = np.zeros(num_classes)
    class_pred_pos = np.zeros(num_classes)
    class_true_pos = np.zeros(num_classes)

    # 为每个类别计算边界精度
    for class_id in range(0, num_classes-1):  # 0通常为背景类，跳过
        # 创建当前类别的二值掩码
        pred_binary = (pred_mask == class_id).astype(np.uint8)
        true_binary = (true_mask == class_id).astype(np.uint8)

        # 提取边界
        true_boundary = _extract_boundary(true_binary)
        pred_boundary = _extract_boundary(pred_binary)

        # 计算真实边界点的距离变换图
        dist_map = distance_transform_edt(~true_boundary)

        # 计算匹配指标
        tp_mask = pred_boundary & (dist_map <= dist_threshold)

        # 累加统计量
        class_tp[class_id] = np.sum(tp_mask)
        class_pred_pos[class_id] = np.sum(pred_boundary)
        class_true_pos[class_id] = np.sum(true_boundary)

    # 计算每个类别的边界F1
    class_f1s = []
    for class_id in range(0, num_classes-1):
        precision = class_tp[class_id] / class_pred_pos[class_id] if class_pred_pos[class_id] > 0 else 0.0
        recall = class_tp[class_id] / class_true_pos[class_id] if class_true_pos[class_id] > 0 else 0.0

        if precision + recall > 0:
            class_f1 = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1 = 0.0

        class_f1s.append(class_f1)

    # 计算整体边界F1（跳过背景类）
    overall_f1 = np.mean(class_f1s) if class_f1s else 0.0
    print('overall_f1:',overall_f1)

    return overall_f1, class_f1s


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """提取二值掩码的单像素边界"""
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = mask - eroded
    return boundary > 0

def main():


    net = Net().cuda()

    net.load_state_dict(torch.load(args['load_path']) , strict = False )  # , strict = False  加载预训练模型的权重参数
    net = net.cuda()
    net.eval()
    print(NET_NAME + ' Model loaded.')
    pred_path = os.path.join(args['data_dir'], 'Eval', NET_NAME)
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')

    val_set = GID.Loader(args['data1_dir'], 'val', sliding_crop=False, size_context=args['size_context'],
                         size_local=args['size_local'])
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    predict(net, val_loader, pred_path, f)
    f.close()


def predict(net, pred_loader, pred_path, f_out=None):
    acc_meter = AverageMeter()
    TP_meter = AverageMeter()
    pred_meter = AverageMeter()
    label_meter = AverageMeter()
    Union_meter = AverageMeter()
    output_info = f_out is not None
    mean_f1 = 0

    for vi, data in enumerate(pred_loader):
        with torch.no_grad():
            img, label = data
            if args['gpu']:
             
                img = img.cuda().float()
                label = label.cuda().float()

            _,output = net(img)

        output = output.detach().cpu()
        pred = torch.argmax(output, dim=1)
        pred = pred.numpy().squeeze()
        if args['s_class']:
            class_map = F.softmax(output, dim=1)[:, args['s_class'], :, :]
            class_map = class_map.numpy().squeeze()
            class_map = norm_gray(class_map)

        label = label.detach().cpu().numpy()
        acc, _ = accuracy(pred, label)
        acc_meter.update(acc)
        pred_color = GID.Index2Color(pred)
        img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        img = norm_gray(img)

        pred_name = os.path.join(pred_path, '%d_TC.png' % vi)
        io.imsave(pred_name, pred_color)

        if args['s_class']:
            saliency_map = cv2.applyColorMap(class_map, cv2.COLORMAP_JET)
            pred_name = os.path.join(pred_path, '%d_saliency.png' % vi)
            saliency_map = (img * 0.5 + saliency_map * 0.5).astype('uint8')
            io.imsave(pred_name, saliency_map)

      
        label_squeezed = np.squeeze(label)
        overall_f1, class_f1s = compute_multiclass_boundary_f1(pred, label_squeezed)
        mean_f1 += overall_f1


        TP, pred_hist, label_hist, union_hist = CaclTP(pred, label, GID.num_classes + 1)
        TP_meter.update(TP)
        pred_meter.update(pred_hist)
        label_meter.update(label_hist)
        Union_meter.update(union_hist)
        print('Eval num %d/%d, Acc %.2f' % (vi, len(pred_loader), acc * 100))
        if output_info:
            f_out.write('Eval num %d/%d, Acc %.2f\n' % (vi, len(pred_loader), acc * 100))

    precision = TP_meter.sum / (label_meter.sum + 1e-10) + 1e-10
    recall = TP_meter.sum / (pred_meter.sum + 1e-10) + 1e-10
    F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    F1 = np.array(F1)
    IoU = TP_meter.sum / Union_meter.sum
    IoU = np.array(IoU)

    boundary_f1 = mean_f1/vi;

    print(output.shape)

    print('Acc %.2f' % (acc_meter.avg * 100))
    avg_F = F1.mean()
    mIoU = IoU.mean()
    print('Avg F1 %.2f' % (avg_F * 100))
    print(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    print('mIoU %.2f' % (mIoU * 100))
    print(np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    print('boundary_f1 %.2f' % (boundary_f1 * 100))
    if output_info:
        f_out.write('Acc %.2f\n' % (acc_meter.avg * 100))
        f_out.write('Avg F1 %.2f\n' % (avg_F * 100))
        f_out.write(
            np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
        f_out.write('\nmIoU %.2f\n' % (mIoU * 100))
        f_out.write(
            np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    return avg_F


if __name__ == '__main__':
    main()