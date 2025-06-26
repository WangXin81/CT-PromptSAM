import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys,gc

sys.path.append('..')
import time
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
#from ptflops import get_model_complexity_info
working_path = os.path.dirname(os.path.abspath(__file__))
from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, AverageMeter
from utils.misc import evaluate
from utils.metircs import Evaluator
import numpy as np
from utils.utils import *
from utils.loss import *
import monai
from thop import profile
# Choose model and data
##################################################
import data_loader.GID as GID
from models.CT_Net import CT_Net as Net
import multiprocessing
from segment_anything import SamPredictor, sam_model_registry
import torch
import torchvision
from thop import profile
from segment_anything.modeling import *
from functools import partial
torch.manual_seed(2025)
np.random.seed(2025)
NET_NAME = 'TCNET'
DATA_NAME = 'GID'
##################################################


# Change training parameters here
args = {
    'train_batch_size': 16,  # 16
    'val_batch_size': 16,  # 32
    'lr': 0.0001, 
    'lr-sam': 0.0005,
    'lr-seg': 0.01,
    'min_lr': 0,
    'epochs': 100,
    'gpu': True,
    'size_local': 256,
    'size_context': 256 * 3,
    'momentum': 0.9,
    'crop_nums': 1000,
    'weight_decay': 1e-05,
    'lr_decay_power': 0.9,
    'print_freq': 50,
    'save_pred': True,
    'num_workers': 0,
    'data_dir': 'datapath',
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME, '1'),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME)
}

if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
writer = SummaryWriter(args['log_dir'])


# torch.backends.cudnn.enable=True
# torch.backends.cudnn.benchmark=True

def main():
    multiprocessing.set_start_method('spawn', force=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")  

    checkpoint = '.../sam_vit_b_01ec64.pth'
    pretrained_state_dict = torch.load(checkpoint)


    #'vit-b'
    num_class_sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)


    num_class_sam_model = num_class_sam_model.to(device)
    num_class_model_state_dict = num_class_sam_model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name in num_class_model_state_dict and num_class_model_state_dict[name].shape == param.shape:
            num_class_model_state_dict[name] = param
    # 加载适配的state_dict:s
    for param in num_class_sam_model.image_encoder.parameters():
        param.requires_grad = False
    for param in num_class_sam_model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in num_class_sam_model.mask_decoder.parameters():
        param.requires_grad = True
    for param in num_class_sam_model.net.parameters():
        param.requires_grad = True
    num_class_sam_model.load_state_dict(num_class_model_state_dict, strict=False)

    seg_params = list(map(id, num_class_sam_model.net.parameters()))

    base_params = filter(lambda p: id(p) not in seg_params, num_class_sam_model.parameters())

    paramsnew = [{'params': base_params, 'lr': args['lr-sam']},
              {'params': num_class_sam_model.net.parameters(), 'lr': args['lr-seg'] }]


    dummy_input = torch.randn(1, 3, 256, 256)
    gt = torch.randn(1, 6, 256, 256)
    dummy_input = dummy_input.cuda()
    #gt = gt.cuda()
    flops, params = profile(num_class_sam_model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    params = 0
    for name, param in num_class_sam_model.named_parameters():
        if param.requires_grad==True:
            params += param.nelement()
    print("params : ", params)


    train_set = GID.Loader(args['data_dir'], 'train', sliding_crop=False, crop_nums=args['crop_nums'], random_flip=True,
                           size_context=args['size_context'], size_local=args['size_local'])
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=args['num_workers'],shuffle=True)
    val_set = GID.Loader(args['data_dir'], 'val', sliding_crop=False, size_context=args['size_context'],
                         size_local=args['size_local'])
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=args['num_workers'], shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=255).cuda()


    optimizer = optim.SGD(paramsnew, lr=args['lr-sam'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    optimizer1 = torch.optim.Adam(num_class_sam_model.net.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-05, amsgrad=False)
    optimizer2 = torch.optim.AdamW(num_class_sam_model.mask_decoder.parameters(), lr=0.0005, weight_decay=0.0005)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1,                                               
                                                     T_max=args['epochs'] * 194,
                                                     eta_min=args['min_lr'])
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2,
                                                      T_max=args['epochs'] * 194,
                                                      eta_min=args['min_lr'])


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args['epochs'] * 194,
                                                     eta_min=args['min_lr'])

    train(train_loader, num_class_sam_model, scheduler1,scheduler2, criterion, optimizer1,optimizer2, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, scheduler1,scheduler2, criterion, optimizer1,optimizer2, val_loader):
    bestaccT = 0
    bestaccV = 0.5
    bestmiou = 0
    bestmiou5 = 0
    bestloss = 1
    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    bce_loss_fn = nn.BCELoss()
    while True:
        gc.collect()
        torch.cuda.empty_cache()


        net.train()

        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()
        seg_loss = monai.losses.DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

        miou_meter = AverageMeter()
        evaluator = Evaluator(6)
        evaluator.reset()
        sam = AutomaticWeightedLoss(2)

        LBABDA_BDY = 0.1
        LBABDA_OBJ = 1.0

        if torch.cuda.is_available():
            device = torch.device("cuda")  # GPU设备对象

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            imgs, labels = data
            imgs = imgs.to(device)
            imgs = imgs.float()
            labels = labels.to(device)
            labels = labels.long()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

         
            alpha = calc_alpha(running_iter, all_iters)

    
            seg_logits, outputs = net(imgs, labels)

            outputs = outputs.to(device)
          

            loss1 = criterion(outputs, labels)
            loss2 = dice_loss(outputs, labels)

            labels0 = labels.unsqueeze(1)
            loss_SAM = seg_loss(outputs, labels0)

            loss3 = criterion(seg_logits, labels)
            loss4 = dice_loss(seg_logits, labels)
    
            loss = loss3 + loss4 + 0.1 * loss_SAM
        
            loss.backward()

            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()


            preds = torch.argmax(outputs, dim=1)
     
            preds = preds.numpy()
    
            acc_curr_meter = AverageMeter()

       
            miou_curr_meter = AverageMeter()

        
            for (pred, label) in zip(preds, labels):
             

                acc, valid_sum = accuracy(pred, label)
                acc_curr_meter.update(acc)
                evaluator.add_batch(label, pred)


            acc_meter.update(acc_curr_meter.avg)
        
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer1.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer1.param_groups[0]['lr'], running_iter)

     
        tmIou = evaluator.Mean_Intersection_over_Union()
        print('mIou:', tmIou)
        tmiou = np.nanmean(tmIou)  
 
        tmiou5 = (tmIou[0] + tmIou[1] + tmIou[2] + tmIou[3] + tmIou[4]) / 5
     

        print(' Train_miou %.4f, Train_miou5 %.4f' \
              % (tmiou * 100, tmiou5 * 100))

        f = open('.../result_1.txt', 'a')
    
        f.write("======================Epoch %d======================\n" % curr_epoch)
        f.write('six class iou %.4f\n' % tmiou)
        f.write('five class iou %.4f\n' % tmiou5)
        f.write('Impervious surfaces iou %.4f\n' % tmIou[0])
        f.write('Building iou %.4f\n' % tmIou[1])
        f.write('Low vegetation iou %.4f\n' % tmIou[2])
        f.write('Tree iou %.4f\n' % tmIou[3])
        f.write('Car iou %.4f\n' % tmIou[4])
        f.write('background iou %.4f\n' % tmIou[5])

        acc_v, loss_v, miou, miou5 = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if miou5 > 0.74:
            save_path = os.path.join(args['chkpt_dir'],
                            NET_NAME + '_%de_0_%.2f.pth' % (curr_epoch,miou5 * 100))
            torch.save(net.state_dict(), save_path)


        print('Total time: %.1fs Best rec: Train %.2f, Val %.2f, Val_loss %.4f, Val_miou %.4f, Val_miou5 %.4f' \
              % (time.time() - begin_time, acc_meter.avg * 100, acc_v * 100, loss_v, miou * 100, miou5 * 100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    miou_meter = AverageMeter()
    evaluator = Evaluator(6)
    evaluator.reset()

    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU设备对象


    for vi, data in enumerate(val_loader):
       
        imgs, labels= data
 
        imgs = imgs.to(torch.float32)
        labels = labels.to(torch.long)
    
        imgs = imgs.to(device)
        imgs = imgs.float()
        labels = labels.to(device)
        labels = labels.long()


        with torch.no_grad():
 
            seg_logits, outputs= net(imgs, labels)
            outputs = outputs.to(device)

            loss1 = criterion(outputs, labels)
            loss2 = dice_loss(outputs, labels)

            loss3 = criterion(seg_logits, labels)
            loss4 = dice_loss(seg_logits, labels)
            loss = loss1 + loss2

      

        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()


        preds = torch.argmax(outputs, dim=1)
        preds = preds.numpy()

        for (pred, label) in zip(preds, labels):
            acc, valid_sum = accuracy(pred, label)
            acc_meter.update(acc)
            evaluator.add_batch(label, pred)
      

        if args['save_pred'] and vi == 0:
            pred_color = GID.Index2Color(preds[0])
            pred_path = os.path.join(args['pred_dir'], NET_NAME + '.png')
            io.imsave(pred_path, pred_color)
            print('Prediction saved!')
     

    mIou = evaluator.Mean_Intersection_over_Union()

    print('mIou:', mIou)
    miou = np.nanmean(mIou) 
   
    miou5 = (mIou[0] + mIou[1] + mIou[2] + mIou[3] + mIou[4]) / 5

    f = open('.../result_1.txt', 'a')
  
    f.write("======================Epoch %d======================\n" % curr_epoch)
    f.write('six class iou %.4f\n' % miou)
    f.write('five class iou %.4f\n' % miou5)
    f.write('Impervious surfaces iou %.4f\n' % mIou[0])
    f.write('Building iou %.4f\n' % mIou[1])
    f.write('Low vegetation iou %.4f\n' % mIou[2])
    f.write('Tree iou %.4f\n' % mIou[3])
    f.write('Car iou %.4f\n' % mIou[4])
    f.write('background iou %.4f\n' % mIou[5])

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Accuracy: %.2f miou: %.2f miou5: %.2f' % (
    curr_time, val_loss.average(), acc_meter.average() * 100,
    miou * 100, miou5 * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)
    writer.add_scalar('val_miou', miou, curr_epoch)
    writer.add_scalar('val_miou5', miou5, curr_epoch)

    return acc_meter.avg, val_loss.avg, miou, miou5


def calc_alpha(curr_iter, all_iters, weight=1.0):
    r = (1.0 - float(curr_iter) / all_iters) ** 2.0
    return weight * r


def adjust_learning_rate(optimizer, curr_iter, all_iter):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
