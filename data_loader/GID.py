import os
import numpy as np
import torch
import torchvision
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import cv2

import skimage
from skimage.segmentation import find_boundaries
from PIL import Image
import time

num_classes = 5
GID_COLORMAP = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],[255, 255, 0], [255, 0, 0]]
GID_CLASSES = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car','background']



GID_MEAN_IRRGB = np.array([126.72, 90.76, 95.96, 87.58])
GID_STD_IRRGB = np.array([62.91, 59.24, 58.03, 57.23])
# GID_MEAN_RGB = np.array([90.76, 95.96, 87.58])
# GID_STD_RGB = np.array([59.24, 58.03, 57.23])

GID_MEAN_RGB=np.array([123.675, 116.28, 103.53])
GID_STD_RGB=np.array([58.395, 57.12, 57.375])


def IRRGB2RGBIR(img):
    IR, R, G, B = cv2.split(img)
    return cv2.merge((R, G, B, IR))

def normalize_image(im):
    #return (im - GID_MEAN_IRRGB) / GID_STD_IRRGB
    return (im - GID_MEAN_RGB) / GID_STD_RGB

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


# colormap2label = np.zeros(256 ** 3)
# for i, cm in enumerate(GID_COLORMAP):
#     colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
#     # if i>=2:
#     #     colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = 1


def Index2Color(pred):
    colormap = np.asarray(GID_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels

#将颜色标签映射为索引标签
def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    colormap2label = np.zeros(256 ** 3)
    for i, cm in enumerate(GID_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        # if i>=1:
        #     colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = 1

    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)



def rescale_images(imgs, scale, order=0):
    rescaled_imgs = []
    for im in imgs:
        rescaled_imgs.append(rescale_image(im, scale, order))
    return rescaled_imgs

#对输入的图像进行缩小处理
def rescale_image(img, scale=8, order=0):
    flag = cv2.INTER_NEAREST #设置默认的插值方法为最近邻插值
    if order==1: flag = cv2.INTER_LINEAR #将插值方法设置为双线性插值
    elif order==2: flag = cv2.INTER_AREA  #将插值方法设置为区域插值
    elif order>2:  flag = cv2.INTER_CUBIC  #将插值方法设置为三次样条插值
    im_rescaled = cv2.resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interpolation=flag)
    return im_rescaled



#
def read_RSimages(data_dir, mode, rescale_ratio=False):
    assert mode in ['train', 'val', 'test']
    data_list = []
    img_dir = os.path.join(data_dir, mode, 'image')
    item_list = os.listdir(img_dir)
    for item in item_list:
        if (item[-4:]=='.tif'): data_list.append(os.path.join(img_dir, item))
    data_length = int(len(data_list))
    count=0
   
    data, labels= [], []
    for it in data_list:
        img_path = it
        mask_path = img_path.replace('image', 'label')


        img = io.imread(img_path)
        label1=io.imread(mask_path )
        label = Color2Index(io.imread(mask_path ))
    


        data.append(img)
        labels.append(label)
      


        count+=1
        if not count%1000: print('%d/%d images loaded.'%(count, data_length))
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
   
    return data, labels

class Loader(data.Dataset):
    def __init__(self, data_dir, mode, random_crop=False, crop_nums=40, random_flip=False, sliding_crop=False,\
        size_context=512, size_local=256, scale=4):
        self.size_context = size_context
        self.size_local = size_local
        self.scale = scale
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop

      
        data, labels= read_RSimages(data_dir, mode)

   
        self.data = data
        self.labels = labels
 


        if sliding_crop:
         
            self.data_s, self.labels_s = transform.slidding_crop_WC(self.data, self.labels, scale)

       

        if self.random_crop:
            self.len = crop_nums*len(self.data)
        else:
            self.len = len(self.data)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = int(idx/self.crop_nums)
            data_s, label_s, label_mask_s, data, label = transform.random_crop2(self.data_s[idx], self.labels_s[idx],\
                self.data[idx], self.labels[idx], self.size_context, self.size_local, self.scale)



        else:
            data = self.data[idx]
            label = self.labels[idx]
          
        #随机翻转
        if self.random_flip:
             data, label = transform.rand_flip2( data, label )

   
        data = normalize_image(data)
    
        data = torch.from_numpy(data.transpose((2, 0, 1)))
     

      
        return data, label

    def __len__(self):
        return self.len

