import math
import random
import numpy as np
#random.seed(0)
#np.random.seed(0)
import cv2
import torch
import skimage
from skimage import transform as sktransf
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image

#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries

num_classes = 5
PD_COLORMAP = [[255, 255, 255], [0, 0, 255], [0, 255, 255],
                [0, 255, 0], [255, 255, 0], [255, 0, 0] ]

# num_classes = 5
# PD_COLORMAP = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],[255, 255, 0], [255, 0, 0] ]

def Index2Color(pred):
    colormap = np.asarray(PD_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def rand_flip(img, label):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

# def rand_flip2(img_s, label_s, label_mask_s, img, label, image_boundary, Objects_first_few):
# #def rand_flip2(img_s, label_s, label_mask_s, img, label):
#     r = random.random()
#     # showIMG(img.transpose((1, 2, 0)))
#     if r < 0.25:
#         return img_s, label_s, label_mask_s, img, label, image_boundary, Objects_first_few
#     elif r < 0.5:
#         return np.flip(img_s, axis=0).copy(), np.flip(label_s, axis=0).copy(), np.flip(label_mask_s, axis=0).copy(), \
#                np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(image_boundary, axis=0).copy(), np.flip(Objects_first_few, axis=0).copy()
#     elif r < 0.75:
#         return np.flip(img_s, axis=1).copy(), np.flip(label_s, axis=1).copy(), np.flip(label_mask_s, axis=1).copy(), \
#                np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(image_boundary, axis=1).copy(), np.flip(Objects_first_few, axis=1).copy()
#     else:
#         return img_s[::-1, ::-1, :].copy(), label_s[::-1, ::-1].copy(), label_mask_s[::-1, ::-1].copy(), \
#                img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), image_boundary[::-1, ::-1].copy(), Objects_first_few[::-1, ::-1].copy()
def rand_flip2(img, label):
#def rand_flip2(img_s, label_s, label_mask_s, img, label):


    # min_scale_factor = 0.5
    # max_scale_factor = 2
    # scale_step_size = 0.25
    #
    # # option 1  #np.random.random_sample() 随机给出设定的size尺寸的位于[0,1)半开半闭区间上的随机数
    # # scale_factor = np.random.random_sample() * (self.max_scale_factor
    # #     - self.min_scale_factor) + self.min_scale_factor
    # # option 2 #在区间[min,max]之之前，以步长为step生成相应可能的值，在随机选择第一个作为缩放的比例
    # num_steps = int((max_scale_factor - min_scale_factor) / scale_step_size + 1)  # 生成多少个可能的值
    # scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps).tolist()
    # np.random.shuffle(scale_factors)
    # scale_factor = scale_factors[0]
    # w = int(round(scale_factor * img.shape[1]))
    # h = int(round(scale_factor * img.shape[0]))
    #
    # img1 = Image.fromarray(img)
    # img1 = F.resize(img1, (w, h), F.InterpolationMode.BILINEAR)
    # label1 = Image.fromarray(label)
    # label1 = F.resize(label1, (w, h), F.InterpolationMode.NEAREST)
    #
    # img = np.array(img1)
    # label = np.array(label1)

    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label
    elif r < 0.5:
        return  np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()





# r = random.random()
    # # showIMG(img.transpose((1, 2, 0)))
    # if r < 0.25:
    #     return img_s, label_s, label_mask_s, img, label
    # elif r < 0.5:
    #     return np.flip(img_s, axis=0).copy(), np.flip(label_s, axis=0).copy(), np.flip(label_mask_s, axis=0).copy(), \
    #            np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy()
    # elif r < 0.75:
    #     return np.flip(img_s, axis=1).copy(), np.flip(label_s, axis=1).copy(), np.flip(label_mask_s, axis=1).copy(), \
    #            np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy()
    # else:
    #     return img_s[::-1, ::-1, :].copy(), label_s[::-1, ::-1].copy(), label_mask_s[::-1, ::-1].copy(), \
    #            img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

def rand_flip_MSC(img_s_0, label_s_0, img_s_1, label_s_1,img_s_2, label_s_2, img, label):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img_s_0, label_s_0, img_s_1, label_s_1,img_s_2, label_s_2, img, label
    elif r < 0.5:
        return np.flip(img_s_0, axis=0).copy(), np.flip(label_s_0, axis=0).copy(), np.flip(img_s_1, axis=0).copy(), np.flip(label_s_1, axis=0).copy(), np.flip(img_s_2, axis=0).copy(), np.flip(label_s_2, axis=0).copy(), np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img_s_0, axis=1).copy(), np.flip(label_s_0, axis=1).copy(), np.flip(img_s_1, axis=1).copy(), np.flip(label_s_1, axis=1).copy(), np.flip(img_s_2, axis=1).copy(), np.flip(label_s_2, axis=1).copy(), np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img_s_0[::-1, ::-1, :].copy(), label_s_0[::-1, ::-1].copy(), img_s_1[::-1, ::-1, :].copy(), label_s_1[::-1, ::-1].copy(), img_s_2[::-1, ::-1, :].copy(), label_s_2[::-1, ::-1].copy(), img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

def rand_flip_mix(img, label, x_s):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label, x_s
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(x_s, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(x_s, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), x_s[::-1, ::-1, :].copy()

def rand_rotate(img, label):
    r = random.randint(0,179)
    # print(r)
    # showIMG(img.transpose((1, 2, 0)))
    img_rotate = np.asarray(sktransf.rotate(img, r, order=1, mode='symmetric',
                                            preserve_range=True), np.float)
    label_rotate = np.asarray(sktransf.rotate(label, r, order=0, mode='constant',
                                               cval=0, preserve_range=True), np.uint8)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return img_rotate, label_rotate

def rand_rotate_crop(img, label):
    r = random.randint(0,179)
    image_height, image_width = img.shape[0:2]
    im_rotated = rotate_image(img, r, order=1)
    l_rotated = rotate_image(label, r, order=0)
    crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(r))
    im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
    l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return im_rotated_cropped, l_rotated_cropped

def rotate_image(image, angle, order=0):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    flag = cv2.INTER_NEAREST
    if order == 1: flag = cv2.INTER_LINEAR
    elif order == 2: flag = cv2.INTER_AREA
    elif order > 2: flag = cv2.INTER_CUBIC

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=flag
    )

    return result

def rand_rotate_mix(img, label, x_s):
    r = random.randint(0,179)
    # print(r)
    # showIMG(img.transpose((1, 2, 0)))
    img_rotate = np.asarray(sktransf.rotate(img, r, order=1, mode='symmetric',
                                            preserve_range=True), np.float)
    label_rotate = np.asarray(sktransf.rotate(label, r, order=0, mode='constant',
                                               cval=0, preserve_range=True), np.uint8)
    x_s_rotate = np.asarray(sktransf.rotate(x_s, r, order=0, mode='symmetric',
                                               cval=0, preserve_range=True), np.uint8)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return img_rotate, label_rotate, x_s_rotate

def create_crops(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img, label,  in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels
    
def create_crops_onlyimgs(ims, size):
    crop_imgs = []
    for img in ims:
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs

def sliding_crop_single_img(img, size):
    crop_imgs = []
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    assert h >= c_h and w >= c_w, "Cannot crop area from image."
    h_rate = h/c_h
    w_rate = w/c_w
    h_times = math.ceil(h_rate)
    w_times = math.ceil(w_rate)
    stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
    stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
    for j in range(h_times):
        for i in range(w_times):
            s_h = int(j*c_h - j*stride_h)
            if(j==(h_times-1)): s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i*c_w - i*stride_w)
            if(i==(w_times-1)): s_w = w - c_w
            e_w = s_w + c_w
            # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
            # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
            crop_imgs.append(img[s_h:e_h, s_w:e_w, :])

    #print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [(mask == i) for i in range(num_classes)]
    mask_onehot = np.array(_mask).astype(np.uint8)
    return mask_onehot

#在输入的标签图像中找到特定条件下的目标区域，并标记出来。
# 通过对指定类别的像素轮廓进行处理，最终得到了处理后的标签图像。
def find_all_mask(label,b):
    #num_class
    mask_onehot = mask_to_onehot(label,6)

    #print('mask_onehot',mask_onehot.shape)  #mask_onehot (6, 192, 192)
    #print('mask_onehot', mask_onehot.shape[0])

    mask = np.zeros_like(label)
    for m in range(mask_onehot.shape[0]): #6
        if m==0 or m==5:
            continue
        l = mask_onehot[m,:,:]

        # l: [[1 1 1... 0 0 1]
        #     [1 1 1... 0 0 0]
        #     [1 1 1... 0 0 0]
        #     ...
        #     [0 0 0... 1 1 1]
        #     [0 0 0... 1 1 1]
        #     [0 0 0... 1 1 1]]

        #print('l:',l)
        contours,_ = cv2.findContours(l,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            a = [x,y,x+w,y+h]
            if (a[1]>=b[3]) or (a[3]<=b[1]) or (a[0]>=b[2]) or (a[2]<=b[0]):
                continue
            else:
                #如果外接矩形与边界b有交集，利用drawContours函数在mask上绘制当前轮廓，
                # 将对应位置的像素值设为1
                cv2.drawContours(mask,contours,i,1,-1)

    label = cv2.add(label,np.zeros(np.shape(label),dtype=np.uint8),mask=mask)
    # img = cv2.add(img,np.zeros(np.shape(img),dtype=np.uint8),mask=mask)

    return label


def SAMAug(tI, mask_generator):
    masks = mask_generator.generate(tI)
    #print('masks:', masks)
    # mask生成返回一个mask列表，其中每个mask是一个包含有关mask的各种数据的字典。这些键包括：
    # segmentation：mask
    # area：mask的面积（以像素为单位）
    # bbox：mask的边界框（XYWH格式）
    # predicted_iou：模型对mask质量的预测
    # point_coords：生成此mask的抽样输入点
    # stability_score：mask质量的附加衡量指标
    # crop_box：用于生成此mask的图像裁剪（XYWH格式）

    # {'segmentation': array([[False, False, False, ..., False, False, False],
    #                         [False, False, False, ..., False, False, False],
    #                         [False, False, False, ..., False, False, False],
    #                         ...,
    #                         [False, False, False, ..., False, False, False],
    #                         [False, False, False, ..., False, False, False],
    #                         [False, False, False, ..., False, False, False]]),
    #  'area': 8668, 'bbox': [183, 61, 72, 171],
    #  'predicted_iou': 1.0210113525390625,
    #  'point_coords': [[236.0, 76.0]],
    #  'stability_score': 0.9673715233802795,
    #  'crop_box': [0, 0, 256, 256]}

    if len(masks) == 0:
        return None, None
    tI = skimage.img_as_float(tI)

    BoundaryPrior = np.zeros((tI.shape[0], tI.shape[1]))
    BoundaryPrior_output = np.zeros((tI.shape[0], tI.shape[1]))

    Objects_first_few = np.zeros((tI.shape[0], tI.shape[1]))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    idx = 1  # K:X中物体对象的最大数量
    for ann in sorted_anns:
        if ann['area'] < 50:  # S:单个物体所能包含的像素个数
            continue
        if idx == 51:
            break
        m = ann['segmentation']
        color_mask = idx
        #print(color_mask)
        Objects_first_few[m] = color_mask  # color_mask:对象标识符索引
        idx = idx + 1

    for maskindex in range(len(masks)):
        thismask = masks[maskindex]['segmentation']
        mask_ = np.zeros((thismask.shape))
        mask_[np.where(thismask == True)] = 1
        BoundaryPrior = BoundaryPrior + find_boundaries(mask_, mode='thick')  # 寻找边界，返回的二进制矩阵中，边界位置的像素值为1，非边界位置的像素值为0。

    BoundaryPrior[np.where(BoundaryPrior > 0)] = 1
    BoundaryPrior_index = np.where(BoundaryPrior > 0)
    Objects_first_few[BoundaryPrior_index] = 0
    BoundaryPrior_output[np.where(BoundaryPrior > 0)] = 255
    #BoundaryPrior_output = BoundaryPrior_output.astype(np.uint8)
    return BoundaryPrior_output, Objects_first_few


def slidding_crop_WC(imgs_s, labels_s, ims, labels, crop_size_global, crop_size_local, scale=8):
    crop_imgs_s = []
    crop_labels_s = []
    crop_labels_mask_s = []
    crop_imgs = []
    crop_labels = []
    c_h = crop_size_local
    c_w = crop_size_local
    label_dims = len(labels[0].shape)
    for img_s, label_s, img, label in zip(imgs_s, labels_s, ims, labels):
        # 从这里开始，先对img（原图）进行裁剪
        h = img.shape[0]
        w = img.shape[1]
        # offest:为了保证裁剪的小图和大图之间的关系，小图在大图的正中间
        offset = int((crop_size_global - crop_size_local) / 2)
        # 判断裁剪窗口的大小会不会超过原影像的大小
        if h < crop_size_local or w < crop_size_local:
            print("Cannot crop area {} from image with size ({}, {})".format(str(crop_size_local), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            continue

        h_rate = h / crop_size_local
        w_rate = w / crop_size_local
        # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整，防止h_rate、w_rate是非整数
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times == 1:
            stride_h = 0
        else:
            # 确定滑动窗口的滑动步幅，沿h方向
            # 虽然知道是这么回事儿吧，但是，没看懂这个公式几个意思，大概算了一下，c_h=256时，stride_h=7
            # 向后面看了一下，这个stride_h应该是重叠的像素个数
            stride_h = math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
        if w_times == 1:
            stride_w = 0
        else:
            # 确定滑动窗口的滑动步幅，沿w方向
            stride_w = math.ceil(c_w * (w_times - w_rate) / (w_times - 1))
        # 开始裁剪
        for j in range(h_times):
            for i in range(w_times):
                # 确定原图上的裁剪窗口的左上角（s_h,s_w）和右下角（e_h,e_w）
                s_h = int(j * c_h - j * stride_h)
                if (j == (h_times - 1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i * c_w - i * stride_w)
                if (i == (w_times - 1)): s_w = w - c_w
                e_w = s_w + c_w
                # 确定缩放图上的裁剪窗口的左上角（s_h_s,s_w_s）和右下角（e_h_s,e_w_s）
                # 看到这儿感觉，它裁剪出来的大图和小图的关系不是小图在大图正中间，而是小图在大图的左上角，右边和下边是多出来的语义信息
                s_h_s = int(s_h / scale)
                s_w_s = int(s_w / scale)
                e_h_s = int((e_h + 2 * offset) / scale)
                e_w_s = int((e_w + 2 * offset) / scale)
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                if label_dims == 2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                    l_s = label_s[s_h_s:e_h_s, s_w_s:e_w_s]

                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                    crop_labels_s.append(label_s[s_h_s:e_h_s, s_w_s:e_w_s, :])

                i_s = img_s[s_h_s:e_h_s, s_w_s:e_w_s, :]
                l_m_s = find_all_mask(l_s, [64, 64, 128, 128])
                crop_imgs_s.append(i_s)
                crop_labels_s.append(l_s)
                crop_labels_mask_s.append(l_m_s)

    print('Sliding crop finished. %d images created.' % len(crop_imgs))
    return crop_imgs_s, crop_labels_s, crop_labels_mask_s, crop_imgs, crop_labels


#-----------------------------------------------------train----------------------------------------------------------
# def slidding_crop_WC(imgs_s, labels_s, ims, labels, crop_size_global, crop_size_local, scale=8, mode='train'):
#     crop_imgs_s = []
#     crop_labels_s = []
#     crop_labels_mask_s = []
#     crop_imgs = []
#     crop_labels = []
#
#     image_boundary = []
#     Objects_first_few = []
#     n=1
#
#     c_h = crop_size_local
#     c_w = crop_size_local
#     label_dims = len(labels[0].shape)
#
# #add
#     # device = 'cuda'
#     # sam = sam_model_registry["vit_h"](checkpoint="/root/autodl-tmp/TCNet_GID/model/sam/sam_vit_h_4b8939.pth")
#     # sam.to(device=device)
#     #
#     # mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.96)
#
#     for img_s, label_s, img, label in zip(imgs_s, labels_s, ims, labels):
#         # 从这里开始，先对img（原图）进行裁剪
#         h = img.shape[0]
#         w = img.shape[1]
#         # offest:为了保证裁剪的小图和大图之间的关系，小图在大图的正中间
#         offset = int((crop_size_global-crop_size_local)/2)
#         # 判断裁剪窗口的大小会不会超过原影像的大小
#         if h < crop_size_local or w < crop_size_local:
#             print("Cannot crop area {} from image with size ({}, {})".format(str(crop_size_local), h, w))
#             crop_imgs.append(img)
#             crop_labels.append(label)
#             continue
#
#         h_rate = h/crop_size_local
#         w_rate = w/crop_size_local
#         # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整，防止h_rate、w_rate是非整数
#         h_times = math.ceil(h_rate)
#         w_times = math.ceil(w_rate)
#         if h_times==1:
#             stride_h=0
#         else:
#             # 确定滑动窗口的滑动步幅，沿h方向
#             # 虽然知道是这么回事儿吧，但是，没看懂这个公式几个意思，大概算了一下，c_h=256时，stride_h=7
#             # 向后面看了一下，这个stride_h应该是重叠的像素个数
#             stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
#         if w_times==1:
#             stride_w=0
#         else:
#             # 确定滑动窗口的滑动步幅，沿w方向
#             stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
#         # 开始裁剪
#         for j in range(h_times):
#             for i in range(w_times):
#                 # 确定原图上的裁剪窗口的左上角（s_h,s_w）和右下角（e_h,e_w）
#                 # 确定缩放图上的裁剪窗口的左上角（s_h_s,s_w_s）和右下角（e_h_s,e_w_s）
#                 # 看到这儿感觉，它裁剪出来的大图和小图的关系不是小图在大图正中间，而是小图在大图的左上角，右边和下边是多出来的语义信息
#                 s_h = int(j*c_h - j*stride_h)
#                 if(j==(h_times-1)): s_h = h - c_h
#                 e_h = s_h + c_h
#                 s_w = int(i*c_w - i*stride_w)
#                 if(i==(w_times-1)): s_w = w - c_w
#                 e_w = s_w + c_w
#                 s_h_s = int(s_h/scale)
#                 s_w_s = int(s_w/scale)
#                 e_h_s = int((e_h+2*offset)/scale)
#                 e_w_s = int((e_w+2*offset)/scale)
#                 # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
#                 # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
#
#
#                 BoundaryPrior_output, Objects_first = SAMAug(img[s_h:e_h, s_w:e_w, :], mask_generator)
#                 if BoundaryPrior_output is not None:
#                     image_boundary.append(BoundaryPrior_output)
#                     img1=Image.fromarray(np.uint8(BoundaryPrior_output))
#                     img1.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/boundary/'+ str(n) +'.tif')
#
#                     Objects_first_few.append(Objects_first)
#                     img2 = Image.fromarray(np.uint8(Objects_first))
#                     img2.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/object/' + str(n) + '.tif')
#
#                     if n%100==0:
#                         print('crop:',n)
#
#
#                     crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
#                     img3 = Image.fromarray(np.uint8(img[s_h:e_h, s_w:e_w, :]))
#                     img3.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/image/' + str(n) + '.tif')
#
#                     #将array转化为图片保存到指定路径
#                     # if(len(crop_imgs)==316):
#                     #     img4=Image.fromarray(np.uint8(img[s_h:e_h, s_w:e_w, :]))
#                     #     img4.save('/root/autodl-tmp/TCNet_GID/results/315.tif')
#
#                     if label_dims==2:
#                         crop_labels.append(label[s_h:e_h, s_w:e_w])
#                         l_s = label_s[s_h_s:e_h_s, s_w_s:e_w_s]
#
#                     else:
#                         crop_labels.append(label[s_h:e_h, s_w:e_w, :])
#                         crop_labels_s.append(label_s[s_h_s:e_h_s, s_w_s:e_w_s, :])
#
#                     img4 = Image.fromarray(np.uint8(Index2Color(label[s_h:e_h, s_w:e_w])))
#                     img4.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/label/' + str(n) + '.tif')
#
#                     i_s = img_s[s_h_s:e_h_s, s_w_s:e_w_s, :]
#                     l_m_s = find_all_mask(l_s,[64,64,128,128])
#                     #print('l_m_s:',l_m_s.shape)
#
#                     crop_imgs_s.append(i_s)
#                     img5 = Image.fromarray(np.uint8(i_s))
#                     img5.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/image_s/' + str(n) + '.tif')
#
#                     crop_labels_s.append(l_s)
#                     #print('l_s', l_s)
#                     img6 = Image.fromarray(np.uint8(Index2Color(l_s)))
#                     img6.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/label_s/' + str(n) + '.tif')
#
#                     #print('l_m_s',l_m_s)
#                     crop_labels_mask_s.append(l_m_s)
#                     img7 = Image.fromarray(np.uint8(Index2Color(l_m_s)))
#                     #img7 = Image.fromarray(np.uint8(label[s_h:e_h, s_w:e_w]))
#                     img7.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/'+ mode +'/label_mask_s/' + str(n) + '.tif')
#
#                     n+=1
#             #     if n==2:
#             #         break
#             # if n==2:
#             #     break
#
#     print('Sliding crop finished. %d images created.' % len(crop_imgs))
#
# #测试问题
#     # print('316',crop_imgs[315].shape)
#     # BoundaryPrior_output, Objects_first = SAMAug(crop_imgs[313], mask_generator)
#     # boundary = torch.from_numpy(BoundaryPrior_output)
#     # print('ok')
#     # print('315', crop_imgs[314].shape)
#     # img1 = Image.open("/root/autodl-tmp/TCNet_GID/results/314.tif")
#     # imgArray = np.array(img1)
#     # BoundaryPrior_output, Objects_first = SAMAug(imgArray, mask_generator)
#     # boundary = torch.from_numpy(BoundaryPrior_output)
#     # print('ok')
#
#
#     return crop_imgs_s, crop_labels_s,crop_labels_mask_s, crop_imgs, crop_labels, image_boundary, Objects_first_few
#

#--------------------------------------------------test-------------------------------------------------------
# def slidding_crop_WC(imgs_s, labels_s, ims, labels, crop_size_global, crop_size_local, scale=8, mode='train'):
#     crop_imgs_s = []
#     crop_labels_s = []
#     crop_labels_mask_s = []
#     crop_imgs = []
#     crop_labels = []
#
#     image_boundary = []
#     Objects_first_few = []
#     n = 1
#
#     c_h = crop_size_local
#     c_w = crop_size_local
#     label_dims = len(labels[0].shape)
#
#
#     for img_s, label_s, img, label in zip(imgs_s, labels_s, ims, labels):
#         # 从这里开始，先对img（原图）进行裁剪
#         h = img.shape[0]
#         w = img.shape[1]
#         # offest:为了保证裁剪的小图和大图之间的关系，小图在大图的正中间
#         offset = int((crop_size_global - crop_size_local) / 2)
#         # 判断裁剪窗口的大小会不会超过原影像的大小
#         if h < crop_size_local or w < crop_size_local:
#             print("Cannot crop area {} from image with size ({}, {})".format(str(crop_size_local), h, w))
#             crop_imgs.append(img)
#             crop_labels.append(label)
#             continue
#
#         h_rate = h / crop_size_local
#         w_rate = w / crop_size_local
#         # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整，防止h_rate、w_rate是非整数
#         h_times = math.ceil(h_rate)
#         w_times = math.ceil(w_rate)
#         if h_times == 1:
#             stride_h = 0
#         else:
#             # 确定滑动窗口的滑动步幅，沿h方向
#             # 虽然知道是这么回事儿吧，但是，没看懂这个公式几个意思，大概算了一下，c_h=256时，stride_h=7
#             # 向后面看了一下，这个stride_h应该是重叠的像素个数
#             stride_h = math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
#         if w_times == 1:
#             stride_w = 0
#         else:
#             # 确定滑动窗口的滑动步幅，沿w方向
#             stride_w = math.ceil(c_w * (w_times - w_rate) / (w_times - 1))
#         # 开始裁剪
#         for j in range(h_times):
#             for i in range(w_times):
#                 # 确定原图上的裁剪窗口的左上角（s_h,s_w）和右下角（e_h,e_w）
#                 # 确定缩放图上的裁剪窗口的左上角（s_h_s,s_w_s）和右下角（e_h_s,e_w_s）
#                 # 看到这儿感觉，它裁剪出来的大图和小图的关系不是小图在大图正中间，而是小图在大图的左上角，右边和下边是多出来的语义信息
#                 s_h = int(j * c_h - j * stride_h)
#                 if (j == (h_times - 1)): s_h = h - c_h
#                 e_h = s_h + c_h
#                 s_w = int(i * c_w - i * stride_w)
#                 if (i == (w_times - 1)): s_w = w - c_w
#                 e_w = s_w + c_w
#                 s_h_s = int(s_h / scale)
#                 s_w_s = int(s_w / scale)
#                 e_h_s = int((e_h + 2 * offset) / scale)
#                 e_w_s = int((e_w + 2 * offset) / scale)
#                 # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
#                 # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
#
#
#                 if n % 100 == 0:
#                     print('crop:', n)
#
#                 crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
#                 img3 = Image.fromarray(np.uint8(img[s_h:e_h, s_w:e_w, :]))
#                 img3.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/test/image/' + str(n) + '.tif')
#
#                 # 将array转化为图片保存到指定路径
#                 # if(len(crop_imgs)==316):
#                 #     img4=Image.fromarray(np.uint8(img[s_h:e_h, s_w:e_w, :]))
#                 #     img4.save('/root/autodl-tmp/TCNet_GID/results/315.tif')
#
#                 if label_dims == 2:
#                     crop_labels.append(label[s_h:e_h, s_w:e_w])
#                     l_s = label_s[s_h_s:e_h_s, s_w_s:e_w_s]
#
#                 else:
#                     crop_labels.append(label[s_h:e_h, s_w:e_w, :])
#                     crop_labels_s.append(label_s[s_h_s:e_h_s, s_w_s:e_w_s, :])
#
#                 img4 = Image.fromarray(np.uint8(Index2Color(label[s_h:e_h, s_w:e_w])))
#                 img4.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/test/label/' + str(n) + '.tif')
#
#                 i_s = img_s[s_h_s:e_h_s, s_w_s:e_w_s, :]
#                 l_m_s = find_all_mask(l_s, [64, 64, 128, 128])
#                 # print('l_m_s:',l_m_s.shape)
#
#                 crop_imgs_s.append(i_s)
#                 img5 = Image.fromarray(np.uint8(i_s))
#                 img5.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/test/image_s/' + str(n) + '.tif')
#
#                 crop_labels_s.append(l_s)
#                 # print('l_s', l_s)
#                 img6 = Image.fromarray(np.uint8(Index2Color(l_s)))
#                 img6.save('/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/test/label_s/' + str(n) + '.tif')
#
#                 # print('l_m_s',l_m_s)
#                 crop_labels_mask_s.append(l_m_s)
#                 img7 = Image.fromarray(np.uint8(Index2Color(l_m_s)))
#                 # img7 = Image.fromarray(np.uint8(label[s_h:e_h, s_w:e_w]))
#                 img7.save(
#                     '/root/autodl-tmp/TCNet_GID/dataset/potsdam/data/test/label_mask_s/' + str(n) + '.tif')
#
#                 n += 1
#             #     if n==2:
#             #         break
#             # if n==2:
#             #     break
#
#     print('Sliding crop finished. %d images created.' % len(crop_imgs))
#
#     # 测试问题
#     # print('316',crop_imgs[315].shape)
#     # BoundaryPrior_output, Objects_first = SAMAug(crop_imgs[313], mask_generator)
#     # boundary = torch.from_numpy(BoundaryPrior_output)
#     # print('ok')
#     # print('315', crop_imgs[314].shape)
#     # img1 = Image.open("/root/autodl-tmp/TCNet_GID/results/314.tif")
#     # imgArray = np.array(img1)
#     # BoundaryPrior_output, Objects_first = SAMAug(imgArray, mask_generator)
#     # boundary = torch.from_numpy(BoundaryPrior_output)
#     # print('ok')
#
#     return crop_imgs_s, crop_labels_s, crop_labels_mask_s, crop_imgs, crop_labels


def slidding_crop_WC_zl(imgs_s, labels_s, ims, labels, crop_size_global, crop_size_local, scale=8):
    crop_imgs_s = []
    crop_labels_s = []
    crop_imgs = []
    crop_labels = []
    c_h = crop_size_local
    c_w = crop_size_local
    label_dims = len(labels[0].shape)
    for img_s, label_s, img, label in zip(imgs_s, labels_s, ims, labels):
        # 从这里开始，先对img（原图）进行裁剪
        h = img.shape[0]
        w = img.shape[1]
        # offest:为了保证裁剪的小图和大图之间的关系，小图在大图的正中间
        offset = int((crop_size_global-crop_size_local)/2)
        # 判断裁剪窗口的大小会不会超过原影像的大小
        if h < crop_size_local or w < crop_size_local:
            print("Cannot crop area {} from image with size ({}, {})".format(str(crop_size_local), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            continue
        
        h_rate = h/crop_size_local
        w_rate = w/crop_size_local
        # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整，防止h_rate、w_rate是非整数
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            # 确定滑动窗口的滑动步幅，沿h方向
            # 虽然知道是这么回事儿吧，但是，没看懂这个公式几个意思，大概算了一下，c_h=256时，stride_h=7
            # 向后面看了一下，这个stride_h应该是重叠的像素个数
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            # 确定滑动窗口的滑动步幅，沿w方向
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        # 开始裁剪
        for j in range(h_times):
            for i in range(w_times):
                # 确定原图上的裁剪窗口的左上角（s_h,s_w）和右下角（e_h,e_w）
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # 确定缩放图上的裁剪窗口的左上角（s_h_s,s_w_s）和右下角（e_h_s,e_w_s）
                # 看到这儿感觉，它裁剪出来的大图和小图的关系不是小图在大图正中间，而是小图在大图的左上角，右边和下边是多出来的语义信息
                s_h_s = int(s_h/scale)
                s_w_s = int(s_w/scale)
                e_h_s = s_h_s + c_h
                e_w_s = s_w_s + c_w
                if e_h_s>img_s.shape[0]:
                    s_h_s = img_s.shape[0] - c_h
                    e_h_s = s_h_s + c_h
                if e_w_s>img_s.shape[1]:
                    s_w_s = img_s.shape[1] - c_w
                    e_w_s = s_w_s + c_w                   
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                crop_imgs_s.append(img_s[s_h_s:e_h_s, s_w_s:e_w_s, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                    crop_labels_s.append(label_s[s_h_s:e_h_s, s_w_s:e_w_s])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                    crop_labels_s.append(label_s[s_h_s:e_h_s, s_w_s:e_w_s, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs_s, crop_labels_s, crop_imgs, crop_labels


def center_crop(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    for img, label in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h = int(h/2 - c_h/2)
        e_h = s_h + c_h
        s_w = int(w/2 - c_w/2)
        e_w = s_w + c_w
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Center crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels

def five_crop(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    for img, label in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h = int(h/2 - c_h/2)
        e_h = s_h + c_h
        s_w = int(w/2 - c_w/2)
        e_w = s_w + c_w
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

        crop_imgs.append(img[0:c_h, 0:c_w, :])
        crop_labels.append(label[0:c_h, 0:c_w, :])
        crop_imgs.append(img[h-c_h:h, w-c_w:w, :])
        crop_labels.append(label[h-c_h:h, w-c_w:w, :])
        crop_imgs.append(img[0:c_h, w-c_w:w, :])
        crop_labels.append(label[0:c_h, w-c_w:w, :])
        crop_imgs.append(img[h-c_h:h, 0:c_w, :])
        crop_labels.append(label[h-c_h:h, 0:c_w, :])

    print('Five crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels

def data_padding(imgs, labels, scale=32):
    for idx, img in enumerate(imgs):
        label = labels[idx]
        shape_before = img.shape
        h, w = img.shape[:2]
        h_padding = h%scale
        w_padding = w%scale
        need_padding = h_padding>0 and w_padding>0
        if need_padding:
            h_padding = (scale-h_padding)/2
            h_padding1 = math.ceil(h_padding)
            h_padding2 = math.floor(h_padding)
            
            w_padding = (scale-w_padding)/2
            w_padding1 = math.ceil(w_padding)
            w_padding2 = math.floor(w_padding)
            img = np.pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
            label = np.pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'constant')
            shape_after = img.shape
            print('img padding: [%d, %d]->[%d, %d]'%(shape_before[0],shape_before[1],shape_after[0],shape_after[1]))
            imgs[idx] = img
            labels[idx] = label
    return imgs, labels

#将输入的图像和标签填充到指定的尺寸大小
def data_padding_fixsize(imgs, labels, size):
    for idx, img in enumerate(imgs):
        label = labels[idx]
        h, w = img.shape[:2]  #获取当前图像的高度和宽度
        h_padding = size[0]
        w_padding = size[1]

        #对高度的填充值进行向上取整和向下取整
        h_padding1 = math.ceil(h_padding)
        h_padding2 = math.floor(h_padding)

        #对宽度的填充值进行向上取整和向下取整
        w_padding1 = math.ceil(w_padding)
        w_padding2 = math.floor(w_padding)
        
        img = np.pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
        label = np.pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'constant')
        imgs[idx] = img
        labels[idx] = label
    return imgs, labels

def five_crop_mix(ims, labels, x_s, size, scale=8):
    crop_imgs = []
    crop_labels = []
    crop_xs = []
    for img, label, x_s in zip(ims, labels, x_s):
        h = img.shape[0]
        w = img.shape[1]
        h_s = int(h/scale)
        w_s = int(w/scale)
        c_h = size[0]
        c_w = size[1]
        c_h_s = int(c_h/scale)
        c_w_s = int(c_w/scale)
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h_s = int(h_s/2 - c_h_s/2)
        e_h_s = s_h_s + c_h_s
        s_w_s = int(w_s/2 - c_w_s/2)
        e_w_s = s_w_s + c_w_s
        s_h = s_h_s*scale
        s_w = s_w_s*scale
        e_h = s_h+c_h
        e_w = s_w+c_w
        
        crop_xs.append(x_s[:, s_h_s:e_h_s, s_w_s:e_w_s])
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

        crop_xs.append(x_s[:, :c_h_s, :c_w_s])
        crop_imgs.append(img[:c_h, :c_w, :])
        crop_labels.append(label[:c_h, :c_w, :])
        
        crop_xs.append(x_s[:, -c_h_s:, -c_w_s:])
        crop_imgs.append(img[-c_h:, -c_w:, :])
        crop_labels.append(label[-c_h:, -c_w:, :])
        
        crop_xs.append(x_s[:, :c_h_s, -c_w_s:])
        crop_imgs.append(img[:c_h, -c_w:, :])
        crop_labels.append(label[:c_h, -c_w:, :])
        
        crop_xs.append(x_s[:, -c_h_s:, :c_w_s])
        crop_imgs.append(img[-c_h:, :c_w, :])
        crop_labels.append(label[-c_h:, :c_w, :])

    print('Five crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_xs

def sliding_crop(img, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        crop_imgs = []
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                crop_im = img[s_h:e_h, s_w:e_w, :]
                crop_imgs.append(crop_im)

                # crop_imgs_f = []
                # for im in crop_imgs:
                #     crop_imgs_f.append(cv2.flip(im, -1))

                # crops = np.concatenate((np.array(crop_imgs)), axis=0)
                # print(crops.shape)
        return crop_imgs

def random_crop(img, label, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im, crop_label

def random_crop2(img_s, label_s, img, label, crop_size_global, crop_size_local, scale):
    # print(img.shape)
    # 记录缩放后的图像的大小
    h_s, w_s = img_s.shape[:2]  #640*640
    # 记录原图大小
    h, w = img.shape[:2]
    padding_size = int((crop_size_global-crop_size_local)/scale)   #(768-256)/4=512/4=128
    # 在缩放后的图像上裁剪的实际大小
    crop_size_s = int(crop_size_global/scale)   #768/4=192
    
    if h_s < crop_size_s or w_s < crop_size_s or h < crop_size_local or w < crop_size_local:
        print('Crop failed. Size error.')
    else:
        # 生成随机数
        h_seed = random.randint(0, h_s-crop_size_s)
        w_seed = random.randint(0, w_s-crop_size_s)
        # 根据生成的随机数，确定裁剪窗口
        start_h_s = h_seed
        end_h_s = start_h_s+crop_size_s
        start_w_s = w_seed
        end_w_s = start_w_s+crop_size_s 
        # 对缩放后的图像进行裁剪       
        crop_im_s = img_s[start_h_s:end_h_s, start_w_s:end_w_s, :]
        crop_label_s = label_s[start_h_s:end_h_s, start_w_s:end_w_s]

        crop_label_mask_s = find_all_mask(crop_label_s, [60,60,130,130])
        #print('start_h_s%d, end_h_s%d, start_w_s%d, end_w_s%d'%(start_h_s,end_h_s,start_w_s,end_w_s))
        # 根据生成的随机数，反算回在原图的位置，确定裁剪小窗口的大小
        start_h = h_seed*scale
        end_h = start_h+crop_size_local
        if end_h>h:
            start_h = h - crop_size_local
            end_h = h
        start_w = w_seed*scale
        end_w = start_w+crop_size_local
        if end_w>w:
            start_w = w - crop_size_local
            end_w = w
        #print('start_h%d, end_h%d, start_w%d, end_w%d'%(start_h,end_h,start_w,end_w))
        crop_im = img[start_h:end_h, start_w:end_w, :]
        crop_label = label[start_h:end_h, start_w:end_w]

        #crop_im_s为
        return crop_im_s, crop_label_s,crop_label_mask_s, crop_im, crop_label


def random_crop_zl(img_s, label_s, img, label, crop_size_global, crop_size_local, scale):
    # print(img.shape)
    # 记录缩放后的图像的大小
    h_s, w_s = img_s.shape[:2]
    # 记录原图大小
    h, w = img.shape[:2]
    padding_size = int((crop_size_global-crop_size_local)/scale)
    # 强制命令裁剪大小一致
    crop_size_s = crop_size_local
    
    if h_s < crop_size_s or w_s < crop_size_s or h < crop_size_local or w < crop_size_local:
        print('Crop failed. Size error.')
    else:
        # 生成随机数
        h_seed = random.randint(0, h_s-crop_size_s)
        w_seed = random.randint(0, w_s-crop_size_s)
        # 根据生成的随机数，确定裁剪窗口
        start_h_s = h_seed
        end_h_s = start_h_s+crop_size_s
        start_w_s = w_seed
        end_w_s = start_w_s+crop_size_s 
        # 对缩放后的图像进行裁剪       
        crop_im_s = img_s[start_h_s:end_h_s, start_w_s:end_w_s, :]
        crop_label_s = label_s[start_h_s:end_h_s, start_w_s:end_w_s]
        #print('start_h_s%d, end_h_s%d, start_w_s%d, end_w_s%d'%(start_h_s,end_h_s,start_w_s,end_w_s))
        # 根据生成的随机数，反算回在原图的位置，确定裁剪小窗口的大小
        start_h = h_seed*scale
        end_h = start_h+crop_size_local
        start_w = w_seed*scale
        end_w = start_w+crop_size_local
        #print('start_h%d, end_h%d, start_w%d, end_w%d'%(start_h,end_h,start_w,end_w))
        crop_im = img[start_h:end_h, start_w:end_w, :]
        crop_label = label[start_h:end_h, start_w:end_w]
        
        return crop_im_s, crop_label_s, crop_im, crop_label

def recombine_img(img, t, k_r):
    """
    img: 等待重新组装的影像
    t: 按照行、列方向各分成几份
    k_r: 空洞系数
    """
    # 核心思想：img代表最大范围，但是整块输入进去太大了，因此将它按照行列方向分割成t*t块，以最中间那块为中心向外扩散，第一次取最中间的9块，第二次令空洞系数为1，隔1块取一个以此类推

    h, w= img.shape[:2]
    # 这一步是判断将影像按照size可以分成多少份
    size = int(h / t)
    # 定义一个重组之后的影像
    if len(img.shape) == 3:
        output = np.zeros([size * 3, size * 3, img.shape[2]], np.uint8)
        # 将原影像分割然后按照需求重组
        for i in range(3):
            for j in range(3):
                tttt = img[(int(t / 2) + (i - 1) * (1 + k_r)) * size:(1 + int(t / 2) + (i - 1) * (1 + k_r)) * size,
                       (int(t / 2) + (j - 1) * (1 + k_r)) * size:(1 + int(t / 2) + (
                               j - 1) * (1 + k_r)) * size, :]
                output[(i * size):((i + 1) * size), (j * size):((j + 1) * size), :] = tttt
    else:
        output = np.zeros([size * 3, size * 3], np.uint8)
        # 将原影像分割然后按照需求重组
        for i in range(3):
            for j in range(3):
                tttt = img[(int(t / 2) + (i - 1) * (1 + k_r)) * size:(1 + int(t / 2) + (i - 1) * (1 + k_r)) * size,
                       (int(t / 2) + (j - 1) * (1 + k_r)) * size:(1 + int(t / 2) + (
                               j - 1) * (1 + k_r)) * size]
                output[(i * size):((i + 1) * size), (j * size):((j + 1) * size)] = tttt
    return output


def random_crop_MS(img_s, label_s, img, label, crop_size_global, crop_size_local, scale):
    # img_s: 缩放后的图像
    # img：原图
    # 记录缩放后的图像的大小
    h_s, w_s = img_s.shape[:2]
    # 记录原图大小
    h, w = img.shape[:2]
    padding_size = int((crop_size_global - crop_size_local) / scale)
    # 在缩放后的图像上裁剪的实际大小
    crop_size_s = int(crop_size_global / scale)

    if h_s < crop_size_s or w_s < crop_size_s or h < crop_size_local or w < crop_size_local:
        print('Crop failed. Size error.')
    else:
        # 生成随机数
        h_seed = random.randint(0, h_s - crop_size_s)
        w_seed = random.randint(0, w_s - crop_size_s)
        # t表示的是将crop_im_s_ori按照行、列进行几等分
        t = 7
        
        # 只求最中心那一块
        start_h = int((h_seed + (crop_size_s * int((2 + 1) / t))) * scale)
        end_h = start_h + crop_size_local
        start_w = int((w_seed + (crop_size_s * int((2 + 1) / t))) * scale)
        end_w = start_w + crop_size_local
        crop_im = img[start_h:end_h, start_w:end_w, :]
        crop_label = label[start_h:end_h, start_w:end_w]
        print('crop_im.shape',crop_im.shape)

        # 根据生成的随机数，确定裁剪窗口
        start_h_s = h_seed
        end_h_s = start_h_s + crop_size_s
        start_w_s = w_seed
        end_w_s = start_w_s + crop_size_s
        # 对缩放后的图像进行裁剪
        crop_im_s_ori = img_s[start_h_s:end_h_s, start_w_s:end_w_s, :]
        crop_label_s_ori = label_s[start_h_s:end_h_s, start_w_s:end_w_s]


        # 空洞率为0
        crop_im_s_0 = recombine_img(crop_im_s_ori, t, 0)
        crop_label_s_0 = recombine_img(crop_label_s_ori, t, 0)
        # 空洞率为1
        crop_im_s_1 = recombine_img(crop_im_s_ori, t, 1)
        crop_label_s_1 = recombine_img(crop_label_s_ori, t, 1)
        # 空洞率为2
        crop_im_s_2 = recombine_img(crop_im_s_ori, t, 2)
        crop_label_s_2 = recombine_img(crop_label_s_ori, t, 2)
        # 放到一个列表里面
        crop_im_s = [crop_im_s_0, crop_im_s_1, crop_im_s_2]
        crop_label_s = [crop_label_s_0, crop_label_s_1, crop_label_s_2]
        print('crop_im_s_0.shape',crop_im_s_0.shape)

        # # 理论上，我们要找大范围最中间的那个区域，把它裁成3*3
        # # 根据生成的随机数，反算回在原图的位置，确定裁剪小窗口的大小
        # # start_h_s+crop_size_s
        # crop_im = []
        # crop_label = []
        # for i in range(3):
        #     start_h = int((h_seed + (crop_size_s * (2 + i) / t)) * scale)
        #     end_h = start_h + crop_size_local
        #     for j in range(3):
        #         start_w = int((w_seed + (crop_size_s * (2 + j) / t)) * scale)
        #         end_w = start_w + crop_size_local
        #         crop_im_i = img[start_h:end_h, start_w:end_w, :]
        #         crop_label_i = label[start_h:end_h, start_w:end_w]
        #         crop_im.append(crop_im_i)
        #         crop_label.append(crop_label_i)



        return crop_im_s, crop_label_s, crop_im, crop_label

def random_crop_mix(img, label, x_s, size, scale=8):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    c_h_s = int(c_h/scale)
    c_w_s = int(c_w/scale)
    h_times = int(h/scale - c_h_s)
    w_times = int(w/scale - c_w_s)
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h_s = random.randint(0, h_times)
        s_h = s_h_s * scale
        s_w_s = random.randint(0, w_times)
        s_w = s_w_s * scale
        e_h_s = s_h_s + c_h_s
        e_w_s = s_w_s + c_w_s
        e_h = s_h + c_h
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        crop_xs = x_s[:, s_h_s:e_h_s, s_w_s:e_w_s]
        # print('%d %d %d %d' % (s_h, e_h, s_w, e_w))
        # print('%d %d %d %d' % (s_h_s, e_h_s, s_w_s, e_w_s))
        return crop_im, crop_label, crop_xs

def create_crops_mix(ims, labels, x_s, size, scale=1/8):
    crop_imgs = []
    crop_labels = []
    crop_x_s = []
    for img, label, x in zip(ims, labels, x_s):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        c_h_s = int(c_h*scale)
        c_w_s = int(c_w*scale)
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                s_h_s = int(s_h*scale)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                e_h_s = s_h_s + c_h_s
                s_w = int(i*c_w - i*stride_w)
                s_w_s = int(s_w*scale)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                e_w_s = s_w_s + c_w_s
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                crop_x_s.append(x[:, s_h_s:e_h_s, s_w_s:e_w_s])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_x_s

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def Rotate_Aug(imgs, labels, step=20, start_angle=20, max_angle=179):
    for idx in range(len(imgs)):
        im = imgs[idx]
        l = labels[idx]
        image_height, image_width = im.shape[0:2]
        for i in range(start_angle, max_angle, step):
            im_rotated = rotate_image(im, i, order=3)
            l_rotated  = rotate_image(l,  i, order=0)
            crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(i))
            im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
            l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
            imgs.append(im_rotated_cropped)
            labels.append(l_rotated_cropped)
        print('Img %d rotated.'%idx)
    print('Rotation finished. %d images in total.'%len(imgs))
    return imgs, labels

def Rotate_Aug_S(im, l, step=20, start_angle=15, max_angle=89):
    imgs = []
    labels = []
    image_height, image_width = im.shape[0:2]
    for i in range(start_angle, max_angle, step):
        im_rotated = rotate_image(im, i, order=1)
        l_rotated  = rotate_image(l,  i, order=0)
        crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(i))
        im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
        l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
        imgs.append(im_rotated_cropped)
        labels.append(l_rotated_cropped)
    print('Rotation finished. %d images added.'%len(imgs))
    return imgs, labels
