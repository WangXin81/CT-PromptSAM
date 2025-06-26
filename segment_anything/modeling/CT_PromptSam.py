# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

from models.CT_Net import CT_Net as Net

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling import utils

class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, ignore_index=255,alpha=0.25, gamma=2, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        valid_mask=torch.where(targets!=ignore_index)
        targets=targets[valid_mask].unsqueeze(0)
        inputs=inputs[valid_mask].unsqueeze(0)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, ignore_index=255,smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        valid_mask=torch.where(targets!=ignore_index)
        targets=targets[valid_mask].unsqueeze(0)
        inputs=inputs[valid_mask].unsqueeze(0)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class CT_PromptSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.net = Net()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.SAM_focal_loss = FocalLoss()
        self.SAM_dice_loss = DiceLoss()
        self.SAM_MSE_loss = torch.nn.MSELoss()

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # def find_contours(sub_mask):
    #     _, thresh = cv2.threshold(sub_mask, 0, 255, cv2.THRESH_BINARY)
    #     return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    #
    # def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    #     pred_mask = (pred_mask >= 0.5).float()
    #     intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    #     union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    #     epsilon = 1e-7
    #     batch_iou = intersection / (union + epsilon)
    #     batch_iou = batch_iou.unsqueeze(1)
    #     return batch_iou

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        #gt: List[Dict[str, Any]],
        # res: List[Dict[str, Any]],
        # seg_logits: List[Dict[str, Any]],
        #flag: bool,
        #multimask_output: bool,
        #multimask_output: False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        #input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        res, seg_logits = self.net(batched_input)    #seg_logits: torch.Size([B, 6, 256, 256])
        pred_sem_seg = seg_logits.argmax(dim=1, keepdim=True)   #pred_sem_seg: torch.Size([B, 1, 256, 256])

        outputs = []
        i = 0
        device = "cuda:0"

        layer = torch.nn.AvgPool2d(4, stride=4)


#mask

        # for image_record,  pred_sem_seg_0, res_0, seg_logits_0 in zip(batched_input,   pred_sem_seg, res,seg_logits):
        #
        #     one_mask_logits = seg_logits_0.reshape(1, 6, 256, 256)   #one_mask_logits: torch.Size([1, 6, 256, 256])
        #     one_mask_logits = layer(one_mask_logits)
        #
        #     with torch.no_grad():
        #         # gt_channel = gt[:, :, channel]
        #         image_record = image_record.reshape((1, 3, 256, 256))
        #         image_embedding = self.image_encoder(image_record)
        #
        #     with torch.no_grad():
        #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #             points=None,
        #             boxes=None,
        #             masks=one_mask_logits,
        #             #masks=None,
        #         )
        #
        #         #dense_embeddings = res_0.reshape(1, 256, 16, 16)
        #
        #     mask_predictions, _ = self.mask_decoder(
        #         image_embeddings=image_embedding.to(device),
        #         image_pe=self.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=False,
        #     )
        #     predictions = F.interpolate(mask_predictions,
        #                                      size=(batched_input.shape[2], batched_input.shape[3]),
        #                                      mode="bilinear",
        #                                      align_corners=False)


#with mask and points
        # for image_record,  pred_sem_seg_0, res_0, seg_logits_0 in zip(batched_input,   pred_sem_seg, res,seg_logits):
        #     # if "point_coords" in image_record:
        #     #     points = (image_record["point_coords"], image_record["point_labels"])
        #     # else:
        #     #     points = None
        #     #print('image_record:', image_record)
        #     #print( seg_logits.shape)
        #
        #     #one_mask = pred_sem_seg.reshape(256, 256)
        #
        #     #reshape和layer操作的目的是为了匹配decoder的相关输入形状
        #     #seg_logits: torch.Size([6, 256, 256])
        #
        #     # one_mask_logits = seg_logits_0.reshape(1, 6, 256, 256)   #one_mask_logits: torch.Size([1, 6, 256, 256])
        #     # one_mask_logits = layer(one_mask_logits)
        #     #print('pred_sem_seg_0.shape:',pred_sem_seg_0)
        #     #pred_sem_seg_0 = pred_sem_seg_0.reshape(1, 1, 256, 256)
        #     pred_sem_seg_0 = pred_sem_seg_0.cpu().numpy()
        #     #one_mask_logits = layer(one_mask_logits)
        #     #one_mask_logits = layer(one_mask_logits)   #one_mask_logits: torch.Size([1, 6, 64, 64])
        #     predictions = []
        #     one_mask_logits = seg_logits_0.reshape(1, 6, 256, 256)   #one_mask_logits: torch.Size([1, 6, 256, 256])
        #     one_mask_logits = layer(one_mask_logits)
        #     for channel in range(6):
        #         with torch.no_grad():
        #             # gt_channel = gt[:, :, channel]
        #             image_record = image_record.reshape((1, 3, 256, 256))
        #             image_embedding = self.image_encoder(image_record)
        #
        #         mask_box = (pred_sem_seg_0 == channel).astype(np.uint8)  # mask_box: (1, 256, 256)
        #         mask_box1 = mask_box.reshape(1, 1, 256, 256)
        #         mask_box1 = torch.as_tensor(mask_box1, dtype=torch.float, device=device)
        #         mask_box1 = layer(mask_box1)
        #
        #         left_clicks, right_clicks = utils.get_random_points(pred_sem_seg_0, channel)
        #         all_points = np.concatenate((left_clicks, right_clicks), axis=0)
        #         all_points = np.array(all_points)
        #         point_labels = np.array([1] * left_clicks.shape[0] + [0] * right_clicks.shape[0], dtype=int)
        #         # print('all_points :', all_points)
        #         # print('point_labels :', point_labels)
        #         if len(all_points) == 0:  # if the model can't generate any sparse prompts
        #
        #             with torch.no_grad():
        #                 sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #                     points=None,
        #                     boxes=None,
        #                     #masks=one_mask_logits,
        #                     masks=None,
        #                 )
        #             # print('sparse_embeddings:',sparse_embeddings.shape)
        #             dense_embeddings = res_0.reshape(1, 256, 16, 16)
        #             # predicted masks
        #             mask_predictions, _ = self.mask_decoder(
        #                 image_embeddings=image_embedding.to(device),
        #                 image_pe=self.prompt_encoder.get_dense_pe(),
        #                 sparse_prompt_embeddings=sparse_embeddings,
        #                 dense_prompt_embeddings=dense_embeddings,
        #                 multimask_output=False,
        #             )
        #             mask_predictions = F.interpolate(mask_predictions,
        #                                              size=(batched_input.shape[2], batched_input.shape[3]),
        #                                              mode="bilinear",
        #                                              align_corners=False)
        #             predictions.append(mask_predictions)
        #
        #
        #             # mask_predictions = torch.zeros((1, 1, 256, 256))
        #             # mask_predictions = torch.as_tensor(mask_predictions, dtype=torch.float, device=device)
        #             # predictions.append(mask_predictions)
        #         else:
        #
        #             transform = ResizeLongestSide(self.image_encoder.img_size)
        #             # embeddings = seg_logits_0.reshape(1, 6, 256, 256)
        #             # embeddings = layer(embeddings)
        #             # embeddings_torch = torch.as_tensor(embeddings, dtype=torch.float, device=device)
        #
        #             # with torch.no_grad():
        #             #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #             #         points=None,
        #             #         boxes=None,
        #             #         masks=mask_box1 ,
        #             #     )
        #             # # #dense_embeddings = embeddings_torch
        #             # #dense_embeddings = res_0.reshape(1, 256, 16, 16)
        #             # # #dense_embeddings = dense_embeddings.reshape(1, 256, 16, 16)
        #             # # # print('dense_embeddings :',dense_embeddings .shape)
        #             # mask_predictions, _ = self.mask_decoder(
        #             #     image_embeddings=image_embedding.to(device),
        #             #     image_pe=self.prompt_encoder.get_dense_pe(),
        #             #     sparse_prompt_embeddings=sparse_embeddings,
        #             #     dense_prompt_embeddings=dense_embeddings,
        #             #     multimask_output=False,
        #             # )
        #
        #             # print('image_record.shape[0]:', image_record.shape)
        #             # print('image_record.shape[0]:', image_record.shape[0])
        #             all_points = transform.apply_coords(all_points, (image_record.shape[2], image_record.shape[3]))
        #             all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
        #             point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
        #             all_points, point_labels = all_points[None, :, :], point_labels[None, :]
        #
        #             points = (all_points, point_labels)
        #
        #             #mask_predictions = layer(mask_predictions)
        #             with torch.no_grad():
        #                 sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #                     points=points,
        #                     boxes=None,
        #                     #masks=one_mask_logits,
        #                     masks=mask_box1,
        #                 )
        #             # print('sparse_embeddings:',sparse_embeddings.shape)
        #             #dense_embeddings = res_0.reshape(1, 256, 16, 16)
        #             # predicted masks
        #             mask_predictions, _ = self.mask_decoder(
        #                 image_embeddings=image_embedding.to(device),
        #                 image_pe=self.prompt_encoder.get_dense_pe(),
        #                 sparse_prompt_embeddings=sparse_embeddings,
        #                 dense_prompt_embeddings=dense_embeddings,
        #                 multimask_output=False,
        #             )
        #             #print('mask_predictions:', mask_predictions.shape)  #mask_predictions: torch.Size([1, 1, 64, 64])
        #             # if multimask_output == True:
        #             #
        #             #     mask_predictions = utils.post_process_segmentation_map(mask_predictions)
        #             #     #print('mask_predictions:', mask_predictions.shape)
        #             #     mask_predictions = torch.as_tensor(mask_predictions, dtype=torch.float, device=device)
        #             #     mask_predictions = mask_predictions.reshape(1, 1, 64, 64)
        #
        #             # print(image_record.shape)
        #             mask_predictions = F.interpolate(mask_predictions, size=(batched_input.shape[2], batched_input.shape[3]), mode="bilinear",
        #                                              align_corners=False)
        #             # print('mask_predictions:',mask_predictions.shape)
        #             predictions.append(mask_predictions)
        #
        #     predictions = torch.cat(predictions, dim=1)
        #
       
#no points
        SAM_focal_loss = 0
        SAM_dice_loss = 0
        loss_SAM = 0
        loss = 0


        for image_record, pred_sem_seg_0, res_0, seg_logits_0 in zip(batched_input, pred_sem_seg, res, seg_logits):

            pred_sem_seg_0 = pred_sem_seg_0.cpu().numpy()
            #res_0 = layer(res_0)
            #per_gt = per_gt.cpu().numpy()
            predictions = []

            for channel in range(6):
                with torch.no_grad():
                    # gt_channel = gt[:, :, channel]
                    image_record = image_record.reshape((1, 3, 256, 256))
                    image_embedding = self.image_encoder(image_record)

                mask_box = (pred_sem_seg_0 == channel).astype(np.uint8)  # mask_box: (1, 256, 256)
                mask_box1 = mask_box.reshape(1, 1, 256, 256)
                mask_box1 = torch.as_tensor(mask_box1, dtype=torch.float, device=device)
                mask_box1 = layer(mask_box1)

                left_clicks, right_clicks = utils.get_random_points(pred_sem_seg_0, channel)
                all_points = np.concatenate((left_clicks, right_clicks), axis=0)
                all_points = np.array(all_points)
                point_labels = np.array([1] * left_clicks.shape[0] + [0] * right_clicks.shape[0], dtype=int)


                if len(all_points) == 0:
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                            points=None,
                            boxes=None,
                            # masks=one_mask_logits,
                            masks=mask_box1,
                        )
                    print('dense_embeddings:',dense_embeddings.shape)
                    dense_embeddings += res_0.reshape(1, 256, 16, 16)
                        # predicted masks
                    mask_predictions, _ = self.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    mask_predictions = F.interpolate(mask_predictions,
                                                     size=(batched_input.shape[2], batched_input.shape[3]),
                                                     mode="bilinear",
                                                     align_corners=False)

                    predictions.append(mask_predictions)
                else:
                    transform = ResizeLongestSide(self.image_encoder.img_size)
                    all_points = transform.apply_coords(all_points, (image_record.shape[2], image_record.shape[3]))
                    all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                    point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                    all_points, point_labels = all_points[None, :, :], point_labels[None, :]

                    points = (all_points, point_labels)
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points= points,
                        boxes=None,
                        # masks=one_mask_logits,
                        masks=mask_box1,
                    )
                    # print('sparse_embeddings:',sparse_embeddings.shape)
                    # dense_embeddings = res_0.reshape(1, 256, 16, 16)
                    # predicted masks
                    mask_predictions, _ = self.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    mask_predictions = F.interpolate(mask_predictions,
                                                     size=(batched_input.shape[2], batched_input.shape[3]),
                                                     mode="bilinear",
                                                     align_corners=False)


                    predictions.append(mask_predictions)


            predictions = torch.cat(predictions, dim=1)


#共同部分


            if i==0:
                outputs = predictions
                i+=1
            else:

                outputs = torch.cat([outputs,predictions], dim=0 )
                i+=1


                # one_gt_mask = np.zeros((256, 256))
                # one_gt_mask[contour_box[1]:contour_box[3], contour_box[0]:contour_box[2]] = np.array(
                #     gt).reshape(256, 256)[
                #                                                                             contour_box[1]:
                #                                                                             contour_box[3],
                #                                                                             contour_box[0]:
                #                                                                             contour_box[2]]
                #
                # one_gt_mask = torch.tensor(one_gt_mask).reshape(1,one_gt_mask.shape[0],one_gt_mask.shape[1])
                # # calculate iou
                # batch_iou = self.calc_iou(masks, one_gt_mask)
                # SAM_focal_loss += self.SAM_focal_loss(masks, one_gt_mask)
                # SAM_dice_loss += self.SAM_dice_loss(masks, one_gt_mask)
                # SAM_MSE_loss += F.mse_loss(iou_predictions, batch_iou, reduction='sum')
                # loss_SAM = (SAM_focal_loss + SAM_dice_loss + SAM_MSE_loss) / (len(one_contours) * 16)
            #outputs.append(masks)

            #print('outputs:',outputs.shape)
        #outputs = np.array(outputs)

        #return seg_logits,  loss_SAM

        return seg_logits, outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
