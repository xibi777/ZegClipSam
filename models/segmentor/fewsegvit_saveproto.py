from importlib.resources import path
from unittest.mock import patch
from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
# from mmseg.models.segmentors.base import BaseSegmentor
# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from .few_base import FewBaseSegmentor
from .few_encoder_decoder import FewEncoderDecoder
from mmcv.runner import BaseModule, auto_fp16

from .untils import tokenize
from utils import transform

import numpy as np
import tqdm
import cv2
import itertools

import os
import matplotlib.pyplot as plt

@SEGMENTORS.register_module()
class FewSegViTSave(FewEncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 pretrained,
                 context_length,
                 base_class,
                 novel_class,
                 both_class,
                 split,
                 shot,
                 supp_dir=None,
                 supp_path=None, # only for evluation on base+novel classes
                 tau=0.07,
                 ft_backbone=False,
                 exclude_key=None,
                #  init_cfg=None,
                 **args):
        super(FewSegViTSave, self).__init__(**args)
        
        self.pretrained = pretrained

        self.tau = tau

        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)

        self.split = split
        self.shot = shot
        self.supp_dir = supp_dir
        self.supp_path = supp_path # only for evaluation

        if len(self.base_class) != len(self.both_class): # few-shot setting
            self._visiable_mask(self.base_class, self.novel_class, self.both_class)

        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]
        self.val_supp_transform = transform.Compose([
            transform.Resize(size=max(512, 512)),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        self.val_supp_aug_transform = transform.Compose([
            transform.RandScale(scale=[0.75, 1.5]),
            # transform.Resize(size=max(784, 784)),
            # transform.Crop(size=(512, 512), crop_type='rand'),
            transform.Resize(size=max(512, 512)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        if self.training:
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
        else:
            self.backbone.eval()

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([255]*256)
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting:', self.visibility_seen_mask) 
    
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _decode_head_forward_train(self, feat, img_metas, gt_semantic_seg, qs_epoch=None):
        """Run forward function and calculate loss for decode head in
        training."""

        losses = dict()
        
        loss_decode = self.decode_head.forward_train(feat, 
                                                    img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg,)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
        
    def _decode_head_forward_test(self, x, img_metas, novel_queries=None):
        """Run forward function and calculate loss for decode head in
        inference."""
        if novel_queries is not None:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg, novel_queries)
        else:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            ## register novel prototypies:
            if len(self.base_class) != len(self.both_class): #generalized few-shot setting
                return self.forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img):
        """Extract features from images."""
        visual_feat = self.backbone(img)
        return visual_feat

    def extract_base_proto_epoch(self, qs, patch_features, targets):
        ## qs(base, 768), patch(bs, 768, 32, 32), gt(bs, 512, 512)
        layers = len(patch_features)
        assert patch_features[0].shape[0] == targets.shape[0]
        # targets = F.interpolate(targets.unsqueeze(1).float(), size=patch_features.shape[-2:], mode='nearest').squeeze(1).int() ## (32, 32)
        # resnet50: patch (bs, 2048. 512, 512)
        qs_epoch = torch.zeros_like(qs[:, -1]) # [21, dim] resnet:[21,512] (2048-512) with proj
        num_base = torch.zeros(qs.shape[0]).to(qs_epoch.device)  #(21)

        n = 0 #bs
        for targets_per_image in targets:
            # for n th image in a batch
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != 255]
            if len(gt_cls) != 0:
                for cls in gt_cls:
                    num_base[cls] += 1
                    binary_mask = torch.zeros(512, 512).to(targets.device)
                    binary_mask[targets_per_image == cls] = 1
                    for layer in range(layers):
                        patch_features_layer_cls = patch_features[layer][n].unsqueeze(0)
                        patch_features_layer_cls = F.interpolate(patch_features_layer_cls, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                        proto_cls = (torch.einsum("dhw,hw->dhw", patch_features_layer_cls.squeeze(), binary_mask).sum(-1).sum(-1)) / binary_mask.sum()
                        qs_epoch[cls] += proto_cls
                    
                        ## update the saved base protos and nums
                        self.decode_head.base_nums[cls, layer] += 1
                        self.decode_head.base_protos[cls, layer] += proto_cls
            n += 1

        # norm for each base classes
        qs_epoch[num_base!=0] = qs_epoch[num_base!=0] / num_base[num_base!=0].unsqueeze(-1) #(15, 768)

        return qs_epoch / layers


    def forward_train(self, img, img_metas, gt_semantic_seg):
        assert gt_semantic_seg.unique() not in self.novel_class
        if len(self.base_class) != len(self.both_class): # few-shot setting
            gt_semantic_seg = torch.Tensor(self.visibility_seen_mask)
            
        visual_feat = self.extract_feat(img) # (bs, 1025, 768)
        qs_epoch = self.extract_base_proto_epoch(self.decode_head.base_protos, visual_feat, gt_semantic_seg.squeeze()) # V1: from dino+vpt better

        losses = dict()
        loss_decode = self._decode_head_forward_train(visual_feat, img_metas, gt_semantic_seg, qs_epoch)
        losses.update(loss_decode)
        return losses
        
    def encode_decode(self, img, img_metas, novel_queries=None): 
        visual_feat = self.extract_feat(img)

        if novel_queries is not None:
            out = self._decode_head_forward_test(visual_feat, img_metas, novel_queries)
        else:
            out = self._decode_head_forward_test(visual_feat, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, novel_protoes=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if novel_protoes is not None:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta, novel_protoes)
                else:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds



