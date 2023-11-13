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
import mmcv
import warnings
from .untils import tokenize
from utils import transform

import numpy as np
import tqdm
import cv2
import itertools

import os
import matplotlib.pyplot as plt

@SEGMENTORS.register_module()
class BinaryFewSegViT(FewEncoderDecoder):
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
        super(BinaryFewSegViT, self).__init__(**args)
        
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

        if self.training:
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
        else:
            self.backbone.eval()

        self.pair_test = 0 # 0-1000

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
                                                    qs_epoch,
                                                    self.train_cfg,)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
        
    def _decode_head_forward_test(self, x, img_metas, novel_queries=None):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg, novel_queries, self.supp_cls)
        return seg_logits

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, gt_semantic_seg=None, support_info=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, gt_semantic_seg, support_info, **kwargs)
        else:
            ## register novel prototypies:
            if len(self.base_class) != len(self.both_class): 
                if self.decode_head.use_stages == 1:
                    self.novel_queries = self.extract_novel_proto_binary(self.supp_dir, self.supp_path) #binary
                else:
                    self.novel_queries = self.extract_novel_proto_binary_multi(self.supp_dir, self.supp_path)
                # self.novel_queries = self.extract_novel_proto_generalized(self.supp_dir, self.supp_path) #generalized
                self.pair_test += 1
                return self.forward_test(img, img_metas, self.novel_queries, **kwargs) #generalized
            else:
                self.pair_test += 1
                return self.forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img):
        """Extract features from images."""
        visual_feat = self.backbone(img)
        return visual_feat

    def extract_novel_proto_binary(self, dir, npy_path):
        ##  From support set!
        sup_npy = np.load(npy_path, allow_pickle=True)
        sup_name = sup_npy[self.pair_test]
        cls_num_img = int(1000/len(self.novel_class)) # voc:200, coco:50 #1000
        cls_label = self.novel_class[int(self.pair_test/cls_num_img)] #200!!! means the target class

        if len(sup_name) == 5: ## 5-shot
            for k in range(5):
                sup_name_k = sup_name[k]
                if len(self.CLASSES) == 81:
                    image_path = dir + '/JPEGImages/val2014/' + str(sup_name_k) + '.jpg'
                    label_path = dir + '/Annotations/val_contain_crowd/' + str(sup_name_k) + '.png'
                elif len(self.CLASSES) == 21:
                    image_path = dir + '/JPEGImages/' + str(sup_name_k) + '.jpg'
                    label_path = dir + '/Annotations/' + str(sup_name_k) + '.png'
                else:
                    assert AttributeError('Do not support this dataset')

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
                image = np.float32(image) # (0-255)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

                ## check the image and label
                image, label = self.val_supp_transform(image, label)
                # try: image = image.unsqueeze(0).to(self.backbone.class_token.device)
                # except: image = image.unsqueeze(0).to(self.backbone.cls_token.device)
                image = image.unsqueeze(0).to(self.decode_head.base_qs.device)
                # label[label==0] = 255 ## ignore the ground truth label
                # label[label!=255] -= 1
                label = label.unsqueeze(0)

                if k==0:
                    images = image
                    labels = label
                else:
                    images = torch.concat((images, image), dim=0)
                    labels = torch.concat((labels, label), dim=0)

            # get all patch features
            with torch.no_grad():
                patch_embeddings = self.extract_feat(images)[0][0]  ## V1: (bs, dim, 32, 32) dino+vpt better
            # patch_embeddings = self.extract_feat(image)[-1] ## V2: only from the original dino

            # obtain the mask
            # label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int()
            patch_embeddings = F.interpolate(patch_embeddings, size=label.shape[-2:], mode='nearest')
            binary_labels = torch.zeros_like(labels).to(patch_embeddings.device)
            # print('label:', label.unique())
            # print('cls:', cls_label)
            binary_labels[labels == cls_label] = 1
            assert binary_labels.sum() != 0
            
            bg_labels = torch.zeros_like(binary_labels)
            bg_labels[binary_labels == 0] = 1

            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            # novel_proto = ((torch.einsum("bdhw,bhw->bdhw", patch_embeddings, binary_labels).sum(-1).sum(-1)) / (binary_labels.sum(-1).sum(-1).unsqueeze(-1))).mean(dim=0)  # dim
            # fake_proto = ((torch.einsum("bdhw,bhw->bdhw", patch_embeddings, fake_labels).sum(-1).sum(-1)) / (fake_labels.sum(-1).sum(-1).unsqueeze(-1))).mean(dim=0)
            
            for n_p in range(5):
                if n_p ==0 :
                    novel_proto = (torch.einsum("dhw,hw->dhw", patch_embeddings[n_p].squeeze(), binary_labels[n_p].to(patch_embeddings[n_p].device)).sum(-1).sum(-1)) / binary_labels[n_p].sum()  # dim
                    bg_proto = (torch.einsum("dhw,hw->dhw", patch_embeddings[n_p].squeeze(), bg_labels[n_p].to(patch_embeddings[n_p].device)).sum(-1).sum(-1)) / bg_labels[n_p].sum()  # dim
                else:
                    novel_proto += (torch.einsum("dhw,hw->dhw", patch_embeddings[n_p].squeeze(), binary_labels[n_p].to(patch_embeddings[n_p].device)).sum(-1).sum(-1)) / binary_labels[n_p].sum()  # dim
                    bg_proto += (torch.einsum("dhw,hw->dhw", patch_embeddings[n_p].squeeze(), bg_labels[n_p].to(patch_embeddings[n_p].device)).sum(-1).sum(-1)) / bg_labels[n_p].sum()  # dim
                    
            bg_novel_proto = torch.concat((bg_proto.unsqueeze(0)/5, (novel_proto.unsqueeze(0)/5)), dim=0)

        else: # 1-shot
            if len(self.CLASSES) == 81:
                image_path = dir + '/JPEGImages/val2014/' + str(sup_name) + '.jpg'
                label_path = dir + '/Annotations/val_contain_crowd/' + str(sup_name) + '.png'
            elif len(self.CLASSES) == 21:
                image_path = dir + '/JPEGImages/' + str(sup_name) + '.jpg'
                label_path = dir + '/Annotations/' + str(sup_name) + '.png'

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image) # (0-255)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

            ## check the image and label
            image, label = self.val_supp_transform(image, label)
            try: image = image.unsqueeze(0).to(self.backbone.class_token.device)
            except: image = image.unsqueeze(0).to(self.backbone.cls_token.device)
            # label[label==0] = 255 ## ignore the ground truth label
            # label[label!=255] -= 1

            # get all patch features
            with torch.no_grad():
                patch_embeddings = self.extract_feat(image)[0][0]  ## V1: (1, dim, 32, 32) dino+vpt better
            # patch_embeddings = self.extract_feat(image)[-1] ## V2: only from the original dino

            # obtain the mask
            patch_embeddings = F.interpolate(patch_embeddings, size=label.shape[-2:], mode='nearest')
            binary_label = torch.zeros_like(label)
            binary_label[label == cls_label] = 1
            # print('i:', self.pair_test)
            # print('label:', label.unique())
            # print('cls:', cls_label)
            assert binary_label.sum() != 0
            # if binary_label.sum() != 0 :
            #     novel_proto = (patch_embeddings.squeeze().sum(-1).sum(-1))/(512*512)
            # else:
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            novel_proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim
            
            bg_label = torch.zeros_like(binary_label)
            bg_label[binary_label == 0] = 1
            # fake_label[label == 255] = 0
            bg_proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), bg_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / bg_label.sum()  # dim
            bg_novel_proto = torch.concat((bg_proto.unsqueeze(0), novel_proto.unsqueeze(0)), dim=0)
            
        # return novel_proto
        self.supp_cls = cls_label #???
        return bg_novel_proto

    def extract_novel_proto_binary_multi(self, dir, npy_path):
        ##  From support set!
        sup_npy = np.load(npy_path, allow_pickle=True)
        sup_name = sup_npy[self.pair_test]
        cls_num_img = int(1000/len(self.novel_class)) # voc:200, coco:50 #1000
        cls_label = self.novel_class[int(self.pair_test/cls_num_img)] #200!!! means the target class

        if len(sup_name) == 5: ## 5-shot
            for k in range(5):
                sup_name_k = sup_name[k]
                if len(self.CLASSES) == 81:
                    image_path = dir + '/JPEGImages/val2014/' + str(sup_name_k) + '.jpg'
                    label_path = dir + '/Annotations/val_contain_crowd/' + str(sup_name_k) + '.png'
                elif len(self.CLASSES) == 21:
                    image_path = dir + '/JPEGImages/' + str(sup_name_k) + '.jpg'
                    label_path = dir + '/Annotations/' + str(sup_name_k) + '.png'
                else:
                    assert AttributeError('Do not support this dataset')

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
                image = np.float32(image) # (0-255)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

                ## check the image and label
                image, label = self.val_supp_transform(image, label)
                # try: image = image.unsqueeze(0).to(self.backbone.class_token.device)
                # except: image = image.unsqueeze(0).to(self.backbone.cls_token.device)
                image = image.unsqueeze(0).to(self.decode_head.base_qs.device)
                # label[label==0] = 255 ## ignore the ground truth label
                # label[label!=255] -= 1
                label = label.unsqueeze(0)

                if k==0:
                    images = image
                    labels = label
                else:
                    images = torch.concat((images, image), dim=0)
                    labels = torch.concat((labels, label), dim=0)

            # get all patch features
            with torch.no_grad():
                novel_support_feat = self.extract_feat(images)[0]  ## V1: (bs, dim, 32, 32) dino+vpt better

            binary_labels = torch.zeros_like(labels)
            binary_labels[labels == cls_label] = 1
            assert binary_labels.sum() != 0
            
            bg_labels = torch.zeros_like(binary_labels)
            bg_labels[binary_labels == 0] = 1

            for n_p in range(5):
                for i_stage in range(self.decode_head.use_stages):
                    if i_stage < (len(novel_support_feat)-1):
                        patch_token_cls_i = novel_support_feat[i_stage][1][n_p].unsqueeze(0).clone().detach()
                    else:
                        patch_token_cls_i = novel_support_feat[i_stage][n_p].unsqueeze(0).clone().detach()
                    # obtain the mask
                    patch_token_cls_i = F.interpolate(patch_token_cls_i, size=label.shape[-2:], mode='bilinear', align_corners=False).squeeze() ## (512, 512)
                    
                    novel_proto_i = (torch.einsum("dhw,hw->dhw", patch_token_cls_i.squeeze(), binary_labels[n_p].to(patch_token_cls_i.device)).sum(-1).sum(-1)) / binary_labels[n_p].sum()  # dim
                    bg_proto_i = (torch.einsum("dhw,hw->dhw", patch_token_cls_i.squeeze(), bg_labels[n_p].to(patch_token_cls_i.device)).sum(-1).sum(-1)) / bg_labels[n_p].sum()  # dim
                    bg_novel_proto_i_np = torch.concat((bg_proto_i.unsqueeze(0), novel_proto_i.unsqueeze(0)), dim=0).unsqueeze(0)
                    
                    if i_stage == 0:
                        bg_novel_proto_i = bg_novel_proto_i_np
                    else:
                        bg_novel_proto_i = torch.concat([bg_novel_proto_i, bg_novel_proto_i_np], dim=0)
                
                if n_p ==0 :
                    all_bg_novel_proto = bg_novel_proto_i.unsqueeze(0)
                else:
                    all_bg_novel_proto = torch.concat([all_bg_novel_proto, bg_novel_proto_i.unsqueeze(0)], dim=0)
                    
            bg_novel_proto = all_bg_novel_proto.mean(0)

        else: # 1-shot
            if len(self.CLASSES) == 81:
                image_path = dir + '/JPEGImages/val2014/' + str(sup_name) + '.jpg'
                label_path = dir + '/Annotations/val_contain_crowd/' + str(sup_name) + '.png'
            elif len(self.CLASSES) == 21:
                image_path = dir + '/JPEGImages/' + str(sup_name) + '.jpg'
                label_path = dir + '/Annotations/' + str(sup_name) + '.png'

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image) # (0-255)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

            ## check the image and label
            image, label = self.val_supp_transform(image, label)
            try: image = image.unsqueeze(0).to(self.backbone.class_token.device)
            except: image = image.unsqueeze(0).to(self.backbone.cls_token.device)
            
            # obtain the mask
            binary_label = torch.zeros_like(label)
            binary_label[label == cls_label] = 1
            assert binary_label.sum() != 0

            bg_label = torch.zeros_like(binary_label)
            bg_label[binary_label == 0] = 1
                        
            # get all patch features
            with torch.no_grad():
                novel_support_feat = self.extract_feat(image)[0]  ## V1: (1, dim, 32, 32) dino+vpt better

            # fake_label[label == 255] = 0
            for i_stage in range(self.decode_head.use_stages):
                if i_stage < (len(novel_support_feat)-1):
                    patch_token_cls_i = novel_support_feat[i_stage][1].clone().detach()
                else:
                    patch_token_cls_i = novel_support_feat[i_stage].clone().detach()
                # obtain the mask
                patch_token_cls_i = F.interpolate(patch_token_cls_i, size=label.shape[-2:], mode='bilinear', align_corners=False).squeeze() ## (512, 512)
                
                novel_proto_i = (torch.einsum("dhw,hw->dhw", patch_token_cls_i.squeeze(), binary_label.to(patch_token_cls_i.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim
                bg_proto_i = (torch.einsum("dhw,hw->dhw", patch_token_cls_i.squeeze(), bg_label.to(patch_token_cls_i.device)).sum(-1).sum(-1)) / bg_label.sum()  # dim
                bg_novel_proto_i = torch.concat((bg_proto_i.unsqueeze(0), novel_proto_i.unsqueeze(0)), dim=0).unsqueeze(0)
                
                if i_stage == 0:
                    bg_novel_proto = bg_novel_proto_i
                else:
                    bg_novel_proto = torch.concat([bg_novel_proto, bg_novel_proto_i], dim=0)
            
        # return novel_proto
        self.supp_cls = cls_label #???
        return bg_novel_proto.transpose(0, 1)

    def extract_bg_base_epoch(self, patch_features, masks, bs):
        ## qs(base, 768), patch(bs, 768, 32, 32), gt(bs, 512, 512)
        assert patch_features.shape[0] == masks.shape[0]
        support_shot = (patch_features.shape[0]/bs)
        patch_features = F.interpolate(patch_features, size=masks.shape[-2:], mode='bilinear', align_corners=False) ## (512, 512)
        # targets = F.interpolate(targets.unsqueeze(1).float(), size=patch_features.shape[-2:], mode='nearest').squeeze(1).int() ## (32, 32)

        bg_base_protos = torch.zeros(bs, 2, patch_features.shape[1]).to(patch_features.device)  #(c, bg+base, dim)
        n = 0
        for mask in masks:
            # for n th image in a batch
            c_n = n % bs
            
            binary_mask_bg = torch.ones_like(mask)
            binary_mask_bg[mask == 1] = 0
            
            # assert binary_mask_bg.sum() != 0 and mask.sum() != 0
            if binary_mask_bg.sum() != 0:
                bg_proto = (torch.einsum("dhw,hw->dhw", patch_features[n].squeeze(), binary_mask_bg).sum(-1).sum(-1)) / binary_mask_bg.sum()
                bg_base_protos[c_n, 0] += bg_proto
            if mask.sum() != 0:
                base_proto = (torch.einsum("dhw,hw->dhw", patch_features[n].squeeze(), mask).sum(-1).sum(-1)) / mask.sum()
                bg_base_protos[c_n, 1] += base_proto
                
            n += 1
        return bg_base_protos / support_shot

    def forward_train(self, img, img_metas, gt_semantic_seg, support_info):
        # print('check_weights_clip_vpt:', self.backbone.prompt_embeddings.sum())
        # print('check_weights_clip_vpt:', self.backbone.prompt_embeddings.requires_grad)
        ## check whether this image includes novel classes, gt:(bs, 1, 512, 512)
        bs = img.shape[0]
        assert gt_semantic_seg.unique() not in self.novel_class
        
        ## get the support image from support_info
        if support_info:
            for i_s in range(len(support_info)):
                if i_s == 0:
                    img_support = support_info[i_s]['img']
                    gt_semantic_seg_support = support_info[i_s]['gt_semantic_seg']
                else:
                    img_support = torch.concat([img_support, support_info[i_s]['img']], dim=0)
                    gt_semantic_seg_support = torch.concat([gt_semantic_seg_support, support_info[i_s]['gt_semantic_seg']], dim=0)
                
        gt_semantic_seg[gt_semantic_seg!=0] == 1 # set all the mask info binary
        gt_semantic_seg_support[gt_semantic_seg_support!=0] == 1 # set all the mask info binary

        # print('gt:', gt_semantic_seg.unique())
        visual_feat = self.extract_feat(img) # (bs*(1+s), 1025, 768)
        with torch.no_grad():
            visual_feat_support = self.extract_feat(img_support)[0][0].clone().detach()
        feat = []
        feat.append(visual_feat)
        qs_epoch = self.extract_bg_base_epoch(visual_feat_support, gt_semantic_seg_support.squeeze(), bs)

        losses = dict()
        loss_decode = self._decode_head_forward_train(feat, img_metas, gt_semantic_seg, qs_epoch)
        losses.update(loss_decode)
        return losses
        
    def encode_decode(self, img, img_metas, novel_queries=None):
        visual_feat = self.extract_feat(img)

        feat = []
        feat.append(visual_feat)

        out = self._decode_head_forward_test(feat, img_metas, novel_queries)
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
        num_classes = 2
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img)) # some part may be split into several images
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
                               (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

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

    def inference(self, img, img_meta, rescale, novel_protoes=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if novel_protoes is not None:
            if self.test_cfg.mode == 'slide':
                seg_logit = self.slide_inference(img, img_meta, rescale, novel_protoes)
            else:
                seg_logit = self.whole_inference(img, img_meta, rescale, novel_protoes)
        else:
            if self.test_cfg.mode == 'slide':
                seg_logit = self.slide_inference(img, img_meta, rescale)
            else:
                seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1) ###???
        output = seg_logit
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = seg_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = seg_logit.flip(dims=(2, ))
        return output

    def simple_test(self, img, img_meta, rescale=True, novel_protoes=None):
        """Simple test with single image."""
        if novel_protoes is not None:
            seg_logit = self.inference(img, img_meta, rescale, novel_protoes)
        else:
            seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1) ###???
        # seg_pred = seg_logit
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        # assert palette.shape[0] == 2
        # assert palette.shape[1] == 3
        # assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

