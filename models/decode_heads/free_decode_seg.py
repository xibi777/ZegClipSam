from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from math import cos, pi

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@HEADS.register_module()
class FreeHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            crop_train=False,
            **kwargs,
    ):
        super(FreeHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        # self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx

        # nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), embed_dims)))

        self.class_embed = nn.Linear(embed_dims, self.num_classes + 1)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, qs_epoch, train_cfg):
        seg_logits = self.forward(inputs, qs_epoch=qs_epoch)

        gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, novel_queries=None):
        if novel_queries is not None:
            return self.forward(inputs, novel_queries=novel_queries)
        else:
            return self.forward(inputs)

    def forward(self, inputs, qs_epoch=None, novel_queries=None):
        patch_token = inputs[0][0][0]
        cls_token = inputs[0][1]
        
        # x = []
        # for stage_ in patch_token[:self.use_stages]:
        #     x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        # x.reverse()
        bs = patch_token.shape[0]

        attns = []
        maps_size = []

        ### obtain the mask by calculating the similarity between proto (self.base_qs) and patch embeddings
        # base_qs(20, 768) patch(bs, 768, 32, 32)
        patch_token = patch_token.transpose(1, 3)
        pseudo_masks = (patch_token @ (self.base_qs).T).transpose(1, 3) #(bs, 32, 32, 20) ->(bs, 20, 32, 32)
        attns.append(pseudo_masks) #()
        patch_token = patch_token.transpose(1, 3)

        fc_out = self.class_embed(cls_token).repeat(21, 1, 1).permute(1, 0, 2) # (bs, 21, 21)

        if not self.training:
            # REGISTER NOVEL: concat the novel queries in the right position
            q = torch.zeros([len(self.all_idx), self.dim]).to(patch_token.device)
            if novel_queries is not None:
                q[self.seen_idx] = self.base_qs.clone()
                q[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
            else:
                q[:] = self.base_qs.clone()
            q = q.repeat(bs, 1, 1).transpose(0, 1)
        else:
            q = self.base_qs.repeat(bs, 1, 1).transpose(0, 1)
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False
        
        outputs_seg_masks = []
        maps_size.append(attns[0].size()[-2:])
        size = maps_size[-1]

        for i_attn, attn in enumerate(attns):
            if True:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

        pred = F.interpolate(outputs_seg_masks[-1],
                                          size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
                                          
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = self.base_qs
            out["pred_logits"] = fc_out
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0)
            return out["pred"]                  
        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def update_m(self, end_m=1.0, base_m=0.995): # 
        max_iter = 10000
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss