    
from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

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
from math import cos, pi

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy
import itertools

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

 
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
  
class MLPFuse(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, dim_list):
        super().__init__()
        self.num_layers = len(dim_list) - 1
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip(dim_list[:-1], dim_list[1:])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_Proj(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # eg: 2, [64,32], 1
        super().__init__()
        self.num_layers = len(hidden_dim) + 1
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dim, hidden_dim + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

@HEADS.register_module()
class SaveHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            max_iter=40000,
            embed_dims=768,
            num_layers=0,
            num_heads=0,
            use_stages=1,
            withRD=False,
            use_proj=True,
            crop_train=False,
            rd_type=None,
            decode_type=None,
            **kwargs,
    ):
        super(SaveHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.max_iter = max_iter
        
        # nhead = num_heads
        dim = embed_dims
        
        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        # self.input_proj = input_proj
        # self.proj_norm = proj_norm

        delattr(self, 'conv_seg')
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_protos", torch.zeros((len(self.seen_idx), use_stages, in_channels)))
        self.register_buffer("base_nums", torch.zeros((len(self.seen_idx), use_stages)))

        self.q_proj = nn.Linear(dim, dim)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs, gt_semantic_seg=None, novel_clip_feats=None, novel_labels=None):
        patch_tokens = inputs[0][-1] #(bs, 768, 32, 32) patch embeddings from the last layer
        
        bs, dim, p, _ = patch_tokens.size()
        patch_tokens = patch_tokens.reshape(bs, dim , -1)
        
        # get rd and update base_qs
        if self.training:
            q = self.q_proj(self.base_protos[:, -1].expand(bs, -1, -1))
            self.cur_iter += 1
        else:
            q = self.q_proj(self.base_protos.expand(bs, -1, -1))
        
        if len(self.all_idx) == 21:
            max_iter = 2000 #200
        elif len(self.all_idx) == 81:
            max_iter = 8 #800
           
        if self.cur_iter == max_iter:
            print('saving protos......')
            ## save the protos
            save_protos = self.base_protos / (self.base_nums.unsqueeze(-1)) # check the value
            save_protos = save_protos.clone().cpu().numpy()
            if len(self.all_idx) == 21:
                save_path = '/media/data/ziqin/data_fss/init_protos/voc_protos_dino.npy'
            elif len(self.all_idx) == 81:
                save_path = '/media/data/ziqin/data_fss/init_protos/coco_protos_dino.npy'
            np.save(save_path, save_protos)
        
        # get prediction and loss
        pred_logits = torch.einsum("bdn,bcd->bcn", patch_tokens, q) # matching directly
        c = pred_logits.shape[1]
        pred_logits = pred_logits.reshape(bs, c, p, p)
        # pred_logits = patch_tokens @ text_token.t()

        pred = F.interpolate(pred_logits, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
                                          
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = q.transpose(0, 1).unsqueeze(0) #(1, bs, c, 768)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.0 is the best   
            return out["pred"]   
        return out                 
        
    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

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
        
        
    def update_m(self, end_m=1.0, base_m=0.996):
        max_iter = 20000
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m
    
    def forward_test(self, inputs, img_metas, test_cfg, novel_clip_feats=None, novel_labels=None):
        if novel_clip_feats is not None:
            return self.forward(inputs, novel_clip_feats=novel_clip_feats, novel_labels=novel_labels)
        else:
            return self.forward(inputs)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs, gt_semantic_seg)
        gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses    