import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import random

# manual_seed=321
# torch.manual_seed(manual_seed)
# torch.cuda.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)
# random.seed(manual_seed)

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
import cv2
from models.decode_heads.utils import positional_encoding
from PIL import Image

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

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size() #(4,2048,60,60)
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)) #(4.512.60,60)
        return torch.cat(out, 1)    #(4,2014+512*4,60,60)

@HEADS.register_module()
class CAPLHeadSeg(BaseDecodeHead):
    def __init__(self, 
            img_size,
            in_channels,
            channels,
            seen_idx,
            all_idx, 
            bins = (1, 2, 3, 6), 
            dropout = 0.1, 
            use_stages=1,
            BatchNorm=nn.BatchNorm2d,
            **kwargs,):
        super(CAPLHeadSeg, self).__init__(
            in_channels=in_channels, channels=channels, **kwargs)
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.novel_idx = self.all_idx.copy()
        self.use_stages=use_stages
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        ### module from original CAPL code:
        self.ppm = PPM(in_channels, int(in_channels/len(bins)), bins, BatchNorm) ##decoder
        in_channels *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels, 768, kernel_size=3, padding=1, bias=False),
            BatchNorm(768),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(768, 768, kernel_size=1)
        )
        # if self.training:
        #     self.aux = nn.Sequential(
        #         nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        #         BatchNorm(256),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=dropout),
        #         nn.Conv2d(256, 256, kernel_size=1)
        #     )

        main_dim = 768
        aux_dim = 768
        self.main_proto = nn.Parameter(torch.randn(len(self.all_idx), main_dim).cuda()) # 21,512
        # self.aux_proto = nn.Parameter(torch.randn(self.all_idx, aux_dim).cuda()) # 21,512
        gamma_dim = 1
        self.gamma_conv = nn.Sequential(
            nn.Linear(in_channels, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(768, gamma_dim)
        )
        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # print('gt:', gt_semantic_seg.unique())
        gt_semantic_seg[gt_semantic_seg==-1] = 255
        seg_logits = self.forward(inputs, y=gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, novel_queries=None):
        if novel_queries is not None:
            return self.forward(inputs, gened_proto=novel_queries)
        else:
            return self.forward(inputs)

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred


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

    def forward(self, inputs, y=None, gened_proto=None):
        patch_token = inputs[0][0] # (4, 768, 32, 32)
        
        x = []
        for stage_ in patch_token[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        maps_size = []
        qs = []

        # for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
        #     lateral = norm_(proj_(x_))
        #     if idx == 0:
        #         laterals.append(lateral)
        #     else:
        #         if laterals[idx - 1].size()[1] == lateral.size()[1]:
        #             laterals.append(lateral + laterals[idx - 1])
        #         else:
        #             # nearest interpolate
        #             l_ = self.d3_to_d4(laterals[idx - 1])
        #             l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
        #             l_ = self.d4_to_d3(l_)
        #             laterals.append(l_ + lateral)

        # lateral = laterals[-1]
        x = x[-1].reshape(bs, 32, 32, -1).permute(0,3,1,2) ##(2,1024,768)->(2,768,32,32)

        x = self.ppm(x)#(4,768*2,32,32) 
        x = self.cls(x)#(4,768,32,32)
        raw_x = x.clone()

        ## lateram for decoder pmm
        if self.training:
            #### training
            fake_num = x.size(0) // 2              
            ori_new_proto, replace_proto = self.generate_fake_proto(proto=self.main_proto, x=x[fake_num:], y=y[fake_num:])                    
            x = self.get_pred(x, ori_new_proto)    

            x_pre = x.clone()
            refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x)
            post_refine_proto = refine_proto.clone() #(4, 21, 512)
            post_refine_proto[:, self.seen_idx] = post_refine_proto[:, self.seen_idx] + ori_new_proto[self.seen_idx].unsqueeze(0)
            post_refine_proto[:, self.novel_idx] = post_refine_proto[:, self.novel_idx] * 0 + ori_new_proto[self.novel_idx].unsqueeze(0)
            x = self.get_pred(raw_x, post_refine_proto)         #(4, 21, 60, 60) 
            pred = F.interpolate(x, size=512, mode='bilinear', align_corners=True)[:,self.seen_idx]

        else: #eval:
            #### evaluation
            # if len(gened_proto.size()[:]) == 3:
                # gened_proto = gened_proto[0] #(1, 5, 768)
            gened_proto = torch.from_numpy(ç).to(x.device).to(x.dtype)  #(5,768)
            # gened_proto = gened_proto / (torch.norm(gened_proto, 2, 1, True) + 1e-12)
            
            refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x) #(1,20,768)
            # refine_proto[:, :len(self.seen_idx)] = refine_proto[:, :len(self.seen_idx)] + gened_proto[self.seen_idx].unsqueeze(0)
            # refine_proto[:, self.novel_idx] = refine_proto[:, self.novel_idx] * 0 + gened_proto[self.novel_idx].unsqueeze(0)
            refine_proto[:, len(self.seen_idx):] = refine_proto[:, len(self.seen_idx):] * 0 + gened_proto.unsqueeze(0)
            refine_proto_ = refine_proto.clone()
            
            refine_proto_[:, self.seen_idx] = refine_proto[:, :len(self.seen_idx)]
            refine_proto_[:, self.novel_idx] = refine_proto[:, len(self.seen_idx):]
            
            x = self.get_pred(raw_x, refine_proto_)
            pred = F.interpolate(x, size=512, mode='bilinear', align_corners=True)

        outputs_seg_masks = []
        outputs_seg_masks.append(pred)
                                          
        out = {"pred_masks": pred}

        if self.training:
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor
            return out["pred"]                  
        return out
    

    def post_refine_proto_v2(self, proto, x):
        raw_x = x.clone()        
        b, c, h, w = raw_x.shape[:]
        pred = self.get_pred(x, proto).view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 2)

        pred_proto = pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)
        pred_proto_norm = F.normalize(pred_proto, 2, -1)    # b, n, c
        proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # 1, n, c
        pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)   # b, n, 1
        pred_weight = pred_weight * (pred_weight > 0).float()
        pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, n, c
        return pred_proto

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]  
            cls_num = proto.size(1)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, -1, True)  # b, n, c
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:         
            cls_num = proto.size(0)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, 1, True)
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 10

    def WG(self, x, y, proto, target_cls):
        b, c, h, w = x.size()[:]
        tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest') 
        out = x.clone()
        unique_y = list(tmp_y.unique())         
        new_gen_proto = proto.data.clone()
        for tmp_cls in unique_y:
            if tmp_cls == 255: 
                continue
            tmp_mask = (tmp_y.float() == tmp_cls.float()).float()
            tmp_p = (out * tmp_mask).sum(0).sum(-1).sum(-1) / tmp_mask.sum(0).sum(-1).sum(-1)
            new_gen_proto[tmp_cls.long(), :] = tmp_p 
        return new_gen_proto        

    def generate_fake_proto(self, proto, x, y):
        b, c, h, w = x.size()[:]
        tmp_y = F.interpolate(y.float(), size=(h,w), mode='nearest')
        unique_y = list(tmp_y.unique())
        raw_unique_y = list(tmp_y.unique())
        if 0 in unique_y:
            unique_y.remove(0)
        if 255 in unique_y:
            unique_y.remove(255)

        novel_num = len(unique_y) // 2
        fake_novel = random.sample(unique_y, novel_num)
        for fn in fake_novel:
            unique_y.remove(fn) 
        fake_context = unique_y
        
        new_proto = self.main_proto.clone()
        new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)
        for fn in fake_novel:
            tmp_mask = (tmp_y == fn).float()
            tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
            fake_vec[fn.long()] = 1
            new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec
        replace_proto = new_proto.clone()

        for fc in fake_context:
            tmp_mask = (tmp_y == fc).float()
            tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)              
            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
            fake_vec[fc.long()] = 1
            raw_feat = new_proto[fc.long()].clone()
            all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze(0)  # 1, 1024
            ratio = F.sigmoid(self.gamma_conv(all_feat))[0]   # n, 512
            new_proto = new_proto * (1 - fake_vec) + ((raw_feat* ratio + tmp_feat* (1 - ratio)).unsqueeze(0) * fake_vec)

        if random.random() > 0.5 and 0 in raw_unique_y:
            tmp_mask = (tmp_y == 0).float()
            tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)  #512             
            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
            fake_vec[0] = 1
            raw_feat = new_proto[0].clone()
            all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze(0)  # 1, 1024         
            ratio = F.sigmoid(self.gamma_conv(all_feat))[0]   # 512
            new_proto = new_proto * (1 - fake_vec) + ((raw_feat * ratio + tmp_feat * (1 - ratio)).unsqueeze(0)  * fake_vec)

        return new_proto, replace_proto
