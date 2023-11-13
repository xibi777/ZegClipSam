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
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)    



@HEADS.register_module()
class PSPHeadSeg(BaseDecodeHead):
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
            out_indices=[11],
            cls_type='ave',
            bins=(1, 2, 3, 6),
            dropout=0.1,
            zoom_factor=8,
            use_proj=True,
            crop_train=False,
            **kwargs,
    ):
        super(PSPHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.cls_type = cls_type
        self.out_indices = out_indices

        nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)
            
        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, nn.BatchNorm2d)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )
        # if self.training:
        #     self.aux = nn.Sequential(
        #         nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=dropout),
        #         nn.Conv2d(256, 256, kernel_size=1)
        #     )

        # main_dim = 512
        # aux_dim = 256
        # self.main_proto = nn.Parameter(torch.randn(self.classes, main_dim).cuda())
        # self.aux_proto = nn.Parameter(torch.randn(self.classes, aux_dim).cuda())
        gamma_dim = 1
        
        self.gamma_conv = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, gamma_dim)
        )  
            

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)))
        ## bg
        # self.bg_qs = nn.Parameter(torch.randn(1, in_channels))

        self.q_proj = nn.Linear(in_channels * 2, embed_dims)
        # self.q_proj = nn.Linear(embed_dims * 2 + 12, embed_dims) ## MULTIHEAD
        ## ADDED FC for prototype
        # self.proto_proj = nn.Linear(embed_dims, embed_dims)
    
    def init_proto(self):
        print('Initialized prototypes with ResNet50 model')
        if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
            path = '/media/data/ziqin/data_fss/init_protos/voc_protos_rn50.npy'
        elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
            path = '/media/data/ziqin/data_fss/init_protos/coco_protos_rn50.npy'
        if self.use_stages == 1:
            init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, -1][self.seen_idx] ##for 11
        else:
            assert AttributeError('Using MultiATMSingleHeadSeg when you need fusing multiple relationship descriptors')
            # init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, -self.use_stages:][self.seen_idx] ##for 11
        self.base_qs.data = init_protos

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
        # get the target of each cliped region
        # ann_path = img_metas[0]['filename'].replace('jpg','png').replace('JPEGImages', 'Annotations')
        # self.gt_ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        # self.gt_label = np.unique(self.gt_ann)
        # self.gt_label[self.gt_label==0] = 255 ## ignore the ground truth label
        # self.gt_label[self.gt_label!=255] -= 1
        # self.gt_label = np.delete(self.gt_label, np.where(self.gt_label == 255))
        if novel_queries is not None:
            return self.forward(inputs, novel_queries=novel_queries)
        else:
            return self.forward(inputs)

    def get_cls_token(self, patch_token, protos):
        # patch_token(bs, 768, 32, 32) -> (bs, L, 768) protos(15/20, 768)
        B, D, _ ,_ = patch_token.size()
        patch_token = patch_token.reshape(B, D, -1).permute(0, 2, 1)
        L = patch_token.size(1)
        
        mu = protos.mean(dim=0) #(768)
        cls_token = torch.cosine_similarity(patch_token.reshape(-1,D),mu).reshape(B,L) # (bs, L)
        cls_token = cls_token.softmax(dim=-1)
        cls_token = (cls_token.unsqueeze(-1) * patch_token).sum(1) # (bs, L, D) -> (bs, D)
        return cls_token

    def forward(self, inputs, qs_epoch=None, novel_queries=None):
        if inputs[0][1].shape[-1] == 2048: # only for resnet, not for vit
            if self.cur_iter == 0:
                self.init_proto()
                
        patch_token = inputs[0][0]
        if self.cls_type == 'ave':
            cls_token = patch_token[0].mean(-1).mean(-1) #(bs, dim)
        elif self.cls_type == 'weighted':
            cls_token = patch_token[0] # (bs, dim, 32, 32)

        if not self.training:
            # REGISTER NOVEL: concat the novel queries in the right position
            both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                both_proto[self.seen_idx] = self.base_qs.clone()
                # print('Novel!:', novel_queries.sum(-1))
                both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
            else:
                both_proto[:] = self.base_qs.clone()
            cls_token = inputs[0][1]

        x = patch_token[0]

        if not self.training:
            q = self.q_proj(self.get_qs(both_proto, cls_token)) #.transpose(0, 1)

        else:
            #### the momentum updated bg !!!!!!!! ()
            q = self.q_proj(self.get_qs(self.base_qs, cls_token)) #(bs, c, 512)
            ## update self.base_qs
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False

        x = self.ppm(x) #(bs, 4096, 16, 16)
        x = self.cls(x) #(bs, 512, 16, 16)
        b, dim, h, w = x.size()[:]
        c = q.shape[1]
        # norm both q and x
        q = q / torch.norm(q, 2, -1, True)
        x = x / torch.norm(x, 2, 1, True)
        x = x.contiguous().view(b, dim, h*w)
        pred = (q @ x).reshape(b, c, h, w)

        pred = F.interpolate(pred, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # print('pred:', pred.shape)
                                          
        out = {"pred_masks": pred}

        if self.training:
            out["qs_base"] = q
            # outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            # out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.2) ## Change the balance factor： 0.2 is the best   
            return out["pred"]   
        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0): 
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def update_m(self, end_m=1.0, base_m=0.996):
        max_iter = 20000
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m

    def get_qs(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base_class, 768) cls: (bs, 768)
        if self.cls_type == 'weighted': # cls is the patch_embeddings, cls (bs, 768, 32,32)
            bs = cls.shape[0]
            cls = cls.flatten(-2, -1).permute(0, 2, 1) # (bs, 768, 32*32) -> (bs, n, d)
            # q1 = torch.einsum("bdn,cd->bcn", cls, q) ## check the value
            q1 = torch.einsum("bnd,cd->bcnd", cls, q)
            
            cls_norm = cls.permute(0, 2, 1) / torch.norm(cls.permute(0, 2, 1), dim=1, keepdim=True) # (bs, d, n)
            q_norm = q / torch.norm(q, dim=-1, keepdim=True) #(c, d)
            
            ## Version1: sigmoid
            # similarity = torch.bmm(q_norm.expand(bs, -1, -1), cls_norm).sigmoid()## (bs, c, n)
            # similarity = similarity / (similarity.sum(-1).unsqueeze(-1))
            ## Version2: softmax
            similarity = (torch.bmm(q_norm.expand(bs, -1, -1), cls_norm)/0.1).softmax(-1)
            
            q1 = (q1 * similarity.unsqueeze(-1)).sum(dim=-2)
            
            q = q.expand(bs, -1, -1)
            q_ = torch.concat((q1 * 100, q), dim=-1) # (bs, 20, 768+768)
            
        else:   
            C, dim = q.shape
            bs, _ = cls.shape
            q = q.expand(bs, -1, -1)
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100 ## check the value
            q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
        return q_

    def get_qs_save(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, 512) cls: (bs, 512)
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)

        # if q1.shape[1] == 20: ##voc
        #     rd_path = '/media/data/ziqin/work_dirs_fss/voc_vit_split0_rd.npy'
        # elif q1.shape[1] == 80: ##voc
        #     rd_path = '/media/data/ziqin/work_dirs_fss/coco_vit_split0_rd.npy'
        # else:
        #     assert AttributeError('Wrong dataset')

        rd_path = '/media/data/ziqin/code/FewViT/work_dirs_fss/tsne/voc_vit_split0_rd.npy'
        cls_path = '/media/data/ziqin/code/FewViT/work_dirs_fss/tsne/voc_vit_split0_cls.npy'
        proto_path = '/media/data/ziqin/code/FewViT/work_dirs_fss/tsne/voc_vit_split0_proto.npy'
            
        ## save the relationship descriptor #
        if int(self.test_iter) < 2000:
            for gt_cls in self.gt_label:
                rd_i = q1.clone().detach().squeeze().cpu().numpy()[gt_cls]
                proto_i = q.clone().detach().squeeze().cpu().numpy()[gt_cls]
                cls_i = cls.clone().detach().squeeze().cpu().numpy()

                self.save_rd[gt_cls].append(rd_i)
                self.save_proto[gt_cls].append(proto_i)
                self.save_cls[gt_cls].append(cls_i)

        elif int(self.test_iter) == 2000:
            np.save(rd_path, self.save_rd)
            np.save(proto_path, self.save_proto)
            np.save(cls_path, self.save_cls)
        # self.test_iter += 1
        return q_

    def get_qs_multihead(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, 768) cls: (bs, 768) 
        C, dim = q.shape
        bs, _ = cls.shape
        head = 12
        q = q.expand(bs, -1, -1) #(bs, base, 768)
        q1 = torch.einsum("bd,bcd->bcd", cls, q) #(bs, base, 768)
        # rd = q1.reshape(bs, C, head, -1) #(bs, base, 12, 64)
        # cls = cls.reshape(bs, head, -1) #(bs, 12, 64)
        mh = q1.reshape(bs, C, head, -1).sum(-1) #(bs, base, 12, 64) ->(bs, base, 12)
        # mh = torch.cosine_similarity(rd, cls.unsqueeze(1)) # (bs, base, 12)
        # mh = torch.einsum("bhd,bchd->bchd", cls, rd).sum(-1)
        q_ = torch.concat((q1, q, mh), dim=-1) # (bs, 20, 768+768+12)
        return q_

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


