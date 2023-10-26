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

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: # not do
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        ## self attn of q
        ## ADDED self-attn between queries: ##Do I need to add?
        # tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        ## cross attn between q and cls
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5 ## fixed: 0.1020620

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() # B, 15, 768
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) #(bs, 15, 768) - (bs, 15, 8, 96) - (bs, 8, 15, 96)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) #(bs, 8, 1024, 96)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3) #(bs, 8, 1024, 96)

        attn = (q @ k.transpose(-2, -1)) * self.scale #(bs, 8, 15, 96)@ (bs, 8, 96, 1024) -> (bs, 8, 15, 1024)
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1) ### do softmax on spatial dim
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C) # (bs, 8, 15, 1024) @ (bs, 8, 1024, 96)->(bs, 8, 15, 96)-> (bs, 15, 8, 96) ->(bs, 15, 768)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads

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

@HEADS.register_module()
class PlusHeadSeg(BaseDecodeHead):
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
            fuse_pred=1,
            use_proj=True,
            crop_train=False,
            **kwargs,
    ):
        super(PlusHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.fuse_pred = fuse_pred

        nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, self.dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(self.dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=self.dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        
        self.f_layer = TPN_DecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=self.dim * 4)
        # self.f_layer = nn.Linear(self.dim, self.dim, bias=False)
        
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)) * 0.01) ## * 0.01
        ## bg
        # self.bg_qs = nn.Parameter(torch.randn(1, in_channels))
        
        # self.q_proj = nn.Linear(in_channels, embed_dims)
        # self.q_proj = nn.Linear(in_channels * 2, embed_dims)
        # self.q_proj = nn.Linear(embed_dims * 2 + 12, embed_dims) ## MULTIHEAD
        ## ADDED FC for prototype
        # self.proto_proj = nn.Linear(embed_dims, embed_dims)

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
    
    def get_raw_pred(self, protos, cls_tokens, patch_tokens):
        c, dim = protos.shape
        bs, _ = cls_tokens.shape
        _, _, p, p = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(bs, dim, -1)
        
        # if f_layer is a transformer layer
        protos_ = self.f_layer(protos.expand(bs, -1, -1).transpose(0, 1), patch_tokens.permute(2, 0, 1))[0].transpose(1, 0) # (c, 768) -> (bs, c, 768)
        
        # if f_layer is a weight
        # protos_ = self.f_layer(protos).expand(bs, -1, -1)
        
        rd = torch.einsum("bd,bcd->bcd", cls_tokens, protos_)
        raw_scores = torch.bmm(rd, patch_tokens).reshape(bs, c, p, p)
        return raw_scores
        

    def forward(self, inputs, qs_epoch=None, novel_queries=None):
        patch_token = inputs[0][0]
        # cls_token = inputs[0][1] 

        if self.training:
            cls_token = inputs[0][1]
            # cls_token = self.get_cls_token(patch_token[0], self.base_qs.clone())
            ### [f(proto, patch).cls]T patch
            raw_pred = self.get_raw_pred(self.base_qs, cls_token, patch_token[-1])
        else:
            # REGISTER NOVEL: concat the novel queries in the right position
            both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                both_proto[self.seen_idx] = self.base_qs.clone()
                # print('Novel!:', novel_queries.sum(-1))
                both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
            else:
                both_proto[:] = self.base_qs.clone()
                
            ## the prototype of bg and person is large the others
            # both_proto[0] = both_proto[0]/10
            # both_proto[15] = both_proto[15]/10
            
            
            cls_token = inputs[0][1]
            # cls_token = self.get_cls_token(patch_token[0], both_proto.clone())
            raw_pred = self.get_raw_pred(both_proto, cls_token, patch_token[-1])
            
        x = []
        for stage_ in patch_token[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        attns = []
        maps_size = []
        qs = []

        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]

        if not self.training:
            # obtain the relationship descriptor between q and cls
            # q = self.q_proj(self.get_qs(both_proto, cls_token)).transpose(0, 1)
            
            # only use q=prototype (bs, c, 768)
            q = both_proto.expand(bs, -1, -1).transpose(0, 1)
        else:
            #### the momentum updated bg !!!!!!!! ()
            # q = self.q_proj(self.get_qs(self.base_qs, cls_token)).transpose(0, 1)
            q = self.base_qs.expand(bs, -1, -1).transpose(0, 1)
            
            ## update self.base_qs
            mom = self.update_m()
            self.cur_iter += 1
            if self.cur_iter % 100 == 0:
                print('check momentum:', qs_epoch.abs().mean(dim=-1))
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False

        for idx, decoder_ in enumerate(self.decoder):
            q_, attn_ = decoder_(q, lateral.transpose(0, 1))
            for q, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2) 
                attn = self.d3_to_d4(attn)
                maps_size.append(attn.size()[-2:])
                qs.append(q.transpose(0, 1))
                attns.append(attn)
        qs = torch.stack(qs, dim=0)

        outputs_seg_masks = []
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
        # print('pred:', pred.shape)
        raw_pred = F.interpolate(raw_pred,
                                          size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
        
        pred_final = raw_pred + self.fuse_pred * pred ## need sigmoid?
        out = {"pred_masks": pred_final}
        
        if self.training:
            out["qs_base"] = qs
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.2 is the best   
            return out["pred"]   
        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0): 
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def update_m(self, end_m=1.0, base_m=0.996): # 0.996
        if len(self.novel_idx) == 5: # for voc
            max_iter = 10000
        elif len(self.novel_idx) == 20: # for coco
            max_iter = 40000
        if self.cur_iter % 100 == 0:
            print('check prototype value:', self.base_qs.abs().mean(dim=-1))
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m

    # def get_qs(self, q, cls):
    #     # q_ = [q.cls, q]
    #     # q: (base, 512) cls: (bs, 512)
    #     C, dim = q.shape
    #     bs, _ = cls.shape
    #     q = q.expand(bs, -1, -1)
    #     q1 = torch.einsum("bd,bcd->bcd", cls, q)
    #     q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
    #     return q_

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


@HEADS.register_module()
class PlusHeadSegOnlyRaw(BaseDecodeHead):
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
        super(PlusHeadSegOnlyRaw, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx

        self.dim = embed_dims

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)
        
        # self.f_layer = TPN_DecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=self.dim * 4)
        self.f_layer = nn.Linear(self.dim, self.dim, bias=False)
        
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)) * 0.01) ## * 0.01
        ## bg
        # self.bg_qs = nn.Parameter(torch.randn(1, in_channels))
        
        # self.q_proj = nn.Linear(in_channels, embed_dims)
        # self.q_proj = nn.Linear(in_channels * 2, embed_dims)
        # self.q_proj = nn.Linear(embed_dims * 2 + 12, embed_dims) ## MULTIHEAD
        ## ADDED FC for prototype
        # self.proto_proj = nn.Linear(embed_dims, embed_dims)

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
    
    def get_raw_pred(self, protos, cls_tokens, patch_tokens):
        c, dim = protos.shape
        bs, _ = cls_tokens.shape
        _, _, p, p = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(bs, dim, -1)
        
        # if f_layer is a transformer layer
        # protos_ = self.f_layer(protos.expand(bs, -1, -1).transpose(0, 1), patch_tokens.permute(2, 0, 1))[0].transpose(1, 0) # (c, 768) -> (bs, c, 768)
        
        # if f_layer is a weight
        # protos_ = self.f_layer(protos).expand(bs, -1, -1)
        
        protos_ = protos.expand(bs, -1, -1)
        rd = torch.einsum("bd,bcd->bcd", cls_tokens, protos_)
        rd = self.f_layer(rd)
        raw_scores = torch.bmm(rd, patch_tokens).reshape(bs, c, p, p)
        return rd, raw_scores
        

    def forward(self, inputs, qs_epoch=None, novel_queries=None):
        patch_token = inputs[0][0]
        # cls_token = inputs[0][1] 

        if self.training:
            cls_token = inputs[0][1]
            # cls_token = self.get_cls_token(patch_token[0], self.base_qs.clone())
            ### [f(proto, patch).cls]T patch
            qs, raw_pred = self.get_raw_pred(self.base_qs, cls_token, patch_token[-1])
        else:
            # REGISTER NOVEL: concat the novel queries in the right position
            both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                both_proto[self.seen_idx] = self.base_qs.clone()
                # print('Novel!:', novel_queries.sum(-1))
                both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
            else:
                both_proto[:] = self.base_qs.clone()
            
            cls_token = inputs[0][1]
            # cls_token = self.get_cls_token(patch_token[0], both_proto.clone())
            qs, raw_pred = self.get_raw_pred(both_proto, cls_token, patch_token[-1])
            
        pred = F.interpolate(raw_pred, size=(self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False)
        
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = qs
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.2 is the best   
            return out["pred"]   
        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0): 
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def update_m(self, end_m=1.0, base_m=0.996): # 0.996
        if len(self.novel_idx) == 5: # for voc
            max_iter = 10000
        elif len(self.novel_idx) == 20: # for coco
            max_iter = 40000
        if self.cur_iter % 100 == 0:
            print('check prototype value:', self.base_qs.abs().mean(dim=-1))
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m

    # def get_qs(self, q, cls):
    #     # q_ = [q.cls, q]
    #     # q: (base, 512) cls: (bs, 512)
    #     C, dim = q.shape
    #     bs, _ = cls.shape
    #     q = q.expand(bs, -1, -1)
    #     q1 = torch.einsum("bd,bcd->bcd", cls, q)
    #     q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
    #     return q_

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

