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
        ## ADDED attn between queries: ##Do I need to add?
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
class ATMSingleHeadSeg(BaseDecodeHead):
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
            cls_type='cls',
            use_proj=True,
            crop_train=False,
            backbone_type='vit', #defualt
            finetune=False,
            **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.out_indices = out_indices
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.backbone_type = backbone_type
        self.finetune = finetune

        nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        
        self.cls_type = cls_type

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
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        
        if self.use_stages == 1:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), in_channels)))
            self.q_proj = nn.Linear(in_channels * 2, embed_dims)
        else:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), self.use_stages, in_channels)))
            q_proj = []
            for i in range(self.use_stages):
                q_proj_i = nn.Linear(in_channels * 2, embed_dims)
                self.add_module("q_proj_{}".format(i + 1), q_proj_i)
                q_proj.append(q_proj_i)
            self.q_proj = q_proj
            
        
    def init_proto(self):
        if self.backbone_type == 'dino': ## dino
            print('Initialized prototypes with DINO model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos_dino.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos_dino.npy'
        elif self.backbone_type == 'vit': ## vit
            print('Initialized prototypes with ViT model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos.npy'
        elif self.backbone_type == 'rn50': ## vit
            print('Initialized prototypes with ViT model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos_rn50.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos_rn50.npy'
                
        # only initialized the base classes in training and finetuning stratege
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
        # if inputs[0][1].shape[-1] == 768: # only for vit, not for resnet
        if self.cur_iter == 0:
            self.init_proto()
        
        if self.use_stages == 1:
            ## only use the last layer
            patch_token = inputs[0][0] #(bs, dim, 32, 32)
            if self.cls_type == 'cls':
                cls_token = inputs[0][1] #(bs, dim)
            elif self.cls_type == 'ave':
                cls_token = patch_token[0].mean(-1).mean(-1) #(bs, dim)
            elif self.cls_type == 'weighted':
                cls_token = patch_token[0] # (bs, dim, 32, 32)
                
        else:
            ## combine the patch_token from different layers
            patch_token = torch.stack([inputs[0][0][i_stage][1] for i_stage in range(self.use_stages-1)])
            patch_token = torch.concat([patch_token, inputs[0][0][-1].unsqueeze(0)]) #(use_stage, bs, dim, 32, 32) (9,10,11)layer
            cls_token = torch.stack([inputs[0][0][i_stage][0] for i_stage in range(self.use_stages-1)])
            cls_token = torch.concat([cls_token, inputs[0][1].unsqueeze(0)]) #(use_stage, bs, dim) (9,10,11)layer

        if not self.training: # only for testing
            if self.use_stages == 1:
                both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            else:
                both_proto = torch.zeros([len(self.all_idx), self.use_stages, self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                both_proto[self.seen_idx] = self.base_qs.clone()
                # print('Novel!:', novel_queries.sum(-1))
                both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
            else:
                both_proto[:] = self.base_qs.clone()

        x = []
        for stage_ in patch_token[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse() # 11 - 10 - 9 layer
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

        # lateral = laterals[-1]

        if not self.training:
            ## ADDED:
            # both_proto = self.proto_proj(both_proto)
            # q = both_proto.repeat(bs, 1, 1).transpose(0, 1) # V1 or V2
            
            #### how about use Learnable bg??
            # bg_qs = self.bg_qs / self.bg_qs.norm(dim=1, keepdim=True)
            # both_proto = torch.concat((bg_qs, both_proto[1:]),dim=0)
            if self.use_stages == 1:
                q_stage = self.q_proj(self.get_qs(both_proto, cls_token)).transpose(0, 1)
            else:
                rd = self.get_multi_qs(both_proto, cls_token)
                q_stage = torch.stack([self.q_proj[rd_stage](rd[rd_stage]).transpose(0, 1) for rd_stage in range(self.use_stages)]) # ()
            # q = self.q_proj(self.get_qs_save(both_proto, cls_token)).transpose(0, 1)
            # q = self.q_proj(self.get_qs_multihead(both_proto, cls_token)).transpose(0, 1) # V3

        else:
            ## V0: learnable q  
            # q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1) # for base classes while training
            # print('q:', q.sum())

            ## V1/V2: update base_qs by the protopyes from base images
            # q = self.base_qs.repeat(bs, 1, 1).transpose(0, 1)
            # self.cur_iter += 1
            # mom = self.update_m()
            # self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

            ## V3: q is generate by combine self.base_qs and cls token and use updata base_qs by the protopyes from base images
            # base_qs = self.base_qs
            # base_qs = self.proto_proj(self.base_qs) #ADDED
            
            #### how about use Learnable bg??
            # bg_qs = self.bg_qs / self.bg_qs.norm(dim=1, keepdim=True)
            # base_qs_epoch = torch.concat((bg_qs, self.base_qs[1:]),dim=0)
            # q = self.q_proj(self.get_qs(base_qs_epoch, cls_token)).transpose(0, 1)
            
            #### the momentum updated bg !!!!!!!! ()
            if self.use_stages == 1:
                q_stage = self.q_proj(self.get_qs(self.base_qs, cls_token)).transpose(0, 1)
            else:
                rd = self.get_multi_qs(self.base_qs, cls_token)
                q_stage = torch.stack([self.q_proj[rd_stage](rd[rd_stage]).transpose(0, 1) for rd_stage in range(self.use_stages)]) # ()
                
            # q = self.q_proj(self.get_qs_multihead(self.base_qs, cls_token)).transpose(0, 1)
            ## update self.base_qs
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q_stage).any()==False and torch.isinf(q_stage).any()==False

        # query_path = '/media/data/ziqin/code/FewViT/work_dirs_fss/tsne/voc_vit_split0_query.npy'
        # if int(self.test_iter) < 2000:
        #     for gt_cls in self.gt_label:
        #         query_i = q.clone().detach().squeeze().cpu().numpy()[gt_cls]
        #         self.save_query[gt_cls].append(query_i)
        # elif int(self.test_iter) == 2000:
        #     np.save(query_path, self.save_query)
        # self.test_iter += 1
        
        ## check the update of decoder
        # print('dc:', self.decoder[0].layers[0].linear1.weight.sum())
        # print('dc:', self.decoder[0].layers[0].linear1.weight.requires_grad)

        # reverse q
        # if self.use_stages > 1:
        #     q_stage = torch.flip(q_stage, dims=[0]) # for 9,10,11 to 11,10,9
        
        for idx, decoder_ in enumerate(self.decoder):
            if len(self.decoder) > 1:
                q_, attn_ = decoder_(q_stage[idx], laterals[idx].transpose(0, 1))
            else:
                q_, attn_ = decoder_(q_stage, laterals[idx].transpose(0, 1))
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
                                   
        out = {"pred_masks": pred}
        if self.training:
            out["qs_base"] = qs
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.2 is the best   
            # out["pred"] = self.semantic_inference_multi(outputs_seg_masks, self.seen_idx, 0.0) ## Change the balance factor： 0.2 is the best
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

    def get_multi_qs(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, stage, 512) cls: (stage, bs, 512)
        c, s, dim = q.shape
        s, bs, _ = cls.shape
        q = q.expand(bs, -1, -1, -1).permute(2, 0, 1, 3) # (s, bs, c, dim)
        q1 = torch.einsum("sbd,sbcd->sbcd", cls, q) * 100 ## check the value
        q_ = torch.concat((q1, q), dim=-1) # (stage, bs, c, 512+512)
        return q_
    
    def get_qs_save(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, 512) cls: (bs, 512)
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100
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
class ATMSingleHeadSegWORD(BaseDecodeHead):
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
            use_proj=True,
            crop_train=False,
            backbone_type='vit', #defualt
            **kwargs,
    ):
        super(ATMSingleHeadSegWORD, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.out_indices = out_indices
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.backbone_type = backbone_type

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
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        
        if self.use_stages == 1:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), in_channels)))
            self.q_proj = nn.Linear(in_channels, embed_dims)
        else:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), self.use_stages, in_channels)))
            q_proj = []
            for i in range(self.use_stages):
                q_proj_i = nn.Linear(in_channels, embed_dims)
                self.add_module("q_proj_{}".format(i + 1), q_proj_i)
                q_proj.append(q_proj_i)
            self.q_proj = q_proj
        
    def init_proto(self):
        if self.backbone_type == 'dino': ## dino
            print('Initialized prototypes with DINO model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos_dino.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos_dino.npy'
        elif self.backbone_type == 'vit': ## vit
            print('Initialized prototypes with ViT model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos.npy'
        
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
        if novel_queries is not None:
            return self.forward(inputs, novel_queries=novel_queries)
        else:
            return self.forward(inputs)

    def forward(self, inputs, qs_epoch=None, novel_queries=None):
        if inputs[0][1].shape[-1] == 768: # only for vit, not for resnet
            if self.cur_iter == 0:
                self.init_proto()
        
        if self.use_stages == 1:
            ## only use the last layer
            patch_token = inputs[0][0] #(bs, dim, 32, 32)
            cls_token = inputs[0][1] #(bs, dim)
        else:
            ## combine the patch_token from different layers
            patch_token = torch.stack([inputs[0][0][i_stage][1] for i_stage in range(self.use_stages-1)])
            patch_token = torch.concat([patch_token, inputs[0][0][-1].unsqueeze(0)]) #(use_stage, bs, dim, 32, 32)
            cls_token = torch.stack([inputs[0][0][i_stage][0] for i_stage in range(self.use_stages-1)])
            cls_token = torch.concat([cls_token, inputs[0][1].unsqueeze(0)]) #(use_stage, bs, dim)

        if not self.training:
            if self.use_stages == 1:
                both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            else:
                both_proto = torch.zeros([len(self.all_idx), self.use_stages, self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                both_proto[self.seen_idx] = self.base_qs.clone()
                # print('Novel!:', novel_queries.sum(-1))
                both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) / 10 ## how to test multi???
            else:
                both_proto[:] = self.base_qs.clone()

        x = []
        for stage_ in patch_token[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse() # 11 - 10 - 9 layer
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

        # lateral = laterals[-1]

        if not self.training:
            if self.use_stages == 1:
                q_stage = self.q_proj(both_proto.repeat(bs, 1, 1)).transpose(0, 1)
        else:
            if self.use_stages == 1:
                q_stage = self.q_proj(self.base_qs.repeat(bs, 1, 1)).transpose(0, 1)
                
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q_stage).any()==False and torch.isinf(q_stage).any()==False

        if self.use_stages > 1:
            q_stage = torch.flip(q_stage, dims=[0]) # for 9,10,11 to 11,10,9
        
        for idx, decoder_ in enumerate(self.decoder):
            if len(self.decoder) > 1:
                q_, attn_ = decoder_(q_stage[idx], laterals[idx].transpose(0, 1))
            else:
                q_, attn_ = decoder_(q_stage, laterals[idx].transpose(0, 1))
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
                                          
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = qs
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.2) ## Change the balance factor： 0.2 is the best   \
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


# @HEADS.register_module()
# class BinaryATMSingleHeadSeg(BaseDecodeHead):
#     def __init__(
#             self,
#             img_size,
#             in_channels,
#             seen_idx,
#             all_idx,
#             embed_dims=768,
#             num_layers=3,
#             num_heads=8,
#             use_stages=1,
#             out_indices=[11],
#             use_proj=True,
#             crop_train=False,
#             backbone_type='vit', #defualt
#             **kwargs,
#     ):
#         super(BinaryATMSingleHeadSeg, self).__init__(
#             in_channels=in_channels, **kwargs)

#         self.image_size = img_size
#         self.use_stages = use_stages
#         self.out_indices = out_indices
#         self.crop_train = crop_train
#         self.seen_idx = seen_idx
#         self.all_idx = all_idx
#         self.backbone_type = backbone_type

#         nhead = num_heads
#         self.dim = embed_dims
#         input_proj = []
#         proj_norm = []
#         atm_decoders = []

#         self.novel_idx = self.all_idx.copy()
#         for i_idx in self.seen_idx:
#             self.novel_idx.remove(i_idx)

#         for i in range(self.use_stages):
#             # FC layer to change ch
#             if use_proj:
#                 proj = nn.Linear(self.in_channels, self.dim)
#                 trunc_normal_(proj.weight, std=.02)
#             else:
#                 proj = nn.Identity()
#             self.add_module("input_proj_{}".format(i + 1), proj)
#             input_proj.append(proj)
#             # norm layer
#             if use_proj:
#                 norm = nn.LayerNorm(self.dim)
#             else:
#                 norm = nn.Identity()
#             self.add_module("proj_norm_{}".format(i + 1), norm)
#             proj_norm.append(norm)
#             # decoder layer
#             decoder_layer = TPN_DecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=self.dim * 4)
#             decoder = TPN_Decoder(decoder_layer, num_layers)
#             self.add_module("decoder_{}".format(i + 1), decoder)
#             atm_decoders.append(decoder)

#         self.input_proj = input_proj
#         self.proj_norm = proj_norm
#         self.decoder = atm_decoders
#         delattr(self, 'conv_seg')
        
#         self.register_buffer("cur_iter", torch.Tensor([0]))
#         self.register_buffer("base_qs", torch.randn((len(self.seen_idx), embed_dims)))

#         self.q_proj = nn.Linear(embed_dims * 2, embed_dims)
#         # self.q_proj = nn.Linear(embed_dims * 2 + 12, embed_dims) ## MULTIHEAD

#     def init_weights(self):
#         for n, m in self.named_modules():
#             if isinstance(m, nn.Linear):
#                 trunc_normal_init(m, std=.02, bias=0)
#             elif isinstance(m, nn.LayerNorm):
#                 constant_init(m, val=1.0, bias=0.0)

#     def forward_train(self, inputs, img_metas, gt_semantic_seg, bg_base_epoch, train_cfg):
#         seg_logits = self.forward(inputs, bg_base_epoch)

#         # gt_semantic_seg[gt_semantic_seg==-1] = 255
#         losses = self.losses(seg_logits, gt_semantic_seg)

#         return losses

#     def forward_test(self, inputs, img_metas, test_cfg, novel_queries=None, supp_cls=None):
#         # get the target of each cliped region
#         # ann_path = img_metas[0]['filename'].replace('jpg','png').replace('JPEGImages', 'Annotations').replace('/val2014/','/val_contain_crowd/')
#         # self.gt_label = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
#         # self.gt_label[self.gt_label==0] = 255 ## ignore the ground truth label
#         # self.gt_label[self.gt_label!=255] -= 1
#         # # self.gt_label = np.delete(self.gt_label, np.where(self.gt_label == 255))
#         # self.gt_label[self.gt_label!=supp_cls] = 255 # set the other object into 255, only focus on one target object
        
#         # self.gt_label[self.gt_label==supp_cls] = 254
#         # self.gt_label[self.gt_label!=254] = 0
#         # self.gt_label[self.gt_label==254] = 255
#         # im = Image.fromarray(self.gt_label)
#         # im = im.convert('L')
#         # if len(self.novel_idx) == 5:
#         #     im.save('/media/data/ziqin/work_dirs_fss/visualization_binary/gt_voc/' + img_metas[0]['filename'].split('/').pop(-1))
#         # elif len(self.novel_idx) == 20:
#         #     im.save('/media/data/ziqin/work_dirs_fss/visualization_binary/gt_coco/' + img_metas[0]['filename'].split('/').pop(-1))
#         return self.forward_binary(inputs, novel_queries=novel_queries)

#     def get_cls_token(self, patch_token, protos):
#         # patch_token(bs, 768, 32, 32) -> (bs, L, 768) protos(bs, 2(bg+base), 768)
#         B, D, _ ,_ = patch_token.size()
#         patch_token = patch_token.reshape(B, D, -1).permute(0, 2, 1)
#         L = patch_token.size(1)
        
#         mu = protos.mean(dim=0) #(768)
#         cls_token = torch.cosine_similarity(patch_token.reshape(-1,D),mu).reshape(B,L) # (bs, L)
#         cls_token = cls_token.softmax(dim=-1)
#         cls_token = (cls_token.unsqueeze(-1) * patch_token).sum(1) # (bs, L, D) -> (bs, D)
#         return cls_token

#     def semantic_inference(self, mask_pred, seen_idx, weight): #-0.2?
#         mask_pred = mask_pred.sigmoid()
#         mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
#         return mask_pred

#     def semantic_inference_binary(self, mask_pred, weight):
#         mask_pred = mask_pred.sigmoid()
#         mask_pred[:, 0] = mask_pred[:, 0] - weight
#         return mask_pred

#     def update_m(self, end_m=1.0, base_m=0.996):
#         max_iter = 20000
#         m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
#         return m

#     def get_qs(self, q, cls):
#         # q_ = [q.cls, q]
#         # q: (base, 512) cls: (bs, 512)
#         C, dim = q.shape
#         bs, _ = cls.shape
#         q = q.expand(bs, -1, -1)
#         q1 = torch.einsum("bd,bcd->bcd", cls, q)
#         q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
#         return q_
    
#     def get_qs_bgbase(self, q, cls):
#         # q: (bs, 2(bg+base), 512) cls: (bs, 512)
#         bs, dim = cls.shape
#         q1 = torch.einsum("bd,bcd->bcd", cls, q)
#         q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
#         return q_

#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_seg_masks):
#         return [
#             {"pred_masks": a}
#             # for a in zip(outputs_seg_masks[:-1])
#             for a in outputs_seg_masks[:-1]
#         ]

#     def d3_to_d4(self, t):
#         n, hw, c = t.size()
#         if hw % 2 != 0:
#             t = t[:, 1:]
#         h = w = int(math.sqrt(hw))
#         return t.transpose(1, 2).reshape(n, c, h, w)

#     def d4_to_d3(self, t):
#         return t.flatten(-2).transpose(-1, -2)

#     @force_fp32(apply_to=('seg_logit',))
#     def losses(self, seg_logit, seg_label, num_classes=None):
#         """Compute segmentation loss."""
#         if isinstance(seg_logit, dict):
#             # atm loss
#             seg_label = seg_label.squeeze(1)

#             loss = self.loss_decode(
#                 seg_logit,
#                 seg_label,
#                 ignore_index = self.ignore_index)

#             loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
#             return loss

#     def forward_binary(self, inputs, qs_epoch=None, novel_queries=None):
#         patch_token = inputs[0][0]
#         # cls_token = inputs[0][1]
#         if self.training:
#             cls_token = inputs[0][1]
#             # cls_token = self.get_cls_token(patch_token[0], self.base_qs.clone())
#         else:
#             # REGISTER NOVEL: concat the novel queries in the right position
#             # novel_proto = novel_queries.clone().unsqueeze(0)
#             # fake_proto = torch.randn_like(novel_proto)
#             bg_novel_proto = novel_queries.clone()
#             cls_token = inputs[0][1]
#             # cls_token = self.get_cls_token(patch_token[0], both_proto.clone())

#         x = []
#         for stage_ in patch_token[:self.use_stages]:
#             x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
#         x.reverse()
#         bs = x[0].size()[0]

#         laterals = []
#         attns = []
#         maps_size = []
#         qs = []

#         for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
#             lateral = norm_(proj_(x_))
#             if idx == 0:
#                 laterals.append(lateral)
#             else:
#                 if laterals[idx - 1].size()[1] == lateral.size()[1]:
#                     laterals.append(lateral + laterals[idx - 1])
#                 else:
#                     # nearest interpolate
#                     l_ = self.d3_to_d4(laterals[idx - 1])
#                     l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
#                     l_ = self.d4_to_d3(l_)
#                     laterals.append(l_ + lateral)

#         lateral = laterals[-1]

#         if not self.training:
#             # q = self.q_proj(self.get_qs(novel_proto.unsqueeze(0), cls_token)).transpose(0, 1)
#             q = self.q_proj(self.get_qs(bg_novel_proto, cls_token)).transpose(0, 1)
#             # q = self.q_proj(self.get_qs_multihead(novel_proto.unsqueeze(0), cls_token)).transpose(0, 1) # V3
#         else:
#             ### do not support training
#             q = self.q_proj(self.get_qs(bg_novel_proto, cls_token)).transpose(0, 1)
#             # q = self.q_proj(self.get_qs_multihead(self.base_qs, cls_token)).transpose(0, 1)
#             # self.cur_iter += 1
#             # mom = self.update_m()
#             # self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

#         assert torch.isnan(q).any()==False and torch.isinf(q).any()==False

#         for idx, decoder_ in enumerate(self.decoder):
#             q_, attn_ = decoder_(q, lateral.transpose(0, 1))
#             for q, attn in zip(q_, attn_):
#                 attn = attn.transpose(-1, -2) 
#                 attn = self.d3_to_d4(attn)
#                 maps_size.append(attn.size()[-2:])
#                 qs.append(q.transpose(0, 1))
#                 attns.append(attn)
#         qs = torch.stack(qs, dim=0)

#         outputs_seg_masks = []
#         size = maps_size[-1]

#         for i_attn, attn in enumerate(attns):
#             if True:
#                 outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
#             else:
#                 outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
#                                          F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

#         pred = F.interpolate(outputs_seg_masks[-1],
#                                           size=(self.image_size, self.image_size),
#                                           mode='bilinear', align_corners=False)
                                          
#         out = {"pred_masks": pred}
        
#         if self.training:
#             out["qs_base"] = qs
#             outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
#             out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
#         else:
#             out["pred"] = self.semantic_inference_binary(out["pred_masks"], 0.0) ## Change the balance factor： 0.0 is the best
#             return out["pred"]              
#         return out

#     def forward(self, inputs, bg_base_epoch):  #protos(bs, 2(bg+base), 768)
#         patch_token = inputs[0][0]
#         cls_token = inputs[0][1]
        
#         x = []
#         for stage_ in patch_token[:self.use_stages]:
#             x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
#         x.reverse()
#         bs = x[0].size()[0]

#         laterals = []
#         attns = []
#         maps_size = []
#         qs = []

#         for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
#             lateral = norm_(proj_(x_))
#             if idx == 0:
#                 laterals.append(lateral)
#             else:
#                 if laterals[idx - 1].size()[1] == lateral.size()[1]:
#                     laterals.append(lateral + laterals[idx - 1])
#                 else:
#                     # nearest interpolate
#                     l_ = self.d3_to_d4(laterals[idx - 1])
#                     l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
#                     l_ = self.d4_to_d3(l_)
#                     laterals.append(l_ + lateral)

#         lateral = laterals[-1]

#         q = self.q_proj(self.get_qs_bgbase(bg_base_epoch, cls_token)).transpose(0, 1)

#         assert torch.isnan(q).any()==False and torch.isinf(q).any()==False

#         for idx, decoder_ in enumerate(self.decoder):
#             q_, attn_ = decoder_(q, lateral.transpose(0, 1))
#             for q, attn in zip(q_, attn_):
#                 attn = attn.transpose(-1, -2) 
#                 attn = self.d3_to_d4(attn)
#                 maps_size.append(attn.size()[-2:])
#                 qs.append(q.transpose(0, 1))
#                 attns.append(attn)
#         qs = torch.stack(qs, dim=0)

#         outputs_seg_masks = []
#         size = maps_size[-1]

#         for i_attn, attn in enumerate(attns):
#             if True:
#                 outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
#             else:
#                 outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
#                                          F.interpolate(attn, size=size, mode='bilinear', align_corners=False))

#         pred = F.interpolate(outputs_seg_masks[-1],
#                                           size=(self.image_size, self.image_size),
#                                           mode='bilinear', align_corners=False)
                                          
#         out = {"pred_masks": pred}

#         if self.training:
#             out["qs_base"] = qs
#             outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
#             out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
#         else:
#             out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor
#             return out["pred"]                  
#         return out
             

@HEADS.register_module()
class MultiATMSingleHeadSeg(BaseDecodeHead):
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
            out_indices=[9, 10, 11],
            cls_type='cls',
            use_proj=True,
            crop_train=False,
            backbone_type='vit', #defualt
            finetune=False,
            **kwargs,
    ):
        super(MultiATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.out_indices = out_indices
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.backbone_type = backbone_type
        self.finetune = finetune

        nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        
        self.cls_type = cls_type

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        # for i in range(self.use_stages):
        for i in range(1):
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
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        if self.use_stages == 1:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), in_channels)))
        else:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), self.use_stages, in_channels)))
        self.q_proj = nn.Linear(in_channels * 2 * use_stages, embed_dims)
        if use_stages >1:
            self.patch_proj = nn.Linear(in_channels * use_stages, embed_dims)
        
    def init_proto(self):
        if self.backbone_type == 'dino': ## dino
            print('Initialized prototypes with DINO model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos_dino.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos_dino.npy'
        elif self.backbone_type == 'vit': ## vit
            print('Initialized prototypes with ViT model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos.npy'
        
        if self.use_stages == 1:
            init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, self.out_indices][self.seen_idx] ##for 11
        else:
            save_init_proto_idx = np.array([3, 6, 9, 10, 11])
            indices = np.where(np.isin(save_init_proto_idx, np.array(self.out_indices)))[0]
            init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, indices][self.seen_idx].squeeze() ##for 11
            
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
        if inputs[0][1].shape[-1] == 768: # only for vit, not for resnet
            if self.cur_iter == 0:
                self.init_proto()
        
        if self.use_stages == 1:
            ## only use the last layer
            patch_token = inputs[0][0] #(bs, dim, 32, 32)
            if self.cls_type == 'cls':
                cls_token = inputs[0][1] #(bs, dim)
            elif self.cls_type == 'ave':
                cls_token = patch_token[0].mean(-1).mean(-1) #(bs, dim)
            elif self.cls_type == 'weighted':
                cls_token = patch_token[0] # (bs, dim, 32, 32)
        else:
            ## combine the patch_token from different layers
            patch_token = torch.stack([inputs[0][0][i_stage][1] for i_stage in range(self.use_stages-1)])
            patch_token = torch.concat([patch_token, inputs[0][0][-1].unsqueeze(0)]) #(use_stage, bs, dim, 32, 32)
            if self.cls_type == 'cls':
                cls_token = torch.stack([inputs[0][0][i_stage][0] for i_stage in range(self.use_stages-1)])
                cls_token = torch.concat([cls_token, inputs[0][1].unsqueeze(0)]) #(use_stage, bs, dim)
            elif self.cls_type == 'ave':
                cls_token = patch_token.mean(-1).mean(-1) #(use_stage, bs, dim)
            elif self.cls_type == 'weighted':
                cls_token = patch_token # (use_stage, bs, dim, 32, 32)
            
            ## proj patch
            patch_token = self.patch_proj(patch_token.permute(1, 0, 2, 3, 4).flatten(1, 2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            patch_token = [patch_token]

        if not self.training:
            if self.use_stages == 1:
                both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_token[0].device)
            else:
                both_proto = torch.zeros([len(self.all_idx), self.use_stages, self.in_channels]).to(patch_token[0].device)
            if novel_queries is not None:
                if len(self.seen_idx) != 0:
                    both_proto[self.seen_idx] = self.base_qs.clone()
                    both_proto[self.novel_idx] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
                else:  #cross
                    both_proto[:] = torch.from_numpy(novel_queries).to(self.base_qs.dtype).to(self.base_qs.device)
            else:
                both_proto[:] = self.base_qs.clone()

        x = []
        # for stage_ in patch_token[:self.use_stages]:
        for stage_ in patch_token:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        bs = x[0].size()[0]

        attns = []
        maps_size = []
        qs = []
        
        if not self.training:
            if self.use_stages == 1:
                q_stage = self.q_proj(self.get_qs(both_proto, cls_token)).transpose(0, 1)
            else:
                rd = self.get_multi_qs(both_proto, cls_token)
                # q_stage = torch.stack([self.q_proj[rd_stage](rd[rd_stage]).transpose(0, 1) for rd_stage in range(self.use_stages)]) # ()
                q_stage = self.q_proj(rd).transpose(0, 1)
        else:      
            #### the momentum updated bg !!!!!!!! ()
            if self.use_stages == 1:
                q_stage = self.q_proj(self.get_qs(self.base_qs, cls_token)).transpose(0, 1)
            else:
                rd = self.get_multi_qs(self.base_qs, cls_token)
                # q_stage = torch.stack([self.q_proj[rd_stage](rd[rd_stage]).transpose(0, 1) for rd_stage in range(self.use_stages)]) # (stage, class, dim)
                q_stage = self.q_proj(rd).transpose(0, 1)
                
            ## update self.base_qs
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)

        assert torch.isnan(q_stage).any()==False and torch.isinf(q_stage).any()==False
        
        for idx, decoder_ in enumerate(self.decoder):
            if len(self.decoder) > 1:
                q_, attn_ = decoder_(q_stage[idx], x[idx].transpose(0, 1))
            else:
                q_, attn_ = decoder_(q_stage, x[idx].transpose(0, 1))
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

        # if self.use_stages > 1:
        #     fuse_stage_mask = torch.stack([attns[i] for i in list(range(len(attns)))[self.use_stages-1::self.use_stages]]) #()
        #     pred = []
        #     for i_stage in range(self.use_stages):
        #         pred.append(F.interpolate(fuse_stage_mask[i_stage], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False))
        #     pred = torch.stack(pred)
        #     pred = pred.mean(dim=0) ### but before sigmoid
            
        pred = F.interpolate(outputs_seg_masks[-1],
                                        size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
                                          
        out = {"pred_masks": pred}

        if self.training:
            out["qs_base"] = qs
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
    
    def semantic_inference_multi(self, mask_preds, seen_idx, weight=0.0): 
        mask_preds = mask_preds.sigmoid()
        mask_pred = mask_preds.mean(dim=0, keepdim=True)
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def update_m(self, end_m=1.0, base_m=0.996):
        max_iter = 20000
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m

    def get_qs(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, 512) cls: (bs, 512)
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100 ## check the value
        q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
        return q_

    def get_multi_qs(self, q, cls):
        # q_ = [q.cls, q]
        if self.cls_type == 'weighted': # cls is the patch_embeddings, cls (use_stage, bs, 768, 32,32)
            # q: (c, stage, 512) cls: (stage, bs, 768, 32,32)
            c, _, dim = q.shape
            cls = cls.flatten(-2, -1).permute(0, 1, 3, 2) # (use_stage, bs, 768, 32*32) -> (use_stage, bs, n, d)
            s, bs, n, _ = cls.shape
            q1 = torch.einsum("sbnd,scd->sbcnd", cls, q.permute(1, 0, 2)) #(s, bs, c, n, d)
            cls_norm = cls.permute(0, 1, 3, 2) / torch.norm(cls.permute(0, 1, 3, 2), dim=2, keepdim=True) # (s, bs, d, n)
            q_norm = q / torch.norm(q, dim=-1, keepdim=True) #(c, s, d)
            q_norm = q_norm.expand(bs, -1, -1, -1).permute(2, 0, 1, 3) #(s, b, c, d)
            
            ## Version1 sigmoid
            # similarity = torch.bmm(q_norm.reshape(s*bs, c, dim), cls_norm.reshape(s*bs, dim, n)).sigmoid().reshape(s, bs, c, n)## (s, bs, c, n)
            # similarity = similarity / (similarity.sum(-1).unsqueeze(-1))
            
            ## Version2 softmax
            similarity = (torch.bmm(q_norm.reshape(s*bs, c, dim), cls_norm.reshape(s*bs, dim, n))/0.1).softmax(-1).reshape(s, bs, c, n)
            
            q1 = (q1 * similarity.unsqueeze(-1)).sum(dim=-2) # (s, bs, c, d)
            
            q = q.expand(bs, -1, -1, -1).permute(2, 0, 1, 3)
            q_ = torch.concat((q1, q), dim=-1)# (stage, bs, c, 512+512)
            q_ =  q_.permute(1, 2, 0, 3).reshape(bs, c, -1)# (stage, bs, c, (512+512)*stage)
            
        else:   
            c, s, dim = q.shape
            s, bs, _ = cls.shape
            q = q.expand(bs, -1, -1, -1).permute(2, 0, 1, 3) # (s, bs, c, dim)
            q1 = torch.einsum("sbd,sbcd->sbcd", cls, q) * 100 ## check the value
            q_ = torch.concat((q1, q), dim=-1) # (stage, bs, c, 512+512)
            q_ =  q_.permute(1, 2, 0, 3).reshape(bs, c, -1) # (stage, bs, c, (512+512)*stage)
        return q_
    
    def get_qs_save(self, q, cls):
        # q_ = [q.cls, q]
        # q: (base, 512) cls: (bs, 512)
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100
        q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)

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
        mh = q1.reshape(bs, C, head, -1).sum(-1) #(bs, base, 12, 64) ->(bs, base, 12)
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
class BinaryATMSingleHeadSeg(BaseDecodeHead):
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
            out_indices=[9, 10, 11],
            cls_type='cls',
            use_proj=True,
            crop_train=False,
            backbone_type='vit', #defualt
            **kwargs,
    ):
        super(BinaryATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.out_indices = out_indices
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.backbone_type = backbone_type
        self.cls_type = cls_type
        
        nhead = num_heads
        self.dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        for i in range(1):
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
        delattr(self, 'conv_seg')
        
        self.register_buffer("cur_iter", torch.Tensor([0]))
        if self.use_stages == 1:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), in_channels)))
            # self.register_buffer("base_qs", torch.zeros((81, in_channels)))
        else:
            self.register_buffer("base_qs", torch.zeros((len(self.seen_idx), self.use_stages, in_channels)))
            
        self.q_proj = nn.Linear(in_channels * 2 * use_stages, embed_dims)
        if use_stages >1:
            self.patch_proj = nn.Linear(in_channels * use_stages, embed_dims)
        
    def init_proto(self):
        if self.backbone_type == 'dino': ## dino
            print('Initialized prototypes with DINO model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos_dino.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos_dino.npy'
        elif self.backbone_type == 'vit': ## vit
            print('Initialized prototypes with ViT model')
            if len(self.seen_idx) == 16 or len(self.seen_idx) == 21: # voc
                path = '/media/data/ziqin/data_fss/init_protos/voc_protos.npy'
            elif len(self.seen_idx) == 61 or len(self.seen_idx) == 81:
                path = '/media/data/ziqin/data_fss/init_protos/coco_protos.npy'
        
        if self.use_stages == 1:
            init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, self.out_indices][self.seen_idx] ##for 11
        else:
            save_init_proto_idx = np.array([3, 6, 9, 10, 11])
            indices = np.where(np.isin(save_init_proto_idx, np.array(self.out_indices)))[0]
            init_protos = torch.from_numpy(np.load(path)).to(self.base_qs.dtype).to(self.base_qs.device)[:, indices][self.seen_idx].squeeze() ##for 11
            
        self.base_qs.data = init_protos        

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, bg_base_epoch, train_cfg):
        seg_logits = self.forward(inputs, bg_base_epoch)

        # gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, novel_queries=None, supp_cls=None):
        return self.forward_binary(inputs, novel_queries=novel_queries)

    def get_cls_token(self, patch_token, protos):
        # patch_token(bs, 768, 32, 32) -> (bs, L, 768) protos(bs, 2(bg+base), 768)
        B, D, _ ,_ = patch_token.size()
        patch_token = patch_token.reshape(B, D, -1).permute(0, 2, 1)
        L = patch_token.size(1)
        
        mu = protos.mean(dim=0) #(768)
        cls_token = torch.cosine_similarity(patch_token.reshape(-1,D),mu).reshape(B,L) # (bs, L)
        cls_token = cls_token.softmax(dim=-1)
        cls_token = (cls_token.unsqueeze(-1) * patch_token).sum(1) # (bs, L, D) -> (bs, D)
        return cls_token

    def semantic_inference(self, mask_pred, seen_idx, weight): #-0.2?
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    def semantic_inference_binary(self, mask_pred, weight):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:, 0] = mask_pred[:, 0] - weight
        
        ## get the max logit
        # mask_pred = torch.concat([mask_pred[:,:-1].max(dim=1)[0].unsqueeze(1), mask_pred[:,-1].unsqueeze(1)], dim=1)
        mask_pred = torch.concat([mask_pred[:,:-1].max(dim=1)[0].unsqueeze(1), (1-mask_pred[:,:-1].max(dim=1)[0].unsqueeze(1))], dim=1)

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
    
    def get_multi_qs(self, q, cls):
        # q_ = [q.cls, q]
        if self.cls_type == 'weighted': # cls is the patch_embeddings, cls (use_stage, bs, 768, 32,32)
            # q: (c, stage, 512) cls: (stage, bs, 768, 32,32)
            c, _, dim = q.shape
            cls = cls.flatten(-2, -1).permute(0, 1, 3, 2) # (use_stage, bs, 768, 32*32) -> (use_stage, bs, n, d)
            s, bs, n, _ = cls.shape
            q1 = torch.einsum("sbnd,scd->sbcnd", cls, q.permute(1, 0, 2)) #(s, bs, c, n, d)
            cls_norm = cls.permute(0, 1, 3, 2) / torch.norm(cls.permute(0, 1, 3, 2), dim=2, keepdim=True) # (s, bs, d, n)
            q_norm = q / torch.norm(q, dim=-1, keepdim=True) #(c, s, d)
            q_norm = q_norm.expand(bs, -1, -1, -1).permute(2, 0, 1, 3) #(s, b, c, d)
            
            ## Version1 sigmoid
            # similarity = torch.bmm(q_norm.reshape(s*bs, c, dim), cls_norm.reshape(s*bs, dim, n)).sigmoid().reshape(s, bs, c, n)## (s, bs, c, n)
            # similarity = similarity / (similarity.sum(-1).unsqueeze(-1))
            
            ## Version2 softmax
            similarity = (torch.bmm(q_norm.reshape(s*bs, c, dim), cls_norm.reshape(s*bs, dim, n))/0.1).softmax(-1).reshape(s, bs, c, n)
            
            q1 = (q1 * similarity.unsqueeze(-1)).sum(dim=-2) # (s, bs, c, d)
            
            q = q.expand(bs, -1, -1, -1).permute(2, 0, 1, 3)
            q_ = torch.concat((q1, q), dim=-1)# (stage, bs, c, 512+512)
            q_ =  q_.permute(1, 2, 0, 3).reshape(bs, c, -1)# (stage, bs, c, (512+512)*stage)
            
        else:   
            c, s, dim = q.shape
            s, bs, _ = cls.shape
            q = q.expand(bs, -1, -1, -1).permute(2, 0, 1, 3) # (s, bs, c, dim)
            q1 = torch.einsum("sbd,sbcd->sbcd", cls, q) * 100 ## check the value
            q_ = torch.concat((q1, q), dim=-1) # (stage, bs, c, 512+512)
            q_ =  q_.permute(1, 2, 0, 3).reshape(bs, c, -1) # (stage, bs, c, (512+512)*stage)
        return q_
    
    def get_qs_bgbase(self, q, cls):
        # q: (bs, 2(bg+base), 512) cls: (bs, 512)
        bs, dim = cls.shape
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1) # (bs, 20, 512+512)
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

    def forward_binary(self, inputs, qs_epoch=None, novel_queries=None):
        if self.use_stages == 1:
            ## only use the last layer
            patch_token = inputs[0][0] #(bs, dim, 32, 32)
            if self.cls_type == 'cls':
                cls_token = inputs[0][1] #(bs, dim)
            elif self.cls_type == 'ave':
                cls_token = patch_token[0].mean(-1).mean(-1) #(bs, dim)
            elif self.cls_type == 'weighted':
                cls_token = patch_token[0] # (bs, dim, 32, 32)
        else:
            ## combine the patch_token from different layers
            patch_token = torch.stack([inputs[0][0][i_stage][1] for i_stage in range(self.use_stages-1)])
            patch_token = torch.concat([patch_token, inputs[0][0][-1].unsqueeze(0)]) #(use_stage, bs, dim, 32, 32)
            if self.cls_type == 'cls':
                cls_token = torch.stack([inputs[0][0][i_stage][0] for i_stage in range(self.use_stages-1)])
                cls_token = torch.concat([cls_token, inputs[0][1].unsqueeze(0)]) #(use_stage, bs, dim)
            elif self.cls_type == 'ave':
                cls_token = patch_token.mean(-1).mean(-1) #(use_stage, bs, dim)
            elif self.cls_type == 'weighted':
                cls_token = patch_token # (use_stage, bs, dim, 32, 32)
            
            ## proj patch
            patch_token = self.patch_proj(patch_token.permute(1, 0, 2, 3, 4).flatten(1, 2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            patch_token = [patch_token]
            
        bg_novel_proto = novel_queries.clone()

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
            if self.use_stages == 1:
                q = self.q_proj(self.get_qs(bg_novel_proto, cls_token)).transpose(0, 1)
            else:
                q = self.q_proj(self.get_multi_qs(bg_novel_proto, cls_token)).transpose(0, 1)
        else:
            assert AttributeError('Only for evaluation')
            
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
                                          
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = qs
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference_binary(out["pred_masks"], 0.0) ## Change the balance factor： 0.0 is the best
            return out["pred"]              
        return out

    def forward(self, inputs, bg_base_epoch):  #protos(bs, 2(bg+base), 768)
        patch_token = inputs[0][0]
        cls_token = inputs[0][1]
        
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

        q = self.q_proj(self.get_qs_bgbase(bg_base_epoch, cls_token)).transpose(0, 1)

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
                                          
        out = {"pred_masks": pred}

        if self.training:
            out["qs_base"] = qs
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor
            return out["pred"]                  
        return out
             
