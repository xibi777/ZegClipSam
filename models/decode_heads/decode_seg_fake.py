    
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
       
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
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

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2
    
       
class FullTPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(FullTPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # self attn
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross attn
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
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads

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
class FakeHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
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
        super(FakeHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        
        self.rd_type = rd_type
        # nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        
        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

        self.input_proj = input_proj
        self.proj_norm = proj_norm

        delattr(self, 'conv_seg')
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)))

        # self.withRD = withRD
        # if withRD:
        #     self.q_proj = nn.Linear(dim * 2, dim)
        # else:
        #     self.q_proj = nn.Linear(dim, dim)
        if 'qclsq' in self.rd_type or 'qcls_norm_q' in self.rd_type and 'pca' not in self.rd_type:
            self.q_proj = nn.Linear(dim * 2, dim)
        elif self.rd_type == 'qcls_pca':
            self.q_proj = nn.Linear(10, dim) ## for voc
        elif self.rd_type == 'qclsq_pca':
            self.q_proj = nn.Linear(dim + 10, dim) ## for voc
        else:
            self.q_proj = nn.Linear(dim, dim)
            
        if self.rd_type == 'combine':
            self.combine_proj = nn.Linear(4, 1)
        
        if self.rd_type == 'mlpcom':
            self.combine_proj = MLP_Proj(2, [64, 32], 1)
            
        self.decode_type = decode_type
        if decode_type == 'mlp':
            self.decoder = MLP(dim, int(dim/4), dim, num_layers=3)
        elif decode_type == 'attn':
            decoder_layer = FullTPN_DecoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4)
            self.decoder = TPN_Decoder(decoder_layer, num_layers=1)
        elif decode_type == 'conv':
            self.decoder = FCNHead(num_convs=2, kernel_size=1)
        elif decode_type == 'mlpfuse':
            self.decoder = MLPFuse([dim*3, int(dim*3/4), int(dim/4), dim])
        else:
            pass

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs, gt_semantic_seg=None, novel_clip_feats=None, novel_labels=None):
        patch_tokens = inputs[0][0][-1] #(bs, 768, 32, 32) patch embeddings from the last layer
        cls_token = inputs[0][1]
        ori_patch_tokens = inputs[0][2]
        
        bs, dim, p, _ = patch_tokens.size()
        patch_tokens = patch_tokens.reshape(bs, dim , -1)
        
        # get qs_epoch
        # calculate relationship descriptor RD=qclsq, q is from average targeted patch embeddings
        if self.training:
            qs_epoch = self.extract_base_proto_epoch(self.base_qs, patch_tokens.reshape(bs, dim, p, p).clone().detach(), gt_semantic_seg.squeeze())
        else:
            if not hasattr(self, 'both_proto'):
                if novel_clip_feats is not None: # few-shot
                    way, shot, _, _, _ = novel_clip_feats.size()
                    novel_clip_feats = novel_clip_feats.reshape(way*shot, dim ,p*p)
                    print('\n' + '------------Registering the prototypes of novel classes-----------')
                    self.novel_queries = self.extract_novel_proto(novel_clip_feats.reshape(way, shot, dim, p, p), novel_labels)
                    # print('------------Registering the aug prototypes of novel classes-----------')
                    # self.novel_queries = self.extract_aug_novel_proto(self.supp_dir, self.supp_path, way=len(self.novel_class), shot=self.shot)
                    print('Done! The dimension of novel prototypes is: ', self.novel_queries.shape)
                    # REGISTER NOVEL: concat the novel queries in the right position
                    self.both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_tokens[0].device)
                    self.both_proto[self.seen_idx] = self.base_qs.clone()
                    self.both_proto[self.novel_idx] = torch.from_numpy(self.novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
                else: #fully supervised
                    self.both_proto = self.base_qs.clone()
             
        # get rd and update base_qs
        if self.training:
            q = self.q_proj(self.get_qs(self.base_qs, cls_token, self.rd_type))
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)
        else:
            q = self.q_proj(self.get_qs(self.both_proto, cls_token, self.rd_type)) #bcd , need normalize??

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False
               
        # refine the patch embeddings
        # if multi-stage:
        if len(inputs[0][0])>1:
            all_patch_tokens = torch.cat(inputs[0][0], dim=1).reshape(bs, len(inputs[0][0])*dim, p*p) ## ()
        else:
            all_patch_tokens = patch_tokens
        if self.decode_type is not None:
            if 'mlp' in self.decode_type:
                all_patch_tokens = self.decoder(all_patch_tokens.transpose(2,1))
                all_patch_tokens = all_patch_tokens.transpose(2,1)
            elif self.decode_type=='attn':    
                all_patch_tokens_list, _ = self.decoder(all_patch_tokens.transpose(2, 1), all_patch_tokens.transpose(2, 1)) # q/k/v=patch embedding
                all_patch_tokens = all_patch_tokens_list[-1].transpose(2,1) #(2, 768, 32*32)
            else:
                assert AttributeError('Donot support this decode type')
        else:
            pass 
        
        # get prediction and loss
        pred_logits = torch.einsum("bdn,bcd->bcn", all_patch_tokens, q) # matching directly
        c = pred_logits.shape[1]
        pred_logits = pred_logits.reshape(bs, c, p, p)
        # pred_logits = patch_tokens @ text_token.t()

        pred = F.interpolate(pred_logits, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
                                          
        out = {"pred_masks": pred}
        
        if self.training:
            out["qs_base"] = q.transpose(0, 1).unsqueeze(0) #(1, bs, c, 768)
            # outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            # out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.0 is the best   
            return out["pred"]   
        return out                 
        
    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.2 #reduce background, for learnable bg use add bg 0.2
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

    def get_qs(self, q, cls, type):
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        if type == 'qclsq':# q = [q.cls, q]
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10 # added or not?
            q_ = torch.concat((q1, q), dim=-1)
        elif type == 'qcls_norm_q': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
        elif type == 'qcls': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) * 10
        elif type == 'qcls_norm': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
        elif type == 'q_cls':
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.abs(cls - q)
        elif type == 'q':
            q_ = q.to(cls.dtype)
        elif type == 'combine':
            # qcls,|q-cls|,q,cls
            qcls = torch.einsum("bd,bcd->bcd", cls, q) * 10
            q_cls = torch.abs(cls.expand(C, -1, -1).permute(1, 0, 2) - q)
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((qcls.unsqueeze(-1), q_cls.unsqueeze(-1), q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 4)
            q_ = self.combine_proj(q_).squeeze(-1)
        elif type == 'mlpcom':
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 2)
            q_ = self.combine_proj(q_).squeeze(-1) #(bs, c, dim)
        elif type == 'qcls_norm_aug': # reduce the mean also on novel classes
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_norm_q_aug': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q_ = torch.bmm(q1, v[:, :, :])
            # a = q_pca.squeeze() / torch.norm(q_pca.squeeze(), dim=-1, keepdim=True)
            # similarity = torch.mm(a, a.T)
        elif type == 'qclsq_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q1_pca = torch.bmm(q1, v[:, :, :])
            q_ = torch.concat((q1_pca, q), dim=-1) # q + 512
        else:
            assert AttributeError('Donot support this rd type')
        return q_


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
        # get the target of each cliped region
        # ann_path = img_metas[0]['filename'].replace('jpg','png').replace('JPEGImages', 'Annotations')
        # self.gt_ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        # self.gt_label = np.unique(self.gt_ann)
        # self.gt_label[self.gt_label==0] = 255 ## ignore the ground truth label
        # self.gt_label[self.gt_label!=255] -= 1
        # self.gt_label = np.delete(self.gt_label, np.where(self.gt_label == 255))
        if novel_clip_feats is not None:
            return self.forward(inputs, novel_clip_feats=novel_clip_feats, novel_labels=novel_labels)
        else:
            return self.forward(inputs)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs, gt_semantic_seg)
        gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses    
    
    def extract_base_proto_epoch(self, qs, patch_features, targets):
        ## qs(base, 768), patch(bs, 768, 32, 32), gt(bs, 512, 512)
        assert patch_features.shape[0] == targets.shape[0]
        patch_features = F.interpolate(patch_features, size=targets.shape[-2:], mode='bilinear', align_corners=False) ## (512, 512)
        # targets = F.interpolate(targets.unsqueeze(1).float(), size=patch_features.shape[-2:], mode='nearest').squeeze(1).int() ## (32, 32)
        # resnet50: patch (bs, 2048. 512, 512)
        qs_epoch = torch.zeros_like(qs) # [15, dim] resnet:[15,512] (2048-512) with proj
        num_base = torch.zeros(qs.shape[0]).to(qs_epoch.device)  #(15)

        n = 0
        for targets_per_image in targets:
            # for n th image in a batch
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != 255]
            if len(gt_cls) != 0:
                for cls in gt_cls:
                    num_base[cls] += 1
                    binary_mask = torch.zeros_like(patch_features[0,0])
                    binary_mask[targets_per_image == cls] = 1
                    proto_cls = (torch.einsum("dhw,hw->dhw", patch_features[n].squeeze(), binary_mask).sum(-1).sum(-1)) / binary_mask.sum()
                    qs_epoch[cls, :] = proto_cls
            n += 1

        # norm for each base classes
        qs_epoch[num_base!=0] = qs_epoch[num_base!=0] / num_base[num_base!=0].unsqueeze(-1) #(15, 768)
        return qs_epoch


    def extract_novel_proto(self, novel_patch_embeddings, novel_labels):
        ## load Image and Annotations, no augmentation but how to handle the crop??
        ## seed from GFS-Seg: 5
        way, shot, dim, p, p = novel_patch_embeddings.size()
        all_novel_queries = np.zeros([way, 768]) # [way, dim]
        novel_labels = novel_labels.reshape(way*shot, 512, 512)

        # generate label for each image
        labels = [[i]*shot for i in range(way)]
        labels = list(itertools.chain.from_iterable(labels))

        n = 0
        for patch_embeddings in novel_patch_embeddings.reshape(way*shot, dim, p, p):
            # obtain the mask
            novel_label = F.interpolate(novel_labels[n].unsqueeze(0).unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int()
            binary_label = torch.zeros_like(novel_label)
            binary_label[novel_label == self.novel_idx[labels[n]]] = 1
            assert binary_label.sum() != 0
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim

            # print('proto:', str(int(n/num_each_supp))+str(labels[n]))
            all_novel_queries[labels[n], :] += proto.clone().cpu().numpy()
            n += 1
        
        # norm for 1shot or 5shot
        all_novel_queries /= shot

        return all_novel_queries/15 #??????


@HEADS.register_module()
class MaskFakeHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
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
        super(MaskFakeHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        
        self.rd_type = rd_type
        # nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        
        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

        self.input_proj = input_proj
        self.proj_norm = proj_norm

        delattr(self, 'conv_seg')
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)))

        # self.withRD = withRD
        # if withRD:
        #     self.q_proj = nn.Linear(dim * 2, dim)
        # else:
        #     self.q_proj = nn.Linear(dim, dim)
        if 'qclsq' in self.rd_type or 'qcls_norm_q' in self.rd_type and 'pca' not in self.rd_type:
            self.q_proj = nn.Linear(dim * 2, dim)
        elif self.rd_type == 'qcls_pca':
            self.q_proj = nn.Linear(10, dim) ## for voc
        elif self.rd_type == 'qclsq_pca':
            self.q_proj = nn.Linear(dim + 10, dim) ## for voc
        else:
            self.q_proj = nn.Linear(dim, dim)
            
        if self.rd_type == 'combine':
            self.combine_proj = nn.Linear(4, 1)
        
        if self.rd_type == 'mlpcom':
            self.combine_proj = MLP_Proj(2, [64, 32], 1)
            
        self.decode_type = decode_type
        if decode_type == 'mlp':
            self.decoder = MLP(dim, int(dim/4), dim, num_layers=3)
        elif decode_type == 'attn':
            decoder_layer = FullTPN_DecoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4)
            self.decoder = TPN_Decoder(decoder_layer, num_layers=1)
        elif decode_type == 'conv':
            self.decoder = FCNHead(num_convs=2, kernel_size=1)
        else:
            pass

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs, novel_prototypes=None):
        patch_tokens = inputs[0][0] #(bs, 768, 32, 32)
        cls_token = inputs[1]
        
        ## mask_cls
        if self.training:
            new_cls_token = inputs[2]
            new_cls_img_idx = inputs[3]
            new_gt_cls = inputs[4]
        
        bs, dim, p, _ = patch_tokens.size()
        patch_tokens = patch_tokens.reshape(bs, dim , -1)
        
        # get qs_epoch
        # calculate relationship descriptor RD=qclsq, q is from average targeted patch embeddings
        if self.training:
            qs_epoch = self.extract_base_proto_epoch(self.base_qs, new_cls_token, new_cls_img_idx, new_gt_cls)
        else:
            if not hasattr(self, 'both_proto'):
                if novel_prototypes is not None: # few-shot
                    print('\n' + '------------Registering the prototypes of novel classes-----------')
                    self.novel_queries = novel_prototypes.detach().clone().cpu().numpy() * 10 ###???????????? why need to *10
                    # print('------------Registering the aug prototypes of novel classes-----------')
                    # self.novel_queries = self.extract_aug_novel_proto(self.supp_dir, self.supp_path, way=len(self.novel_class), shot=self.shot)
                    print('Done! The dimension of novel prototypes is: ', self.novel_queries.shape)
                    # REGISTER NOVEL: concat the novel queries in the right position
                    self.both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_tokens[0].device)
                    self.both_proto[self.seen_idx] = self.base_qs.clone()
                    self.both_proto[self.novel_idx] = torch.from_numpy(self.novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
                else: #fully supervised
                    self.both_proto = self.base_qs.clone()
             
        # get rd and update base_qs
        if self.training:
            q = self.q_proj(self.get_qs(self.base_qs, cls_token, self.rd_type))
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)
        else:
            q = self.q_proj(self.get_qs(self.both_proto, cls_token, self.rd_type)) #bcd , need normalize??

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False
               
        # refine the patch embeddings        
        if self.decode_type is not None:
            if self.decode_type=='mlp':
                patch_tokens = self.decoder(patch_tokens.transpose(2,1))
                patch_tokens = patch_tokens.transpose(2,1)
            elif self.decode_type=='attn':    
                patch_tokens_list, _ = self.decoder(patch_tokens.transpose(2, 1), patch_tokens.transpose(2, 1)) # q/k/v=patch embedding
                patch_tokens = patch_tokens_list[-1].transpose(2,1) #(2, 768, 32*32)
            else:
                assert AttributeError('Donot support this decode type')
        else:
            pass 
        
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
            # outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            # out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.2) ## Change the balance factor： 0.0 is the best   
            return out["pred"]   
        return out                 
        
    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.0 #reduce background, for learnable bg use add bg 0.2
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

    def get_qs(self, q, cls, type):
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        if type == 'qclsq':# q = [q.cls, q]
            q1 = torch.einsum("bd,bcd->bcd", cls, q) # added or not?
            q_ = torch.concat((q1, q), dim=-1)
        elif type == 'qcls_norm_q': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
        elif type == 'qcls': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q)
        elif type == 'qcls_norm': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
        elif type == 'q_cls':
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.abs(cls - q)
        elif type == 'q':
            q_ = q.to(cls.dtype)
        elif type == 'combine':
            # qcls,|q-cls|,q,cls
            qcls = torch.einsum("bd,bcd->bcd", cls, q)
            q_cls = torch.abs(cls.expand(C, -1, -1).permute(1, 0, 2) - q)
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((qcls.unsqueeze(-1), q_cls.unsqueeze(-1), q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 4)
            q_ = self.combine_proj(q_).squeeze(-1)
        elif type == 'mlpcom':
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 2)
            q_ = self.combine_proj(q_).squeeze(-1) #(bs, c, dim)
        elif type == 'qcls_norm_aug': # reduce the mean also on novel classes
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_norm_q_aug': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q)
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q_ = torch.bmm(q1, v[:, :, :])
            # a = q_pca.squeeze() / torch.norm(q_pca.squeeze(), dim=-1, keepdim=True)
            # similarity = torch.mm(a, a.T)
        elif type == 'qclsq_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q)
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q1_pca = torch.bmm(q1, v[:, :, :])
            q_ = torch.concat((q1_pca, q), dim=-1) # q + 512
        else:
            assert AttributeError('Donot support this rd type')
        return q_


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
        max_iter = 20000 ## ????? should be changed??
        m = end_m - (end_m - base_m) * (cos(pi * self.cur_iter / float(max_iter)) + 1) / 2
        return m
    
    def forward_test(self, inputs, img_metas, test_cfg, novel_prototypes=None):
        # get the target of each cliped region
        # ann_path = img_metas[0]['filename'].replace('jpg','png').replace('JPEGImages', 'Annotations')
        # self.gt_ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        # self.gt_label = np.unique(self.gt_ann)
        # self.gt_label[self.gt_label==0] = 255 ## ignore the ground truth label
        # self.gt_label[self.gt_label!=255] -= 1
        # self.gt_label = np.delete(self.gt_label, np.where(self.gt_label == 255))
        if novel_prototypes is not None:
            return self.forward(inputs, novel_prototypes)
        else:
            return self.forward(inputs)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs, gt_semantic_seg)
        gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses    
    
    def extract_base_proto_epoch(self, qs, new_cls, img_idx, new_gt_cls):
        qs_epoch = torch.zeros_like(qs) # [15, dim] resnet:[15,512] (2048-512) with proj
        
        new_gt_cls_unique = torch.tensor(new_gt_cls).unique()
        if len(new_gt_cls_unique) != 0:
            for cls in new_gt_cls_unique:
                qs_epoch[cls] = new_cls[(torch.Tensor(new_gt_cls) == cls).nonzero()].mean(dim=0) #norm
        return qs_epoch


    def extract_novel_proto(self, novel_patch_embeddings, novel_labels):
        ## load Image and Annotations, no augmentation but how to handle the crop??
        ## seed from GFS-Seg: 5
        way, shot, dim, p, p = novel_patch_embeddings.size()
        all_novel_queries = np.zeros([way, 768]) # [way, dim]
        novel_labels = novel_labels.reshape(way*shot, 512, 512)

        # generate label for each image
        labels = [[i]*shot for i in range(way)]
        labels = list(itertools.chain.from_iterable(labels))

        n = 0
        for patch_embeddings in novel_patch_embeddings.reshape(way*shot, dim, p, p):
            # obtain the mask
            novel_label = F.interpolate(novel_labels[n].unsqueeze(0).unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int()
            binary_label = torch.zeros_like(novel_label)
            binary_label[novel_label == self.novel_idx[labels[n]]] = 1
            assert binary_label.sum() != 0
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim

            # print('proto:', str(int(n/num_each_supp))+str(labels[n]))
            all_novel_queries[labels[n], :] += proto.clone().cpu().numpy()
            n += 1
        
        # norm for 1shot or 5shot
        all_novel_queries /= shot

        return all_novel_queries/10 #??????
    

@HEADS.register_module()
class BianryFakeHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
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
        super(BianryFakeHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        
        self.rd_type = rd_type
        # nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        
        self.novel_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.novel_idx.remove(i_idx)

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

        self.input_proj = input_proj
        self.proj_norm = proj_norm

        delattr(self, 'conv_seg')
        self.register_buffer("cur_iter", torch.Tensor([0]))
        self.register_buffer("base_qs", torch.randn((len(self.seen_idx), in_channels)))

        # self.withRD = withRD
        # if withRD:
        #     self.q_proj = nn.Linear(dim * 2, dim)
        # else:
        #     self.q_proj = nn.Linear(dim, dim)
        if 'qclsq' in self.rd_type or 'qcls_norm_q' in self.rd_type and 'pca' not in self.rd_type:
            self.q_proj = nn.Linear(dim * 2, dim)
        elif self.rd_type == 'qcls_pca':
            self.q_proj = nn.Linear(10, dim) ## for voc
        elif self.rd_type == 'qclsq_pca':
            self.q_proj = nn.Linear(dim + 10, dim) ## for voc
        else:
            self.q_proj = nn.Linear(dim, dim)
            
        if self.rd_type == 'combine':
            self.combine_proj = nn.Linear(4, 1)
        
        if self.rd_type == 'mlpcom':
            self.combine_proj = MLP_Proj(2, [64, 32], 1)
            
        self.decode_type = decode_type
        if decode_type == 'mlp':
            self.decoder = MLP(dim, int(dim/4), dim, num_layers=3)
        elif decode_type == 'attn':
            decoder_layer = FullTPN_DecoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4)
            self.decoder = TPN_Decoder(decoder_layer, num_layers=1)
        elif decode_type == 'conv':
            self.decoder = FCNHead(num_convs=2, kernel_size=1)
        else:
            pass

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs, gt_semantic_seg=None, novel_clip_feats=None, novel_labels=None):
        patch_tokens = inputs[0][0][0] #(bs, 768, 32, 32)
        cls_token = inputs[0][1]
        ori_patch_tokens = inputs[0][2]
        
        bs, dim, p, _ = patch_tokens.size()
        patch_tokens = patch_tokens.reshape(bs, dim , -1)
        
        # get qs_epoch
        # calculate relationship descriptor RD=qclsq, q is from average targeted patch embeddings
        if self.training:
            qs_epoch = self.extract_base_proto_epoch(self.base_qs, patch_tokens.reshape(bs, dim, p, p).clone().detach(), gt_semantic_seg.squeeze())
        else:
            if not hasattr(self, 'both_proto'):
                if novel_clip_feats is not None: # few-shot
                    way, shot, _, _, _ = novel_clip_feats.size()
                    novel_clip_feats = novel_clip_feats.reshape(way*shot, dim ,p*p)
                    print('\n' + '------------Registering the prototypes of novel classes-----------')
                    self.novel_queries = self.extract_novel_proto(novel_clip_feats.reshape(way, shot, dim, p, p), novel_labels)
                    # print('------------Registering the aug prototypes of novel classes-----------')
                    # self.novel_queries = self.extract_aug_novel_proto(self.supp_dir, self.supp_path, way=len(self.novel_class), shot=self.shot)
                    print('Done! The dimension of novel prototypes is: ', self.novel_queries.shape)
                    # REGISTER NOVEL: concat the novel queries in the right position
                    self.both_proto = torch.zeros([len(self.all_idx), self.in_channels]).to(patch_tokens[0].device)
                    self.both_proto[self.seen_idx] = self.base_qs.clone()
                    self.both_proto[self.novel_idx] = torch.from_numpy(self.novel_queries).to(self.base_qs.dtype).to(self.base_qs.device) ## how to test multi???
                else: #fully supervised
                    self.both_proto = self.base_qs.clone()
             
        # get rd and update base_qs
        if self.training:
            q = self.q_proj(self.get_qs(self.base_qs, cls_token, self.rd_type))
            self.cur_iter += 1
            mom = self.update_m()
            self.base_qs = (mom * self.base_qs.to(qs_epoch.device) + (1-mom) * qs_epoch)
        else:
            q = self.q_proj(self.get_qs(self.both_proto, cls_token, self.rd_type)) #bcd , need normalize??

        assert torch.isnan(q).any()==False and torch.isinf(q).any()==False
               
        # refine the patch embeddings        
        if self.decode_type is not None:
            if self.decode_type=='mlp':
                patch_tokens = self.decoder(patch_tokens.transpose(2,1))
                patch_tokens = patch_tokens.transpose(2,1)
            elif self.decode_type=='attn':    
                patch_tokens_list, _ = self.decoder(patch_tokens.transpose(2, 1), patch_tokens.transpose(2, 1)) # q/k/v=patch embedding
                patch_tokens = patch_tokens_list[-1].transpose(2,1) #(2, 768, 32*32)
            else:
                assert AttributeError('Donot support this decode type')
        else:
            pass 
        
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
            # outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)# (3, bs, 15, 14, 14)
            # out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.0) ## Change the balance factor： 0.0 is the best   
            return out["pred"]   
        return out                 
        
    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,0] = mask_pred[:,0] - 0.2 #reduce background, for learnable bg use add bg 0.2
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

    def get_qs(self, q, cls, type):
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        if type == 'qclsq':# q = [q.cls, q]
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10 # added or not?
            q_ = torch.concat((q1, q), dim=-1)
        elif type == 'qcls_norm_q': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
        elif type == 'qcls': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) * 10
        elif type == 'qcls_norm': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
        elif type == 'q_cls':
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.abs(cls - q)
        elif type == 'q':
            q_ = q.to(cls.dtype)
        elif type == 'combine':
            # qcls,|q-cls|,q,cls
            qcls = torch.einsum("bd,bcd->bcd", cls, q) * 10
            q_cls = torch.abs(cls.expand(C, -1, -1).permute(1, 0, 2) - q)
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((qcls.unsqueeze(-1), q_cls.unsqueeze(-1), q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 4)
            q_ = self.combine_proj(q_).squeeze(-1)
        elif type == 'mlpcom':
            q = q.to(cls.dtype)
            cls = cls.expand(C, -1, -1).permute(1, 0, 2)
            q_ = torch.concat((q.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1) #(bs, c, dim, 2)
            q_ = self.combine_proj(q_).squeeze(-1) #(bs, c, dim)
        elif type == 'qcls_norm_aug': # reduce the mean also on novel classes
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_norm_q_aug': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
            if self.training:
                q_ = q_[:, self.seen_idx, :] # only use the seen part, but class mean is calculated from all classes
        elif type == 'qcls_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q_ = torch.bmm(q1, v[:, :, :])
            # a = q_pca.squeeze() / torch.norm(q_pca.squeeze(), dim=-1, keepdim=True)
            # similarity = torch.mm(a, a.T)
        elif type == 'qclsq_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 10
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q1_pca = torch.bmm(q1, v[:, :, :])
            q_ = torch.concat((q1_pca, q), dim=-1) # q + 512
        else:
            assert AttributeError('Donot support this rd type')
        return q_


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
        # get the target of each cliped region
        # ann_path = img_metas[0]['filename'].replace('jpg','png').replace('JPEGImages', 'Annotations')
        # self.gt_ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        # self.gt_label = np.unique(self.gt_ann)
        # self.gt_label[self.gt_label==0] = 255 ## ignore the ground truth label
        # self.gt_label[self.gt_label!=255] -= 1
        # self.gt_label = np.delete(self.gt_label, np.where(self.gt_label == 255))
        if novel_clip_feats is not None:
            return self.forward(inputs, novel_clip_feats=novel_clip_feats, novel_labels=novel_labels)
        else:
            return self.forward(inputs)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs, gt_semantic_seg)
        gt_semantic_seg[gt_semantic_seg==-1] = 255
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses    
    
    def extract_base_proto_epoch(self, qs, patch_features, targets):
        ## qs(base, 768), patch(bs, 768, 32, 32), gt(bs, 512, 512)
        assert patch_features.shape[0] == targets.shape[0]
        patch_features = F.interpolate(patch_features, size=targets.shape[-2:], mode='bilinear', align_corners=False) ## (512, 512)
        # targets = F.interpolate(targets.unsqueeze(1).float(), size=patch_features.shape[-2:], mode='nearest').squeeze(1).int() ## (32, 32)
        # resnet50: patch (bs, 2048. 512, 512)
        qs_epoch = torch.zeros_like(qs) # [15, dim] resnet:[15,512] (2048-512) with proj
        num_base = torch.zeros(qs.shape[0]).to(qs_epoch.device)  #(15)

        n = 0
        for targets_per_image in targets:
            # for n th image in a batch
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != 255]
            if len(gt_cls) != 0:
                for cls in gt_cls:
                    num_base[cls] += 1
                    binary_mask = torch.zeros_like(patch_features[0,0])
                    binary_mask[targets_per_image == cls] = 1
                    proto_cls = (torch.einsum("dhw,hw->dhw", patch_features[n].squeeze(), binary_mask).sum(-1).sum(-1)) / binary_mask.sum()
                    qs_epoch[cls, :] = proto_cls
            n += 1

        # norm for each base classes
        qs_epoch[num_base!=0] = qs_epoch[num_base!=0] / num_base[num_base!=0].unsqueeze(-1) #(15, 768)
        return qs_epoch


    def extract_novel_proto(self, novel_patch_embeddings, novel_labels):
        ## load Image and Annotations, no augmentation but how to handle the crop??
        ## seed from GFS-Seg: 5
        way, shot, dim, p, p = novel_patch_embeddings.size()
        all_novel_queries = np.zeros([way, 768]) # [way, dim]
        novel_labels = novel_labels.reshape(way*shot, 512, 512)

        # generate label for each image
        labels = [[i]*shot for i in range(way)]
        labels = list(itertools.chain.from_iterable(labels))

        n = 0
        for patch_embeddings in novel_patch_embeddings.reshape(way*shot, dim, p, p):
            # obtain the mask
            novel_label = F.interpolate(novel_labels[n].unsqueeze(0).unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int()
            binary_label = torch.zeros_like(novel_label)
            binary_label[novel_label == self.novel_idx[labels[n]]] = 1
            assert binary_label.sum() != 0
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim

            # print('proto:', str(int(n/num_each_supp))+str(labels[n]))
            all_novel_queries[labels[n], :] += proto.clone().cpu().numpy()
            n += 1
        
        # norm for 1shot or 5shot
        all_novel_queries /= shot

        return all_novel_queries/15 #??????

