    
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

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)


    def forward(self, inputs_both, self_training=None):
        patch_tokens = inputs_both[0][0][0] #(bs, dim, 1024)
        cls_token = inputs_both[0][1]
        text_tokens = inputs_both[1]

        bs, dim, p, _ = patch_tokens.size()
        patch_tokens = patch_tokens.reshape(bs, dim , -1)

        # inputs[0].retain_grad() ##get grad
        # runner.model.module.backbone.prompt_proj.weight.grad

        # if self.withRD:
        #     q = self.q_proj(self.get_qs(text_tokens, cls_token)) #bcd
        #     pred_logits = torch.einsum("bdn,bcd->bcn", patch_tokens, q)
        # else:
        #     q = self.q_proj(text_tokens.to(patch_tokens.dtype))
        #     pred_logits = torch.einsum("bdn,cd->bcn", patch_tokens, q)
        
        if self.decode_type is not None:
            if self.decode_type=='mlp':
                patch_tokens = self.decoder(patch_tokens.transpose(2,1))
                patch_tokens = patch_tokens.transpose(2,1)
            elif self.decode_type=='attn':    
                patch_tokens_list, _ = self.decoder(patch_tokens.transpose(2, 1), patch_tokens.transpose(2, 1)) # q/k/v=patch embedding
                patch_tokens = patch_tokens_list[-1].transpose(2,1)
            else:
                assert AttributeError('Donot support this decode type')
        else:
            pass 
            
        q = self.q_proj(self.get_qs(text_tokens, cls_token, self.rd_type)) #bcd , need normalize??
        pred_logits = torch.einsum("bdn,bcd->bcn", patch_tokens, q) # matching directly
        c = pred_logits.shape[1]

        pred_logits = pred_logits.reshape(bs, c, p, p)
        # pred_logits = patch_tokens @ text_token.t()
        
        pred = F.interpolate(pred_logits, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
                                          
        out = {"pred_masks": pred}

        
        if self.training:
            return out
        else:
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx) #(bs, 20, 224, 224)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.2)
            return out["pred"]                  
        
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

    def get_qs(self, q, cls, type):
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        if type == 'qclsq':# q = [q.cls, q]
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100 # added or not?
            q_ = torch.concat((q1, q), dim=-1)
        elif type == 'qcls_norm_q': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) # for norm, do it need *100????????
            # norm the rd between all c classes
            q_norm = q_.mean(dim=1).unsqueeze(1)
            q_ = q_ - q_norm
            q_ = torch.concat((q_, q), dim=-1)
        elif type == 'qcls': # maybe need an scalor
            q_ = torch.einsum("bd,bcd->bcd", cls, q) * 100
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
            qcls = torch.einsum("bd,bcd->bcd", cls, q) * 100
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
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100
            _, _, v = torch.pca_lowrank(q1, q=10, center=True, niter=2)
            q_ = torch.bmm(q1, v[:, :, :])
            # a = q_pca.squeeze() / torch.norm(q_pca.squeeze(), dim=-1, keepdim=True)
            # similarity = torch.mm(a, a.T)
        elif type == 'qclsq_pca': # maybe need an scalor
            q1 = torch.einsum("bd,bcd->bcd", cls, q) * 100
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
        
