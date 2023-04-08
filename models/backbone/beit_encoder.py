# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

from .mmcv_custom.checkpoint import load_checkpoint
from .utils import *
import math
from functools import reduce
from operator import mul
from torch.nn import Dropout

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape #(bs, 1025, 768)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # attn = attn + relative_position_bias.unsqueeze(0)
            attn[:,:,0,0] = attn[:,:,0,0] + relative_position_bias.unsqueeze(0)[:, :, 0, 0]
            attn[:, :, -(self.window_size[0]*self.window_size[1]):, -(self.window_size[0]*self.window_size[1]):]\
                 = attn[:, :, -(self.window_size[0]*self.window_size[1]):, -(self.window_size[0]*self.window_size[1]):]\
                     + relative_position_bias.unsqueeze(0)[:, :, -(self.window_size[0]*self.window_size[1]):, -(self.window_size[0]*self.window_size[1]):]

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


# @BACKBONES.register_module()
# class BEiT(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
#                  use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
#                  out_indices=[3, 5, 7, 11]):
#         super().__init__()
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
#         else:
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#         self.out_indices = out_indices

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         if use_abs_pos_emb:
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         else:
#             self.pos_embed = None
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         if use_shared_rel_pos_bias:
#             self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
#         else:
#             self.rel_pos_bias = None

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.use_rel_pos_bias = use_rel_pos_bias
#         self.use_checkpoint = use_checkpoint
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                 init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
#             for i in range(depth)])

#         if self.pos_embed is not None:
#             trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         # trunc_normal_(self.mask_token, std=.02)
#         self.out_indices = out_indices

#         if patch_size == 16:
#             self.fpn1 = nn.Sequential(
#                 nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#                 nn.SyncBatchNorm(embed_dim),
#                 nn.GELU(),
#                 nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#             )

#             self.fpn2 = nn.Sequential(
#                 nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#             )

#             self.fpn3 = nn.Identity()

#             self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
#         elif patch_size == 8:
#             self.fpn1 = nn.Sequential(
#                 nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#             )

#             self.fpn2 = nn.Identity()

#             self.fpn3 = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#             )

#             self.fpn4 = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=4, stride=4),
#             )
#         self.apply(self._init_weights)
#         self.fix_init_weight()

#     def fix_init_weight(self):
#         def rescale(param, layer_id):
#             param.div_(math.sqrt(2.0 * layer_id))

#         for layer_id, layer in enumerate(self.blocks):
#             rescale(layer.attn.proj.weight.data, layer_id + 1)
#             rescale(layer.mlp.fc2.weight.data, layer_id + 1)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def init_weights(self, pretrained=None):
        
#         def _init_weights(m):
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#         if isinstance(pretrained, str):
#             self.apply(_init_weights)
#             logger = get_root_logger()
#             # load_checkpoint(self, pretrained, strict=False, logger=logger)
#         elif pretrained is None:
#             self.apply(_init_weights)
#         else:
#             raise TypeError('pretrained must be a str or None')

#     def get_num_layers(self):
#         return len(self.blocks)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def forward_features(self, x):
#         B, C, H, W = x.shape
#         x, (Hp, Wp) = self.patch_embed(x)
#         batch_size, seq_len, _ = x.size()

#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         if self.pos_embed is not None:
#             x = x + self.pos_embed
#         x = self.pos_drop(x)

#         rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
#         features = []
#         for i, blk in enumerate(self.blocks):
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, rel_pos_bias)
#             else:
#                 x = blk(x, rel_pos_bias)
#             if i in self.out_indices:
#                 xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
#                 features.append(xp.contiguous())

#         ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
#         for i in range(len(features)):
#             features[i] = ops[i](features[i])

#         return tuple(features)

#     def forward(self, x):
#         x = self.forward_features(x)
#         return 


@BACKBONES.register_module()
class PromptBEiT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[3, 5, 7, 11], 
                 pretrained=None, num_tokens=20, prompt_dim=768, total_d_layer=11, **kwargs):
        super().__init__()
        self.pretrained = pretrained

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        self.out_indices = out_indices

        # if patch_size == 16:
        #     self.fpn1 = nn.Sequential(
        #         nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #         nn.SyncBatchNorm(embed_dim),
        #         nn.GELU(),
        #         nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #     )

        #     self.fpn2 = nn.Sequential(
        #         nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #     )

        #     self.fpn3 = nn.Identity()

        #     self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # elif patch_size == 8:
        #     self.fpn1 = nn.Sequential(
        #         nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #     )

        #     self.fpn2 = nn.Identity()

        #     self.fpn3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #     )

        #     self.fpn4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=4, stride=4),
        #     )

        self.apply(self._init_weights)
        self.fix_init_weight()

        ## Setting of visual prompt tuning
        self.norm = norm_layer(embed_dim)
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer
        self.num_layers = depth

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    # def init_weights(self, pretrained=None):
    #     print('==========> Loading parameters from pretrained model DEiT <===========')
    #     pretrained = pretrained or self.pretrained
    #     if isinstance(pretrained, str):
    #         state_dict = torch.load(pretrained, map_location='cpu')['model']
    #         # remove `module.` prefix
    #         state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    #         if 'pos_embed' in state_dict.keys():
    #             if self.pos_embed.shape != state_dict['pos_embed'].shape:
    #                 # (1025, 768)                      (197, 768)  
    #                 print(f'Resize the pos_embed shape from {state_dict["pos_embed"].shape} to {self.pos_embed.shape}')
    #                 N = state_dict['pos_embed'].shape[1] - 1

    #                 cls_pos = state_dict["pos_embed"][:, 0:1, :]
    #                 # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
    #                 spatial_pos = state_dict["pos_embed"][:, 1:, :]
    #                 w0 = self.w // self.patch_embed.patch_size
    #                 h0 = self.h // self.patch_embed.patch_size
    #                 # we add a small number to avoid floating point error in the interpolation
    #                 # see discussion at https://github.com/facebookresearch/dino/issues/8
    #                 w0, h0 = w0 + 0.1, h0 + 0.1
    #                 spatial_pos = nn.functional.interpolate(
    #                 spatial_pos.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), self.dim).permute(0, 3, 1, 2),
    #                 scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),mode='bicubic',)
    #                 assert int(w0) == spatial_pos.shape[-2] and int(h0) == spatial_pos.shape[-1]

    #                 spatial_pos = spatial_pos.permute(0, 2, 3, 1).view(1, -1, self.dim)
    #                 positional_embedding = torch.cat([cls_pos, spatial_pos], dim=1)
    #                 # print('pos_emb:', positional_embedding.shape)
    #                 state_dict['pos_embed'] = positional_embedding
    #                 assert self.pos_embed.shape == state_dict['pos_embed'].shape

    #         u, w = self.load_state_dict(state_dict, strict=False)
    #         print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned, u is the new build model, w is the loaded model
  
    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x, (H, W) = self.patch_embed(x) 
        B, _, C = x.size() #(B, 32*32, 768)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x) #(B, 1025, 768)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        ## get proto for q only from dino
        x_p = x.clone().detach()
        with torch.no_grad():
            for i, blk in enumerate(self.blocks):
                if self.pretrained:
                    x_p = checkpoint.checkpoint(blk, x_p, rel_pos_bias)
                else:
                    x_p = blk(x_p, rel_pos_bias)
            proto_embedding = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()

        if self.total_d_layer >=0:
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]), dim=1) # (B, 1+n_prompt+n_patch, D)

        features = []
        outs = []
        x = x.permute(1, 0, 2)  # NLD -> LND (1+prompt+n_patches, B, D)
        
        # print('check weights change:', self.prompt_embeddings.sum())
        # print('check weights not change 1:', self.blocks[0].attn.proj.weight.sum())
        # print('check weights not change 2:', self.mask_token.sum())

        if self.total_d_layer == 0: #shallow
            for i, blk in enumerate(self.blocks):
                
                if self.pretrained:
                    x = checkpoint.checkpoint(blk, x, rel_pos_bias)
                else:
                    x = blk(x, rel_pos_bias)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0, 2)[:, 1+self.num_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                        features.append(xp.contiguous())
        elif self.total_d_layer > 0: # deep
            x, features = self.forward_deep_prompt(x, features, H, W)
        elif self.total_d_layer < 0:
            x, features = self.forward_reverse_deep_prompt(x, features, H, W)
        else:
            AttributeError('Input correct total_d_layer')

        x = x.permute(1,0,2)
        x = self.norm(x) #(bs, 1025, 768)

        global_embedding = x[:, 0]
        visual_embedding = x[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W
        # features.append([global_embedding, visual_embedding])
        if len(self.out_indices) == 1: # return the final features after proj
            # visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
            features.append(visual_embedding) #len(features) = 1, [B, 512, 32, 32]

        ## get embedding:
        # global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        # proto_embedding = proto_embedding / proto_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm

        outs.append(tuple(features))
        outs.append(global_embedding) 
        outs.append(proto_embedding) 
        return outs
    
    def forward_deep_prompt(self, embedding_output, features, H, W, out_last=False): #embedding_output=x=(1+n_prompt+n_patches, B, D)
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = (self.blocks[i](embedding_output.permute(1, 0, 2))).permute(1, 0, 2) #(n_prompt, B, D)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states = (self.blocks[i](hidden_states.permute(1, 0, 2))).permute(1, 0, 2)
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[-(H*W):, :, :]
                ), dim=0) #(1+n_patches, B, D)
                hidden_states = (self.blocks[i](hidden_states.permute(1, 0, 2))).permute(1, 0, 2)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2): #10
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = self.prompt_norm(hidden_states) #(1=prompt+1024, 4, 768)
        if out_last:
            return before_last_feats
        else:
            return encoded, features #only for saving middle features

    def forward_reverse_deep_prompt(self, embedding_output, features, H, W, out_last=False): #embedding_output=x=(1+n_prompt+n_patches, B, D)
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.deep_prompt_embeddings.shape[0])-1 # (12-9)-1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = (self.blocks[i](embedding_output.permute(1, 0, 2))).permute(1, 0, 2) #(n_prompt, B, D)
            elif 0<i<=deep_num_no:
                hidden_states = (self.blocks[i](hidden_states.permute(1, 0, 2))).permute(1, 0, 2) #(n_prompt, B, D)
            else: ## with deep prompts
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-deep_num_no-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[-(H*W):, :, :]
                ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states = (self.blocks[i](hidden_states.permute(1, 0, 2))).permute(1, 0, 2)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    # xp = hidden_states.permute(1, 0, 2)[:, 1+self.num_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2): #10
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features #only for saving middle features

