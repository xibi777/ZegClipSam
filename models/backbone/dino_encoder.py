# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
# from charset_normalizer import utils

import torch
import torch.nn as nn

from .utils import *

from functools import reduce
from operator import mul

from mmseg.models.builder import BACKBONES
from torch.nn import Dropout


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def save_attn_map(self, attn):
        from PIL import Image
        import matplotlib.pyplot as plt
        savepath = './work_dirs_fss/head_attn/'
        b, h, hw1, hw1 = attn.size()
        for i in range(h):
            attn_map = attn[:, i, 0, 1:].reshape(32, 32) #(1, 1024)
            plt.imshow(attn_map.cpu().numpy().squeeze())
            plt.savefig(savepath + str(i) + '.png')


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# class VisionTransformer(nn.Module):
#     """ Vision Transformer """
#     def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim

#         self.patch_embed = PatchEmbed(
#             img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         # Classifier head
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def interpolate_pos_encoding(self, x, w, h):
#         npatch = x.shape[1] - 1
#         N = self.pos_embed.shape[1] - 1
#         if npatch == N and w == h:
#             return self.pos_embed
#         class_pos_embed = self.pos_embed[:, 0]
#         patch_pos_embed = self.pos_embed[:, 1:]
#         dim = x.shape[-1]
#         w0 = w // self.patch_embed.patch_size
#         h0 = h // self.patch_embed.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#     def prepare_tokens(self, x):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)  # patch linear embedding

#         # add the [CLS] token to the embed patch tokens
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # add positional encoding to each token
#         x = x + self.interpolate_pos_encoding(x, w, h)

#         return self.pos_drop(x)

#     def forward(self, x):
#         x = self.prepare_tokens(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         return x[:, 0]

#     def get_last_selfattention(self, x):
#         x = self.prepare_tokens(x)
#         for i, blk in enumerate(self.blocks):
#             if i < len(self.blocks) - 1:
#                 x = blk(x)
#             else:
#                 # return attention of the last block
#                 return blk(x, return_attention=True)

#     def get_intermediate_layers(self, x, n=1):
#         x = self.prepare_tokens(x)
#         # we return the output tokens from the `n` last blocks
#         output = []
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)
#             if len(self.blocks) - i <= n:
#                 output.append(self.norm(x))
#         return output

@BACKBONES.register_module()
class PromptVisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), out_indices=[3, 5, 7, 11], pretrained=None,
                 num_tokens=20, prompt_dim=768, total_d_layer=11, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.w = img_size
        self.h = img_size
        self.dim = embed_dim
        self.num_layers = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

        self.out_indices = out_indices

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        print('==========> Loading parameters from pretrained model DINO <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["pos_embed"].shape} to {self.pos_embed.shape}')
                    N = state_dict['pos_embed'].shape[1] - 1

                    cls_pos = state_dict["pos_embed"][:, 0:1, :]
                    # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = state_dict["pos_embed"][:, 1:, :]
                    w0 = self.w // self.patch_embed.patch_size
                    h0 = self.h // self.patch_embed.patch_size
                    # we add a small number to avoid floating point error in the interpolation
                    # see discussion at https://github.com/facebookresearch/dino/issues/8
                    w0, h0 = w0 + 0.1, h0 + 0.1
                    spatial_pos = nn.functional.interpolate(
                    spatial_pos.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), self.dim).permute(0, 3, 1, 2),
                    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),mode='bicubic',)
                    assert int(w0) == spatial_pos.shape[-2] and int(h0) == spatial_pos.shape[-1]

                    spatial_pos = spatial_pos.permute(0, 2, 3, 1).view(1, -1, self.dim)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=1)
                    # print('pos_emb:', positional_embedding.shape)
                    state_dict['pos_embed'] = positional_embedding
                    assert self.pos_embed.shape == state_dict['pos_embed'].shape

            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x) # (bs, 1025, 768)
        B = x.shape[0]
        H = W = int(np.sqrt((x.shape[1]-1)))

        ## get proto for q only from dino
        x_p = x.clone().detach() #(b, 1025, 768)
        with torch.no_grad():
            for blk in self.blocks:
                x_p = blk(x_p)
            proto_embedding = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()

        ## check if freeze the backbone:
        # print('bb:', self.blocks[0].attn.proj.weight.sum())
        # print('bb:', self.blocks[0].attn.proj.weight.requires_grad)
        # print('p1:', self.prompt_embeddings.sum())
        # print('p1:', self.prompt_embeddings.requires_grad)        
        # print('p2:', self.deep_prompt_embeddings.sum())
        # print('p2:', self.deep_prompt_embeddings.requires_grad)

        if self.total_d_layer >=0:
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1) # (B, 1+n_prompt+n_patch, D)
        
        features = []
        outs = []
        x = x.permute(1, 0, 2)  # NLD -> LND (1+prompt+n_patches, B, D)

        if self.total_d_layer == 0: #shallow
            for i, blk in enumerate(self.blocks):
                x = blk(x.permute(1, 0, 2)).permute(1, 0, 2)
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
            visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
            features.append(visual_embedding) #len(features) = 1, [B, 512, 32, 32]

        ## get embedding:
        global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        proto_embedding = proto_embedding / proto_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm

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

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

@BACKBONES.register_module()
class BaseVisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), out_indices=[3, 5, 7, 11], pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.w = img_size
        self.h = img_size
        self.dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.out_indices = out_indices

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["pos_embed"].shape} to {self.pos_embed.shape}')
                    N = state_dict['pos_embed'].shape[1] - 1

                    cls_pos = state_dict["pos_embed"][:, 0:1, :]
                    # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = state_dict["pos_embed"][:, 1:, :]
                    w0 = self.w // self.patch_embed.patch_size
                    h0 = self.h // self.patch_embed.patch_size
                    # we add a small number to avoid floating point error in the interpolation
                    # see discussion at https://github.com/facebookresearch/dino/issues/8
                    w0, h0 = w0 + 0.1, h0 + 0.1
                    spatial_pos = nn.functional.interpolate(
                    spatial_pos.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), self.dim).permute(0, 3, 1, 2),
                    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),mode='bicubic',)
                    assert int(w0) == spatial_pos.shape[-2] and int(h0) == spatial_pos.shape[-1]

                    spatial_pos = spatial_pos.permute(0, 2, 3, 1).view(1, -1, self.dim)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=1)
                    print('pos_emb:', positional_embedding.shape)
                    state_dict['pos_embed'] = positional_embedding
                    assert self.pos_embed.shape == state_dict['pos_embed'].shape

            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x) # (bs, 1025, 768)
        B = x.shape[0]
        H = W = int(np.sqrt((x.shape[1]-1)))

        ## get original proto from dino
        x_p = x.clone().detach()
        with torch.no_grad():
            for blk in self.blocks:
                x_p = blk(x_p)
            x_p = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()

        features = []
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.out_indices) > 1: # return the middle features of visual CLIP
                if i in self.out_indices:
                    xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

        x = self.norm(x) #LayerNorm: (bs, 1025, 768)
        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W
        # features.append([global_embedding, visual_embedding])
        if len(self.out_indices) == 1: # return the final features after proj
            features.append(visual_embedding) #len(features) = 1, [B, 512, 32, 32]

        outs.append(tuple(features))
        outs.append(global_embedding) 
        outs.append(x_p) 
        return outs

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# def vit_tiny(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_small(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_base(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
