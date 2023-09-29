"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
import itertools
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import _load_weights
import math
import torch.nn.functional as F
from functools import reduce
from operator import mul
from mmseg.models.builder import BACKBONES
from torch.nn import Dropout

from .utils import *
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)) #(bs, 12, 1025, 1025)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

    def save_attn_map(self, attn):
        from PIL import Image
        import matplotlib.pyplot as plt
        savepath = './work_dirs_fss/head_attn_vit/'
        b, h, hw1, hw1 = attn.size()
        for i in range(h):
            attn_map = attn[:, i, 0, 1:].reshape(32, 32) #(1, 1024)
            plt.imshow(attn_map.cpu().numpy().squeeze())
            plt.savefig(savepath + str(i) + '.png')


class MaskMultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, masks):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores_ori = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)) #(bs, 12, 1025, 1025)
        scores = self.drop(F.softmax(scores_ori, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous() #(bs, 1025, 12, 64)
        # -merge-> (B, S, D)
        h = merge_last(h, 2) #(bs, 1025, 768)
        self.scores = scores
        
        # get new cls token with 
        with torch.no_grad():
            if masks is not None:
                img_idx, new_masks, new_scores, new_v, new_x = self.prepocess(masks, scores_ori, v, x)
                new_masks = new_masks[:, None, None, :].float()
                new_scores = new_scores - 10000.0 * (1.0 - new_masks)
                new_scores = self.drop(F.softmax(new_scores, dim=-1))
                new_h = (new_scores @ new_v).transpose(1, 2).contiguous()
                new_h = merge_last(new_h, 2)
        return h, new_h, img_idx, new_x

    def prepocess(self, masks, scores_ori, v_ori, x_ori): # repeat scores
        img_idx = []
        new_masks = []
        new_scores = []
        new_v = []
        new_x = []
        bs, head, dim , _ = scores_ori.size()
        for i in range(len(masks)):
            masks_i = masks[i].reshape(masks[i].shape[0], -1).float() #n_cls_i, 32*32
            new_masks_i = torch.zeros(masks[i].shape[0], scores_ori.shape[-1])
            new_masks_i[:, 1:(masks_i.shape[1]+1)] = masks_i
            new_masks.append(new_masks_i)
            img_idx.append([i] * masks_i.shape[0])
            # new_scores.append(torch.stack([scores_ori[i]] * masks_i.shape[0], dim=0).reshape(masks_i.shape[0], head, dim, dim))
            new_scores.append(scores_ori[i].unsqueeze(0).repeat(masks_i.shape[0], 1, 1, 1))
            new_v.append(v_ori[i].unsqueeze(0).repeat(masks_i.shape[0], 1, 1, 1))
            new_x.append(x_ori[i].unsqueeze(0).repeat(masks_i.shape[0], 1, 1))
        
        img_idx = list(itertools.chain(*img_idx))
        new_masks = torch.stack(new_masks, dim=0).reshape(len(img_idx), -1).to(masks_i.dtype).to(masks_i.device)
        new_scores = torch.stack(new_scores, dim=0).reshape(len(img_idx), head, dim, dim).to(masks_i.dtype).to(masks_i.device)
        new_v = torch.stack(new_v, dim=0).reshape(len(img_idx), head, dim, v_ori.shape[-1]).to(v_ori.dtype).to(v_ori.device)
        new_x = torch.stack(new_x, dim=0).reshape(len(img_idx), dim, x_ori.shape[-1]).to(x_ori.dtype).to(x_ori.device)
        return img_idx, new_masks, new_scores, new_v, new_x

    def save_attn_map(self, attn):
        from PIL import Image
        import matplotlib.pyplot as plt
        savepath = './work_dirs_fss/head_attn_vit/'
        b, h, hw1, hw1 = attn.size()
        for i in range(h):
            attn_map = attn[:, i, 0, 1:].reshape(32, 32) #(1, 1024)
            plt.imshow(attn_map.cpu().numpy().squeeze())
            plt.savefig(savepath + str(i) + '.png')


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x

class MaskBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MaskMultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_ori, mask):
        h, new_h, img_mask_idx, x_new = self.attn(self.norm1(x_ori), mask)
        
        h = self.drop(self.proj(h))
        x = x_ori + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        with torch.no_grad():
            # to get new cls
            new_h = self.drop(self.proj(new_h))
            x_new = x_new + new_h
            new_h = self.drop(self.pwff(self.norm2(x_new)))
            x_new = x_new + new_h
        
        return x, x_new, img_mask_idx

class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return 

class MaskTransformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers - 1)])
        self.blocks.append(MaskBlock(dim, num_heads, ff_dim, dropout)) # the last layer used masked patch to get cls token

    def forward(self, x, mask=None):
        for block in self.blocks:
            x, new_x, img_mask, idx = block(x, mask)
        return
    
class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

@BACKBONES.register_module()
class PromptImageNetViT(nn.Module):

    def __init__(
        self,  
        image_size = 512,
        patches = 16,
        dim = 768,
        ff_dim = 3072,
        num_heads = 12,
        num_layers = 12,
        dropout_rate = 0.1,
        positional_embedding = '1d',
        in_channels = 3, 
        out_indices=[3, 5, 7, 11], 
        pretrained = None,
        ## ADDED
        num_tokens = 10, 
        prompt_dim = 768, 
        total_d_layer=11, 
        **kwargs
    ):
        super().__init__()

        self.pretrained = pretrained
        self.patch_size = patches
        self.n_layers = num_layers
        self.d_model = self.dim = dim
        self.d_ff = ff_dim
        self.n_heads = num_heads
        self.image_size = image_size                

        # Image and patch sizes
        h, w = (image_size, image_size)  # image sizes
        fh, fw = (patches, patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw + 1

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.w = image_size
        self.h = image_size
        # Initialize weights
        # cls and pos tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        # trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

        # self.apply(self._init_weights)
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer
        self._init_prompt(patches, self.num_tokens, self.prompt_dim, self.total_d_layer)

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
        print('==========> Loading parameters from pretrained model ViT <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            # remove `module.` prefix
            # state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if 'positional_embedding.pos_embedding' in state_dict.keys():
                if self.positional_embedding.pos_embedding.shape != state_dict['positional_embedding.pos_embedding'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding.pos_embedding"].shape} to {self.positional_embedding.pos_embedding.shape}')
                    N = state_dict['positional_embedding.pos_embedding'].shape[1] - 1

                    cls_pos = state_dict["positional_embedding.pos_embedding"][:, 0:1, :]
                    # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = state_dict["positional_embedding.pos_embedding"][:, 1:, :]
                    w0 = self.w // self.patch_size
                    h0 = self.h // self.patch_size
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
                    state_dict['positional_embedding.pos_embedding'] = positional_embedding
                    assert self.positional_embedding.pos_embedding.shape == state_dict['positional_embedding.pos_embedding'].shape

            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        x = self.patch_embedding(x)  # b,d,gh,gw 
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(B, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.positional_embedding(x)  # b,gh*gw+1,d

        mask = None
        
        ## get proto for q only from dino
        # x_p = x.clone().detach()
        # with torch.no_grad():
        #     for blk in self.transformer.blocks:
        #         x_p = blk(x_p, mask)
        #     proto_embedding = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()
        
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
            for i, blk in enumerate(self.transformer.blocks):
                x = blk(x.permute(1, 0, 2), mask).permute(1, 0, 2)
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
        # proto_embedding = proto_embedding / proto_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        
        outs.append(tuple(features))
        outs.append(global_embedding) 
        # outs.append(proto_embedding) 
        return outs

    def forward_deep_prompt(self, embedding_output, features, H, W, out_last=False): #embedding_output=x=(1+n_prompt+n_patches, B, D)
        mask = None
        B = embedding_output.shape[1]
        for i in range(self.num_layers):
            if i == 0:
                hidden_states = (self.transformer.blocks[i](embedding_output.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[-(H*W):, :, :]
                ), dim=0) #(1+n_patches, B, D)
                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
            
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
        mask = None
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.deep_prompt_embeddings.shape[0])-1 # (12-9)-1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = (self.transformer.blocks[i](embedding_output.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
            elif 0<i<=deep_num_no:
                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
            else: ## with deep prompts
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-deep_num_no-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[-(H*W):, :, :]
                ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
            
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


@BACKBONES.register_module()
class BaseImageNetViT(nn.Module):

    def __init__(
        self,  
        image_size = 512,
        patches = 16,
        dim = 768,
        ff_dim = 3072,
        num_heads = 12,
        num_layers = 12,
        dropout_rate = 0.1,
        positional_embedding = '1d',
        in_channels = 3, 
        out_indices=[3, 5, 7, 11], 
        pretrained = None,
        **kwargs
    ):
        super().__init__()

        self.pretrained = pretrained
        self.patch_size = patches
        self.n_layers = num_layers
        self.d_model = self.dim = dim
        self.d_ff = ff_dim
        self.n_heads = num_heads
        self.image_size = image_size
        self.out_indices = out_indices                

        # Image and patch sizes
        h, w = (image_size, image_size)  # image sizes
        fh, fw = (patches, patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw + 1

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.w = image_size
        self.h = image_size
        # Initialize weights
        # cls and pos tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        # trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

        # self.apply(self._init_weights)
        self.num_layers = num_layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        print('==========> Loading parameters from pretrained model ViT <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            # remove `module.` prefix
            # state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if 'positional_embedding.pos_embedding' in state_dict.keys():
                if self.positional_embedding.pos_embedding.shape != state_dict['positional_embedding.pos_embedding'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding.pos_embedding"].shape} to {self.positional_embedding.pos_embedding.shape}')
                    N = state_dict['positional_embedding.pos_embedding'].shape[1] - 1

                    cls_pos = state_dict["positional_embedding.pos_embedding"][:, 0:1, :]
                    # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = state_dict["positional_embedding.pos_embedding"][:, 1:, :]
                    w0 = self.w // self.patch_size
                    h0 = self.h // self.patch_size
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
                    state_dict['positional_embedding.pos_embedding'] = positional_embedding
                    assert self.positional_embedding.pos_embedding.shape == state_dict['positional_embedding.pos_embedding'].shape

            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        x = self.patch_embedding(x)  # b,d,gh,gw 
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(B, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.positional_embedding(x)  # b,gh*gw+1,d

        ## get proto for q only from dino
        mask = None
        x_p = x.clone().detach()
        with torch.no_grad():
            for blk in self.transformer.blocks:
                x_p = blk(x_p, mask)
            x_p = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()

        features = []
        outs = []
        x = x.permute(1, 0, 2)  # NLD -> LND (1+prompt+n_patches, B, D)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x.permute(1, 0, 2), mask).permute(1, 0, 2)
            if len(self.out_indices) > 1: # return the middle features of visual CLIP
                if i in self.out_indices:
                    xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

        x = x.permute(1,0,2)
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



@BACKBONES.register_module()
class MaskPromptImageNetViT(nn.Module):

    def __init__(
        self,  
        image_size = 512,
        patches = 16,
        dim = 768,
        ff_dim = 3072,
        num_heads = 12,
        num_layers = 12,
        dropout_rate = 0.1,
        positional_embedding = '1d',
        in_channels = 3, 
        out_indices=[3, 5, 7, 11], 
        pretrained = None,
        ## ADDED
        num_tokens = 10, 
        prompt_dim = 768, 
        total_d_layer=11, 
        **kwargs
    ):
        super().__init__()

        self.pretrained = pretrained
        self.patch_size = patches
        self.n_layers = num_layers
        self.d_model = self.dim = dim
        self.d_ff = ff_dim
        self.n_heads = num_heads
        self.image_size = image_size                

        # Image and patch sizes
        h, w = (image_size, image_size)  # image sizes
        fh, fw = (patches, patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw + 1

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        self.transformer = MaskTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.w = image_size
        self.h = image_size
        # Initialize weights
        # cls and pos tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        # trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

        # self.apply(self._init_weights)
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer
        self._init_prompt(patches, self.num_tokens, self.prompt_dim, self.total_d_layer)

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
        print('==========> Loading parameters from pretrained model ViT <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            # remove `module.` prefix
            # state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            if 'positional_embedding.pos_embedding' in state_dict.keys():
                if self.positional_embedding.pos_embedding.shape != state_dict['positional_embedding.pos_embedding'].shape:
                    # (1025, 768)                      (197, 768)  
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding.pos_embedding"].shape} to {self.positional_embedding.pos_embedding.shape}')
                    N = state_dict['positional_embedding.pos_embedding'].shape[1] - 1

                    cls_pos = state_dict["positional_embedding.pos_embedding"][:, 0:1, :]
                    # spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = state_dict["positional_embedding.pos_embedding"][:, 1:, :]
                    w0 = self.w // self.patch_size
                    h0 = self.h // self.patch_size
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
                    state_dict['positional_embedding.pos_embedding'] = positional_embedding
                    assert self.positional_embedding.pos_embedding.shape == state_dict['positional_embedding.pos_embedding'].shape

            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned

    def forward(self, x, masks):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        all_masks, img_mask_labels = self.prepare_targets(masks)
        
        x = self.patch_embedding(x)  # b,d,gh,gw 
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(B, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.positional_embedding(x)  # b,gh*gw+1,d

        ## get proto for q only from encoder
        # x_p = x.clone().detach() #(bs, 1025, 768)
        ## get class_mask:
        
        # use the patch embedding without any prompt finetuning
        # with torch.no_grad():
        #     for blk in self.transformer.blocks:
        #         x_p = blk(x_p)
        #     proto_embedding = self.norm(x_p)[:, -(H*W):].reshape(B, H, W, -1).permute(0, 3, 1, 2).detach()
            
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
            for i, blk in enumerate(self.transformer.blocks):
                if i != (len(self.transformer)-1):
                    x = blk(x.permute(1, 0, 2)).permute(1, 0, 2)
                else:
                    x = blk(x.permute(1, 0, 2), all_masks).permute(1, 0, 2)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0, 2)[:, 1+self.num_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                        features.append(xp.contiguous())
        elif self.total_d_layer > 0: # deep
            x, features, mask_cls, img_mask_idx = self.forward_deep_prompt(x, features, all_masks, H, W)
        elif self.total_d_layer < 0:
            x, features = self.forward_reverse_deep_prompt(x, features, all_masks, H, W)
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
        # proto_embedding = proto_embedding / proto_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        
        outs.append(tuple(features))
        outs.append(global_embedding) 
        # outs.append(proto_embedding) 
        outs.append(mask_cls)
        outs.append(img_mask_idx)
        outs.append(img_mask_labels)
        return outs

    def forward_deep_prompt(self, embedding_output, features, all_masks, H, W, out_last=False): #embedding_output=x=(1+n_prompt+n_patches, B, D)
        
        B = embedding_output.shape[1]
        for i in range(self.num_layers):
            if i == self.num_layers-1: ## the last layer
                mask = all_masks
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states, new_hidden_states, img_mask_idx = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask))
                hidden_states = hidden_states.permute(1, 0, 2)
            else:
                mask = None
                if i == 0:
                    hidden_states = (self.transformer.blocks[i](embedding_output.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
                elif i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0) #(1+n_prompt+n_patches, B, D)

                    hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
                else:
                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        hidden_states[-(H*W):, :, :]
                    ), dim=0) #(1+n_patches, B, D)
                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
            
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
            return encoded, features, new_hidden_states[:, 1, :], img_mask_idx #only for saving middle features

    def forward_reverse_deep_prompt(self, embedding_output, features, H, W, out_last=False): #embedding_output=x=(1+n_prompt+n_patches, B, D)
        mask = None
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.deep_prompt_embeddings.shape[0])-1 # (12-9)-1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = (self.transformer.blocks[i](embedding_output.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
            elif 0<i<=deep_num_no:
                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2) #(n_prompt, B, D)
            else: ## with deep prompts
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-deep_num_no-1]).expand(B, -1, -1)).permute(1, 0, 2) #(n_prompt, B, D)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[-(H*W):, :, :]
                ), dim=0) #(1+n_prompt+n_patches, B, D)

                hidden_states = (self.transformer.blocks[i](hidden_states.permute(1, 0, 2), mask)).permute(1, 0, 2)
            
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
        
    def prepare_targets(self, targets):
        all_masks = []
        all_gt_cls = []
        
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != 255] #self.ignore_index
            masks = []
            
            # downsize target
            # targets_per_image = F.interpolate(targets_per_image.unsqueeze(0).float(),
            #                     size=(32, 32), mode='bilinear', align_corners=False).squeeze().long()
            targets_per_image = F.interpolate(targets_per_image.unsqueeze(0).float(),
                                size=(32, 32), mode='nearest').squeeze().long()
            
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            # if len(gt_cls) == 0:
            #     masks.append(targets_per_image == 255)
            masks = torch.stack(masks, dim=0)
            all_masks.append(masks)
            all_gt_cls.append(gt_cls)
        
        # all_cls = torch.stack(all_cls).reshape(-1)
        # all_masks = torch.stack(all_masks).reshape(len(img_idx), h, w)
        all_gt_cls = torch.stack(all_gt_cls).reshape(-1).tolist()
        return all_masks, all_gt_cls
