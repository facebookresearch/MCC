# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath

from util.pos_embed import get_2d_sincos_pos_embed

class MCCDecoderAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.args = args
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, unseen_size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = torch.zeros((1, 1, N, N), device=attn.device)
        mask[:, :, :, -unseen_size:] = float('-inf')
        for i in range(unseen_size):
            mask[:, :, -(i + 1), -(i + 1)] = 0
        attn = attn + mask
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MCCDecoderBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = MCCDecoderAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, args=args)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, unseen_size):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), unseen_size)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class XYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.win_size = 8

        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, seen_xyz, valid_seen_xyz):
        emb = self.pos_embed(seen_xyz)

        emb[~valid_seen_xyz] = 0.0
        emb[~valid_seen_xyz] += self.invalid_xyz_token

        B, H, W, C = emb.shape
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)

        emb = emb + self.two_d_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)


class DecodeXYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Linear(3, embed_dim)

    def forward(self, unseen_xyz):
        return self.pos_embed(unseen_xyz)


class MCC(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 rgb_weight=1.0, occupancy_weight=1.0, args=None):
        super().__init__()

        self.rgb_weight = rgb_weight
        self.occupancy_weight = occupancy_weight
        self.args = args

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_xyz = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.xyz_pos_embed = XYZPosEmbed(embed_dim)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=args.drop_path
            ) for i in range(depth)])

        self.blocks_xyz = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=args.drop_path
            ) for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.norm_xyz = norm_layer(embed_dim)
        self.cached_enc_feat = None

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(
            embed_dim * 2,
            decoder_embed_dim,
            bias=True
        )

        self.decoder_xyz_pos_embed = DecodeXYZPosEmbed(decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            MCCDecoderBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=args.drop_path,
                args=args,
            ) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if self.args.regress_color:
            self.decoder_pred = nn.Linear(decoder_embed_dim, 3 + 1, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, 256 * 3 + 1, bias=True) # decoder to patch

        self.loss_occupy = nn.BCEWithLogitsLoss()
        if self.args.regress_color:
            self.loss_rgb = nn.MSELoss()
        else:
            self.loss_rgb = nn.CrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_token_xyz, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x, seen_xyz, valid_seen_xyz):

        # get tokens
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        y = self.xyz_pos_embed(seen_xyz, valid_seen_xyz)

        ##### forward E_XYZ #####
        # append cls token
        cls_token_xyz = self.cls_token_xyz
        cls_tokens_xyz = cls_token_xyz.expand(y.shape[0], -1, -1)

        y = torch.cat((cls_tokens_xyz, y), dim=1)
        # apply Transformer blocks
        for blk in self.blocks_xyz:
            y = blk(y)
        y = self.norm_xyz(y)

        ##### forward E_RGB #####
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # combine encodings
        x = torch.cat([x, y], dim=2)
        return x

    def forward_decoder(self, x, unseen_xyz):
        # embed tokens
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        # 3D pos embed
        unseen_xyz = self.decoder_xyz_pos_embed(unseen_xyz)
        x = torch.cat([x, unseen_xyz], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, unseen_xyz.shape[1])

        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)
        # remove cls & seen token
        pred = pred[:, -unseen_xyz.shape[1]:, :]

        return pred

    def forward_loss(self, pred, unseen_occupy, unseen_rgb):
        loss = self.loss_occupy(
            pred[:, :, :1].reshape((-1, 1)),
            unseen_occupy.reshape((-1, 1)).float()
        ) * self.occupancy_weight

        if unseen_occupy.sum() > 0:
            if self.args.regress_color:
                pred_rgb = pred[:, :, 1:][unseen_occupy.bool()]
                gt_rgb = unseen_rgb[unseen_occupy.bool()]
            else:
                pred_rgb = pred[:, :, 1:][unseen_occupy.bool()].reshape((-1, 256))
                gt_rgb = torch.round(unseen_rgb[unseen_occupy.bool()] * 255).long().reshape((-1,))

            rgb_loss = self.loss_rgb(pred_rgb, gt_rgb) * self.rgb_weight
            loss = loss + rgb_loss
        return loss


    def clear_cache(self):
        self.cached_enc_feat = None

    def forward(self, seen_images, seen_xyz, unseen_xyz, unseen_rgb, unseen_occupy, valid_seen_xyz,
                cache_enc=False):

        unseen_xyz = shrink_points_beyond_threshold(unseen_xyz, self.args.shrink_threshold)

        if self.cached_enc_feat is None:
            seen_images = preprocess_img(seen_images)
            seen_xyz = shrink_points_beyond_threshold(seen_xyz, self.args.shrink_threshold)
            latent = self.forward_encoder(seen_images, seen_xyz, valid_seen_xyz)

        if cache_enc:
            if self.cached_enc_feat is None:
                self.cached_enc_feat = latent
            else:
                latent = self.cached_enc_feat

        pred = self.forward_decoder(latent, unseen_xyz)
        loss = self.forward_loss(pred, unseen_occupy, unseen_rgb)
        return loss, pred


def get_mcc_model(**kwargs):
    return MCC(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


def shrink_points_beyond_threshold(xyz, threshold):
    xyz = xyz.clone().detach()
    dist = (xyz ** 2.0).sum(axis=-1) ** 0.5
    affected = (dist > threshold) * torch.isfinite(dist)
    xyz[affected] = xyz[affected] * (
        threshold * (2.0 - threshold / dist[affected]) / dist[affected]
    )[..., None]
    return xyz


def preprocess_img(x):
    if x.shape[2] != 224:
        assert x.shape[2] == 800
        x = F.interpolate(
            x,
            scale_factor=224./800.,
            mode="bilinear",
        )
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape((1, 3, 1, 1))
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((1, 3, 1, 1))
    imgs_normed = (x - resnet_mean) / resnet_std
    return imgs_normed


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
