from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            # in_channels=config['decoder_channels'][-1],
            # out_channels=config['n_classes'],
            # in_channels= 16,
            in_channels= 16,
            out_channels=2,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

# # coding=utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import copy
# import logging
# import math
#
# from os.path import join as pjoin
#
# import torch
# import torch.nn as nn
# import numpy as np
#
# # from .Block import Block
# from collections import OrderedDict
# from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from torch.nn.modules.utils import _pair
# from scipy import ndimage
# from . import vit_seg_configs as configs
# from .vit_seg_modeling_resnet_skip import ResNetV2
#
#
#
# logger = logging.getLogger(__name__)
#
#
# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"
# #
# # # class Residual(nn.Module):
# # #     def __init__(self, inp_dim, out_dim):
# # #         super(Residual, self).__init__()
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.bn1 = nn.BatchNorm2d(inp_dim)
# # #         self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
# # #         self.bn2 = nn.BatchNorm2d(int(out_dim/2))
# # #         self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
# # #         self.bn3 = nn.BatchNorm2d(int(out_dim/2))
# # #         self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
# # #         self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
# # #         if inp_dim == out_dim:
# # #             self.need_skip = False
# # #         else:
# # #             self.need_skip = True
# # #
# # #     def forward(self, x):
# # #         if self.need_skip:
# # #             residual = self.skip_layer(x)
# # #         else:
# # #             residual = x
# # #         out = self.bn1(x)
# # #         out = self.relu(out)
# # #         out = self.conv1(out)
# # #         out = self.bn2(out)
# # #         out = self.relu(out)
# # #         out = self.conv2(out)
# # #         out = self.bn3(out)
# # #         out = self.relu(out)
# # #         out = self.conv3(out)
# # #         out += residual
# # #         return out
# #
# #
# # # class ChannelPool(nn.Module):
# # #     def forward(self, x):
# # #         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
# #
# #
# # # class Conv(nn.Module):
# # #     def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
# # #         super(Conv, self).__init__()
# # #         self.inp_dim = inp_dim
# # #         self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
# # #         self.relu = None
# # #         self.bn = None
# # #         if relu:
# # #             self.relu = nn.ReLU(inplace=True)
# # #         if bn:
# # #             self.bn = nn.BatchNorm2d(out_dim)
# # #
# # #     def forward(self, x):
# # #         assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
# # #         x = self.conv(x)
# # #         if self.bn is not None:
# # #             x = self.bn(x)
# # #         if self.relu is not None:
# # #             x = self.relu(x)
# # #         return x
# #
# #
# #
# # # class BiF(nn.Module):
# # #     def __init__(self, ch, r_2 ,drop_rate=0.2):
# # #         super(BiF, self).__init__()
# # #
# # #         ch_1 = ch_2 = ch_int = ch_out = ch
# # #
# # #         # channel attention for F_g, use SE Block
# # #         self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
# # #         self.sigmoid = nn.Sigmoid()
# # #
# # #         # spatial attention for F_l
# # #         self.compress = ChannelPool()
# # #         self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
# # #
# # #         # bi-linear modelling for both
# # #         self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
# # #         self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
# # #         self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
# # #     #         self.W = DANetHead(ch_int, ch_int)
# # #
# # #         self.relu = nn.ReLU(inplace=True)
# # #
# # #         self.residual = Residual(ch_1+ch_2+ch_int, ch_out)
# # #
# # #         self.dropout = nn.Dropout2d(drop_rate)
# # #         self.drop_rate = drop_rate
# # #
# # #         self.pam = PAM_Module(ch_int)
# # #
# # #         self.cam = CAM_Module(ch_int)
# # #
# # #
# # #
# # #     def forward(self, g, x):
# # #         # bilinear pooling
# # #         W_g = self.W_g(g)
# # #         W_x = self.W_x(x)
# # #         bp = self.W(W_g*W_x)
# # #
# # #         # spatial attention for cnn branch
# # #         g_in = g
# # #         gS = self.cam(g)
# # #         g = self.compress(g)
# # #         g = self.spatial(g)
# # #         g = self.sigmoid(g) * g_in
# # #
# # #
# # #         # channel attetion for transformer branch
# # #         x_in = x
# # #         x = self.pam(x)
# # #         x = x.mean((2, 3), keepdim=True)
# # #         x = self.fc1(x)
# # #         x = self.relu(x)
# # #         x = self.fc2(x)
# # #         g = self.pam(x_in)
# # #         x = self.sigmoid(x) * x_in
# # #
# # #         fuse = self.residual(torch.cat([g, x, bp], 1))
# # #
# # #
# # #         if self.drop_rate > 0:
# # #             return self.dropout(fuse)
# # #         else:
# # #             return fuse
# #
# #
# #
# #
# #
# # # def norm(planes, mode='gn', groups=16):
# # #     if mode == 'bn':
# # #         return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
# # #     elif mode == 'gn':
# # #         return nn.GroupNorm(groups, planes)
# # #     else:
# # #         return nn.Sequential()
# #
# #
# #
# # def np2th(weights, conv=False):
# #     """Possibly convert HWIO to OIHW."""
# #     if conv:
# #         weights = weights.transpose([3, 2, 0, 1])
# #     return torch.from_numpy(weights)
# #
# #
# #
# #
# # def swish(x):
# #     return x * torch.sigmoid(x)
# #
# #
# # ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
# #
# #
# # # class DANetHeadC(nn.Module):
# # #     def __init__(self, in_channels, out_channels):
# # #         super(DANetHead2, self).__init__()
# # #         inter_channels = in_channels // 4
# # #
# # #         self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
# # #                                     norm(inter_channels),
# # #                                     nn.ReLU())
# # #
# # #         self.sa = PAM_Module(inter_channels)
# # #         self.sc = CAM_Module(inter_channels)
# # #
# # #         self.conv6 = nn.Sequential(nn.Dropout2d(0.02, False), nn.Conv2d(inter_channels*2, out_channels, 1),
# # #                                    nn.ReLU())
# # #
# # #     def forward(self, x):
# # #         feat = self.conv5(x)
# # #         sa_feat = self.sa(feat)
# # #         sc_feat = self.sc(feat)
# # #
# # #         # Concatenate the attention-weighted features
# # #         feat_concat = torch.cat([sa_feat, sc_feat], dim=1)
# # #
# # #         output = self.conv6(feat_concat)
# # #
# # #         return output
# #
# #
# #
# # #  DANetHead
# # # class DANetHead(nn.Module):
# # #     def __init__(self, in_channels, out_channels):
# # #         super(DANetHead, self).__init__()
# # #         inter_channels = in_channels // 16
# # # #         inter_channels = in_channels  # test
# # #
# # #         self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
# # #                                     norm(inter_channels),
# # #                                     nn.ReLU())
# # #
# # #         self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
# # #                                     norm(inter_channels),
# # #                                     nn.ReLU())
# # #
# # #         self.sa = PAM_Module(inter_channels)
# # #         self.sc = CAM_Module(inter_channels)
# # #         self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
# # #                                     norm(inter_channels),
# # #                                     nn.ReLU())
# # #         self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
# # #                                     norm(inter_channels),
# # #                                     nn.ReLU())
# # #
# # #         self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
# # #                                    nn.ReLU())
# # #         self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
# # #                                    nn.ReLU())
# # #
# # #         self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
# # #                                    nn.ReLU())
# # #
# # #
# # #     def forward(self, x):
# # #
# # #         feat1 = self.conv5a(x)
# # #         sa_feat = self.sa(feat1)
# # #         sa_conv = self.conv51(sa_feat)
# # #         sa_output = self.conv6(sa_conv)
# # #
# # #         feat2 = self.conv5c(x)
# # #         sc_feat = self.sc(feat2)
# # #         sc_conv = self.conv52(sc_feat)
# # #         sc_output = self.conv7(sc_conv)
# # #
# # #         feat_sum = sa_conv + sc_conv
# # #
# # #         sasc_output = self.conv8(feat_sum)
# # #
# # #
# # #         return sasc_output
# #
# #
# # class Attention(nn.Module):
# #     def __init__(self, config, vis):
# #         super(Attention, self).__init__()
# #         self.vis = vis
# #         self.num_attention_heads = config.transformer["num_heads"]
# #         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
# #         self.all_head_size = self.num_attention_heads * self.attention_head_size
# #
# #         self.query = Linear(config.hidden_size, self.all_head_size)
# #         self.key = Linear(config.hidden_size, self.all_head_size)
# #         self.value = Linear(config.hidden_size, self.all_head_size)
# #
# #         self.out = Linear(config.hidden_size, config.hidden_size)
# #         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
# #         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
# #
# #         self.softmax = Softmax(dim=-1)
# #
# #     def transpose_for_scores(self, x):
# #         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
# #         x = x.view(*new_x_shape)
# #         return x.permute(0, 2, 1, 3)
# #
# #     def forward(self, hidden_states):
# #         mixed_query_layer = self.query(hidden_states)
# #         mixed_key_layer = self.key(hidden_states)
# #         mixed_value_layer = self.value(hidden_states)
# #
# #         query_layer = self.transpose_for_scores(mixed_query_layer)
# #         key_layer = self.transpose_for_scores(mixed_key_layer)
# #         value_layer = self.transpose_for_scores(mixed_value_layer)
# #
# #         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
# #         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
# #         attention_probs = self.softmax(attention_scores)
# #         weights = attention_probs if self.vis else None
# #         attention_probs = self.attn_dropout(attention_probs)
# #
# #         context_layer = torch.matmul(attention_probs, value_layer)
# #         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
# #         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
# #         context_layer = context_layer.view(*new_context_layer_shape)
# #         attention_output = self.out(context_layer)
# #         attention_output = self.proj_dropout(attention_output)
# #         return attention_output, weights
# #
# #
# # class Mlp(nn.Module):
# #     def __init__(self, config):
# #         super(Mlp, self).__init__()
# #         self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
# #         self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
# #         self.act_fn = ACT2FN["gelu"]
# #         self.dropout = Dropout(config.transformer["dropout_rate"])
# #
# #         self._init_weights()
# #
# #     def _init_weights(self):
# #         nn.init.xavier_uniform_(self.fc1.weight)
# #         nn.init.xavier_uniform_(self.fc2.weight)
# #         nn.init.normal_(self.fc1.bias, std=1e-6)
# #         nn.init.normal_(self.fc2.bias, std=1e-6)
# #
# #     def forward(self, x):
# #         # print(x.shape)
# #         x = self.fc1(x)
# #         x = self.act_fn(x)
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         x = self.dropout(x)
# #         return x
# #
# #
# # # from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# # #
# # #
# # # class MlP(nn.Module):
# # #     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
# # #         super().__init__()
# # #         out_features = out_features or in_features
# # #         hidden_features = hidden_features or in_features
# # #         self.fc1 = nn.Linear(in_features, hidden_features)
# # #         self.act = act_layer()
# # #         self.fc2 = nn.Linear(hidden_features, out_features)
# # #         self.drop = nn.Dropout(drop)
# # #
# # #     def forward(self, x):
# # #         x = self.fc1(x)
# # #         x = self.act(x)
# # #         x = self.drop(x)
# # #         x = self.fc2(x)
# # #         x = self.drop(x)
# # #         return x
# # #
# # #
# # # def window_partition(x, window_size):
# # #     """
# # #     Args:
# # #         x: (B, H, W, C)
# # #         window_size (int): window size
# # #
# # #     Returns:
# # #         windows: (num_windows*B, window_size, window_size, C)
# # #     """
# # #     B, H, W, C = x.shape
# # #     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
# # #     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
# # #     return windows
# # #
# # #
# # # def window_reverse(windows, window_size, H, W):
# # #     """
# # #     Args:
# # #         windows: (num_windows*B, window_size, window_size, C)
# # #         window_size (int): Window size
# # #         H (int): Height of image
# # #         W (int): Width of image
# # #
# # #     Returns:
# # #         x: (B, H, W, C)
# # #     """
# # #     B = int(windows.shape[0] / (H * W / window_size / window_size))
# # #     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
# # #     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
# # #     return x
# # #
# # #
# # # class WindowAttention(nn.Module):
# # #     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
# # #     It supports both of shifted and non-shifted window.
# # #
# # #     Args:
# # #         dim (int): Number of input channels.
# # #         window_size (tuple[int]): The height and width of the window.
# # #         num_heads (int): Number of attention heads.
# # #         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
# # #         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
# # #         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
# # #         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
# # #     """
# # #
# # #     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
# # #
# # #         super().__init__()
# # #         self.dim = dim
# # #         self.window_size = window_size  # Wh, Ww
# # #         self.num_heads = num_heads
# # #         head_dim = dim // num_heads
# # #         self.scale = qk_scale or head_dim ** -0.5
# # #
# # #         # define a parameter table of relative position bias
# # #         self.relative_position_bias_table = nn.Parameter(
# # #             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
# # #
# # #         # get pair-wise relative position index for each token inside the window
# # #         coords_h = torch.arange(self.window_size[0])
# # #         coords_w = torch.arange(self.window_size[1])
# # #         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
# # #         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
# # #         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
# # #         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
# # #         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
# # #         relative_coords[:, :, 1] += self.window_size[1] - 1
# # #         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
# # #         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
# # #         self.register_buffer("relative_position_index", relative_position_index)
# # #
# # #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
# # #         self.attn_drop = nn.Dropout(attn_drop)
# # #         self.proj = nn.Linear(dim, dim)
# # #         self.proj_drop = nn.Dropout(proj_drop)
# # #
# # #         trunc_normal_(self.relative_position_bias_table, std=.02)
# # #         self.softmax = nn.Softmax(dim=-1)
# # #
# # #     def forward(self, x, mask=None):
# # #         """
# # #         Args:
# # #             x: input features with shape of (num_windows*B, N, C)
# # #             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
# # #         """
# # #         B_, N, C = x.shape
# # #         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
# # #         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
# # #
# # #         q = q * self.scale
# # #         attn = (q @ k.transpose(-2, -1))
# # #
# # #         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
# # #             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
# # #         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
# # #         attn = attn + relative_position_bias.unsqueeze(0)
# # #
# # #         if mask is not None:
# # #             nW = mask.shape[0]
# # #             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
# # #             attn = attn.view(-1, self.num_heads, N, N)
# # #             attn = self.softmax(attn)
# # #         else:
# # #             attn = self.softmax(attn)
# # #
# # #         attn = self.attn_drop(attn)
# # #
# # #         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
# # #         x = self.proj(x)
# # #         x = self.proj_drop(x)
# # #         return x
# # #
# # #     def extra_repr(self) -> str:
# # #         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
# # #
# # #     def flops(self, N):
# # #         # calculate flops for 1 window with token length of N
# # #         flops = 0
# # #         # qkv = self.qkv(x)
# # #         flops += N * self.dim * 3 * self.dim
# # #         # attn = (q @ k.transpose(-2, -1))
# # #         flops += self.num_heads * N * (self.dim // self.num_heads) * N
# # #         #  x = (attn @ v)
# # #         flops += self.num_heads * N * N * (self.dim // self.num_heads)
# # #         # x = self.proj(x)
# # #         flops += N * self.dim * self.dim
# # #         return flops
# # #
# # #
# # # class SwinTransformerBlock(nn.Module):
# # #     r""" Swin Transformer Block.
# # #
# # #     Args:
# # #         dim (int): Number of input channels.
# # #         input_resolution (tuple[int]): Input resulotion.
# # #         num_heads (int): Number of attention heads.
# # #         window_size (int): Window size.
# # #         shift_size (int): Shift size for SW-MSA.
# # #         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
# # #         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
# # #         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
# # #         drop (float, optional): Dropout rate. Default: 0.0
# # #         attn_drop (float, optional): Attention dropout rate. Default: 0.0
# # #         drop_path (float, optional): Stochastic depth rate. Default: 0.0
# # #         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
# # #         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
# # #     """
# # #
# # #     def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
# # #                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
# # #                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
# # #         super().__init__()
# # #         self.dim = dim
# # #         self.input_resolution = input_resolution
# # #         self.num_heads = num_heads
# # #         self.window_size = window_size
# # #         self.shift_size = shift_size
# # #         self.mlp_ratio = mlp_ratio
# # #         if min(self.input_resolution) <= self.window_size:
# # #             # if window size is larger than input resolution, we don't partition windows
# # #             self.shift_size = 0
# # #             self.window_size = min(self.input_resolution)
# # #         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
# # #
# # #         self.norm1 = norm_layer(dim)
# # #         self.attn = WindowAttention(
# # #             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
# # #             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
# # #
# # #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# # #         self.norm2 = norm_layer(dim)
# # #         mlp_hidden_dim = int(dim * mlp_ratio)
# # #         self.mlp = MlP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
# # #
# # #         if self.shift_size > 0:
# # #             # calculate attention mask for SW-MSA
# # #             H, W = self.input_resolution
# # #             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
# # #             h_slices = (slice(0, -self.window_size),
# # #                         slice(-self.window_size, -self.shift_size),
# # #                         slice(-self.shift_size, None))
# # #             w_slices = (slice(0, -self.window_size),
# # #                         slice(-self.window_size, -self.shift_size),
# # #                         slice(-self.shift_size, None))
# # #             cnt = 0
# # #             for h in h_slices:
# # #                 for w in w_slices:
# # #                     img_mask[:, h, w, :] = cnt
# # #                     cnt += 1
# # #
# # #             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
# # #             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
# # #             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
# # #             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
# # #         else:
# # #             attn_mask = None
# # #
# # #         self.register_buffer("attn_mask", attn_mask)
# # #
# # #     def forward(self, x):
# # # #         print("a", x.shape)
# # #         H, W = self.input_resolution
# # #         B, L, C = x.shape
# # #         assert L == H * W, "input feature has wrong size"
# # #
# # #         shortcut = x
# # #         x = self.norm1(x)
# # #         x = x.view(B, H, W, C)
# # #
# # #         # cyclic shift
# # #         if self.shift_size > 0:
# # #             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
# # #         else:
# # #             shifted_x = x
# # #
# # #         # partition windows
# # #         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
# # #         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
# # #
# # #         # W-MSA/SW-MSA
# # #         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
# # #
# # #         # merge windows
# # #         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
# # #         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
# # #
# # #         # reverse cyclic shift
# # #         if self.shift_size > 0:
# # #             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
# # #         else:
# # #             x = shifted_x
# # #         x = x.view(B, H * W, C)
# # #
# # #         # FFN
# # #         x = shortcut + self.drop_path(x)
# # #         x = x + self.drop_path(self.mlp(self.norm2(x)))
# # # #         print("b", x.shape)
# # #
# # #         return x
# # #
# # #     def extra_repr(self) -> str:
# # #         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
# # #                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
# # #
# # #     def flops(self):
# # #         flops = 0
# # #         H, W = self.input_resolution
# # #         # norm1
# # #         flops += self.dim * H * W
# # #         # W-MSA/SW-MSA
# # #         nW = H * W / self.window_size / self.window_size
# # #         flops += nW * self.attn.flops(self.window_size * self.window_size)
# # #         # mlp
# # #         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
# # #         # norm2
# # #         flops += self.dim * H * W
# # #         return flops
# #
# #
# #
# #
# # #  Transformer中用到了
# # class Embeddings(nn.Module):
# #     """Construct the embeddings from patch, position embeddings.
# #     """
# #     def __init__(self, config, img_size, in_channels=3):
# #         super(Embeddings, self).__init__()
# #         self.hybrid = True
# #         self.config = config
# #         img_size = _pair(img_size)
# #
# #         # self.DAblock1 = DANetHead(768, 768)
# # #         self.bif = BiF(ch = 768, r_2 = 4)
# #
# #         if config.patches.get("grid") is not None:   # ResNet
# #             grid_size = config.patches["grid"]
# #             patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
# #             patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
# #             n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
# #             self.hybrid = True
# #         else:
# #             patch_size = _pair(config.patches["size"])
# #             n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
# #             self.hybrid = False
# #
# #         if self.hybrid:
# #             self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
# #             in_channels = self.hybrid_model.width * 16
# #         self.patch_embeddings = Conv2d(in_channels=in_channels,
# #                                        out_channels=config.hidden_size,
# #                                        kernel_size=patch_size,
# #                                        stride=patch_size)
# #         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
# #
# #         self.dropout = Dropout(config.transformer["dropout_rate"])
# #
# #         # self.swin = SwinTransformerBlock(dim=768, input_resolution=[16,16],
# #         #                          num_heads=4, window_size=8,
# #         #                          shift_size=0,
# #         #                          mlp_ratio=4,
# #         #                          qkv_bias=True, qk_scale=None,
# #         #                          drop=0.2, attn_drop=0.2,
# #         #                         )
# #
# #
# #
# #     def forward(self, x):
# #         if self.hybrid:
# #             x, features = self.hybrid_model(x)
# #         else:
# #             features = None
# #         redual = x
# #
# #         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
# # #         x = self.DAblock1(x)
# #         x = x.flatten(2)
# #         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
# #
# #         embeddings = x + self.position_embeddings
# #         embeddings = self.dropout(embeddings)
# #
# #         return embeddings, features
# #
# #
# # class Block(nn.Module):
# #     def __init__(self, config, vis):
# #         super(Block, self).__init__()
# #
# #         self.hidden_size = config.hidden_size
# #         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
# #         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
# #         self.ffn = Mlp(config)
# #         self.attn = Attention(config, vis)
# #
# #     def forward(self, x):
# #         # print(x.shape)
# #         h = x
# #         x = self.attention_norm(x)
# #
# #         x, weights = self.attn(x)
# #         x = x + h
# #
# #
# #         h = x
# #         x = self.ffn_norm(x)
# #
# #         x = self.ffn(x)
# #
# #         x = x + h
# # #         print(x.shape) # 24 196 768
# #
# #
# #
# #         return x, weights
# #
# #     def load_from(self, weights, n_block):
# #         ROOT = f"Transformer/encoderblock_{n_block}"
# #         with torch.no_grad():
# #             query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
# #             key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
# #             value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
# #             out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
# #
# #             query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
# #             key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
# #             value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
# #             out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
# #
# #             self.attn.query.weight.copy_(query_weight)
# #             self.attn.key.weight.copy_(key_weight)
# #             self.attn.value.weight.copy_(value_weight)
# #             self.attn.out.weight.copy_(out_weight)
# #             self.attn.query.bias.copy_(query_bias)
# #             self.attn.key.bias.copy_(key_bias)
# #             self.attn.value.bias.copy_(value_bias)
# #             self.attn.out.bias.copy_(out_bias)
# #
# #             mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
# #             mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
# #             mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
# #             mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
# #
# #             self.ffn.fc1.weight.copy_(mlp_weight_0)
# #             self.ffn.fc2.weight.copy_(mlp_weight_1)
# #             self.ffn.fc1.bias.copy_(mlp_bias_0)
# #             self.ffn.fc2.bias.copy_(mlp_bias_1)
# #
# #             self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
# #             self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
# #             self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
# #             self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
# #
# # #  Transformer 中的Encoder函数
# # class Encoder(nn.Module):
# #     def __init__(self, config, vis):
# #         super(Encoder, self).__init__()
# #         self.vis = vis
# #         self.layer = nn.ModuleList()
# #         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
# #         for _ in range(config.transformer["num_layers"]):
# #             layer = Block(config, vis)
# #             self.layer.append(copy.deepcopy(layer))
# #
# #     def forward(self, hidden_states):
# #         attn_weights = []
# #         for layer_block in self.layer:
# #             hidden_states, weights = layer_block(hidden_states)
# #             if self.vis:
# #                 attn_weights.append(weights)
# #         encoded = self.encoder_norm(hidden_states)
# #         return encoded, attn_weights
# #
# #
# # class Transformer(nn.Module):
# #     def __init__(self, config, img_size, vis):
# #         super(Transformer, self).__init__()
# #         self.embeddings = Embeddings(config, img_size=img_size)
# #         self.encoder = Encoder(config, vis)
# #
# #     def forward(self, input_ids):
# #         embedding_output, features = self.embeddings(input_ids)
# #         encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
# #         return encoded, attn_weights, features
# #
# #
# #
# # class Conv2dReLU(nn.Sequential):
# #     def __init__(
# #             self,
# #             in_channels,
# #             out_channels,
# #             kernel_size,
# #             padding=0,
# #             stride=1,
# #             use_batchnorm=True,
# #     ):
# #         conv = nn.Conv2d(
# #             in_channels,
# #             out_channels,
# #             kernel_size,
# #             stride=stride,
# #             padding=padding,
# #             bias=not (use_batchnorm),
# #         )
# #         relu = nn.ReLU(inplace=True)
# #
# #         bn = nn.BatchNorm2d(out_channels)
# #
# #         super(Conv2dReLU, self).__init__(conv, bn, relu)
# #
# #
# # class DecoderBlock(nn.Module):
# #     def __init__(
# #             self,
# #             in_channels,
# #             out_channels,
# #             skip_channels=0,
# #             use_batchnorm=True,
# #     ):
# #         super().__init__()
# #
# #         self.conv1 = Conv2dReLU(
# #             in_channels + skip_channels,
# #             out_channels,
# #             kernel_size=3,
# #             padding=1,
# #             use_batchnorm=use_batchnorm,
# #         )
# #         self.conv2 = Conv2dReLU(
# #             out_channels,
# #             out_channels,
# #             kernel_size=3,
# #             padding=1,
# #             use_batchnorm=use_batchnorm,
# #         )
# #         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
# #         # self.bif3 = BiF(ch = 64, r_2 = 4)
# #
# #
# #
# #     def forward(self, x, skip=None):
# #         x = self.up(x)
# # #         print(x.shape) 24 512 28 28,24 256 56 56,24 128 112 112,24 64 224 224
# #         if skip is not None:
# # #             redual = skip
# # #             if skip.size(1) and x.size(1) == 64:
# # #                 skip = self.bif3(x, x)
# # #                 skip = skip + redual
# #             x = torch.cat([x, skip], dim=1)
# #         x = self.conv1(x)
# #         x = self.conv2(x)
# #         return x
# #
# #
# # class SegmentationHead(nn.Sequential):
# #
# #     def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
# #         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
# #         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
# #         super().__init__(conv2d, upsampling)
# #
# # #这部分是reshape块的方法，因此该部分会涉及到DA块的加入，考虑在之前加还是在reshape后加入，至于解码器部分后续在讨论有无必要加入DA模块
# # class DecoderCup(nn.Module):
# #     def __init__(self, config):
# #         super().__init__()
# #         self.config = config
# #         head_channels = 512
# #         self.conv_more = Conv2dReLU(
# #             config.hidden_size,
# #             head_channels,
# #             kernel_size=3,
# #             padding=1,
# #             use_batchnorm=True,
# #         )
# #         decoder_channels = config.decoder_channels
# #         in_channels = [head_channels] + list(decoder_channels[:-1])
# #         out_channels = decoder_channels
# #
# #         if self.config.n_skip != 0:
# #             skip_channels = self.config.skip_channels
# #             for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
# #                 skip_channels[3-i]=0
# #
# #         else:
# #             skip_channels=[0,0,0,0]
# #
# #         blocks = [
# #             DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
# #         ]
# #         self.blocks = nn.ModuleList(blocks)
# #
# #     def forward(self, hidden_states, features=None):
# #         B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
# #         h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
# #         x = hidden_states.permute(0, 2, 1)
# #         x = x.contiguous().view(B, hidden, h, w)
# #         x = self.conv_more(x)
# #         for i, decoder_block in enumerate(self.blocks):
# #             if features is not None:
# #                 skip = features[i] if (i < self.config.n_skip) else None
# #             else:
# #                 skip = None
# #             x = decoder_block(x, skip=skip)
# #         return x
# #
# #
# # class VisionTransformer(nn.Module):
# #     def __init__(self, config, img_size=224, num_classes=0, zero_head=False, vis=False):
# #         super(VisionTransformer, self).__init__()
# #         self.num_classes = num_classes
# #         self.zero_head = zero_head
# #         self.classifier = config.classifier
# #         self.transformer = Transformer(config, img_size, vis)
# #         self.decoder = DecoderCup(config)
# #         self.segmentation_head = SegmentationHead(
# #             # in_channels=config['decoder_channels'][-1],
# #             # out_channels=config['n_classes'],
# #             in_channels= 24 ,
# #             out_channels= 1,
# #             kernel_size=3,
# #         )
# #         self.config = config
# #
# #     def forward(self, x):
# #         if x.size()[1] == 1:
# #             x = x.repeat(1,3,1,1)
# #         x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
# #         x = self.decoder(x, features)
# #         logits = self.segmentation_head(x)
# # #         print("over",logits.shape)
# #         return logits
# #
# #     def load_from(self, weights):
# #         with torch.no_grad():
# #
# #             res_weight = weights
# #             self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
# #             self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
# #
# #             self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
# #             self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
# #
# #             posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
# #
# #             posemb_new = self.transformer.embeddings.position_embeddings
# #             if posemb.size() == posemb_new.size():
# #                 self.transformer.embeddings.position_embeddings.copy_(posemb)
# #             elif posemb.size()[1]-1 == posemb_new.size()[1]:
# #                 posemb = posemb[:, 1:]
# #                 self.transformer.embeddings.position_embeddings.copy_(posemb)
# #             else:
# #                 logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
# #                 ntok_new = posemb_new.size(1)
# #                 if self.classifier == "seg":
# #                     _, posemb_grid = posemb[:, :1], posemb[0, 1:]
# #                 gs_old = int(np.sqrt(len(posemb_grid)))
# #                 gs_new = int(np.sqrt(ntok_new))
# #                 print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
# #                 posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
# #                 zoom = (gs_new / gs_old, gs_new / gs_old, 1)
# #                 posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
# #                 posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
# #                 posemb = posemb_grid
# #                 self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
# #
# #             # Encoder whole
# #             for bname, block in self.transformer.encoder.named_children():
# #                 for uname, unit in block.named_children():
# #                     unit.load_from(weights, n_block=uname)
# #
# #             if self.transformer.embeddings.hybrid:
# #                 self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
# #                 gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
# #                 gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
# #                 self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
# #                 self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
# #
# #                 for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
# #                     for uname, unit in block.named_children():
# #                         unit.load_from(res_weight, n_block=bname, n_unit=uname)
# #
# # CONFIGS = {
# #     'ViT-B_16': configs.get_b16_config(),
# #     'ViT-B_32': configs.get_b32_config(),
# #     'ViT-L_16': configs.get_l16_config(),
# #     'ViT-L_32': configs.get_l32_config(),
# #     'ViT-H_14': configs.get_h14_config(),
# #     'R50-ViT-B_16': configs.get_r50_b16_config(),
# #     'R50-ViT-L_16': configs.get_r50_l16_config(),
# #     'testing': configs.get_testing(),
# # }
# #
# #
# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
# # import torch
# # from torch import nn
# # import copy
# # import logging
# # import math
# #
# # from .block import BiF,DANetHead
# # from os.path import join as pjoin
# # import numpy as np
# # from .swinblock import SwinTransformerBlock
# #
# # from collections import OrderedDict
# # from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# # from torch.nn.modules.utils import _pair
# # from scipy import ndimage
# # from . import vit_seg_configs as configs
# # from .vit_seg_modeling_resnet_skip import ResNetV2
# #
# # __all__ = ['UNet','DDATransformer']
#
#
#
# __all__ = ['UNet', 'NestedUNet','VisionTransformer']
#
#
# class Residual(nn.Module):
#     def __init__(self, inp_dim, out_dim):
#         super(Residual, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.bn1 = nn.BatchNorm2d(inp_dim)
#         self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
#         self.bn2 = nn.BatchNorm2d(int(out_dim/2))
#         self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
#         self.bn3 = nn.BatchNorm2d(int(out_dim/2))
#         self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
#         self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
#         if inp_dim == out_dim:
#             self.need_skip = False
#         else:
#             self.need_skip = True
#
#     def forward(self, x):
#         if self.need_skip:
#             residual = self.skip_layer(x)
#         else:
#             residual = x
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out += residual
#         return out
#
#
# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
#
#
# class Conv(nn.Module):
#     def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
#         super(Conv, self).__init__()
#         self.inp_dim = inp_dim
#         self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
#         self.relu = None
#         self.bn = None
#         if relu:
#             self.relu = nn.ReLU(inplace=True)
#         if bn:
#             self.bn = nn.BatchNorm2d(out_dim)
#
#     def forward(self, x):
#         assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# def norm(planes, mode='bn', groups=16):
#     if mode == 'bn':
#         return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
#     elif mode == 'gn':
#         return nn.GroupNorm(groups, planes)
#     else:
#         return nn.Sequential()
#
# class BiF(nn.Module):
#     def __init__(self, ch, r_2 ,drop_rate=0.2):
#         super(BiF, self).__init__()
#
#         ch_1 = ch_2 = ch_int = ch_out = ch
#
#         # channel attention for F_g, use SE Block
#         self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#         # spatial attention for F_l
#         self.compress = ChannelPool()
#         self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
#
#         # bi-linear modelling for both
#         self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
#         self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
#         self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.residual = Residual(ch_1+ch_2+ch_int, ch_out)
#
#         self.dropout = nn.Dropout2d(drop_rate)
#         self.drop_rate = drop_rate
#
#         self.pam = PAM_Module(ch_int)
#
#         self.cam = CAM_Module(ch_int)
#
#
#
#     def forward(self, g, x):
#         # bilinear pooling
#         W_g = self.W_g(g)
#         W_x = self.W_x(x)
#         bp = self.W(W_g*W_x)
#
#         # spatial
#         g_in = g
#         g_in = self.cam(g_in)
#         g = self.compress(g)
#         g = self.spatial(g)
#         g = self.sigmoid(g) * g_in
#
#
#         # channel
#         x_in = x
#         x_in = self.pam(x_in)
#         x = x.mean((2, 3), keepdim=True)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x) * x_in
#
#         fuse = self.residual(torch.cat([g, x, bp], 1))
#
#
#         if self.drop_rate > 0:
#             return self.dropout(fuse)
#         else:
#             return fuse
#
#
#
# class DANetHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DANetHead, self).__init__()
#         inter_channels = in_channels // 16
#         self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.sa = PAM_Module(inter_channels)
#         self.sc = CAM_Module(inter_channels)
#         self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#         self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#         self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#
#         self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#
#     def forward(self, x):
#         feat1 = self.conv5a(x)
#         sa_feat = self.sa(feat1)
#         sa_conv = self.conv51(sa_feat)
#         sa_output = self.conv6(sa_conv)
#
#         feat2 = self.conv5c(x)
#         sc_feat = self.sc(feat2)
#         sc_conv = self.conv52(sc_feat)
#         sc_output = self.conv7(sc_conv)
#
#         feat_sum = sa_conv + sc_conv
#
#         sasc_output = self.conv8(feat_sum)
#         print(sasc_output.size)
#         print(sasc_output.shape)
#
#         return sasc_output
#
#
# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         return out
#
#
# # class ResBlock(nn.Module):
# #     def __init__(self, in_channels, middle_channels, out_channels):
# #         super().__init__()
# #         self.relu = nn.ReLU(inplace=True)
# #         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
# #         self.bn1 = nn.BatchNorm2d(middle_channels)
# #         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
# #         self.bn2 = nn.BatchNorm2d(out_channels)
#
# #     def forward(self, x):
# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)
#
# #         out = self.conv2(out)
# #         out = self.bn2(out)
# #         out = x + out
# #         out = self.relu(out)
#
# #         return out
#
# class DABlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.DA = DANetHead(middle_channels, middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.DA(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         return out
#
#
# class UNet(nn.Module):
#     def __init__(self, num_classes, input_channels=3, **kwargs):
#         super().__init__()
#
#         nb_filter = [32, 64, 128, 256, 512]
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
#
#         self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#
#         self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
#         self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
#         self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
#
#         self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#
#     def forward(self, input):
#         x0_0 = self.conv0_0(input)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x4_0 = self.conv4_0(self.pool(x3_0))
#
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
#
#         output = self.final(x0_4)
#         return output
#
#
# class NestedUNet(nn.Module):
#     def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
#         super().__init__()
#
#         nb_filter = [32, 64, 128, 256, 512]
#
#         self.deep_supervision = deep_supervision
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#         self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
#         self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
#         self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
#         self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
#
#         self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
#         self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
#
#         self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
#         self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
#
#         self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
#         self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
#
#         self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
#
#         if self.deep_supervision:
#             self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#             self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#         else:
#             self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
#
#     def forward(self, input):
#         # print('input:',input.shape)
#         x0_0 = self.conv0_0(input)
#         # print('x0_0:',x0_0.shape)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         # print('x1_0:',x1_0.shape)
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
#         # print('x0_1:',x0_1.shape)
#
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         # print('x2_0:',x2_0.shape)
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
#         # print('x1_1:',x1_1.shape)
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
#         # print('x0_2:',x0_2.shape)
#
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         # print('x3_0:',x3_0.shape)
#         x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
#         # print('x2_1:',x2_1.shape)
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
#         # print('x1_2:',x1_2.shape)
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
#         # print('x0_3:',x0_3.shape)
#         x4_0 = self.conv4_0(self.pool(x3_0))
#         # print('x4_0:',x4_0.shape)
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
#         # print('x3_1:',x3_1.shape)
#         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
#         # print('x2_2:',x2_2.shape)
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
#         # print('x1_3:',x1_3.shape)
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
#         # print('x0_4:',x0_4.shape)
#
#         if self.deep_supervision:
#             output1 = self.final1(x0_1)
#             output2 = self.final2(x0_2)
#             output3 = self.final3(x0_3)
#             output4 = self.final4(x0_4)
#             return [output1, output2, output3, output4]
#
#         else:
#             output = self.final(x0_4)
#             return output
#
# # coding=utf-8
#
# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
#
# import copy
# import logging
# import math
#
# from os.path import join as pjoin
#
# import torch
# import torch.nn as nn
# import numpy as np
#
# from collections import OrderedDict
# # from .DA import CAM_Module, PAM_Module
# from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from torch.nn.modules.utils import _pair
# from scipy import ndimage
# from . import vit_seg_configs as configs
# # import vit_seg_configs as configs
# from .vit_seg_modeling_resnet_skip import ResNetV2
#
#
# logger = logging.getLogger(__name__)
#
#
# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"
#
# def norm(planes, mode='gn', groups=16):
#     if mode == 'bn':
#         return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
#     elif mode == 'gn':
#         return nn.GroupNorm(groups, planes)
#     else:
#         return nn.Sequential()
#
#
#
# def np2th(weights, conv=False):
#     """Possibly convert HWIO to OIHW."""
#     if conv:
#         weights = weights.transpose([3, 2, 0, 1])
#     return torch.from_numpy(weights)
#
#
#
# def swish(x):
#     return x * torch.sigmoid(x)
#
#
# ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
#
#
#
# #  加入的DANetHead
# class DANetHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DANetHead, self).__init__()
#         inter_channels = in_channels // 16
#         #         inter_channels = in_channels  # test
#
#         self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.sa = PAM_Module(inter_channels)
#         self.sc = CAM_Module(inter_channels)
#         self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#         self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm(inter_channels),
#                                     nn.ReLU())
#
#         self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#         self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#
#         self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
#                                    nn.ReLU())
#
#
#     def forward(self, x):
#         #         x = x.unsqueeze(0)
#
#         feat1 = self.conv5a(x)
#         sa_feat = self.sa(feat1)
#         sa_conv = self.conv51(sa_feat)
#         sa_output = self.conv6(sa_conv)
#
#         feat2 = self.conv5c(x)
#         sc_feat = self.sc(feat2)
#         sc_conv = self.conv52(sc_feat)
#         sc_output = self.conv7(sc_conv)
#
#         feat_sum = sa_conv + sc_conv
#
#         sasc_output = self.conv8(feat_sum)
#
#
#         return sasc_output
#
#
# class Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         self.query = Linear(config.hidden_size, self.all_head_size)
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)
#
#         self.out = Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
#
#         self.softmax = Softmax(dim=-1)
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, hidden_states):
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         weights = attention_probs if self.vis else None
#         attention_probs = self.attn_dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         return attention_output, weights
#
#
# class Mlp(nn.Module):
#     def __init__(self, config):
#         super(Mlp, self).__init__()
#         self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
#         self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
#         self.act_fn = ACT2FN["gelu"]
#         self.dropout = Dropout(config.transformer["dropout_rate"])
#
#         self._init_weights()
#
#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
#
# #  Transformer中用到了
# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self, config, img_size, in_channels=3):
#         super(Embeddings, self).__init__()
#         self.hybrid = None
#         self.config = config
#         img_size = _pair(img_size)
#
#         self.DAblock1 = DANetHead(768, 768)
#         self.bif = BiF(ch = 768, r_2 = 4)
#
#         if config.patches.get("grid") is not None:   # ResNet
#             grid_size = config.patches["grid"]
#             patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
#             patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
#             n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
#             self.hybrid = True
#         else:
#             patch_size = _pair(config.patches["size"])
#             n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
#             self.hybrid = False
#
#         if self.hybrid:
#             self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
#             in_channels = self.hybrid_model.width * 16
#         self.patch_embeddings = Conv2d(in_channels=in_channels,
#                                        out_channels=config.hidden_size,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
#
#         self.dropout = Dropout(config.transformer["dropout_rate"])
#
#
#
#     def forward(self, x):
#         if self.hybrid:
#             x, features = self.hybrid_model(x)
#         else:
#             features = None
#
#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
#         x = self.DAblock1(x)
#         #         x = self.bif(x, x)
#         x = x.flatten(2)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#
#         return embeddings, features
#
#
# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)
#
#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#
#         x, weights = self.attn(x)
#         x = x + h
#
#
#         h = x
#         x = self.ffn_norm(x)
#
#         x = self.ffn(x)
#
#         x = x + h
#
#
#
#         return x, weights
#
#     def load_from(self, weights, n_block):
#         ROOT = f"Transformer/encoderblock_{n_block}"
#         with torch.no_grad():
#             query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#
#             query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
#             key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
#             value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
#             out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
#
#             self.attn.query.weight.copy_(query_weight)
#             self.attn.key.weight.copy_(key_weight)
#             self.attn.value.weight.copy_(value_weight)
#             self.attn.out.weight.copy_(out_weight)
#             self.attn.query.bias.copy_(query_bias)
#             self.attn.key.bias.copy_(key_bias)
#             self.attn.value.bias.copy_(value_bias)
#             self.attn.out.bias.copy_(out_bias)
#
#             mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
#             mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
#             mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
#             mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
#
#             self.ffn.fc1.weight.copy_(mlp_weight_0)
#             self.ffn.fc2.weight.copy_(mlp_weight_1)
#             self.ffn.fc1.bias.copy_(mlp_bias_0)
#             self.ffn.fc2.bias.copy_(mlp_bias_1)
#
#             self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
#             self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
#             self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
#             self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
#
# #  Transformer 中的Encoder函数
# class Encoder(nn.Module):
#     def __init__(self, config, vis):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config, vis)
#             self.layer.append(copy.deepcopy(layer))
#
#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)
#             if self.vis:
#                 attn_weights.append(weights)
#         encoded = self.encoder_norm(hidden_states)
#         return encoded, attn_weights
#
#
# class Transformer(nn.Module):
#     def __init__(self, config, img_size, vis):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config, vis)
#
#     def forward(self, input_ids):
#         embedding_output, features = self.embeddings(input_ids)
#         encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
#         return encoded, attn_weights, features
#
# # test
#
#
#
# class Conv2dReLU(nn.Sequential):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding=0,
#             stride=1,
#             use_batchnorm=True,
#     ):
#         conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=not (use_batchnorm),
#         )
#         relu = nn.ReLU(inplace=True)
#
#         bn = nn.BatchNorm2d(out_channels)
#
#         super(Conv2dReLU, self).__init__(conv, bn, relu)
#
#
# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         self.conv1 = Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#             )
#         self.conv2 = Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.bif3 = BiF(ch = 64, r_2 = 4)
#         self.da = DANetHead(64, 64)
#
#
#     def forward(self, x, skip=None):
#         x = self.up(x)
#         if skip is not None:
#             redual = skip
#             if skip.size(1) and x.size(1) == 64:
#                 skip = self.bif3(x, x)
#                 #                 skip = self.da(x)
#                 skip = skip + redual
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x
#
#
# class SegmentationHead(nn.Sequential):
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         super().__init__(conv2d, upsampling)
#
# #这部分是reshape块的方法，因此该部分会涉及到DA块的加入，考虑在之前加还是在reshape后加入，至于解码器部分后续在讨论有无必要加入DA模块
# class DecoderCup(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         head_channels = 512
#         self.conv_more = Conv2dReLU(
#             config.hidden_size,
#             head_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=True,
#         )
#         decoder_channels = config.decoder_channels
#         in_channels = [head_channels] + list(decoder_channels[:-1])
#         out_channels = decoder_channels
#
#         if self.config.n_skip != 0:
#             skip_channels = self.config.skip_channels
#             for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
#                 skip_channels[3-i]=0
#
#         else:
#             skip_channels=[0,0,0,0]
#
#         blocks = [
#             DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
#         ]
#         self.blocks = nn.ModuleList(blocks)
#
#     def forward(self, hidden_states, features=None):
#         B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
#         h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
#         x = hidden_states.permute(0, 2, 1)
#         x = x.contiguous().view(B, hidden, h, w)
#         x = self.conv_more(x)
#         for i, decoder_block in enumerate(self.blocks):
#             if features is not None:
#                 skip = features[i] if (i < self.config.n_skip) else None
#             else:
#                 skip = None
#             x = decoder_block(x, skip=skip)
#         return x
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self, config, img_size=256, num_classes=1, zero_head=False, vis=False):
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.classifier = config.classifier
#         self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config)
#         self.segmentation_head = SegmentationHead(
#             #in_channels=config['decoder_channels'][-1],
#             in_channels=16,
#             #out_channels=config['n_classes'],
#             out_channels=1,
#             kernel_size=3,
#         )
#         self.config = config
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
#         x = self.decoder(x, features)
#         logits = self.segmentation_head(x)
#
#         return logits
#
#     def load_from(self, weights):
#         with torch.no_grad():
#
#             res_weight = weights
#             self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
#             self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
#
#             self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
#             self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
#
#             posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
#
#             posemb_new = self.transformer.embeddings.position_embeddings
#             if posemb.size() == posemb_new.size():
#                 self.transformer.embeddings.position_embeddings.copy_(posemb)
#             elif posemb.size()[1]-1 == posemb_new.size()[1]:
#                 posemb = posemb[:, 1:]
#                 self.transformer.embeddings.position_embeddings.copy_(posemb)
#             else:
#                 logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
#                 ntok_new = posemb_new.size(1)
#                 if self.classifier == "seg":
#                     _, posemb_grid = posemb[:, :1], posemb[0, 1:]
#                 gs_old = int(np.sqrt(len(posemb_grid)))
#                 gs_new = int(np.sqrt(ntok_new))
#                 print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
#                 posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
#                 zoom = (gs_new / gs_old, gs_new / gs_old, 1)
#                 posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
#                 posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
#                 posemb = posemb_grid
#                 self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
#
#             # Encoder whole
#             for bname, block in self.transformer.encoder.named_children():
#                 for uname, unit in block.named_children():
#                     unit.load_from(weights, n_block=uname)
#
#             if self.transformer.embeddings.hybrid:
#                 self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
#                 gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
#                 gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
#                 self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
#                 self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
#
#                 for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
#                     for uname, unit in block.named_children():
#                         unit.load_from(res_weight, n_block=bname, n_unit=uname)
#
# CONFIGS = {
#     'ViT-B_16': configs.get_b16_config(),
#     'ViT-B_32': configs.get_b32_config(),
#     'ViT-L_16': configs.get_l16_config(),
#     'ViT-L_32': configs.get_l32_config(),
#     'ViT-H_14': configs.get_h14_config(),
#     'R50-ViT-B_16': configs.get_r50_b16_config(),
#     'R50-ViT-L_16': configs.get_r50_l16_config(),
#     'testing': configs.get_testing(),
# }
#
#
#
#
