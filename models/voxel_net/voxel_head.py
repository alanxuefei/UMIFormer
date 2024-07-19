import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, get_norm
from torch import nn
from torch.nn import functional as F

class VoxelHead(nn.Module):
    def __init__(self, cfg):
        super(VoxelHead, self).__init__()

        # fmt: off
        self.voxel_size = cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE
        conv_dims       = cfg.MODEL.VOXEL_HEAD.CONV_DIM
        num_conv        = cfg.MODEL.VOXEL_HEAD.NUM_CONV
        input_channels  = cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS
        self.norm       = cfg.MODEL.VOXEL_HEAD.NORM
        # fmt: on

        assert self.voxel_size % 2 == 0

        self.conv_norm_relus = []
        prev_dim = input_channels
        for k in range(num_conv):
            conv = Conv2d(
                prev_dim,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("voxel_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            prev_dim = conv_dims

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.predictor = Conv2d(
            conv_dims, self.voxel_size, kernel_size=1, stride=1, padding=0
        )

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for voxel prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        # Initialize the MultiheadAttention
        self.multihead_attention = MultiheadAttention(conv_dims, d_model=conv_dims, dropout=0.1)

    def forward(self, x):
        V = self.voxel_size
        x = F.interpolate(x, size=V // 2, mode="bilinear", align_corners=False)
        for layer in self.conv_norm_relus:
            x = layer(x)

        # Use the multi-head attention mechanism
        x = self.multihead_attention(x)

        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Linear projection to d_model
        self.fc = nn.Linear(input_dim, d_model)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Residual connection layers
        self.res_fc = nn.Linear(input_dim, d_model)

    def forward(self, tokens):
        batch_size, num_channels, height, width = tokens.shape

        # Flatten the spatial dimensions and combine them with the batch
        tokens = tokens.view(batch_size, num_channels, -1)  # (B, C, H*W)
        tokens = tokens.permute(0, 2, 1)  # (B, H*W, C)

        # Project num_channels to d_model
        tokens = self.fc(tokens)  # (B, H*W, d_model)

        # Split into query, key, value
        query = key = value = tokens

        # Apply self-attention
        attn_output, _ = self.self_attention(query, key, value)

        # Add residual connection and layer normalization
        tokens = self.norm1(tokens + self.dropout(attn_output))

        # Another residual connection and layer normalization
        tokens = self.norm2(tokens + self.res_fc(tokens))

        # Reshape the output back to (B, d_model, H, W)
        attn_output = tokens.permute(0, 2, 1).contiguous()
        attn_output = attn_output.view(batch_size, self.d_model, height, width)

        return attn_output

# Example usage
# Assume cfg is properly defined somewhere in your code
# voxel_head = VoxelHead(cfg)
# x = torch.randn(1, 256, 56, 56)  # Example input
# output = voxel_head(x)
# print(output.shape)
