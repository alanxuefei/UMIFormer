import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.linear = nn.Linear(patch_size**3, embed_dim)

    def forward(self, x):
        batch_size, _, d, h, w = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                    .unfold(3, self.patch_size, self.patch_size) \
                    .unfold(4, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size**3)
        tokens = self.linear(patches)
        return tokens

class Transformer3D(nn.Module):
    def __init__(self, voxel_dim, n_heads, n_layers, dim_feedforward):
        super(Transformer3D, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=voxel_dim, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.voxel_dim = voxel_dim
        self.patch_size = int(voxel_dim ** (1/3))

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        x = x.permute(1, 0, 2)  # (num_patches, batch_size, voxel_dim)
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)  # (batch_size, num_patches, voxel_dim)
        return output

class VoxelReconstructor(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(VoxelReconstructor, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.linear = nn.Linear(embed_dim, patch_size**3)

    def forward(self, tokens):
        batch_size, num_patches, _ = tokens.size()
        patches = self.linear(tokens)
        patches = patches.view(batch_size, num_patches, self.patch_size, self.patch_size, self.patch_size)
        
        d_patches = h_patches = w_patches = int(num_patches ** (1/3))
        voxel_grid = patches.view(batch_size, 
                                  d_patches, h_patches, w_patches, 
                                  self.patch_size, self.patch_size, self.patch_size)
        voxel_grid = voxel_grid.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        voxel_grid = voxel_grid.view(batch_size, 
                                     d_patches * self.patch_size, 
                                     h_patches * self.patch_size, 
                                     w_patches * self.patch_size)
        return voxel_grid

# Hyperparameters
embed_dim = 512
n_heads = 8
n_layers = 6
dim_feedforward = 2048
patch_size = 8

# Initialize components
patch_embedder = PatchEmbedder(patch_size, embed_dim)
transformer_3d = Transformer3D(embed_dim, n_heads, n_layers, dim_feedforward)
voxel_reconstructor = VoxelReconstructor(embed_dim, patch_size)

# Dummy input
batch_size = 16
voxel_grid = torch.randn(batch_size, 1, 32, 32, 32)  # Example voxel grid

# Forward pass
tokens = patch_embedder(voxel_grid)
transformed_tokens = transformer_3d(tokens)
new_voxel_grid = voxel_reconstructor(transformed_tokens)

# Print output shape
print(new_voxel_grid.shape)  # Expected shape: (batch_size, 32, 32, 32)
