import torch
import torch.nn as nn

from models.voxel_net.backbone import build_backbone
from models.voxel_net.voxel_head import VoxelHead

class VoxNet(nn.Module):
    def __init__(self, cfg=None):
        super(VoxNet, self).__init__()

        self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE)
        cfg.MODEL.VOXEL_HEAD.COMPUTED_INPUT_CHANNELS = feat_dims[-1]
        self.voxel_head = VoxelHead(cfg)

    def forward(self, imgs):
        img_feats = self.backbone(imgs)
        voxel_scores = self.voxel_head(img_feats[-1])
        return voxel_scores

# Example usage
if __name__ == '__main__':
    # Using default configuration
    model = VoxNet()
    # Dummy input
    imgs = torch.randn(1, 3, 224, 224)
    output = model.forward(imgs)
    print(output.shape)
