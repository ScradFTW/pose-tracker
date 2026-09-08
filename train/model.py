import torch.nn as nn

NUM_JOINTS = 14


def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PoseNet(nn.Module):
    """Small from-scratch heatmap-regression CNN.

    256x256 RGB in -> 14x64x64 joint heatmaps out. No pretrained backbone --
    every weight is trained from random init on LSP.
    """

    def __init__(self, num_joints=NUM_JOINTS):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu(3, 32, 2),   # 256 -> 128
            conv_bn_relu(32, 64, 2),  # 128 -> 64
        )
        self.stage1 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 128, 2),  # 64 -> 32
        )
        self.stage2 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_joints, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.up(x)
        return self.head(x)
