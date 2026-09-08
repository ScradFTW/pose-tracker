import torch.nn as nn

NUM_JOINTS = 17


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def make_layer(cin, cout, n_blocks, stride):
    layers = [BasicBlock(cin, cout, stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock(cout, cout, 1))
    return nn.Sequential(*layers)


class ResNetPoseNet(nn.Module):
    """ResNet18-shaped backbone (random init, no pretraining) + 3-stage
    deconv head, in the spirit of Xiao et al. 2018 "Simple Baselines for
    Human Pose Estimation". Every weight trained from scratch on COCO.

    256x256 RGB in -> 17x64x64 joint heatmaps out.
    """

    def __init__(self, num_joints=NUM_JOINTS, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),   # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),                    # 128 -> 64
        )
        self.layer1 = make_layer(64, 64, blocks[0], 1)     # 64
        self.layer2 = make_layer(64, 128, blocks[1], 2)    # 32
        self.layer3 = make_layer(128, 256, blocks[2], 2)   # 16
        self.layer4 = make_layer(256, 512, blocks[3], 2)   # 8

        def deconv(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.head = nn.Sequential(
            deconv(512, 256),  # 8 -> 16
            deconv(256, 256),  # 16 -> 32
            deconv(256, 256),  # 32 -> 64
        )
        self.final = nn.Conv2d(256, num_joints, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return self.final(x)
