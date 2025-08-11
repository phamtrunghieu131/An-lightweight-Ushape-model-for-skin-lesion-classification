import torch
import torch.nn as nn
import torch.nn.functional as F

from prettytable import PrettyTable

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module
    GAP -> FC -> ReLU -> FC -> Sigmoid -> Scale input
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        reduced = int(self.channels / 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)           # (B, C)
        y = self.fc(y).view(b, c, 1, 1)           # (B, C, 1, 1)
        return x * y.expand_as(x)                 # scale input


class MFCAM(nn.Module):
    """
    Multi-scale Feature Channel Attention Module (4-branch)
    Branches:
      - branch1: 1x1 conv -> BN + ReLU -> SE
      - branch2: 3x3 conv -> BN + ReLU -> SE
      - branch3: 3x3 conv -> BN + ReLU -> 1x1 conv -> BN + ReLU -> SE
      - branch4: identity (no conv) -> SE
    Final: sum(branch1, branch2, branch3, branch4)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels

        # Branch 1: 1x1
        self.b1_conv = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.b1_bn = nn.BatchNorm2d(C)
        self.b1_relu = nn.ReLU(inplace=True)
        self.b1_se = SEModule(C)

        # Branch 2: 3x3
        self.b2_conv = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.b2_bn = nn.BatchNorm2d(C)
        self.b2_relu = nn.ReLU(inplace=True)
        self.b2_se = SEModule(C)

        # Branch 3: 3x3 -> 1x1
        self.b3_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.b3_bn1 = nn.BatchNorm2d(C)
        self.b3_relu1 = nn.ReLU(inplace=True)
        self.b3_conv2 = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.b3_bn2 = nn.BatchNorm2d(C)
        self.b3_relu2 = nn.ReLU(inplace=True)
        self.b3_se = SEModule(C)

        # Branch 4: identity -> SE (no conv)
        self.b4_se = SEModule(C)
        self.b4_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        b1 = self.b1_conv(x)
        b1 = self.b1_bn(b1)
        b1 = self.b1_relu(b1)
        b1 = self.b1_se(b1)

        # Branch 2
        b2 = self.b2_conv(x)
        b2 = self.b2_bn(b2)
        b2 = self.b2_relu(b2)
        b2 = self.b2_se(b2)

        # Branch 3
        b3 = self.b3_conv1(x)
        b3 = self.b3_bn1(b3)
        b3 = self.b3_relu1(b3)
        b3 = self.b3_conv2(b3)
        b3 = self.b3_bn2(b3)
        b3 = self.b3_relu2(b3)
        b3 = self.b3_se(b3)

        # Branch 4 (bypass -> SE)
        b4 = self.b4_relu(x)
        b4 = self.b4_se(b4)

        # Sum branches
        out = b1 + b2 + b3 + b4

        return out

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == "__main__":
    model = MFCAM(in_channels=256)
    # x = torch.randn(2, 64, 32, 32)
    # y = model(x)
    # print(y.min())  # (2, 64, 32, 32)
    count_parameters(model)