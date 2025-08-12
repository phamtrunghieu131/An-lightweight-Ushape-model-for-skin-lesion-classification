import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable


class ChannelPermute(nn.Module):
    """
    Channel permutation implemented as channel shuffle (grouped transpose).
    Splits channels into `groups`, reshapes (groups, channels_per_group, ...),
    transposes the first two dims and flattens back.
    """
    def __init__(self, groups: int = 2):
        super().__init__()
        assert groups >= 1, "groups must be >=1"
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        if self.groups == 1 or C % self.groups != 0:
            # fallback: no shuffle if cannot split evenly
            return x
        channels_per_group = C // self.groups
        # reshape -> (B, groups, channels_per_group, H, W)
        x = x.view(B, self.groups, channels_per_group, H, W)
        # transpose group and channel dims -> (B, channels_per_group, groups, H, W)
        x = x.transpose(1, 2).contiguous()
        # flatten back -> (B, C, H, W)
        x = x.view(B, C, H, W)
        return x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (channel + spatial) 
    """
    def __init__(self, channels: int, spatial_kernel: int = 1):
        super().__init__()
        self.channels = channels
        mid = int(channels / 2)

        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )

        # spatial attention conv (1x1 theo hình)
        padding = (spatial_kernel - 1) // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Channel Attention -----
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        mlp_out = self.mlp(avg_pool) + self.mlp(max_pool)  # shared MLP
        channel_att = self.sigmoid(mlp_out)
        x_channel = x * channel_att  # broadcast multiply

        # ----- Spatial Attention -----
        max_map, _ = torch.max(x_channel, dim=1, keepdim=True)
        avg_map = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([max_map, avg_map], dim=1)  # concat theo kênh -> (B,2,H,W)
        spatial_map = self.spatial_conv(spatial_input)
        spatial_att = self.sigmoid(spatial_map)
        out = x_channel * spatial_att
        return out

class DCPBAM(nn.Module):
    """
    Dual Convolutional Permuted Block Attention Module (DC-PBAM)
    Based on the provided diagram:
      - Two 1x1 conv branches
      - Top branch: Conv1x1 -> CP
      - Bottom branch: Conv1x1 -> CBAM -> CP
      - Add both CP outputs -> CBAM -> output
    """
    def __init__(self, in_channels: int, mid_channels: int = None, cp_groups: int = 2, spatial_kernel: int = 1):
        """
        Args:
            in_channels: input channel C
            mid_channels: number of channels used by the 1x1 convs (if None, use in_channels)
            cp_groups: groups used in channel permutation (channel shuffle)
            spatial_kernel: kernel size for CBAM spatial conv (diagram shows 1)
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv_top = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.conv_bottom = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        # CBAM applied on bottom branch before CP and also after addition
        self.cbam_before = CBAM(mid_channels, spatial_kernel=spatial_kernel)
        self.cp1 = ChannelPermute(groups=cp_groups)  # for top branch
        self.cp2 = ChannelPermute(groups=cp_groups)  # for bottom branch after cbam
        self.cbam_after = CBAM(mid_channels, spatial_kernel=spatial_kernel)

        # optional projection if mid_channels != in_channels (to match output C)
        if mid_channels != in_channels:
            self.out_proj = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.out_proj = None

        # optional normalization/activation
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # top branch
        t = self.conv_top(x)          # (B, mid, H, W)
        t = self.cp1(t)               # channel permutation

        # bottom branch
        b = self.conv_bottom(x)
        b = self.cbam_before(b)       # CBAM on bottom branch
        b = self.cp2(b)               # channel permutation

        # fusion
        s = t + b                     # addition (elementwise)
        s = self.cbam_after(s)        # final CBAM
        if self.out_proj is not None:
            s = self.out_proj(s)
        s = self.relu(s)
        return s

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params


# if __name__ == "__main__":
    # model = DCPBAM(in_channels=128, mid_channels=128, cp_groups=4, spatial_kernel=1)
    # x = torch.randn(2, 16, 256, 256)  # batch 2, C=64, H=W=32
    # y = model(x)
    # print(y.shape)  # (2,64,32,32)

    # count_parameters(model)