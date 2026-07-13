import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => ReLU) * 2 (Batch Norm removed)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv, with padding to handle odd numbers like 605"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Compensate for odd dimensions (e.g., 605 -> 302 -> 604 mismatch)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Strict5xOceanUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=2, base_features=16, bottleneck_dim=256):
        """
        A U-Net designed to extract features at the low resolution (605x1072)
        and explicitly upsample by exactly 5x at the very end using PixelShuffle.
        Includes a configurable bottleneck dimension for sensitivity analysis.
        """
        super().__init__()
        self.n_channels = n_channels

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        # Modified to output the explicit bottleneck_dim
        self.down4 = Down(base_features * 8, bottleneck_dim)

        # --- Decoder ---
        # up1 modified to accept the custom bottleneck_dim + the skip connection
        self.up1 = Up(bottleneck_dim + base_features * 8, base_features * 8)
        self.up2 = Up(base_features * 8 + base_features * 4, base_features * 4)
        self.up3 = Up(base_features * 4 + base_features * 2, base_features * 2)
        self.up4 = Up(base_features * 2 + base_features, base_features)

        # --- Super Resolution Head ---
        upscale_factor = 5
        mid_features = n_classes * (upscale_factor**2)  # 2 * 25 = 50 channels

        self.pre_shuffle = nn.Conv2d(base_features, mid_features, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        """
        Expects Input x: [B, 2, 605, 1072]
        Returns Output:  [B, 2, 3025, 5360]
        """
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final 5x Upscale
        x = self.pre_shuffle(x)
        out = self.pixel_shuffle(x)

        return out