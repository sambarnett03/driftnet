import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
    def __init__(self, n_channels=2, n_classes=2, base_features=16):
        """
        A U-Net designed to extract features at the low resolution (605x1072)
        and explicitly upsample by exactly 5x at the very end using PixelShuffle.
        Uses Global Residual Learning to predict sub-grid scale turbulence.
        """
        super().__init__()
        self.n_channels = n_channels

        # --- Encoder (Operates at 605 x 1072) ---
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16)

        # --- Decoder (Operates at 605 x 1072) ---
        self.up1 = Up(base_features * 16 + base_features * 8, base_features * 8)
        self.up2 = Up(base_features * 8 + base_features * 4, base_features * 4)
        self.up3 = Up(base_features * 4 + base_features * 2, base_features * 2)
        self.up4 = Up(base_features * 2 + base_features, base_features)

        # --- Super Resolution Head ---
        self.upscale_factor = 5
        mid_features = n_classes * (self.upscale_factor**2)  # 2 * 25 = 50 channels

        self.pre_shuffle = nn.Conv2d(base_features, mid_features, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

    def forward(self, x):
        """
        Expects Input x: [B, 2, 605, 1072]
        Returns Output:  [B, 2, 3025, 5360]
        """
        # 1. Base physics: Bilinearly upsample the low-res input
        # align_corners=True ensures the geographic grid coordinates don't shift
        base_field = F.interpolate(
            x,
            scale_factor=float(self.upscale_factor),
            mode="bilinear",
            align_corners=True
        )

        # 2. ML turbulence: Extract features to generate the high-frequency residual
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up = self.up1(x5, x4)
        x_up = self.up2(x_up, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up4(x_up, x1)

        x_feat = self.pre_shuffle(x_up)
        residual = self.pixel_shuffle(x_feat)

        # 3. Global Residual Connection: Add the ML turbulence to the smooth base field
        out = base_field + residual

        return out