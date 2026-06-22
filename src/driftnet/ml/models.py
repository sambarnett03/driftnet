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


class Strict2xOceanUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=2, base_features=16):
        """
        A U-Net designed to extract features at the low resolution (605x1072)
        and explicitly upsample by exactly 2x (1210x2144) at the very end.
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
        # Instead of a 1x1 conv, we use a ConvTranspose2d with stride=2.
        # This explicitly doubles the output dimensions: 605x1072 -> 1210x2144.
        self.outc = nn.ConvTranspose2d(
            in_channels=base_features, out_channels=n_classes, kernel_size=2, stride=2
        )

    def forward(self, x):
        """
        Expects Input x: [B, 2, 605, 1072]
        Returns Output:  [B, 2, 1210, 2144]
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

        # Final 2x Upscale
        out = self.outc(x)

        return out
