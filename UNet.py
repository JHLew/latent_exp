import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import interpolate, pad
import torch.nn.functional as F


# normalize (0, 1) to (-1, 1)
def preprocess(t):
    return t * 2 - 1


# denormalize from (-1, 1) to (0, 1)
def postprocess(t):
    return torch.clamp((t + 1) / 2, min=0, max=1)


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize, avgpool=True, latent_dim=None, spatial_size=None):
        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        if avgpool:
            self.pool = nn.AvgPool2d(2)
        else:
            self.pool = nn.Conv2d(inChannels, inChannels, kernel_size=2, stride=2, padding=1)

        if latent_dim is not None and spatial_size is not None:
            self.latent_extractor = nn.Conv2d(inChannels, latent_dim, spatial_size)
            self.extract_latent = True
        else:
            self.extract_latent = False

    def forward(self, x):
        # Average pooling with kernel size 2 (2 x 2).
        # x = F.avg_pool2d(x, 2)
        if self.extract_latent:
            latent = self.latent_extractor(x)

        x = self.pool(x)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        if self.extract_latent:
            return x, latent
        else:
            return x


class up(nn.Module):
    def __init__(self, inChannels, outChannels, skip=True, out_skip=False, modulation=False, latent_dim=None):
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        if skip:
            self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(outChannels, outChannels, 3, stride=1, padding=1)
        self.skip = skip
        self.out_skip = out_skip
        self.use_modulation = modulation

        if modulation:
            if latent_dim is None:
                raise ValueError('latent dimension not specified!')
            self.modulation_1 = Modulation(latent_dim, outChannels)
            self.modulation_2 = Modulation(latent_dim, outChannels)

    def forward(self, x, skpCn=None, latent=None):
        # Convolution + Leaky ReLU
        if self.out_skip:
            res = x
        if self.use_modulation:
            x = self.modulation_1(x, latent)
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if self.use_modulation:
            x = self.modulation_2(x, latent)
        if self.skip:
            x = torch.cat((x, skpCn), 1)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        if self.out_skip:
            x = res + x
        return x


class UNet(nn.Module):
    def __init__(self, inChannels, outChannels, skip=True):
        super(UNet, self).__init__()
        self.skip = skip
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)  # 8
        self.down6 = down(512, 512, 3)  # 4
        self.down7 = down(512, 512, 3)  # 2
        self.down8 = down(512, 512, 3)  # 1
        self.up_2 = up(512, 512, skip)
        self.up_1 = up(512, 512, skip)
        self.up0 = up(512, 512, skip)
        self.up1 = up(512, 512, skip)
        self.up2 = up(512, 256, skip)
        self.up3 = up(256, 128, skip)
        self.up4 = up(128, 64, skip)
        self.up5 = up(64, 32, skip)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.down6(s6)
        s8 = self.down7(s7)
        x = self.down8(s8)

        x = self.up_2(x, s8)
        x = self.up_1(x, s7)
        x = self.up0(x, s6)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = self.conv3(x)
        return postprocess(x)


class UpNet(nn.Module):
    def __init__(self, encoder, latent_dim, outChannels):
        super(UpNet, self).__init__()
        # self.encoder = resnet18(num_classes=latent_dim)
        self.encoder = encoder
        # Initialize neural network blocks.
        self.up0 = up(latent_dim, 512, False)  # 1 -> 2
        self.up1 = up(512, 512, False)  # 2 -> 4
        self.up2 = up(512, 512, False)  # 4 -> 8ffff
        self.up3 = up(512, 256, False)  # 8 -> 16
        self.up4 = up(256, 256, False)  # 16 -> 32
        self.up5 = up(256, 256, False)  # 32 -> 64
        self.up6 = up(256, 256, False)  # 64 -> 128
        self.up7 = up(256, 256, False)  # 128 -> 256
        self.conv3 = nn.Conv2d(256, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        latent = self.encoder(x)
        # latent, ff = self.encoder(x)
        x = latent.unsqueeze(2).unsqueeze(2)
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.conv3(x)
        return postprocess(x)


class Modulation(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(Modulation, self).__init__()
        self.shared_conv = nn.Conv2d(latent_dim, out_dim, 1, 1)
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(out_dim)

        self.gamma_predictor = nn.Conv2d(out_dim, out_dim, 1, 1)
        self.beta_predictor = nn.Conv2d(out_dim, out_dim, 1, 1)

    def forward(self, x, latent):
        s = self.shared_conv(latent)
        s = self.act(s)
        gamma = self.gamma_predictor(s)
        beta = self.beta_predictor(s)

        # x = x * gamma + beta
        x = self.norm(x) * gamma + beta
        return x


class modulated_UPNet(nn.Module):
    def __init__(self, encoder, latent_dim, outChannels, outskip=True):
        super(modulated_UPNet, self).__init__()
        self.encoder = encoder
        # Initialize neural network blocks.
        self.up0 = up(latent_dim, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 1 -> 2
        self.up1 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 2 -> 4
        self.up2 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 4 -> 8
        self.up3 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 8 -> 16
        self.up4 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 16 -> 32
        self.up5 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 32 -> 64
        self.up6 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 64 -> 128
        self.up7 = up(512, 512, False, out_skip=outskip, modulation=True, latent_dim=latent_dim)  # 128 -> 256
        self.conv3 = nn.Conv2d(512, outChannels, 3, stride=1, padding=1)


    def forward(self, x):
        latent = self.encoder(x)
        # latent, ff = self.encoder(x)
        latent = latent.unsqueeze(2).unsqueeze(2)
        x = latent
        x = self.up0(x, None, latent=latent)
        x = self.up1(x, None, latent=latent)
        x = self.up2(x, None, latent=latent)
        x = self.up3(x, None, latent=latent)
        x = self.up4(x, None, latent=latent)
        x = self.up5(x, None, latent=latent)
        x = self.up6(x, None, latent=latent)
        x = self.up7(x, None, latent=latent)
        x = self.conv3(x)

        return postprocess(x)


class Hier_Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Hier_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.down1 = down(32, 64, 5, False, latent_dim, spatial_size=256)    # 128
        self.down2 = down(64, 128, 3, False, latent_dim, spatial_size=128)   # 64
        self.down3 = down(128, 256, 3, False, latent_dim, spatial_size=64)  # 32
        self.down4 = down(256, 512, 3, False, latent_dim, spatial_size=32)  # 16
        self.down5 = down(512, 512, 3, False, latent_dim, spatial_size=16)  # 8
        self.down6 = down(512, 512, 3, False, latent_dim, spatial_size=8)  # 4
        self.down7 = down(512, 512, 3, False, latent_dim, spatial_size=4)  # 2
        self.down8 = down(512, 512, 3, False, latent_dim, spatial_size=2)  # 1

        self.fc1 = nn.Linear(latent_dim * 9, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s0 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s1, l1 = self.down1(s0)
        s2, l2 = self.down2(s1)
        s3, l3 = self.down3(s2)
        s4, l4 = self.down4(s3)
        s5, l5 = self.down5(s4)
        s6, l6 = self.down6(s5)
        s7, l7 = self.down7(s6)
        s8, l8 = self.down8(s7)

        latents = torch.cat([s8, l1, l2, l3, l4, l5, l6, l7, l8], dim=1)[:, :, 0, 0]

        latents = F.leaky_relu(self.fc1(latents), negative_slope=0.1)
        full_latent = self.fc2(latents)

        return full_latent

