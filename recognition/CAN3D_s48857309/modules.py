import torch
import torch.nn as nn

class CAN3D(nn.Module):
    """Progressive dilation network

    Implementation is based on descriptions in:
        W. Dai, et al. “CAN3D: Fast 3D medical image segmentation via compact 
        context aggregation”, 2021, https://doi.org/10.1016/j.media.2022.102562.
    """
    def __init__(self, depth=256, height=256, width=128, outputs = 6):
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width

        self.inlayer = self.create_layer(1, 3, 1, )
        self.downlayer1 = self.create_layer(3, 6, 1, 2)
        self.downlayer2 = self.create_layer(6, 36, 1, 2)
        self.cammodule = nn.Sequential(
            self.create_layer(36, 36, 2, 1, 3, 'identity'),
            self.create_layer(36, 36, 4, 1, 3, 'identity'),
            self.create_layer(36, 36, 8, 1, 3, 'identity'),
            self.create_layer(36, 48, 1, 1, 3),
        )
        self.uplayer1 = self.deconv_2xlayer(48, 48)
        self.uplayer2 = self.deconv_2xlayer(48, 48)

        self.skip = self.create_layer(3, 3, 1, 1, 1)
        self.segmentationlayer = nn.Conv3d(51, outputs, 1)

    def create_layer(self, in_channels, out_channels, dilation, stride=1, kernel_size=3, initialiser='xavier'):
        padding_required = (dilation * (kernel_size - 1) - stride + 1) // 2
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding_required, dilation=dilation, bias=False)
        if initialiser == 'identity' and in_channels == out_channels: # Note: default is xavier, so no action required
            conv.weight.fill_(0.0)
            for i in range(in_channels):
                conv.weight.data[i, i, kernel_size//2, kernel_size//2, kernel_size//2] = 1.0
        return nn.Sequential(
            conv,
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1)
        )

    def deconv_2xlayer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.inlayer(x.unsqueeze(-1))
        u = self.skip(x)
        x = self.downlayer1(x)
        x = self.downlayer2(x)
        x = self.cammodule(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = torch.cat((x, u), dim=1)
        x = self.segmentationlayer(x)
        return x

