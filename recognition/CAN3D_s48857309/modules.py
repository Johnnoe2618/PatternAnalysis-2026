from gettext import install

from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.instancenorm = nn.InstanceNorm3d(channels)
        self.weights = nn.Parameter(torch.rand(2))

    def forward(self, x):
        return self.weights[0] * x + self.weights[1] * self.instancenorm(x)

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

        self.inlayer = self.create_layer(1, 3, 1)
        self.downlayer1 = self.create_layer(3, 6, 1, 2)
        self.downlayer2 = self.create_layer(6, 36, 1, 2)
        self.cammodule = nn.Sequential(
            self.create_layer(36, 36, 2, 1, 3, 'identity', padding=2),
            self.create_layer(36, 36, 4, 1, 3, 'identity', padding=4),
            self.create_layer(36, 36, 8, 1, 3, 'identity', padding=8),
            self.create_layer(36, 48, 1, 1, 3, 'identity'),
        )
        self.uplayer1 = self.deconv_2xlayer(48, 36)
        self.uplayer2 = self.deconv_2xlayer(36, 16)

        self.skip = self.create_layer(3, 3, 1, 1, 1, padding=0)
        self.segmentationlayer = nn.Conv3d(16 + 3, outputs, 1)

    def create_layer(self, in_channels, out_channels, dilation, stride=1, kernel_size=3, initialiser='xavier', padding=1):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        if initialiser == 'identity' and in_channels == out_channels: # Note: default is xavier, so no action required
            with torch.no_grad():
                conv.weight.fill_(0.0)
                for i in range(in_channels):
                    conv.weight.data[i, i, kernel_size//2, kernel_size//2, kernel_size//2] = 1.0
        return nn.Sequential(
            conv,
            AdaptiveInstanceNorm(out_channels),
            nn.LeakyReLU(0.1)
        )

    def deconv_2xlayer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2, padding=0),
            AdaptiveInstanceNorm(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.inlayer(x.view(x.shape[0], 1, self.depth, self.height, self.width))
        u = self.skip(x)
        x = self.downlayer1(x)
        x = self.downlayer2(x)
        x = self.cammodule(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = torch.cat((x, u), dim=1)
        x = self.segmentationlayer(x)
        return nn.functional.softmax(x, dim=1)


class CAN2D:
    """Progressive dilation network

    Adapted architecture from Fisher Yu, by adding AdaIn as Wai Dai suggests, and leaky relu.
    """
    def __init__(self, height=256, width=128, outputs=6):
        self.height = height
        self.width = width

        self.cammodule = nn.Sequential(
            self.create_layer(1, 36, 1),
            self.create_layer(36, 36, 1),
            self.create_layer(36, 36, 2, 1, 3, 'identity', padding=2),
            self.create_layer(36, 36, 4, 1, 3, 'identity', padding=4),
            self.create_layer(36, 36, 8, 1, 3, 'identity', padding=8),
            self.create_layer(36, 36, 16, 1, 3, 'identity', padding=16),
            self.create_layer(36, 36, 16, 1),
            nn.Conv2d(36, outputs, 1)
        )

        def create_layer(self, in_channels, out_channels, dilation, stride=1, kernel_size=3, initialiser='xavier', padding=1):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
            if initialiser == 'identity' and in_channels == out_channels: # Note: default is xavier, so no action required
                with torch.no_grad():
                    conv.weight.fill_(0.0)
                    for i in range(in_channels):
                        conv.weight.data[i, i, kernel_size//2, kernel_size//2] = 1.0
            return nn.Sequential(
                conv,
                AdaptiveInstanceNorm(out_channels),
                nn.LeakyReLU(0.1)
            )


def compute_dsfl(y_probs, y_true, weights=None, alphas=None, smooth = 1e-6, gamma=2):
    """Dice squared focal loss, as described in the (W. Dai, 2021) paper."""
    if weights is None or alphas is None:
        raise ValueError("Weights and alphas must be provided as tensors on the device.")

    voxel_count = y_probs.shape[2] * y_probs.shape[3] * y_probs.shape[4]
    y_probs = y_probs.reshape(y_probs.shape[0], y_probs.shape[1], -1)
    y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1)

    difference = y_probs - y_true
    phi = torch.sum(difference * difference, dim=-1)
    denominator = torch.sum(y_probs * y_probs + y_true * y_true  + smooth, dim=-1)

    dsl_per_class = phi / denominator + phi / voxel_count
    dsl_per_class = torch.mean(dsl_per_class, dim=0)
    fl_per_class = -torch.sum(torch.pow(1.0 - y_probs, gamma) * torch.log(y_probs + smooth), dim=-1) / voxel_count
    fl_per_class = torch.mean(fl_per_class, dim=0)
    dsfl = torch.dot(dsl_per_class, weights) + torch.dot(fl_per_class, alphas)
    return dsfl

def compute_loss(predictions, labels, weights, alphas):
    labels = nn.functional.one_hot(labels, predictions.shape[1]).permute(0, 4, 1, 2, 3)
    return compute_dsfl(predictions, labels, weights, alphas)

def calculate_descrete_dsc(prediction_logits, label, smooth = 1e-6):
    batch_size, num_classes = prediction_logits.shape[0], prediction_logits.shape[1]
    y_preds = torch.argmax(prediction_logits, dim=1)
    y_preds = nn.functional.one_hot(y_preds, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)
    y_true = nn.functional.one_hot(label, num_classes).permute(0, 4, 1, 2, 3).view(batch_size, num_classes, -1)

    intersection = torch.sum(y_preds * y_true, dim=-1)
    cardinality = torch.sum(y_preds + y_true, dim=-1)

    dice_per_class = (2. * intersection + smooth) / (cardinality + smooth)
    sum_dice_per_class = torch.sum(dice_per_class, dim=0)
    return sum_dice_per_class, batch_size
