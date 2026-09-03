import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class DPFEM(nn.Module):

    def __init__(self, in_planes, reduction=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        hidden_dim = max(in_planes // reduction, 8)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.max_pool2 = nn.AdaptiveMaxPool2d(2)
        self.proj_1x1 = nn.Sequential(nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.proj_2x2 = nn.Sequential(nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(hidden_dim * 4, in_planes, kernel_size=1, bias=False), nn.BatchNorm2d(in_planes), nn.ReLU(inplace=True), nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(in_planes), nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        avg1 = self.proj_1x1(self.avg_pool1(x))
        avg2 = self.proj_2x2(self.avg_pool2(x))
        max1 = self.proj_1x1(self.max_pool1(x))
        max2 = self.proj_2x2(self.max_pool2(x))
        avg1_up = F.interpolate(avg1, size=(h, w), mode='bilinear', align_corners=False)
        avg2_up = F.interpolate(avg2, size=(h, w), mode='bilinear', align_corners=False)
        max1_up = F.interpolate(max1, size=(h, w), mode='bilinear', align_corners=False)
        max2_up = F.interpolate(max2, size=(h, w), mode='bilinear', align_corners=False)
        context = torch.cat([avg1_up, avg2_up, max1_up, max2_up], dim=1)
        context = self.fuse(context)
        out = x + self.alpha * context
        return out

class DMRFBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, dilations=[1, 2, 3], use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.num_branches = len(dilations)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dwconvs = nn.ModuleList([nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=d, dilation=d, groups=middle_channels, bias=False), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True)) for d in dilations])
        combined_channels = middle_channels * self.num_branches
        self.fusion_conv = nn.Sequential(nn.Conv2d(combined_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        dw_outs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(out) + out
            dw_outs.append(dw_out)
        dout = torch.cat(dw_outs, dim=1)
        dout = rearrange(dout, 'b (g c) h w -> b (c g) h w', g=self.num_branches)
        out = self.fusion_conv(dout)
        if self.use_residual:
            out = out + self.shortcut(identity)
        out = self.relu(out)
        return out

class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class MRFDPUnet2Plus(nn.Module):

    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ctx_04 = DPFEM(nb_filter[0], alpha=0.1)
        self.conv0_0 = DMRFBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = DMRFBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = DMRFBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = DMRFBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = DMRFBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.ctx_04(x0_4)
        return self.final(x0_4)
