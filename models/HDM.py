import sys
sys.path.append("..")
import torch.nn as nn
import torch
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        # self.ealayer=External_attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        # res=self.ealayer(res)
        res += x
        return res


class Block_onlyca(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_onlyca, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        # self.palayer = PALayer(dim)
        # self.ealayer=External_attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        # res = self.palayer(res)
        # res=self.ealayer(res)
        res += x
        return res

class Block_onlypa(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_onlypa, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        # self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        # self.ealayer=External_attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        # res = self.calayer(res)
        res = self.palayer(res)
        # res=self.ealayer(res)
        res += x
        return res



class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class Group_onlyca(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group_onlyca, self).__init__()
        modules = [Block_onlyca(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class Group_onlypa(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group_onlypa, self).__init__()
        modules = [Block_onlypa(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

from dcn_v2 import DCN

class DCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCNBlock, self).__init__()
        self.dcn = DCN(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1).cuda()

    def forward(self, x):
        return self.dcn(x)


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)

        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

from drconv import DRConv2d,DRConv2d_phy
import torchvision.utils as vutils
class DR_Net_phy(torch.nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(DR_Net_phy, self).__init__()

        self.gps = gps
        self.dim = 64
        kernel_size = 3
        blocks = 9

        # DRConv2d(in_channels, out_channels, kernel_size=3, region_num=5,stride=2,padding=1)

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, self.dim, 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.Conv2d(self.dim, self.dim, 3, 2, 1),
            nn.ReLU(inplace=True)

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.Conv2d(self.dim, self.dim, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.FAB = nn.Sequential(
            Group(conv, self.dim, kernel_size, blocks=blocks),
            Group(conv, self.dim, kernel_size, blocks=blocks),
            Group(conv, self.dim, kernel_size, blocks=blocks),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(

            nn.Conv2d(self.dim, self.dim, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(self.dim, self.dim, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(self.dim, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.drconv = DRConv2d(self.dim, self.dim, kernel_size=3, region_num=5, stride=1, padding=1)
        self.drconv_phy = DRConv2d_phy(self.dim, self.dim, kernel_size=3, region_num=5, stride=1, padding=1)
        self.dcn_block = DCNBlock(self.dim, self.dim)


        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        self.mix3 = Mix(m=-0.5)
        self.mix4 = Mix(m=-0.5)

    def forward(self, x,mask):
        x = self.conv0(x)
        x = self.conv1(x)

        res1 = x
        x = self.conv2(x)
        res2 = x
        x = self.FAB(x)
        # x = self.FAB(x)

        x_dcn1 = self.dcn_block(x)

        temp1 = x_dcn1


        x_dcn1_dcp = self.drconv_phy(temp1,mask)
        x_dcn1 = x_dcn1_dcp
        x_dcn2 = self.dcn_block(x_dcn1)
        temp2 = x_dcn2

        x_dcn2_phy = self.drconv_phy(temp2, mask)
        x_dcn2 = x_dcn2_phy
        x_out_mix = self.mix1(res2, x_dcn2)

        x = self.deconv1(x_out_mix)
        x_up1_mix = self.mix2(res1, x)

        x = self.deconv2(x_up1_mix)

        x = self.output(x)
        return x







