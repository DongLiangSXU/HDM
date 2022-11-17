import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable, Function


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25


        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel,stride,padding, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        # print(px.shape)
        # print(pk.shape)
        po = F.conv2d(px, pk,stride=stride,padding=padding, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel,stride,padding, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    # print(px.shape)
    # print(pk.shape)
    po = F.conv2d(px, pk, **kwargs, groups=batch,stride=stride,padding=padding)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    # print(po.shape)
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, stride,padding, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        # print(kernel.shape)
        out = F.conv2d(x, kernel, **kwargs,stride=stride,padding=padding, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel,stride,padding, **kwargs):
        # print('------------------------------')
        if self.training:
            # print('------------------------------training')
            if self.use_slow:
                return xcorr_slow(x, kernel,stride,padding, kwargs)
            else:
                return xcorr_fast(x, kernel, stride,padding,kwargs)
        else:
            # print('-------------------------------no_train')
            return Corr.apply(x, kernel,stride,padding,1, kwargs)


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8,stride=1,padding=1, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num
        self.stride = stride
        self.padding = padding

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, stride=stride,padding=padding,**kwargs)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input):
        kernel = self.conv_kernel(input)
        # print(kernel.shape)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        # print(kernel.shape)
        output = self.corr(input, kernel,self.stride,self.padding, **self.kwargs)  # B x (r*out) x W x H
        # print(output.shape)
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        # print(output.shape)
        guide_feature = self.conv_guide(input)
        # print(guide_feature.shape)
        # print(output.shape)

        output = self.asign_index(output, guide_feature)
        # print(output.shape)

        return output



class DRConv2d_phy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8,stride=1,padding=1, **kwargs):
        super(DRConv2d_phy, self).__init__()
        self.region_num = region_num
        self.stride = stride
        self.padding = padding

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        # self.asign_index = asign_index.apply

    def forward(self, input,phy_mask):


        kernel = self.conv_kernel(input)
        # print(kernel.shape)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        # print(kernel.shape)
        output = self.corr(input, kernel,self.stride,self.padding, **self.kwargs)  # B x (r*out) x W x H
        # print(output.shape)
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        # print(output.shape)
        output = torch.sum(output*phy_mask,dim=1)
        # print(output.shape)
        return output
