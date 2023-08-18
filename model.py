#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName   : models.py
# @Author     : KK
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.nn import init

    
# 策略网络，输出两个mask
class Generator(nn.Module):
    # 参数不直接体现state_dim和action_dim,但总之默认都是512*512
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn):
        super(Generator, self).__init__()

        ###### shared encoder
        self.enc1 = ConvBlock(in_channels=3,          out_channels=conv_dim* 1, kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*3 --> 256*256*32
        self.enc2 = ConvBlock(in_channels=conv_dim*1, out_channels=conv_dim* 2, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*32 --> 128*128*64
        self.enc3 = ConvBlock(in_channels=conv_dim*2, out_channels=conv_dim* 4, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*64 --> 64*64*128
        self.enc4 = ConvBlock(in_channels=conv_dim*4, out_channels=conv_dim* 8, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*128 --> 32*32*256
        self.enc5 = ConvBlock(in_channels=conv_dim*8, out_channels=conv_dim*16, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*256 --> 16*16*512

        ###### decoder 1 of Policy Network
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim*16, conv_dim*8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 8, conv_dim*4, 1, 1, 0, 1, True, use_sn))
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 4, conv_dim*2, 1, 1, 0, 1, True, use_sn))
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 2, conv_dim*1, 1, 1, 0, 1, True, use_sn))

        self.dec1 = ConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*512 --> 32*32*256
        self.dec2 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*256 --> 64*64*128
        self.dec3 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*128 --> 128*128*64
        self.dec4 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*64 --> 256*256*32
        self.dec5 = nn.Sequential(
            SNConv(in_channels=conv_dim*1, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False),
            SNConv(in_channels=conv_dim*1, out_channels=3,          kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False),
            nn.Sigmoid() # Tanh()
        )

        self.ga5 = GAM(conv_dim*16, conv_dim*16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga4 = GAM(conv_dim* 8, conv_dim* 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga3 = GAM(conv_dim* 4, conv_dim* 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga2 = GAM(conv_dim* 2, conv_dim* 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga1 = GAM(conv_dim* 1, conv_dim* 1, reduction=8, bias=False, use_sn=use_sn, norm=True)

        ##### decoder 2 of Policy Network
        self.upsample21 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim*16, conv_dim*8, 1, 1, 0, 1, True, use_sn))
        self.upsample22 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 8, conv_dim*4, 1, 1, 0, 1, True, use_sn))
        self.upsample23 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 4, conv_dim*2, 1, 1, 0, 1, True, use_sn))
        self.upsample24 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 2, conv_dim*1, 1, 1, 0, 1, True, use_sn))

        self.dec21 = ConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*512 --> 32*32*256
        self.dec22 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*256 --> 64*64*128
        self.dec23 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*128 --> 128*128*64
        self.dec24 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*64 --> 256*256*32
        self.dec25 = nn.Sequential(
            SNConv(in_channels=conv_dim*1, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False),
            SNConv(in_channels=conv_dim*1, out_channels=3,          kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False),
            nn.Sigmoid()
        )

        self.ga25 = GAM(conv_dim*16, conv_dim*16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga24 = GAM(conv_dim* 8, conv_dim* 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga23 = GAM(conv_dim* 4, conv_dim* 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga22 = GAM(conv_dim* 2, conv_dim* 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga21 = GAM(conv_dim* 1, conv_dim* 1, reduction=8, bias=False, use_sn=use_sn, norm=True)

        ###### decoder 3 of Value Network
        self.fc1 = nn.Linear(conv_dim*16 * 16 * 16 * 4, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        ### shared encoder

        x1 = self.enc1(x)                           # [1, 32, 256, 256]
        x2 = self.enc2(x1)                          # [1, 64, 128, 128]
        x3 = self.enc3(x2)                          # [1, 128, 64, 64]
        x4 = self.enc4(x3)                          # [1, 256, 32, 32]
        x5 = self.enc5(x4)                          # [1, 512, 16, 16]
        x5 = self.ga5(x5)                           # [1, 512, 16, 16]

        ### decoder 1
        y1 = self.upsample1(x5)                     # [1, 256, 32, 32]
        y1 = torch.cat([y1, self.ga4(x4)], dim=1)   # [1, 512, 32, 32]
        y1 = self.dec1(y1)                          # [1, 256, 32, 32]

        y2 = self.upsample2(y1)                     # [1, 128, 64, 64]
        y2 = torch.cat([y2, self.ga3(x3)], dim=1)   # [1, 256, 64, 64]
        y2 = self.dec2(y2)                          # [1, 128, 64, 64]

        y3 = self.upsample3(y2)                     # [1, 64, 128, 128]
        y3 = torch.cat([y3, self.ga2(x2)], dim=1)   # [1, 128, 128, 128]
        y3 = self.dec3(y3)                          # [1, 64, 128, 128]

        y4 = self.upsample4(y3)                     # [1, 32, 256, 256]
        y4 = torch.cat([y4, self.ga1(x1)], dim=1)   # [1, 64, 256, 256]
        y4 = self.dec4(y4)                          # [1, 32, 256, 256]

        mask1 = self.dec5(y4.mul(x1))               # [1, 3, 256, 256]

        ### decoder 2
        y21 = self.upsample21(x5)                      # [1, 256, 32, 32]
        y21 = torch.cat([y21, self.ga24(x4)], dim=1)   # [1, 512, 32, 32]
        y21 = self.dec21(y21)                          # [1, 256, 32, 32]

        y22 = self.upsample22(y21)                     # [1, 128, 64, 64]
        y22 = torch.cat([y22, self.ga23(x3)], dim=1)   # [1, 256, 64, 64]
        y22 = self.dec22(y22)                          # [1, 128, 64, 64]

        y23 = self.upsample23(y22)                     # [1, 64, 128, 128]
        y23 = torch.cat([y23, self.ga22(x2)], dim=1)   # [1, 128, 128, 128]
        y23 = self.dec23(y23)                          # [1, 64, 128, 128]

        y24 = self.upsample24(y23)                     # [1, 32, 256, 256]
        y24 = torch.cat([y24, self.ga21(x1)], dim=1)   # [1, 64, 256, 256]
        y24 = self.dec24(y24)                          # [1, 32, 256, 256]

        mask2 = self.dec25(y24.mul(x1))                # [1, 3, 256, 256]

        ### decoder 3
        y31 = x5.view(x5.size(0), -1)  # 将卷积特征展平
        y32 = torch.relu(self.fc1(y31))
        value = self.fc2(y32)

        return mask1, mask2, value

class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()

        # scale 1 and prediction of scale 1         128
        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=2, padding=3, dilation=1,
                              use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [
            dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                                use_bias=False, type=adv_loss_type)]

        # scale 2       64
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3,
                              dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3 and prediction of scale 3          32
        d_3 = [dis_conv_block(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=7, stride=2, padding=3,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim * 4, out_channels=1, kernel_size=7, stride=1, padding=3,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4  16
        d_4 = [dis_conv_block(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=5, stride=2, padding=2,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 5 and prediction of scale 5         8
        d_5 = [dis_conv_block(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=5, stride=2, padding=2,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_5_pred = [dis_pred_conv_block(in_channels=conv_dim * 16, out_channels=1, kernel_size=5, stride=1, padding=2,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)
        self.d5 = nn.Sequential(*d_5)
        self.d5_pred = nn.Sequential(*d_5_pred)

    def forward(self, x):
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        ds5 = self.d5(ds4)
        ds5_pred = self.d5_pred(ds5)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred, ds5_pred]


# ========================================
# Common Blocks
# ========================================

# 判别器的卷积块
def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun,
                   use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = [nn.ReflectionPad2d(padding), SpectralNorm(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                  dilation=dilation, bias=use_bias), use_sn)]
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main


# 判别器预测卷积模块
def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = [nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0, dilation=dilation, bias=use_bias)]
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main


# 计算特征均值与方差
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# Global Attention Module
class GAM(nn.Module):
    """Global attention module"""
    def __init__(self, in_nc, out_nc, reduction=8, bias=False, use_sn=False, norm=False):
        super(GAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_nc*2, out_channels=in_nc//reduction, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_nc//reduction, out_channels=out_nc, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1),
        )
        self.fuse = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=in_nc * 2, out_channels=out_nc, kernel_size=1, stride=1, bias=True, padding=0, dilation=1), use_sn),
        )
        self.in_norm = nn.InstanceNorm2d(out_nc)
        self.norm = norm

    def forward(self, x):
        x_mean, x_std = calc_mean_std(x)
        out = self.conv(torch.cat([x_mean, x_std], dim=1))
        # out = self.conv(x_mean)
        out = self.fuse(torch.cat([x, out.expand_as(x)], dim=1))
        if self.norm:
            out = self.in_norm(out)
        return out


# 生成器卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun,
                 use_sn):
        super(ConvBlock, self).__init__()

        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        block = [
            nn.ReflectionPad2d(padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        ]
        norm_fun = get_norm_fun(norm_fun)
        block.append(norm_fun(out_channels))
        block.append(get_act_fun(act_fun))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


# 插值算法
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out


class DisPredConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
        super(DisPredConvBlock, self).__init__()

        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.blocks = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=0, dilation=dilation, bias=use_bias)
        ]
        if type in ['ls', 'rals']:
            self.blocks.append(nn.Sigmoid())
        elif type in ['hinge', 'rahinge']:
            self.blocks.append(nn.Tanh())
        else:
            raise NotImplementedError("Adversarial loss [{}] is not found.".format(type))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


# 谱归一化
def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# 谱归一化卷积层
class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        """
        Conv2d:
        H_o = [(H_i + 2 * padding - d * (k - 1) - 1) / stride] + 1
        目标：让 H_o = H_i / stride，Reflection的作用是在周围加上指定数目的padding数，设为x，即下面的self.padding
        由目标可以推出：H_o = [(H_i + 2 * x + 2 * padding - d * (k - 1) - 1) / stride] + 1
        padding=0带入：H_o = [(H_i + 2 * x - d * (k - 1) - 1) / stride] + 1 = (H_i/stride) + [(2 * x - d * (k - 1) - 1) / stride] + 1
        让 [(2 * x - d * (k - 1) - 1) / stride] + 1 = 0 -> 2 * x - d * (k - 1) - 1 + stride = 0 -> x = (d * (k - 1) + 1 - stride) / 2
        由于 stride = 1 或者 2，所以 1 - stride = 1 或者 -1，除以2都为0，因此，x = (d * (k - 1)) / 2 = (k + (k - 1) * (d - 1) - 1)
        """
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=0, dilation=dilation, bias=use_bias), use_sn),
        )

    def forward(self, x):
        return self.main(x)


# 获取激活函数
def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()


# identity 等效为特殊权重的卷积层
# https://zhuanlan.zhihu.com/p/353697121#:~:text=identity,x1%E7%9A%84%E5%8D%B7%E7%A7%AF%E5%BD%A2%E5%BC%8F%EF%BC%9A
class Identity(nn.Module):
    def forward(self, x):
        return x


# 获取归一化函数
def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun


# ========================================
# Test
# ========================================
if __name__ == '__main__':
    data = torch.rand([1, 3, 512, 512])
    print("========================== Generator Test ==========================")
    G = Generator(conv_dim=32, norm_fun='InstanceNorm', act_fun='LeakyReLU', use_sn=False)
    mask1, mask2, value = G(data)
    print(data.shape)

    print("========================== Discriminator Test ======================")
    D = Discriminator(conv_dim=32, norm_fun='none', act_fun='LeakyReLU', use_sn=True, adv_loss_type='rals')
    data = mask1
    data = D(data)
    for d in data:
        print(d.shape)
