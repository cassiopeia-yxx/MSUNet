# -*- coding: utf-8 -*-
# Author : ML_XX
# Date : 2025/11/13 10:34
# Description : Core architectural components for MUWNet, providing a research-grade implementation of multi-scale restoration blocks and attention mechanisms.

from natsort.utils import input_string_transform_factory
from basicsr.utils.registry import ARCH_REGISTRY
import math
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import numbers
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from basicsr.archs.mamba import Vmamba


try:
    from thop import profile, clever_format
except ImportError:
    print("thop is not installed. FLOPs and parameter count will not be available.")
    profile = lambda model, inputs, verbose: (0, 0)
    clever_format = lambda x, y: (0, 0)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        stride=stride,
        bias=bias,
    )


class ConvBlock(nn.Module):
    def __init__(self, n_feats, expansion_factor=8):
        super().__init__()
        hidden_dim = n_feats * expansion_factor

        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_feats, 1, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class UpSample(nn.Module):
    """MRUNet 的上采样模块 (x2)"""

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpSample2(nn.Module):
    """MRUNet 的上采样模块 (x4)"""

    def __init__(self, in_channels, out_channels):
        super(UpSample2, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Fusion(nn.Module):
    def __init__(self, features, reduction=4):
        super().__init__()
        self.fuse = nn.Conv2d(features * 3, features, 1)
        self.ca = CALayer(features, reduction=reduction)
        self.tail = conv(features, 3, 1)

    def forward(self, x1, x2, x3):
        x = self.fuse(torch.cat([x1, x2, x3], dim=1))
        x = self.ca(x)
        return self.tail(x)


class ISFF(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ISFF, self).__init__()
        self.gamma = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups=in_nc)
        self.phi = nn.Conv2d(in_nc, in_nc, 3, 1, 1, groups=in_nc)

    def forward(self, x, x_enc, x_dec):
        out = self.gamma(x_enc) * x + self.phi(x_dec) + x
        return out


class Mamba(nn.Module):
    def __init__(self, dim):
        super(Mamba, self).__init__()
        self.mamba = Vmamba(dim=dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """

        identity = x

        # x = x.permute(0, 2, 3, 1).contiguous()  # (B, h, w, C)
        x = self.mamba(x)  # (B, h, w, C)
        # x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, h, w)

        out = identity + x

        return out


class UNetConvBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        downsample,
        use_csff=False,
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff
        self.Mamba = Mamba(dim=in_size)

        if downsample and use_csff:
            self.ISFF_Fusion = ISFF(in_size, in_size)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = x
        out = self.Mamba(out)  # run Mamba
        if enc is not None and dec is not None:
            assert self.use_csff
            out = self.ISFF_Fusion(out, enc, dec)  # run ISFF
        if self.downsample:
            out_down = self.downsample(out)  # run Down
            return out_down
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv = nn.Conv2d(out_size * 2, out_size, 1, bias=False)
        self.conv_block = UNetConvBlock(
            out_size, out_size, downsample=False, use_csff=False
        )

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv(torch.cat([up, bridge], dim=1))
        out = self.conv_block(out)
        return out


class Encoder(nn.Module):
    def __init__(self, n_feat, use_csff=False, depth=3):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(
                    in_size=n_feat * 2 ** (i),
                    out_size=n_feat * 2 ** (i + 1),
                    downsample=True,
                    use_csff=use_csff,
                )
            )

        self.body.append(
            UNetConvBlock(
                in_size=n_feat * 2 ** (depth - 1),
                out_size=n_feat * 2 ** (depth - 1),
                downsample=False,
                use_csff=use_csff,
            )
        )

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x, encoder_outs[i], decoder_outs[-i - 1])
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    res.append(x)
                    x = down(x)
                else:
                    x = down(x)
        return res, x


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, depth=3):
        super(Decoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetUpBlock(
                    in_size=n_feat * 2 ** (depth - i - 1),
                    out_size=n_feat * 2 ** (depth - i - 2),
                )
            )

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, bridges[-i - 1])
            res.append(x)
        return res, x


class Rnet(nn.Module):
    def __init__(self, in_c, n_feat, kernel_size, n_depth):
        super(Rnet, self).__init__()
        self.R_encoder = conv(in_c, n_feat, 1, bias=False)
        self.stage_encoder = Encoder(n_feat, use_csff=True, depth=n_depth)
        self.stage_decoder = Decoder(n_feat, kernel_size, depth=n_depth)

    def forward(self, R, f_encoder, f_decoder):
        R = self.R_encoder(R)
        feat1, f_encoder = self.stage_encoder(R, f_encoder, f_decoder)
        f_decoder, last_out = self.stage_decoder(f_encoder, feat1)
        return last_out, feat1, f_decoder


class init_R(nn.Module):
    def __init__(self, in_c, n_feat, kernel_size, n_depth):
        super(init_R, self).__init__()
        self.R_encoder = conv(in_c, n_feat, 1, bias=False)
        self.init_encoder = Encoder(n_feat, use_csff=False, depth=n_depth)
        self.init_decoder = Decoder(n_feat, kernel_size, depth=n_depth)

    def forward(self, R):
        R = self.R_encoder(R)
        feat1, f_encoder = self.init_encoder(R)
        f_decoder, out_put = self.init_decoder(f_encoder, feat1)
        return feat1, f_decoder, out_put


@ARCH_REGISTRY.register()
class MUWNet(nn.Module):
    def __init__(self, in_c=3, n_feat=32, nums_stage=5, n_depth=3, kernel_size=3):
        super(MUWNet, self).__init__()
        self.nums_stages = nums_stage
        self.rho_s1 = self.make_eta(self.nums_stages, torch.Tensor([0.5]))
        self.L_s1 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.Lt_s1 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phi_s1 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phit_s1 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )

        # 尺度 2 (s2): 0.5
        self.rho_s2 = self.make_eta(self.nums_stages, torch.Tensor([0.5]))
        self.L_s2 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.Lt_s2 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phi_s2 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phit_s2 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )

        # 尺度 3 (s3): 0.25
        self.rho_s3 = self.make_eta(self.nums_stages, torch.Tensor([0.5]))
        self.L_s3 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.Lt_s3 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phi_s3 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )
        self.phit_s3 = nn.ModuleList(
            [ConvBlock(in_c, expansion_factor=8) for _ in range(self.nums_stages)]
        )

        # --- 近端映射网络 (权重共享) ---
        self.init_updateR = init_R(in_c, n_feat, kernel_size, n_depth)
        self.proxNet_R = nn.ModuleList(
            [
                Rnet(in_c, n_feat, kernel_size, n_depth)
                for _ in range(self.nums_stages - 1)
            ]
        )

        # --- 融合与上采样模块 (新增) ---
        self.up_s2 = UpSample(n_feat, n_feat)  # x2
        self.up_s3 = UpSample2(n_feat, n_feat)  # x4
        self.fusion = nn.ModuleList([Fusion(n_feat) for _ in range(self.nums_stages)])

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f)
        return eta

    def forward(self, I):
        output_R = []
        b, c, h, w = I.shape

        input_img = I
        # --- 创建多尺度输入和状态 ---
        I_s1 = input_img
        I_s2 = F.interpolate(
            input_img, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        I_s3 = F.interpolate(
            input_img, scale_factor=0.25, mode="bilinear", align_corners=False
        )

        # === 3. 初始化 R (三个尺度)，从 R0 出发 ===
        R_s1 = I_s1
        R_s2 = I_s2
        R_s3 = I_s3

        # --- 3. 初始化近端网络内部状态 (三个尺度) ---
        feat_R_s1, f_decoder_R_s1 = None, None
        feat_R_s2, f_decoder_R_s2 = None, None
        feat_R_s3, f_decoder_R_s3 = None, None

        ##----------- stage 1 (i=0) ----------------- ##
        # 1. GDM 步骤 (多尺度)
        R_hat_s1 = R_s1 - self.rho_s1[0, :] * self.Lt_s1[0](
            self.phit_s1[0](self.phi_s1[0](self.L_s1[0](R_s1)) - I_s1)
        )
        R_hat_s2 = R_s2 - self.rho_s2[0, :] * self.Lt_s2[0](
            self.phit_s2[0](self.phi_s2[0](self.L_s2[0](R_s2)) - I_s2)
        )
        R_hat_s3 = R_s3 - self.rho_s3[0, :] * self.Lt_s3[0](
            self.phit_s3[0](self.phi_s3[0](self.L_s3[0](R_s3)) - I_s3)
        )

        # 3. ProxNet 步骤 (权重共享)
        feat_R_s1, f_decoder_R_s1, out_feat_s1 = self.init_updateR(R_hat_s1)
        feat_R_s2, f_decoder_R_s2, out_feat_s2 = self.init_updateR(R_hat_s2)
        feat_R_s3, f_decoder_R_s3, out_feat_s3 = self.init_updateR(R_hat_s3)

        # 4. 上采样 & 融合
        out_feat_s2_up = self.up_s2(out_feat_s2)
        out_feat_s3_up = self.up_s3(out_feat_s3)
        out_put_R_fused = self.fusion[0](
            out_feat_s1, out_feat_s2_up, out_feat_s3_up
        )  # (B, 3, H, W)

        # 5. 更新 R (只更新尺度1)
        R_s1 = R_hat_s1 + out_put_R_fused
        output_R.append(R_s1[:, :, :h, :w])

        # 6. 为下一阶段准备多尺度输入
        R_s2 = F.interpolate(R_s1, scale_factor=0.5)
        R_s3 = F.interpolate(R_s1, scale_factor=0.25)

        ##-------------- Stage 2 to k (i=1 到 k-1) ---------------------##
        for i in range(1, self.nums_stages):
            # 1. GDM 步骤 (多尺度)
            R_hat_s1 = R_s1 - self.rho_s1[i, :] * self.Lt_s1[i](
                self.phit_s1[i](self.phi_s1[i](self.L_s1[i](R_s1)) - I_s1)
            )
            R_hat_s2 = R_s2 - self.rho_s2[i, :] * self.Lt_s2[i](
                self.phit_s2[i](self.phi_s2[i](self.L_s2[i](R_s2)) - I_s2)
            )
            R_hat_s3 = R_s3 - self.rho_s3[i, :] * self.Lt_s3[i](
                self.phit_s3[i](self.phi_s3[i](self.L_s3[i](R_s3)) - I_s3)
            )

            # 3. ProxNet 步骤 (权重共享)
            out_feat_s1, feat_R_s1, f_decoder_R_s1 = self.proxNet_R[i - 1](
                R_hat_s1, feat_R_s1, f_decoder_R_s1
            )
            out_feat_s2, feat_R_s2, f_decoder_R_s2 = self.proxNet_R[i - 1](
                R_hat_s2, feat_R_s2, f_decoder_R_s2
            )
            out_feat_s3, feat_R_s3, f_decoder_R_s3 = self.proxNet_R[i - 1](
                R_hat_s3, feat_R_s3, f_decoder_R_s3
            )

            # 4. 上采样 & 融合
            out_feat_s2_up = self.up_s2(out_feat_s2)
            out_feat_s3_up = self.up_s3(out_feat_s3)
            out_put_R_fused = self.fusion[i](
                out_feat_s1, out_feat_s2_up, out_feat_s3_up
            )  # (B, 3, H, W)

            # 5. 更新 R (只更新尺度1)
            R_s1 = R_hat_s1 + out_put_R_fused
            output_R.append(R_s1[:, :, :h, :w])

            # 6. 为下一阶段准备多尺度输入 (如果不是最后阶段)
            if i < self.nums_stages - 1:
                R_s2 = F.interpolate(R_s1, scale_factor=0.5)
                R_s3 = F.interpolate(R_s1, scale_factor=0.25)

        return output_R


if __name__ == "__main__":
    # --- Configuration ---
    N_FEAT = 32
    INPUT_SIZE = (1, 3, 256, 256)
    NUM_STAGES = 5

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Creation ---
    model = MUWNet(n_feat=N_FEAT, nums_stage=NUM_STAGES).to(device)

    # --- Model Analysis ---
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # Test forward pass
    print(f"在设备 {device} 上测试 forward...")
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print("模型成功执行！")
        print(f"总共 {len(outputs)} 个阶段的输出。")
        for i, out in enumerate(outputs):
            print(f"  阶段 {i+1} 输出 shape: {out.shape}")
    except Exception as e:
        print(f"模型 forward 失败: {e}")

    # FLOPs/Params (thop 可能在复杂模型上失败)
    print("尝试计算 FLOPs 和参数...")

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("=" * 70)
    print("Multi-Scale SSURNet (融合 MRUNet 思想)")
    print(f"Input Tensor Size: {INPUT_SIZE}")
    print(f"Running on Device: {device}")
    print("-" * 70)
    print(f"Total Parameters: {params}")
    print(f"Total FLOPs: {flops}")
    print("=" * 70)
