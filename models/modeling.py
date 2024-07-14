import torch
from torch import nn
from models.spectrum_pixel_clustering import Cluster
from models.backbones.resnet import resnet18
from models.backbones.swin import swin_t, Swin_T_Weights
from models.backbones.PVTV2 import pvt_v2_b1
import torch.nn.functional as F
from models.VIT import ViT


class spectral_derivate:
    def __init__(self):
        super().__init__()
        self.wavelength = torch.tensor([466, 480, 500, 520, 536, 550, 566, 580,
                                        596, 610, 626, 640, 656, 670, 686, 700,
                                        716, 730, 746, 760, 776, 790, 806, 820,
                                        836, 850, 866, 880, 896, 910, 926, 940])
        self.delta_n = self.wavelength[1:] - self.wavelength[:-1]

    def sdf(self, x):
        derivate = x[:, 1:, :, :] - x[:, :-1, :, :]
        if x.shape[1] == 32:
            delta_n = self.delta_n.to(x.device)
            return derivate / delta_n.view(1, -1, 1, 1)
        else:
            return derivate


class Encoder(nn.Module):
    def __init__(self, backbone, in_channels, out_channels):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = resnet18(pretrained=True, in_channels=in_channels)
            channels = [64, 64, 128]
        elif backbone == 'pvtv2_b1':
            self.encoder = pvt_v2_b1(
                pre_trained_path="models/pre_trained/pvt_v2_b1.pth",
                in_chans=in_channels)
            channels = [64, 128, 320]
        elif backbone == 'swin_tiny':
            self.encoder = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1, progress=True, in_channel=in_channels)
            channels = [96, 192, 384]
        self.up_sample = nn.ModuleDict({
            'up1': nn.Sequential(
                nn.Conv2d(channels[2], channels[1], kernel_size=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True)
            ),
            'up2': nn.Sequential(
                nn.Conv2d(channels[1] + channels[1], channels[0], kernel_size=1),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            ),
            'up3': nn.Sequential(
                nn.Conv2d(channels[0] + channels[0], out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        })

    def forward(self, x):
        img_feat, res_feats = self.encoder(x)
        img_feat = F.interpolate(self.up_sample['up1'](img_feat), size=res_feats[1].shape[-2:], mode='nearest')
        img_feat = F.interpolate(self.up_sample['up2'](torch.cat([img_feat, res_feats[1]], dim=1)),
                                 size=res_feats[0].shape[-2:], mode='nearest')
        img_feat = F.interpolate(self.up_sample['up3'](torch.cat([img_feat, res_feats[0]], dim=1)), size=x.shape[-2:],
                                 mode='nearest')
        return img_feat


class DSTC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg['in_channels']
        classes = cfg['num_classes']
        clus_dim = cfg['cluster']['dim']
        self.cfg = cfg
        # UNet
        self.UNet = Encoder(backbone=cfg['backbone'], in_channels=in_channels, out_channels=clus_dim)
        # SDF operation
        self.sdf_op = spectral_derivate()
        #
        self.cluster = Cluster(in_channels, classes,
                               proposal=cfg['cluster']['proposal'], fold=cfg['cluster']['fold'], hidden_dim=clus_dim)
        self.vit = ViT(image_size=cfg['img_size'], patch_size=cfg['vit']['patch_size'], num_classes=classes,
                       dim=cfg['vit']['hidden_size'], depth=cfg['vit']['depth'], heads=cfg['vit']['num_heads'],
                       mlp_dim=cfg['vit']['mlp_dim'], channels=clus_dim, dim_head=cfg['vit']['dim_head'],
                       dropout=cfg['vit']['dropout'], emb_dropout=cfg['vit']['attention_dropout'])

    def forward(self, img, gt):
        """
            img: [B, D, H, W]
            gt: [B, H, W]
        """
        # UNet
        img_feat = self.UNet(img)
        # sdf
        sdf_data = self.sdf_op.sdf(img)  # [B, D-1, H, W]
        # ----cluster----
        center_feat, labels, spix_map = self.cluster(img, gt, img_feat, sdf_data)
        # ----vit classification----
        vit_out = self.vit(center_feat)
        return vit_out, labels, spix_map
