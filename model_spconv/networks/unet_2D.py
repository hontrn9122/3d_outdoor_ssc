import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
import torch_scatter
from model_spconv.layers import ConvCapsule2d, Squash


class BEV_UNet(nn.Module):
    def __init__(self, n_class, dilation, bilinear, group_conv, input_batch_norm, dropout, circular_padding, dropblock):
        super(BEV_UNet, self).__init__()
        padding_mode = 'circular' if circular_padding else 'zeros'
        self.inc = inconv(32+16, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(256, 256, dilation, group_conv, circular_padding)
        self.down3 = down(512, 512, dilation, group_conv, circular_padding)
        self.down4 = down(1024, 512, dilation, group_conv, circular_padding)
        
        self.caps1 = ConvCapsule2d(3, 32, 16, 32, 32, padding = 1, share_weight=False, padding_mode=padding_mode)
        self.caps2 = ConvCapsule2d(3, 32, 32, 32, 64, padding = 1, share_weight=False, padding_mode=padding_mode)
        self.caps3 = ConvCapsule2d(3, 32, 64, 20, 32, padding = 1, share_weight=False, padding_mode=padding_mode)
        
        self.up1 = up(1664, 512, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(1024, 256, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(512, 128, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(192, 128, circular_padding, bilinear = bilinear, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(128, n_class)

    def forward(self, voxel, evoxel, x_ds1, x_ds2, x_ds3):
        x = torch.cat([voxel, evoxel], dim=1)
        x1 = self.inc(x)    # [B, 64, 256, 256]
        x2 = self.down1(x1)    # [B, 128, 128, 128]
        x2_cat = torch.cat((x2, self.channel_reduction(x_ds1, x2.shape[1])), dim = 1)    # [B, 128, 128, 128] + [B, 128, 128, 128]
        x3 = self.down2(x2_cat)    # [B, 256, 64, 64]
        x3_cat = torch.cat((x3, self.channel_reduction(x_ds2, x3.shape[1])), dim = 1)    # [B, 256, 64, 64] + [B, 256, 64, 64]
        x4 = self.down3(x3_cat)    # [B, 512, 32, 32]
        x4_cat = torch.cat((x4, self.channel_reduction(x_ds3, x4.shape[1])), dim = 1)    # [B, 512, 32, 32] + [B, 512, 32, 32]
        x5 = self.down4(x4_cat)    # [B, 512, 16, 16]
        
        xcaps0 = x5.reshape(-1, 32, 16, x5.shape[2], x5.shape[3])
        xcaps0 = Squash()(xcaps0, dim=2)
        xcaps1 = self.caps1(xcaps0)
        xcaps2 = self.caps2(xcaps1)
        xcaps3 = self.caps3(xcaps2)
        
        x = xcaps3.reshape(xcaps3.shape[0], -1, xcaps3.shape[-2], xcaps3.shape[-1])
        x = self.up1(x, x4_cat)
        x = self.up2(x, x3_cat)
        x = self.up3(x, x2_cat)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x, xcaps3

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: (B, C1, H, W)
            out_channels: C2

        Returns:

        """
        B, in_channels, H, W = x.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.view(B, out_channels, -1, H, W)
        # x = torch.max(x, dim=2)[0]
        x = x.sum(dim=2)
        return x

    
class CapsBEV_UNet(nn.Module):
    def __init__(self, n_class, dilation, circular_padding=False):
        super(CapsBEV_UNet, self).__init__()
        padding_mode = 'circular' if circular_padding else 'zeros'
        self.inc = inconvCaps(32+16*32, 8, 8, dilation)
        self.down1 = downCaps(8, 8, 8, 16, 128 * 16, dilation)
        self.down2 = downCaps(8, 16, 16, 16, 256 * 8, dilation)
        self.down3 = downCaps(16, 16, 16, 32, 512 * 4, dilation)
        self.down4 = downCaps(16, 32, 32, 32, None, dilation)
        
        self.prim_caps = ConvCapsule2d(3, 32, 32, 20, 64, padding = 1, share_weight=False, padding_mode=padding_mode)
        
        self.up1 = upCaps(20, 64, 20, 64, 16, 32)
        self.up2 = upCaps(20, 64, 20, 32, 16, 16)
        self.up3 = upCaps(20, 32, 20, 32, 8, 16)
        self.up4 = upCaps(20, 32, 20, 32, 8, 8)
        # self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(20*32, 20*32)
        
    def forward(self, voxel, evoxel, x_ds1, x_ds2, x_ds3):
        x = torch.cat([voxel, evoxel], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1, x_ds1)
        x3 = self.down2(x2, x_ds2)
        x4 = self.down3(x3, x_ds3)
        x5 = self.down4(x4)
        
        prim_caps = self.prim_caps(x5)
        x = prim_caps.view(-1, 1280, 16, 16)
        
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x.view(-1, 20, 32, 256, 256).permute(0, 1, 3, 4, 2)
        return x, prim_caps
    
    
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class inconvCaps(nn.Module):
    def __init__(self, in_ch, out_caps, out_dim, dilation):
        super(inconvCaps, self).__init__()
        out_ch = out_caps*out_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # nn.BatchNorm2d(out_ch*2),
            # nn.LeakyReLU(inplace=True),
        )
        self.caps = ConvCapsule2d(3, out_caps, out_dim, out_caps, out_dim, padding = 1, share_weight=False)
        self.squash = Squash()

    def forward(self, x):
        x = self.conv(x)
        caps_shape = list(x.shape)
        caps_shape = caps_shape[0:1] + [self.out_caps, self.out_dim] + caps_shape[2:]
        x = x.reshape(*caps_shape)
        x = self.squash(x, dim=2)
        return self.caps(x)
    

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )                

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
    
class downCaps(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, aid_ch=None, dilation=1, padding_mode='zeros'):
        super(downCaps,self).__init__()
        self.aid_ch = aid_ch
        if aid_ch is not None:
            self.capsconv = nn.ModuleList([
                ConvCapsule2d(2, in_caps, in_dim, out_caps, out_dim//2, stride=2, share_weight=False, padding_mode=padding_mode, squash=False),
                ConvCapsule2d(3, out_caps, out_dim, out_caps, out_dim, padding=1, share_weight=False, padding_mode=padding_mode)
            ])
            self.shortcut = nn.Conv2d(aid_ch, out_caps*out_dim//2, 3, padding=1)
            self.squash = Squash()
        else:
            self.capsconv = nn.Sequential(
                ConvCapsule2d(2, in_caps, in_dim, out_caps, out_dim, stride=2, share_weight=False, padding_mode=padding_mode, squash=True),
                ConvCapsule2d(3, out_caps, out_dim, out_caps, out_dim, padding=1, share_weight=False, padding_mode=padding_mode)
            )
            
        
    def forward(self, x, aid=None):
        assert (aid is None and self.aid_ch is None) or (self.aid_ch is not None and aid is not None)
        
        if self.aid_ch is None:
            return self.capsconv(x)
        
        x = self.capsconv[0](x)
        aid = self.shortcut(aid).view(*list(x.shape))
        x = torch.cat([x, aid], dim=2)
        x = self.squash(x, dim=2)
        return self.capsconv[1](x)
    

class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock=False, drop_p=0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class upCaps(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, scnt_caps, scnt_dim, last_plain=False):
        super(upCaps, self).__init__()
        in_ch = in_caps * in_dim
        out_ch = out_caps * out_dim
        self.up = nn.ConvTranspose2d(in_ch, out_ch//2, 2, stride=2, groups = out_caps)
        self.shortcut = ConvCapsule2d(3, scnt_caps, scnt_dim, out_caps, out_dim//2, padding=1, share_weight=False, squash=False)
        if last_plain:
            self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups = out_caps)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups = out_caps),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        
    def forward(self, x, skip_cnt):
        x = self.up(x)
        shortcut = self.shortcut(skip_cnt)
        shortcut_shape = list(shortcut.shape)
        x = x.view(*shortcut_shape)
        x = torch.cat([x, shortcut], dim=2)
        feat_shape = list(x.shape)
        feat_shape = feat_shape[0:1] + [-1] + feat_shape[3:]
        return self.conv(x.view(*feat_shape))
    
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class outconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return t    