import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.ResNet import *
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_tf_
from mmengine.model import constant_init, kaiming_init,trunc_normal_init,normal_init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.helpers import named_apply
#from segment_anything.modeling.common import LayerNorm2d

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class MFIM(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MFIM, self).__init__()
        self.conv0_0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0_0 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv0_1 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=1, dilation=1,groups=dim // fc_ratio)
        self.bn0_1 = nn.BatchNorm2d(dim // fc_ratio)

        self.conv0_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
     
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim//fc_ratio)
 
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_0 = self.relu(self.bn0_0(self.conv0_0(x)))
        attn0_1 = self.relu(self.bn0_1(self.conv0_1(attn0_0)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_0)))
        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_0)))
        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_0)))
     
        attn = torch.cat([attn0_1, attn0_2 , attn1_2 , attn2_2], dim=1)
      
        return attn


class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):    
        x = x + res

        return x

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=6):
        super(MAF, self).__init__()

 

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(256, num_classes, kernel_size=1))

    def forward(self, x):

        out = self.head(x)

        return out



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class UFRM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(UFRM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6()
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class SSP(nn.Module):
    def __init__(self, kernel_size=7):
        super(SSP, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SCE(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(SCE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU6()
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)



class MSSAM(nn.Module):
    def __init__(self, in_dim, is_bottom=False):
        super().__init__()
     
        self.ms = MFIM(in_dim)
        self.eps = 1e-8
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
      

        self.sce = SCE(in_dim)
        self.ssp = SSP()

    def forward(self, x, skip):

        c = self.sce(x)

        skip1 = self.ms(skip)  
        s = self.ssp(skip1)

        x1 = x * s
        x2 = skip1 * c
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * x1 + fuse_weights[1] * x2

        return x

# #
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SWA(nn.Module):
    def __init__(self, in_channels, n_heads=8, window_size=8):
        super(SWA, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        padded_x = F.pad(x, [self.window_size // 2, self.window_size // 2, self.window_size // 2, self.window_size // 2], mode='reflect')

        proj_query = self.query_conv(x).view(batch_size, self.n_heads, C // self.n_heads, height * width)
        proj_key = self.key_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_key = proj_key.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)
        proj_value = self.value_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_value = proj_value.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)

        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = self.softmax(energy)

        out_window = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        out_window = out_window.permute(0, 1, 3, 2).contiguous().view(batch_size, C, height, width)

        out = self.gamma * out_window + x
        return out



class GDEM(nn.Module):
    def __init__(self, dim=2048, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SWA(in_channels = dim,window_size=8)
        #self.attn2 = SWA(in_channels=dim, window_size=2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim / mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dilation = [[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

    
        self.b4 = GDEM(dim=encode_channels[3])

        self.seg_head = MAF(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)
         
        self.p3 = MSSAM(encode_channels[2], is_bottom=False)
        self.p2 = MSSAM(encode_channels[1], is_bottom=False)
        self.p1 = MSSAM(encode_channels[0], is_bottom=False)
      
        eucb_ks = 3
        self.eucb1 = UFRM(in_channels=encode_channels[3], out_channels=encode_channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.eucb2 = UFRM(in_channels=encode_channels[2], out_channels=encode_channels[1], kernel_size=eucb_ks,
                          stride=eucb_ks // 2)
        self.eucb3 = UFRM(in_channels=encode_channels[-3], out_channels=encode_channels[-4], kernel_size=eucb_ks,
                          stride=eucb_ks // 2)

        self.init_weight()
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=1),
        )

    def forward(self, res1, res2, res3, res4, h, w):

       


        x4 = self.b4(res4)      
        x4 = self.eucb1(x4)
       
        x3 = self.p3(x4, res3)
     
        res = self.mask_downscaling(x3)
                    
        del x4
        del res4

        x3 = self.eucb2(x3)
        x2 = self.p2(x3, res2)
          
        del x3
        del res3

        x = self.eucb3(x2)

        del x2
        del res2
  
        x1 = self.p1(x, res1)

        x = self.seg_head(x1)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return res,x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CT_Net(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.1,
                 num_classes=6,
                 backbone=ResNet50
                 ):
        super().__init__()

        self.backbone = backbone()

        self.decoder = Decoder(encode_channels, decode_channels, dropout=dropout, num_classes=num_classes)


    def forward(self, x):
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)
        res,x = self.decoder(res1, res2, res3, res4, h, w)

        return res, x
