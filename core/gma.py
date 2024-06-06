import torch
from matplotlib import pyplot as plt
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from torch.nn import AdaptiveAvgPool2d


class Sp_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Sp_Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1, groups=dim , bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def Sob(self, feat):
        kernel_dx = [[1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]]

        kernel_dy = [[1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]]

        dim = feat.size()[1]

        dx_window = torch.FloatTensor(kernel_dx).expand(dim, dim, 3, 3)
        dx_weight = nn.Parameter(data=dx_window, requires_grad=False).cuda()

        dy_window = torch.FloatTensor(kernel_dy).expand(dim, dim, 3, 3)
        dy_weight = nn.Parameter(data=dy_window, requires_grad=False).cuda()

        dx = F.conv2d(feat, dx_weight, padding=1)
        dy = F.conv2d(feat, dy_weight, padding=1)
        return dx, dy

    def forward(self, x):

        dx, dy = self.Sob(x)
        b, c, h, w = x.shape

        q_S = self.q_dwconv(self.q(dx + dy))

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q_S = rearrange(q_S, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_S = torch.nn.functional.normalize(q_S, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q_S.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q_S @ k.transpose(-2, -1)) * self.temperature


        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim=128, bias=False):
        super(FFN, self).__init__()


        self.project_m = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        self.project_dwm = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1, groups= output_dim,bias=bias)
        self.project_s = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        self.project_dws = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1, groups=output_dim,
                                     bias=bias)
        ###alter 3x3->1x1 5x5->7x7

        self.adavg_m = AdaptiveAvgPool2d(1)
        self.adavg_s = AdaptiveAvgPool2d(1)

        self.dwconv_m = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, groups=output_dim, bias=bias)
        self.dwconv_s = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, groups=output_dim, bias=bias)
        self.relum = nn.ReLU()
        self.relus = nn.ReLU()

        self.ag = nn.Conv2d(output_dim*2, output_dim, kernel_size=1, stride=1)

    def forward(self, x, y):
        '''
        Args:
            x: motion_features sp_attention
            y: sp_attention

        Returns:
        '''
        _, c, h, w = x.shape
        m = self.project_dwm(self.project_m(x))
        s = self.project_dws(self.project_s(y))
        mw = self.adavg_m(m)
        sw = self.adavg_s(s)
        m_sw = m.mul(sw.expand(-1,c,h,w)) + m
        s_mw = s.mul(mw.expand(-1,c,h,w)) + s
        dw_msw = self.relum(self.dwconv_m(m_sw))
        dw_smw = self.relus(self.dwconv_s(s_mw))
        fms = self.ag(torch.cat([dw_msw, dw_smw], dim=1))
        out = m + torch.sigmoid(fms) * s

        return out

