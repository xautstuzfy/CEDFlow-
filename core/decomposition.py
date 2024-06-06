import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from torch import sigmoid
from torch.nn import AdaptiveAvgPool2d, MaxPool2d, MaxUnpool2d, Sequential, Linear, GELU


class Decomposition(nn.Module):
    def __init__(self, in_dim=256, out_dim=256):
        super(Decomposition, self).__init__()

        self.D1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1)
        self.D3 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):

        x = x.to(torch.float32)

        Matone = torch.ones_like(x)
        HF_Weight = torch.sigmoid((self.D1(x) - self.D3(x)))
        LF = (Matone - HF_Weight) * x
        HF = LF * x
        return LF, HF


class AdaptiveHF(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, bias):
        super(AdaptiveHF,self).__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dw_conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias)

        self.dw_conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias)

        self.epe = nn.Conv2d(in_dim, out_dim, kernel_size=1, groups=out_dim, bias=bias)

        self.project_out = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)


    def forward(self, x, y):

        # x: HF, y : LF

        b, c, h, w = x.shape

        # print("x", x.shape) x torch.Size([4, 256, 46, 62]) head = 4

        q = self.dw_conv1(x)
        k = self.dw_conv2(x)
        lepe = self.epe(x)

        input_y = (torch.sigmoid(y) * x ) + y

        qx = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kx = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vx = rearrange(input_y, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        qx = torch.nn.functional.normalize(qx, dim=-1)
        kx = torch.nn.functional.normalize(kx, dim=-1)

        _, _, C, _ = qx.shape

        # print("c", C)   c -> 64

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)


        attn = (qx @ kx.transpose(-2, -1)) * self.temperature
        #print("attn", attn.shape) attn torch.Size([4, 4, 64, 64])

        # C/2 32
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        # C*2/3
        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        out1 = (attn1 @ vx)
        out2 = (attn2 @ vx)

        out = out1 * self.attn1 + out2 * self.attn2

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        sparseHF = self.project_out( out + lepe)


        return sparseHF


class DwRefineBlock(nn.Module):
    def __init__(self,in_dim, bias ):
        super(DwRefineBlock,self).__init__()

        self.con1 = nn.Conv2d(in_dim , 384 , kernel_size=1, bias=bias)
        self.dwconv2 = nn.Conv2d(384 , 384, kernel_size=3, padding=1,groups=384)
        self.con3 = nn.Conv2d(384, in_dim , kernel_size=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):

        input = x
        input = self.relu1(self.con1(input))
        input = self.relu2(self.dwconv2(input))
        input = self.relu3(self.con3(input))
        output = input + x

        return output


class DecomEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, bias=False):
        super(DecomEncoder,self).__init__()

        self.decomposition = Decomposition(in_dim, out_dim)

        self.adaptiveHF  = AdaptiveHF(in_dim, out_dim, num_heads, bias)


        self.refine = DwRefineBlock(in_dim,bias)

    def forward(self, x):


        LF, HF = self.decomposition(x)

        refine_LF = self.refine(self.refine(LF))

        sparseHF = self.adaptiveHF(HF, LF)


        return  refine_LF, sparseHF
