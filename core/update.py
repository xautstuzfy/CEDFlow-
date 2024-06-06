import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import MaxPool2d, MaxUnpool2d

from gma import FFN
from torch import Tensor, sigmoid

class DwResidualBlock(nn.Module):
    def __init__(self,in_dim, out_dim, bias ):
        super(DwResidualBlock,self).__init__()

        self.dwcon1 = nn.Conv2d(in_dim // 2, out_dim , kernel_size=1, bias=bias)
        self.linear = nn.Linear(out_dim , out_dim)
        self.dwcon3 = nn.Conv2d(out_dim, out_dim // 2, kernel_size=1, bias=bias)
        self.relu1 = nn.ReLU6(inplace=True)
        self.relu2 = nn.ReLU6(inplace=True)
        self.relu3 = nn.ReLU6(inplace=True)

    def forward(self, x):

        input = x
        input = self.relu1(self.dwcon1(input))
        input = input.permute(0,2,3,1)
        input = self.relu2(self.linear(input))
        # print(input.shape) torch.Size([4, 256, 46, 62])
        input = input.permute(0,3,1,2)
        input = self.relu3(self.dwcon3(input))
        output = input + x
        return output

class Compensation(nn.Module):
    def __init__(self,in_dim, out_dim, bias ):
        super(Compensation, self).__init__()

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.maxselect = nn.MaxUnpool2d(2)

        self.dwr1 = DwResidualBlock(in_dim, out_dim, bias)

        self.dwr2 = DwResidualBlock(in_dim, out_dim, bias)

        self.tem_factor = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)

        self.project_out = nn.Conv2d(in_dim, 192, 3, padding=1)

    def forward(self, Lcor, Hcor):

        TopBrachL, BottomBrachL = Lcor.chunk(2, dim=1)

        TopBrachH, BottomBrachH = Hcor.chunk(2, dim=1)


        BottomBrachL = self.dwr1(BottomBrachL)
        TopBrachL    = self.dwr2(TopBrachL)

        Hmax, max_index = self.pool(TopBrachH)

        max_TopBrachH = self.maxselect(Hmax, max_index)

        L = torch.cat([TopBrachL, max_TopBrachH], dim=1)
        H = torch.cat([BottomBrachL, BottomBrachH], dim=1)


        corr = (L + H) * self.tem_factor
        corr = self.project_out(corr)

        return corr

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2

        self.lconvc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)

        self.hconvc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)

        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        # 256 126
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

        self.compensationblock = Compensation(256, 256, bias=False)

    def forward(self, flow, Lcorr, Hcorr):


        Lcor = F.relu(self.lconvc1(Lcorr))

        Hcor = F.relu(self.hconvc1(Hcorr))

        cor = self.compensationblock(Lcor, Hcor)

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128): # hid=128, inp = 256
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h



class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim) ### alter here
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, output_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)  #hid=128, inp = 256
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        # self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

        self.dualffn = FFN(input_dim=hidden_dim, bias=False)

    def forward(self, net, inp, Lcorr, SHcorr, flow, sp_attention):

        "net, inp, Lcorr, SHcorr, flow, sp_attention"

        mf = self.encoder(flow, Lcorr, SHcorr)


        # mf dimesion 128
        ms = self.dualffn(mf, inp)

        inp_cat = torch.cat([ms, sp_attention], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat) #128 256

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



