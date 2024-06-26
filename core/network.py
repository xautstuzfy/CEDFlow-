import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decomposition import DecomEncoder
from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from gma import Sp_Attention

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.num_heads = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim, output_dim=cdim)
        self.att = Sp_Attention(dim=cdim, num_heads=self.num_heads, bias=False)
        self.decomposition = DecomEncoder(in_dim=hdim + cdim, out_dim=hdim + cdim, num_heads=self.num_heads, bias=False)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        #####SDSD 操作#########
        image1 = image1[:, 0:3, :, :]
        image2 = image2[:, 0:3, :, :]
        ####################################

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        Lfmap1, SpHfmap1 = self.decomposition(fmap1)
        Lfmap2, SpHfmap2 = self.decomposition(fmap2)

        Lfmap1 = Lfmap1.float()
        Lfmap2 = Lfmap2.float()

        SpHfmap1 = SpHfmap1.float()
        SpHfmap2 = SpHfmap2.float()
        '''
        fmap1 torch.Size([4, 256, 46, 62])
        fmap2 torch.Size([4, 256, 46, 62])
        
        '''
        Lcorr_fn = CorrBlock(Lfmap1, Lfmap2, radius=self.args.corr_radius)

        SHcorr_fn = CorrBlock(SpHfmap1, SpHfmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            sp_attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            Lcorr = Lcorr_fn(coords1)  # index correlation volume

            SHcorr = SHcorr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, Lcorr, SHcorr, flow, sp_attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
