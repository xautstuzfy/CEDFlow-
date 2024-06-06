import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import imageio

from core.network import RAFTGMA

import datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/ours/{dstype}/'):
                os.makedirs(f'vis_test/ours/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


# @torch.no_grad()
# def validate_chairs(model, iters=6):
#     """ Perform evaluation on the FlyingChairs (test) split """
#     model.eval()
#     epe_list = []
#
#     val_dataset = datasets.FlyingChairs(split='validation')
#     for val_id in range(len(val_dataset)):
#         image1, image2, flow_gt, _ = val_dataset[val_id]
#         image1 = image1[None].cuda()
#         image2 = image2[None].cuda()
#
#         _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
#         epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
#         epe_list.append(epe.view(-1).numpy())
#
#     epe = np.mean(np.concatenate(epe_list))
#     print("Validation Chairs EPE: %f" % epe)
#     return {'chairs_epe': epe}

@torch.no_grad()
def validate_Canon(model, iters=24):
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.Canon(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation Canon EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    return {'canon': epe}

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    return {'chairs': epe}



@torch.no_grad()
def validate_things(model, iters=6):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def separate_inout_sintel_occ():
    """ Peform validation using the Sintel (train) split """
    dstype = 'clean'
    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
    # coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    # coords = torch.stack(coords[::-1], dim=0).float()
    # return coords[None].expand(batch, -1, -1, -1)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, occ, occ_path = val_dataset[val_id]
        _, h, w = image1.size()
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()

        coords_img_2 = coords + flow_gt
        out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
        occ_union = out_of_frame | occ
        in_frame = occ_union ^ out_of_frame

        # Generate union of occlusions and out of frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'occ_plus_out'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, occ_union.int().numpy() * 255)

        # Generate out-of-frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'out_of_frame'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, out_of_frame.int().numpy() * 255)

        # # Generate in-frame occlusions
        # path_list = occ_path.split('/')
        # path_list[-3] = 'in_frame_occ'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, in_frame.int().numpy() * 255)



@torch.no_grad()
def validate_kitti(model, iters=6):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}


# 41 Validation Canon EPE: 20.685, 1px: 0.450, 3px: 0.620, 5px: 0.667
#   Validation Canon EPE: 13.970, 1px: 0.587, 3px: 0.729, 5px: 0.781
#   Validation Chairs EPE: 0.983, 1px: 0.894, 3px: 0.962, 5px: 0.974

# 40 Validation Canon EPE: 20.726, 1px: 0.447, 3px: 0.621, 5px: 0.668
#    Validation Canon EPE: 13.868, 1px: 0.586, 3px: 0.730, 5px: 0.782

# 38 Validation Canon EPE: 20.565, 1px: 0.449, 3px: 0.621, 5px: 0.668
#    Validation Canon EPE: 13.938, 1px: 0.587, 3px: 0.729, 5px: 0.780
#    Validation Chairs EPE: 1.013, 1px: 0.889, 3px: 0.961, 5px: 0.973

# 45 Validation Canon EPE: 6.253, 1px: 0.528, 3px: 0.741, 5px: 0.817 VBOF
#    Validation Canon EPE: 4.388, 1 px: 0.654, 3 px: 0.792, 5 px: 0.870
#    Validation Chairs EPE: 1.145, 1px: 0.861, 3px: 0.952, 5px: 0.968


# decomposition-based 修证后的结果 #FCDNed 0.00025
#45W  Validation Chairs EPE: 0.956, 1px: 0.901, 3px: 0.964, 5px: 0.974 FCND
#45W  Validation Canon EPE: 20.069, 1px: 0.458, 3px: 0.586, 5px: 0.633 CANON
# Validation Canon EPE: 40.248, 1px: 0.265, 3px: 0.353, 5px: 0.369 FUji

# FCDNed-41.pth
# Validation Chairs EPE: 0.983, 1px: 0.894, 3px: 0.962, 5px: 0.974
# Validation Canon EPE: 19.536, 1px: 0.462, 3px: 0.590, 5px: 0.637 CANON
# Validation Canon EPE: 35.807, 1px: 0.301, 3px: 0.402, 5px: 0.419 FUJI
# Validation Canon EPE: 13.970, 1px: 0.587, 3px: 0.729, 5px: 0.781 FUJI2
# Validation Canon EPE: 25.856, 1px: 0.288, 3px: 0.468, 5px: 0.556 Nikon
# Validation Canon EPE: 24.237, 1px: 0.443, 3px: 0.555, 5px: 0.612 Nikon2
# Validation Canon EPE: 15.373, 1px: 0.289, 3px: 0.769, 5px: 0.799 Sony
# Validation Canon EPE: 30.028, 1px: 0.279, 3px: 0.491, 5px: 0.552 Sony2
# Validation Canon EPE: 19.325, 1px: 0.513, 3px: 0.719, 5px: 0.727 Sony3
# Validation Canon EPE: 20.685, 1px: 0.450, 3px: 0.620, 5px: 0.667 All

# FCDNed-38.pth
# Validation Canon EPE: 20.565, 1px: 0.449, 3px: 0.621, 5px: 0.668 All
# Validation Canon EPE: 20.074, 1px: 0.464, 3px: 0.591, 5px: 0.638 Canon
# Validation Canon EPE: 34.821, 1px: 0.302, 3px: 0.408, 5px: 0.426 Fuji
# Validation Canon EPE: 25.854, 1px: 0.284, 3px: 0.464, 5px: 0.551 Nikon
# Validation Canon EPE: 24.345, 1px: 0.440, 3px: 0.548, 5px: 0.606 Nikon2
# Validation Canon EPE: 15.451, 1px: 0.285, 3px: 0.765, 5px: 0.796 Sony
# Validation Canon EPE: 28.577, 1px: 0.278, 3px: 0.503, 5px: 0.564 Sony2
# Validation Canon EPE: 19.000, 1px: 0.514, 3px: 0.727, 5px: 0.736 Sony3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='D:\\CEDp\\checkpoints\\FCDNed-38.pth',help="restore checkpoint")
    parser.add_argument('--dataset',default='canon', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=4, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')
# Validation Canon EPE: 4.429, 1px: 0.653, 3px: 0.791, 5px: 0.870
    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()

    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model, warm_start=True)
    # create_sintel_submission_vis(model, warm_start=True)
    # create_kitti_submission(model)
    # create_kitti_submission_vis(model)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters)

        elif args.dataset == 'canon':
            validate_Canon(model.module, iters=args.iters)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, iters=args.iters)

        elif args.dataset == 'sintel_occ':
            validate_sintel_occ(model.module, iters=args.iters)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, iters=args.iters)
