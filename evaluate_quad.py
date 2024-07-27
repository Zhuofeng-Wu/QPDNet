from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.qpd_net import QPDNet, autocast
import core.Quad_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import os.path as osp
import os
import cv2
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_colormap(depth_range, dpi):
    ##setting for colormap
    diff = depth_range[1] - depth_range[0]
    cm = plt.get_cmap('jet', diff * dpi)
    delta = diff / cm.N
    value = np.arange(depth_range[0], depth_range[1], delta)
    norm = BoundaryNorm(value, ncolors=cm.N)
    norm.clip = False
    cm.set_under('gray')
    return cm, norm

def show_colormap(value, path, depth_range, dpi, figsize=(12, 10)):
    ##color map setting
    cm, norm = set_colormap(depth_range, dpi)

    ##plot color map
    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.imshow(value, cmap=cm, norm=norm)
    plt.colorbar(orientation='vertical')

    ##show or save map
    if (len(path) > 0):
        folder = osp.dirname(path)
        if not osp.exists(folder):
            os.makedirs(folder)
        plt.savefig(path)
    else:
        plt.show()

    ##close plot
    plt.clf()

@torch.no_grad()
def validate_QPD(model, input_image_num, iters=32, mixed_prec=False, save_result=False, val_num=None, val_save_skip=1,image_set='test', path=''):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    aug_params = {}
    
    if path == '':
        val_dataset = datasets.QPD(aug_params, image_set=image_set)
    else:
        val_dataset = datasets.QPD(aug_params, image_set=image_set, root=path)

    out0_1_list, out0_5_list, out1_list, out2_list, out4_list, epe_list, rmse_list = [], [], [], [], [], [], []
    if val_num==None:
        val_num = len(val_dataset)

    path = os.path.basename(os.path.dirname(path))

    for val_id in tqdm(range(val_num)):

        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        ## 4 LRTB,  2 LR    
        if input_image_num == 4:
            image2 = image2.squeeze()
        else:
            image2 = image2.squeeze()[:2]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        
        if flow_pr.shape[0]==2:
            flow_pr = flow_pr[1]-flow_pr[0]
        flow_gt = flow_gt/2

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        if save_result and val_id%val_save_skip==0:
            if not os.path.exists('result\\predictions\\'+path+'\\'):
                os.makedirs('result\\predictions\\'+path+'\\')
            
            flow_gtn = flow_gt.cpu().numpy().squeeze()
            flow_range = flow_gtn.max()-flow_gtn.min()
            flow_max = flow_gtn.max()+flow_range*0.2
            flow_min = flow_gtn.min()-flow_range*0.2
            flow_prn = flow_pr.cpu().numpy().squeeze()

            np.save('result\\predictions\\'+path+'\\'+ str(val_id)+".npy", flow_prn)
            np.save('result\\predictions\\'+path+'\\'+ str(val_id)+"-gt.npy", flow_gtn)
            show_colormap(flow_prn, 'result\\predictions\\'+path+'\\'+ str(val_id)+".png", [flow_min, flow_max], 200, (12,10))
            show_colormap(flow_gtn, 'result\\predictions\\'+path+'\\'+ str(val_id)+"-gt.png", [flow_min, flow_max], 200, (12,10))
            show_colormap(np.abs(flow_gtn-flow_prn), 'result\\predictions\\'+path+'\\'+ str(val_id)+"-error.png", [0, 0.2], 200, (12,10))
        
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        rmse = torch.sum((flow_pr - flow_gt)**2, dim=0)
        epe = epe.flatten()
        rmse = rmse.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        epe_list.append(epe[val].mean().item())
        rmse_list.append(rmse[val].mean().item())
        out0_1 = (epe > 0.1)
        out0_1_list.append(out0_1[val].cpu().numpy())
        out0_5 = (epe > 0.5)
        out0_5_list.append(out0_5[val].cpu().numpy())
        out1 = (epe > 1.0)
        out1_list.append(out1[val].cpu().numpy())
        out2 = (epe > 2.0)
        out2_list.append(out2[val].cpu().numpy())
        out4 = (epe > 4.0)
        out4_list.append(out4[val].cpu().numpy())

    epe_list = np.array(epe_list)
    rmse_list = np.array(rmse_list)
    out1_list = np.concatenate(out1_list)

    epe = np.mean(epe_list)
    rmse = np.sqrt(np.mean(rmse_list))
    d01 = 100 * np.mean(out0_1_list)
    d05 = 100 * np.mean(out0_5_list)
    d1 = 100 * np.mean(out1_list)
    d2 = 100 * np.mean(out2_list)
    d4 = 100 * np.mean(out4_list)

    print("#######################: epe, rmse, d0.1, d0.5, d1, d2, d4")
    print("Validation FlyingThings: %f, %f, %f, %f, %f, %f, %f" % (epe, rmse, d01, d05, d1, d2, d4))
    return {'things-epe': epe, 'things-rmse': rmse, 'things-d0.1': d01, 'things-d0.5': d05, 'things-d1': d1, 'things-d2': d2, 'things-d4': d4}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=False, choices=["QPD"], default="QPD")
    parser.add_argument('--datasets_path', default='dd_dp_dataset_hypersim_377\\', help="test datasets.")    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--input_image_num', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--CAPA', default=True, help="if use Channel wise and pixel wise attention")

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--save_path', default='result/predictions/')
    parser.add_argument('--save_result', default='False')
    args = parser.parse_args()

    args.save_result = args.save_result == str(True)

    model = torch.nn.DataParallel(QPDNet(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        # model.load_state_dict(checkpoint, strict=True)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
        

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'QPD':
        validate_QPD(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, save_result=args.save_result, input_image_num = args.input_image_num, image_set="test", path=args.datasets_path)
