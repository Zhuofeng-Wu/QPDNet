from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.qpd_net import QPDNet

from evaluate_quad import *
import core.Quad_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    b,c,h,w = flow_gt.shape
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

        fp = flow_preds[i]
        i_loss = (fp-(flow_gt/2)).abs()

        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()
    
    fp = flow_preds[-1]
    epe = torch.sum((fp - (flow_gt)/2)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model, last_epoch=-1):
    """ Create the optimizer and learning rate scheduler """
    if last_epoch == -1:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    else:
        max_lr = args.lr
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': max_lr, 'max_lr': args.lr, 
                                  'min_lr': 1e-8}], lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps = args.num_steps+100,
                pct_start=0.01, cycle_momentum=False, anneal_strategy='linear', last_epoch=last_epoch)

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, total_steps):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='result/runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='result/runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='result/runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(QPDNet(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    
    total_steps = 0
    optimizer, scheduler = fetch_optimizer(args, model, total_steps-1)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        total_steps = int((args.restore_ckpt).split('\\')[-1].split("_")[2])+1
        checkpoint = torch.load(args.restore_ckpt)
        # model.load_state_dict(checkpoint, strict=True)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=True)
            optimizer, scheduler = fetch_optimizer(args, model, total_steps-1)
        logging.info(f"Done loading checkpoint")
        
    
    logger = Logger(model, scheduler, total_steps)

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = total_steps
    batch_len = len(train_loader)
    epoch = int(total_steps/batch_len)


    epebest,rmsebest = 1000,1000
    epeepoch,rmseepoch = 0,0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            center_img, lrtblist, flow, valid = [x.cuda() for x in data_blob]

            b,s,c,h,w = lrtblist.shape

            image1 = center_img.contiguous().view(b,c,h,w)
            if args.input_image_num == 4:
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1],lrtblist[:,2],lrtblist[:,3]], dim=0).contiguous()
            else:
                image2 = torch.cat([lrtblist[:,0],lrtblist[:,1]], dim=0).contiguous()

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training
            if args.input_image_num == 42:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=-1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid)
            elif args.input_image_num == 24:
                rot_flow_predictions=[]
                for i in range(len(flow_predictions)):
                    rot_flow_predictions.append(torch.rot90(flow_predictions[i], k=1, dims=[2,3]))   
                loss, metrics = sequence_loss(rot_flow_predictions, flow, valid)
            else:
                loss, metrics = sequence_loss(flow_predictions, flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % batch_len == 0:# and total_steps != 0:
                epoch = int(total_steps/batch_len)
                save_path = Path('result/checkpoints/%d_epoch_%d_%s.pth' % (epoch, total_steps + 1, args.name))
                print('checkpoints/%d_epoch_%d_%s' % (epoch, total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            # ... any other states you need
                            }, save_path)
                
                if total_steps % (batch_len*1) == 0:
                    results = validate_QPD(model.module, iters=args.valid_iters, save_result=True, val_save_skip=30, input_image_num=args.input_image_num, image_set='validation', path=args.datasets_path)

                if epebest>=results['things-epe']:
                    epebest = results['things-epe']
                    epeepoch = epoch
                if rmsebest>=results['things-rmse']:
                    rmsebest = results['things-rmse']
                    rmseepoch = epoch
                logging.info(f"Current Best Result epe epoch {epeepoch}, result: {epebest}")
                logging.info(f"Current Best Result rmse epoch {rmseepoch}, result: {rmsebest}")

                logger.write_dict(results)

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path('result/checkpoints/%d_epoch_%d_%s.pth.gz' % (epoch, total_steps + 1, args.name))
            print()
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)


    print("FINISHED TRAINING")
    logger.close()
    PATH = 'result/checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='QPD-Net', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['QPD'], help="training datasets.")
    parser.add_argument('--datasets_path', default='dd_dp_dataset_hypersim_377\\', help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--input_image_num', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[452, 452], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=8, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    parser.add_argument('--CAPA', default=True, help="if use Channel wise and pixel wise attention")
    

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("result/checkpoints").mkdir(exist_ok=True, parents=True)
    Path("result/predictions").mkdir(exist_ok=True, parents=True)

    train(args)