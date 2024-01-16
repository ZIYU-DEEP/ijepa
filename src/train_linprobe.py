# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.cifar100 import make_cifar100

from src.helper import (
    load_checkpoint,
    load_checkpoint_prober,
    init_model,
    init_encoder,
    init_opt,
    init_opt_prober)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = int(args['data']['batch_size'])
    pin_mem = bool(args['data']['pin_mem'])
    num_workers = int(args['data']['num_workers'])
    root_path = str(args['data']['root_path'])
    image_folder = str(args['data']['image_folder'])
    crop_size = int(args['data']['crop_size'])

    # -- MASK
    patch_size = int(args['mask']['patch_size'])

    # -- OPTIMIZATION
    wd = float(args['optimization']['weight_decay'])
    momentum = float(args['optimization']['momentum'])
    nesterov = bool(args['optimization']['nesterov'])
    base_lr_value = float(args['optimization']['base_lr_value'])
    base_lr_batch_size = int(args['optimization']['base_lr_value'])
    milestones = list(args['optimization']['milestones'])
    gamma = float(args['optimization']['gamma'])
    base_epochs = int(args['optimization']['base_epochs'])
    num_epochs = base_epochs * int(base_batch_size / batch_size)

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # -- PROBE
    out_feat_keys = args['probe']['out_feat_keys']
    n_categories = args['probe']['n_categories']
    load_weights = args['probe']['load_weights']
    load_weights_path = args['probe']['load_weights_path']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    os.makedirs(folder, exist_ok=True)

    with open(dump, 'w+') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        print('pass!')
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss'),
                            ('%.5f', 'mask-A'),
                            ('%.5f', 'mask-B'),
                            ('%d', 'time (ms)'))

    # -- init model
    encoder = init_encoder(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name)

    transform_train = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_test = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    _, supervised_loader, supervised_sampler = make_cifar100(
            transform=transform_train,
            batch_size=batch_size,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            training=True,
            drop_last=True)
    ipe = len(supervised_loader)

    # ########################################
    # Below is the temp code
    # ########################################
    # SET THE ENCODER
    # -- set the model
    encoder = DistributedDataParallel(encoder)

    # -- load the weights
    if load_weights:
        encoder_weights = torch.load(load_weights_path,
                                     map_location='cpu')
        encoder.load_state_dict(encoder_weights['encoder'])

    # -- set it to eval mode
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # SET THE LINEAR PROBING MODEL
    # -- set the in_dim
    in_dim = encoder.module.embed_dim
    for key in out_feat_keys:
        if key.startswith('concatPOOL'):
            v = int(key.replace('concatPOOL', ''))
            in_dim = model.module.embed_dim * v

        if key.startswith('lastPOOL'):
            in_dim = model.module.embed_dim

    # -- set the model
    prober = torch.nn.Linear(in_dim, n_categories).to(device)

    # SET THE OPTIMIZER, SCHEDULER
    # ############### TO MODIFY ####################################
    optimizer, scaler, scheduler, wd_scheduler = init_opt_prober(
        prober=prober,
        weight_decay=wd,
        momentum=momentum,
        nesterov=nesterov,
        batch_size=batch_size,
        base_lr_value=base_lr_value,
        base_lr_batch_size=base_lr_batch_size,
        current_batch_size=current_batch_size,
        milestones=milestones,
        gamma=gamma)

    prober = DistributedDataParallel(prober, static_graph=True)

    # -- load training checkpoint
    if load_model:
        prober, optimizer, scaler, start_epoch = load_checkpoint_prober(
            device=device,
            r_path=load_path,
            prober=prober,
            opt=optimizer,
            scaler=scaler)

        for _ in range(start_epoch*ipe):
            scheduler.step()
            if wd_scheduler: wd_scheduler.step()

    def ckpoint(epoch):
        save_dict = {
            'prober': prober.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        supervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        acc_meter = AverageMeter()

        for itr, (x, y) in enumerate(supervised_loader):

            # ###############################
            x = x.to(device, non_blocking=true)
            y = y.to(device, non_blocking=true)


            #################################

            def train_step():
                # Set lr and weight decay
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step() if wd_scheduler else wd
                # --

                def get_feature():
                    with torch.no_grad():
                        features = encoder(x,
                            out_feat_keys=out_feat_keys)[0].reshape(- 1, in_dim)
                    return features

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    features = get_feature()
                    optimizer.zero_grad()
                    logits = prober(features)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    acc = (logits.argmax(dim=1) == y).float().mean()

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = None
                optimizer.zero_grad()

                return (float(loss), float(acc), _new_lr, _new_wd, grad_stats)

            (loss, acc, _new_lr, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            acc_meter.update(acc)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, acc, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[acc: %.2e] '
                                '[wd: %.2e] '
                                '[lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   acc_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.** 2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        ckpoint(epoch+1)

if __name__ == "__main__":
    main()
