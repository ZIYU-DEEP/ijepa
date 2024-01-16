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
    init_model,
    init_opt)
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
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # -- PROBE
    out_feat_keys = args['probe']['out_feat_keys']
    n_categories = args['probe']['n_categories']
    load_weights = args['probe']['load_weights']
    load_weights_path = args['probe']['load_weights_path']

    dump = os.path.join(folder, 'params-ijepa.yaml')
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
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
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
    # TO MODIFY THIS TO MATCH THE SETTING OF VISSL
    # optimizer, scaler, scheduler, wd_scheduler = init_opt(
    #     encoder=encoder,
    #     predictor=predictor,
    #     wd=wd,
    #     final_wd=final_wd,
    #     start_lr=start_lr,
    #     ref_lr=lr,
    #     final_lr=final_lr,
    #     iterations_per_epoch=ipe,
    #     warmup=warmup,
    #     num_epochs=num_epochs,
    #     ipe_scale=ipe_scale,
    #     use_bfloat16=use_bfloat16)

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
    optimizer, scaler, scheduler, wd_scheduler = init_opt_linprobe()
    prober = DistributedDataParallel(prober, static_graph=True)

    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
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
    # ############### TO MODIFY ####################################

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
                _new_wd = wd_scheduler.step()
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

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            acc_meter.update(acc)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, acc, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[acc: %.2e] '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   acc_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
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
        save_checkpoint(epoch+1)

if __name__ == "__main__":
    main()
