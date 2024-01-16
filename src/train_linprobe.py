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

# ----------------------------------------------------------------------- #
#  PASSED IN PARAMS FROM CONFIG FILE
# ----------------------------------------------------------------------- #

# -- META
use_bfloat16 = args['meta']['use_bfloat16']
model_name = args['meta']['model_name']
load_model = args['meta']['load_checkpoint'] or resume_preempt
r_file = args['meta']['read_checkpoint']
copy_data = args['meta']['copy_data']
pred_depth = args['meta']['pred_depth']
pred_emb_dim = args['meta']['pred_emb_dim']
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

# -- DATA
use_gaussian_blur = args['data']['use_gaussian_blur']
use_horizontal_flip = args['data']['use_horizontal_flip']
use_color_distortion = args['data']['use_color_distortion']
color_jitter = args['data']['color_jitter_strength']
# --
batch_size = args['data']['batch_size']
pin_mem = args['data']['pin_mem']
num_workers = args['data']['num_workers']
root_path = args['data']['root_path']
image_folder = args['data']['image_folder']
crop_size = args['data']['crop_size']
crop_scale = args['data']['crop_scale']
# --

# -- MASK
allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
patch_size = args['mask']['patch_size']  # patch-size for model training
num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
min_keep = args['mask']['min_keep']  # min number of patches in context block
enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
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

import torch
import numpy as np
from tqdm import tqdm

def linear_probe_test(model,
                      data_loader, 
                      device, 
                      n_categories=100,
                      probe_lr=1e-3, 
                      probe_weight_decay=1e-4, 
                      val_epochs=1, 
                      out_feat_keys=['lastPOOL']):
    
    # Set the eval model
    for _, p in model.named_parameters():
        p.requires_grad = False
    
    model.eval()

    # Use the embed_dim as the input dimension for the linear layer
    # Set the correct size 
    in_dim = model.module.embed_dim
    for key in out_feat_keys:
        if key.startswith('concatPOOL'):
            v = int(key.replace('concatPOOL', ''))
            in_dim = model.module.embed_dim * v

        if key.startswith('lastPOOL'):
            in_dim = model.module.embed_dim

    linear_probe = torch.nn.Linear(in_dim, n_categories).to(device)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), 
                                  lr=probe_lr, 
                                  weight_decay=probe_weight_decay, 
                                  amsgrad=True)

    for epoch in range(val_epochs):
        epoch_scores = []

        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # Forward pass through the model and get the output from the last layer
                # Assuming the output is in the form (batch_size, num_patches, embed_dim)
                features = model(x, 
                                 out_feat_keys=out_feat_keys)[0].reshape(-1, in_dim)
                # Take the mean over the num_patches dimension if necessary
                if features.dim() == 3:
                    features = features.mean(dim=1)

            optimizer.zero_grad()
            logits = linear_probe(features)

            loss = torch.nn.functional.cross_entropy(logits, y)
            top1_acc = (logits.argmax(dim=1) == y).float().mean()
            epoch_scores.append(top1_acc.item())
            print(np.mean(epoch_scores))

            loss.backward()
            optimizer.step()

        print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")

    return model
