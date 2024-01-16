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
    load_encoder_weights = args['meta']['load_encoder_weights']
    load_encoder_weights_path = args['meta']['load_encoder_weights_path']
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
    base_lr_batch_size = int(args['optimization']['base_lr_batch_size'])
    milestones = list(args['optimization']['milestones'])
    gamma = float(args['optimization']['gamma'])
    base_epochs = int(args['optimization']['base_epochs'])
    num_epochs = base_epochs * int(base_lr_batch_size / batch_size)

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # -- PROBE
    out_feat_keys = args['probe']['out_feat_keys']
    n_categories = args['probe']['n_categories']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    os.makedirs(folder, exist_ok=True)

    with open(dump, 'w+') as f:
        yaml.dump(args, f)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
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
    if load_encoder_weights:
        logger.info(f'Loading weights from {load_encoder_weights_path}')
        encoder_weights = torch.load(load_encoder_weights_path,
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
            in_dim = encoder.module.embed_dim * v

        if key.startswith('lastPOOL'):
            in_dim = encoder.module.embed_dim

        if key.startswith('direct'):
            in_dim = encoder.module.embed_dim

    # -- set the model
    prober = torch.nn.Linear(in_dim, n_categories).to(device)

    # SET THE OPTIMIZER, SCHEDULER
    # ############### TO MODIFY ####################################
    optimizer = torch.optim.AdamW(prober.parameters(),
                                  lr=1e-3,
                                  weight_decay=1e-4,
                                  amsgrad=True)

    prober = DistributedDataParallel(prober, static_graph=True)

    encoder.eval()
    for epoch in range(num_epochs):
        epoch_scores = []

        for x, y in tqdm(supervised_loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                features = encoder(x)

            optimizer.zero_grad()
            logits = prober(features)
            loss = torch.nn.functional.cross_entropy(logits, y)
            top1_acc = (logits.argmax(dim=1) == y).float().mean()
            epoch_scores.append(top1_acc.item())
            print(np.mean(epoch_scores))
            loss.backward()
            optimizer.step()

        print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")


if __name__ == "__main__":
    main()

# def linear_probe_test(model,
#                       data_loader,
#                       device,
#                       n_categories=100,
#                       probe_lr=1e-3,
#                       probe_weight_decay=1e-4,
#                       val_epochs=5,
#                       batch_size=8):
#     model.eval()
#
#     # Use the embed_dim as the input dimension for the linear layer
#     linear_probe = torch.nn.Linear(model.module.embed_dim, n_categories).to(device)
#
#     optimizer = torch.optim.AdamW(linear_probe.parameters(),
#                                   lr=probe_lr,
#                                   weight_decay=probe_weight_decay,
#                                   amsgrad=True)
#
#     for epoch in range(val_epochs):
#         epoch_scores = []
#
#         for x, y in tqdm(data_loader):
#             x = x.to(device)
#             y = y.to(device)
#
#             with torch.no_grad():
#                 # Forward pass through the model and get the output from the last layer
#                 # Assuming the output is in the form (batch_size, num_patches, embed_dim)
#                 features = model(x)
#                 # Take the mean over the num_patches dimension if necessary
#                 if features.dim() == 3:
#                     features = features.mean(dim=1)
#
#             optimizer.zero_grad()
#             logits = linear_probe(features)
#
#             loss = torch.nn.functional.cross_entropy(logits, y)
#             top1_acc = (logits.argmax(dim=1) == y).float().mean()
#             epoch_scores.append(top1_acc.item())
#             print(np.mean(epoch_scores))
#
#             loss.backward()
#             optimizer.step()
#
#         print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")
#
#     model.train()
#
#     return model
