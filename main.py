import os
import argparse
import json
import time
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from timm.optim import create_optimizer
from timm.utils.metrics import AverageMeter, accuracy

from logger import config_logging
from data import create_dataset
from model import AqResnet as models
from utils_funcs import set_seed

from model.funcs import create_backbone
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from model.AqMethod import AqMethod
from args.setup import parse_args


def main(local_rank, args):
    result_path = args.results_dir
    os.makedirs(result_path, exist_ok=True)

    logging.info('====>  args{} '.format(args))
    
    train_dataset, num_classes, in_channels = create_dataset(args.dataset, args.dataset_path, train=True)
    test_dataset, _, _ = create_dataset(args.dataset, args.dataset_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    backbone = create_backbone(args)

    wandb_logger = WandbLogger(
        name=f"{args.name}",
        project=args.project,
        offline=True if args.wandb_mode == 'offline' else False,
        reinit=True,
        save_dir=result_path,
        config=args,
    )
    wandb_logger.log_hyperparams(args)
    kwargs = {'logger': wandb_logger, 'enable_checkpointing': False, 'detect_anomaly': False, 'max_epochs': args.epoch}
    if args.use_amp:
        kwargs['precision'] = 16
        kwargs['amp_backend'] = 'native'
    if args.disable_tqdm:
        kwargs['enable_progress_bar'] = False
    if args.use_ddp:
        kwargs['gpus'] = 4
        kwargs['strategy'] = 'ddp'
    trainer = Trainer.from_argparse_args(args, **kwargs)
    model = AqMethod(args, backbone)
    if args.method == 'normal':
        model.disable_quantizer()
    trainer.fit(model, train_loader)

    trainer.test(model, test_loader)
    wandb.finish()




if __name__ == "__main__":
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    # pl.seed_everything(args.seed)
    print(args)

    main(0, args)