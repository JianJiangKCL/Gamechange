import os
import sys
import argparse
import json
import time
import math
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from timm.optim import create_optimizer
from timm.utils.metrics import AverageMeter, accuracy

from logger import config_logging
from data import create_dataset
import models as models
from utils import CofStepController, set_seed


def train_epoch(epoch, model, criterion, loader, optimizer, scaler, autocast, device, diff_cof=0.5, max_norm=0):
    cls_loss_meter, weight_diff_meter, feature_diff_meter, loss_meter, acc_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loader = tqdm(loader)
    model.train()
    for images, targets in loader:
        batch_size = targets.shape[0]
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast:
            outputs = model(images)
            cls_loss = criterion(outputs, targets)
            weight_diff = model.accumlate_diff_weight()
            feature_diff = model.accumlate_diff_feature()
            loss = cls_loss + (weight_diff + feature_diff) * diff_cof
            
        acc = accuracy(outputs, targets, topk=(1,))[0]
        
        # if not math.isfinite(loss.item()):
        #     #print(cls_loss.item(), )
        #     #print("Loss is {}, stopping training".format(loss.item()))
        #     sys.exit(1)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        model.update_quantizer()
        
        cls_loss_meter.update(cls_loss.item(), batch_size)
        weight_diff_meter.update(weight_diff)
        feature_diff_meter.update(feature_diff, batch_size)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
                
        loader.set_description(f'Epoch: {epoch+1}  CELoss:{cls_loss_meter.avg:.5f}  w_diff: {weight_diff_meter.avg:.5f}  '
                               f'f_diff: {feature_diff_meter.avg:.5f}  Total: {loss_meter.avg:.5f}  Acc: {acc_meter.avg:.5f}')
        
    logging.info(f'====> Train Epoch {epoch+1}: Train_acc {acc_meter.avg:.5f}')
        
        
@torch.no_grad()
def test_epoch(epoch, model, loader, device):
    acc_meter, weight_diff_meter = AverageMeter(), AverageMeter()
    loader = tqdm(loader)
    model.eval()
    for images, targets in loader:
        batch_size = targets.shape[0]
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        acc = accuracy(outputs, targets, topk=(1,))[0]
        acc_meter.update(acc, batch_size)
        weight_diff = model.accumlate_diff_weight()
        weight_diff_meter.update(weight_diff)
    
        loader.set_description(f'Epoch: {epoch+1}  w_diff: {weight_diff_meter.avg:.5f}  Acc: {acc_meter.avg:.5f}')
    logging.info(f'====> Test Epoch {epoch+1}: Test_acc {acc_meter.avg:.5f}')


def main(args):
    result_path = os.path.join(args.root, args.result_dir, args.tag)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'args.json'), 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
    ckpt_path = os.path.join(result_path, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    log_file = os.path.join(result_path, 'log.txt')
    config_logging(log_file)
    logging.info('====>  args{} '.format(args))
    
    train_dataset, num_classes, in_channels = create_dataset(args.dataset, args.dataset_path, train=True)
    test_dataset, _, _ = create_dataset(args.dataset, args.dataset_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    model = getattr(models, args.model)(in_channels, args.n_dw_emb, args.n_pw_emb, args.n_f_emb, num_classes, gs=1, out_planes=args.out_planes).to(args.device)
    inpt = torch.randn((1, in_channels, 28, 28)).to(args.device)
    model(inpt)
    model.zero_buffer()
    del inpt
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    cof_controller = CofStepController(args.diff_cof, args.cof_gamma, milestones=[20, 100])
    
    autocast = torch.cuda.amp.autocast(enabled=args.amp)
    scaler = GradScaler(enabled=args.amp)
    
    #print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_epoch(epoch, model, criterion, train_loader, optimizer, scaler, autocast, args.device, cof_controller.cof, args.clip_grad)
        test_epoch(epoch, model, test_loader, args.device)
        
        cof_controller.step()
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(ckpt_path, 'epoch_'+str(epoch)+'.pth'))        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--root", default='./', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--dataset_path", default="../datasets/mnist", type=str)
    parser.add_argument("--result_dir", default='results', type=str)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--tag", default='default', type=str)
    
    parser.add_argument("--model", default='QuantNet9', type=str)
    parser.add_argument("--out_planes", default=64, type=int)
    parser.add_argument("--n_dw_emb", default=100, type=int)
    parser.add_argument("--n_pw_emb", default=100, type=int)
    parser.add_argument("--n_f_emb", default=100, type=int)
    parser.add_argument("--diff-cof", default=0.5, type=float)
    parser.add_argument("--cof-gamma", default=5., type=float)
    parser.add_argument("--device", default='cuda', type=str)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--clip-grad', type=float, default=0., metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    
    # Setting for Automatic Mixed Precision
    parser.add_argument('--amp', action='store_true',
                        help='Turn on Automatic Mixed Precision')
    parser.add_argument('--disable-amp', action='store_false', dest='amp',
                        help='turn off Automatic Mixed Precision')
    parser.set_defaults(amp=True)

    args = parser.parse_args()
    set_seed(args.seed, args.device)
    main(args)