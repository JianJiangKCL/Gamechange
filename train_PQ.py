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
from models import AqResnet as models
from utils import CofStepController, set_seed
from models.AqResnet import FeatureQuantizer, QuantConv_DW, QuantConv_PW


def train_epoch(epochs, model, criterion, loader, optimizer, scaler, autocast, device, diff_cof=0.5, max_norm=0):
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
            # weight_diff = torch.Tensor([0])
            feature_diff = model.accumlate_diff_feature()
            loss = cls_loss + (weight_diff + feature_diff) * diff_cof
            # loss = cls_loss + feature_diff * diff_cof
        acc = accuracy(outputs, targets, topk=(1,))[0]
        
        # if not math.isfinite(loss.item()):
        #     #print(cls_loss.item(), )
        #     #print("Loss is {}, stopping training".format(loss.item()))
        #     sys.exit(1)
            
        scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # if max_norm > 0.:
        #     torch.nn.utils.clip_grad_norm_(models.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        model.update_quantizer()

        #confirmed: even w/o meters, memory still leak
        cls_loss_meter.update(cls_loss.item(), batch_size)
        weight_diff_meter.update(weight_diff.item(), batch_size)
        feature_diff_meter.update(feature_diff if isinstance(feature_diff, int) else feature_diff.item(), batch_size)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        #
        description = (f'mem: {torch.cuda.max_memory_allocated()/1024/1024/1024:.2f} '
                               f'epochs: {epochs+1}  CELoss:{cls_loss_meter.avg:.3f}  w_diff: {weight_diff_meter.avg:.3f} '
                               f'f_diff: {feature_diff_meter.avg:.3f}  Total: {loss_meter.avg:.3f}  Acc: {acc_meter.avg:.3f}')
        loader.set_description(description)
    logging.info(f"===>Train epochs {epochs+1}" + description)
        
        
@torch.no_grad()
def test_epoch(epochs, model, loader, device):
    acc_meter, acc_meter2, weight_diff_meter = AverageMeter(), AverageMeter(), AverageMeter()
    loader = tqdm(loader)
    model.eval()
    for images, targets in loader:
        batch_size = targets.shape[0]
        images, targets = images.to(device), targets.to(device)
        outputs, outputs2 = model(images)
        acc = accuracy(outputs, targets, topk=(1,))[0]
        acc_meter.update(acc.item(), batch_size)
        acc2 = accuracy(outputs2, targets, topk=(1,))[0]
        acc_meter2.update(acc2.item(), batch_size)
        weight_diff = model.accumlate_diff_weight()
        weight_diff_meter.update(weight_diff.item(), batch_size)
        # break
        loader.set_description(f'epochs: {epochs+1}  w_diff: {weight_diff_meter.avg:.3f}  Acc: {acc_meter.avg:.3f} Acc2: {acc_meter2.avg:.3f}')
    logging.info(f'====> Test epochs {epochs+1}: Test_acc {acc_meter.avg:.3f}')
    return acc_meter.avg

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
    
    model = getattr(models, args.model)(in_channels, args.n_dw_emb, args.n_pw_emb, args.n_f_emb, num_classes, gs=1,  layers=args.layers).to(args.device)


    # inpt = torch.randn((1, in_channels, 28, 28)).to(args.device)
    # models(inpt)
    # # for m in models.modules():
    # #     if isinstance(m, QuantHelper):
    # #         if not isinstance(m, FeatureQuantizer):
    # #             # m.set_quant_mode(args.quant_mode)
    # #             m.set_quant_mode(False)
    # models.zero_buffer()
    # del inpt

    # pt2_path = 'results/sumbaseline/default/checkpoints/best_model.pth'
    # pt2 = torch.load(pt2_path)
    # models.load_state_dict(pt2, strict=False)
    test_state = model.state_dict()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    cof_controller = CofStepController(args.diff_cof, args.cof_gamma, milestones=[20, 100])
    
    autocast = torch.cuda.amp.autocast(enabled=args.amp)
    scaler = GradScaler(enabled=args.amp)
    
    #print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    flag_warm = True
    best_acc = 0

    from sumResnet import QuantNet9 as SumBaseline
    from train_sumbaseline import test_epoch as test_epoch_sumbaseline
    sumbaseline = SumBaseline(in_channels, num_classes, inplanes=args.inplanes, layers=args.layers).to(args.device)
    sumbaseline.load_state_dict(pt2, strict=False)

    for epochs in range(args.epochs):
        # train_epoch(epochs, models, criterion, train_loader, optimizer, scaler, autocast, args.device, cof_controller.cof, args.clip_grad)
        acc = test_epoch(epochs, model, test_loader, args.device)
        test_epoch_sumbaseline(epochs, sumbaseline, test_loader, args.device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'best_model.pth'))

        if epochs >= args.warmup and flag_warm:
            print("Start to quantize")
            flag_warm = False
            for m in model.modules():
                if isinstance(m, FeatureQuantizer):
                    if args.use_fq:
                        m.set_quant_mode(True)
                if isinstance(m, QuantConv_DW) or isinstance(m, QuantConv_PW):
                    if args.use_wq:
                        m.set_quant_mode(True)
                # if isinstance(m, QuantHelper):
                #     m.set_quant_mode(True)
                    # m.set_quant_mode(True)

        cof_controller.step()
        
        # if (epochs + 1) % args.save_freq == 0 or (epochs + 1) == args.epochs:
    torch.save({
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epochs': epochs + 1
    }, os.path.join(ckpt_path, 'epoch_'+str(epochs)+'.pth'))
    logging.info(f'====> best Test_acc {best_acc:.3f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--root", default='./', type=str)
    parser.add_argument("--dataset", default='mnist', type=str)
    parser.add_argument("--dataset_path", default="../datasets/mnist", type=str)
    parser.add_argument("--result_dir", default='results', type=str)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--tag", default='default', type=str)
    
    parser.add_argument("--models", default='QuantNet9', type=str)
    parser.add_argument("--inplanes", default=64, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--n_dw_emb", default=100, type=int)
    parser.add_argument("--n_pw_emb", default=100, type=int)
    parser.add_argument("--n_f_emb", default=100, type=int)
    parser.add_argument("--diff-cof", default=0.5, type=float)
    parser.add_argument("--cof-gamma", default=5., type=float)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--use_fq", action='store_true')
    parser.add_argument("--use_wq", action='store_true')
    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
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
    # parser.set_defaults(amp=True, use_fq=True, use_wq=True)

    args = parser.parse_args()
    set_seed(args.seed, args.device)
    main(args)