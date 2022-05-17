import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim

from model.funcs import create_backbone
from pytorch_lightning.utilities import AttributeDict
from typing import Union
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
class AqMethod(pl.LightningModule):
    args: AttributeDict

    def __init__(self, args, backbone=None):
        super(AqMethod, self).__init__()
        self.acc_sum = 0
        self.n_sum = 0
        self.accuracy = torchmetrics.Accuracy(top_k=1)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.flag_decay = True
        if backbone is None:
            self.backbone = create_backbone(args)
        else:
            self.backbone = backbone

    # this is used for setting the temporary weights get close to the selected kernels

    def configure_optimizers(self):
        
        op_multi = lambda a, b: int(a * b)
        if self.args.optimizer == 'adam':
            opt = optim.Adam(self.backbone.parameters(), lr=self.args.lr)
            MILESTONES = list((map(op_multi, [0.5], [self.args.epoch])))
        elif self.args.optimizer == 'sgd':
            opt = optim.SGD(self.backbone.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=4e-5, nesterov=True)
            MILESTONES = list((map(op_multi, [0.5, 0.8], [self.args.epoch, self.args.epoch])))
        else:
            raise NotImplementedError
        # scheduler = MultiStepLR(opt, milestones=MILESTONES, gamma=0.1)
        scheduler = CosineAnnealingLR(opt, self.args.epoch, eta_min=0.0006)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        # if self.current_epoch < self.args.warm_epoch:
        #     self.backbone.layer2.straight_mode_()
        # else:
        #     self.backbone.layer2.quant_mode_()
        x, y = batch
        y_hat = self.backbone(x)

        ce_loss = self.criterion(y_hat, y)

        weight_diff = self.backbone.accumlate_diff_weight()

        feature_diff = self.backbone.accumlate_diff_feature()
        # if self.current_epoch > 10:
        #     # if self.flag_decay:
        #     #     self.backbone.set_qtz_decay(0.9999)
        #     #     self.flag_decay = False
        #     loss = ce_loss + (weight_diff * 0.001) + feature_diff * 0.1
        # else:
        # loss = ce_loss + (weight_diff * 0.001) + feature_diff    #* self.args.beta
        loss = ce_loss + (weight_diff + feature_diff) * self.args.beta
        self.backbone.update_quantizer()
        self.accuracy.update(y_hat, y)
        acc = self.accuracy.compute()
        # no need to do the .item() here. no memory leak
        log_data = {

            'loss': loss,
            'ce_loss': ce_loss,
            'w_diff': weight_diff,
            'f_diff': feature_diff,
            'acc': acc,
            'GB': torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

        }

        self.log_dict(log_data, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.global_rank == 0:
            print('accuracy:', self.accuracy.compute())
        self.accuracy.reset()


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.accuracy.update(y_hat, y)
        acc = self.accuracy.compute()
        log_data = {
            'test_loss': test_loss.item(),

            'test_acc': acc
        }

        self.log_dict(log_data, prog_bar=True, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):

        self.accuracy.reset()

    def disable_quantizer(self):
        self.backbone.disable_quantizer()