import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim

from models.funcs import create_backbone
from pytorch_lightning.utilities import AttributeDict
from typing import Union
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import torch
import wandb

from funcs.module_funcs import setup_optimizer, setup_scheduler


class AqMethod(pl.LightningModule):
    args: AttributeDict

    def __init__(self, args, backbone=None):
        super(AqMethod, self).__init__()
        self.acc_sum = 0
        self.n_sum = 0
        self.train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.val_accuracy = torchmetrics.Accuracy(top_k=1)
        self.test_accuracy = torchmetrics.Accuracy(top_k=1)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.flag_decay = True
        if backbone is None:
            self.backbone = create_backbone(args)
        else:
            self.backbone = backbone

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self.backbone)
        scheduler = setup_scheduler(self.args, opt, milestones=self.args.milestones)
        if scheduler is None:
            return opt
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        # if self.current_epoch < self.args.warm_epoch:
        #     self.backbone.layer2.straight_mode_()
        # else:
        #     self.backbone.layer2.quant_mode_()
        x, y = batch
        y_hat = self.backbone(x)
        epoch = self.current_epoch
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
        loss = ce_loss + (weight_diff * self.args.beta_w + feature_diff * self.args.beta_f) * self.args.beta
        self.backbone.update_quantizer()
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute()
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
            print('accuracy:', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        acc = self.test_accuracy.compute()
        log_data = {
            'test_loss': test_loss.item(),

            'test_acc': acc
        }

        self.log_dict(log_data, prog_bar=True, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):

        self.test_accuracy.reset()

    def disable_quantizer(self):
        self.backbone.disable_quantizer()