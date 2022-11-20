# https://github.com/PyTorchLightning/pytorch-lightning/blob/fe34bf2a653ebd50e6a3a00be829e3611f820c3c/pl_examples/bug_report/bug_report_model.py
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader, Dataset
from funcs.setup import set_logger, set_trainer, parse_args
from funcs.module_funcs import setup_scheduler, setup_optimizer
import os



class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.args = args

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()


        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        lr = self.optimizers().param_groups[0]["lr"]
        current_epoch = self.current_epoch
        # print learning rate 2 decimal places
        print(f"epoch, {current_epoch}  {lr:.7f}")

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self)
        scheduler = setup_scheduler(self.args, opt)
        return [opt], [scheduler]


def main():
    args = parse_args()
    args.epochs = 100000
    args.scheduler = 'cosine'
    args.scheduler_interval = 'epoch'
    args.disable_tqdm = 1
    args.lr = 0.1
    train_data = DataLoader(RandomDataset(32, 63), batch_size=1)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel(args)
    result_path = 'results'
    os.makedirs(result_path, exist_ok=True)
    wandb_logger = set_logger(args, result_path)
    # wandb_logger = None

    trainer = set_trainer(args, wandb_logger, result_path)

    trainer.fit(model, train_dataloaders=train_data)#, val_dataloaders=val_data)


if __name__ == "__main__":
    main()