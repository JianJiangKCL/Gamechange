import os
import logging
from torch.utils.data import DataLoader
from data import create_dataset
from funcs.utils_funcs import set_seed
from models.funcs import create_backbone
import wandb
from models.AqMethod import AqMethod
from funcs.setup import parse_args, set_logger, set_trainer


def main(local_rank, args):
    result_path = args.results_dir
    os.makedirs(result_path, exist_ok=True)

    logging.info('====>  args{} '.format(args))
    
    train_dataset, num_classes, in_channels = create_dataset(args.dataset, args.dataset_path, train=True)
    test_dataset, _, _ = create_dataset(args.dataset, args.dataset_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    backbone = create_backbone(args)

    wandb_logger = set_logger(args, result_path)
    trainer = set_trainer(args, wandb_logger, result_path)
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