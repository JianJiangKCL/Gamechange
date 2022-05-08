import argparse

from args.general.dataset import dataset_args
from args.general.train import train_args
from args.general.project import project_args
from args.custom import custom_args
from utils_funcs import load_temp_, load_yaml_

def parse_args() -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    project_args(parser)
    train_args(parser)

    # add custom args
    custom_args(parser)

    # parse args
    args = parser.parse_args()

    # load config
    load_yaml_(args)
    # update args if  temp_args is given
    load_temp_(args)
    #

    return args


