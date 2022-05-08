from argparse import ArgumentParser


def train_args(parser: ArgumentParser):
	parser.add_argument("--seed", type=int, default=1993)
	parser.add_argument("--arch", type=str, default='resnet18')
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--epoch", type=int, default=200)
	parser.add_argument("--warm_epoch", type=int, default=1)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--optimizer", default='adam', type=str)
	parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
	parser.add_argument("--disable_tqdm", action='store_true', help="disable tqdm progress bar")
	parser.add_argument("--distributed", action="store_true", help="distribute over different gpus ")
	parser.add_argument("--gpus", type=int, default=4)
	parser.add_argument("--local_rank", type=int, default=-1)
	parser.add_argument('--dist_url', default='tcp://localhost:12335', type=str, help='url used to set up distributed training')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
	help='path to latest checkpoint (default: none)')
	parser.add_argument('--finetune', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument("--use_amp", choices=[0, 1], default=0, type=int, help="use apex for mixed precision training")
	parser.add_argument("--use_ddp", choices=[0, 1], default=0, type=int, help="used ddp for distributed training")
