import argparse
from torchvision.models import MobileNetV2
import torch
from funcs.module_funcs import setup_scheduler
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.epochs = 100
args.warmup_epochs = 10
args.scheduler = 'cosine'
model = MobileNetV2()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
# to unit test
scheduler, interval = setup_scheduler(args, opt)
for epoch in range(10):
	for batch in range(100):
		for param_group in opt.param_groups:
			lr = param_group['lr']

			param_group['lr'] = lr
		scheduler.step()
		print(param_group['lr'])