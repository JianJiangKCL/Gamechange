from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

def setup_optimizer(args, model):
	if args.optimizer == 'adam':
		opt = optim.Adam(model.parameters(), lr=args.lr)

	elif args.optimizer == 'sgd':
		opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5, nesterov=True)

	elif args.optimizer == 'adamw':
		opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

	else:
		raise NotImplementedError
	return opt


def setup_scheduler(args, opt):
	if args.scheduler == 'cosine':
		scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
		interval = 'step'
	elif args.scheduler == 'warm_cosine':
		scheduler = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, eta_min=1e-5)
		interval = 'step'
	elif args.scheduler == 'multistep':
		op_multi = lambda a, b: int(a * b)
		MILESTONES = list((map(op_multi, [0.5, 0.75, 0.9], [args.epochs] * 3)))
		scheduler = MultiStepLR(opt, milestones=MILESTONES, gamma=0.1)
		interval = 'epoch'
	else:
		raise NotImplementedError

	return scheduler, interval

