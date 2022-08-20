from argparse import ArgumentParser


def custom_args(parser: ArgumentParser):
	parser.add_argument("--n_dw_emb", default=128, type=int)
	parser.add_argument("--n_pw_emb", default=128, type=int)
	parser.add_argument("--n_f_emb", default=128, type=int)
	parser.add_argument("--out_planes", default=64, type=int)
	parser.add_argument("--use_fq", action='store_true')
	parser.add_argument("--use_wq", action='store_true')
	parser.add_argument("--beta", default=0.5, type=float)
	parser.add_argument("--beta_w", default=0.5, type=float)
	parser.add_argument("--beta_f", default=0.5, type=float)
	parser.add_argument("--gs", default=1, type=int, help="group size")
	parser.add_argument("--method", default='aq', choices=['aq', 'normal'])
	parser.add_argument("--num_class", default=5, type=int)

