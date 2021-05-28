"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
from tensorboardX import SummaryWriter

def setup_path(args):
	resPath = "SCCL"
	if args.use_perturbation:
		resPath += ".perturbation"
	resPath += f'.{args.bert}'
	resPath += f'.{args.dataset}'
	resPath += f'.lr{args.lr}'
	resPath += f'.lrscale{args.lr_scale}'
	resPath += f'.tmp{args.temperature}'
	resPath += f'.alpha{args.alpha}'
	resPath += f'.seed{args.seed}/'
	resPath = args.result_path + resPath
	print(f'results path: {resPath}')

	tensorboard = SummaryWriter(resPath)
	return resPath, tensorboard


def statistics_log(tensorboard, losses=None, global_step=0):
	print("[{}]-----".format(global_step))
	if losses is not None:
		for key, val in losses.items():
			tensorboard.add_scalar('train/'+key, val, global_step)
			print("{}:\t {}".format(key, val))



