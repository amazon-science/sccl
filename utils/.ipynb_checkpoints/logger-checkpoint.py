"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
from tensorboardX import SummaryWriter
import random
import torch
import numpy as np


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def setup_path(args):
    resPath = "SCCL"
    resPath += f'.{args.bert}'
    resPath += f'.{args.use_pretrain}'
    resPath += f'.{args.augtype}'
    resPath += f'.{args.dataname}'
    resPath += f".{args.text}"
    resPath += f'.lr{args.lr}'
    resPath += f'.lrscale{args.lr_scale}'
    resPath += f'.{args.objective}'
    resPath += f'.eta{args.eta}'
    resPath += f'.tmp{args.temperature}'
    resPath += f'.alpha{args.alpha}'
    resPath += f'.seed{args.seed}/'
    resPath = args.resdir + resPath
    print(f'results path: {resPath}')

    tensorboard = SummaryWriter(resPath)
    return resPath, tensorboard


def statistics_log(tensorboard, losses=None, global_step=0):
    print("[{}]-----".format(global_step))
    if losses is not None:
        for key, val in losses.items():
            if key in ["pos", "neg", "pos_diag", "pos_rand", "neg_offdiag"]:
                tensorboard.add_histogram('train/'+key, val, global_step)
            else:
                try:
                    tensorboard.add_scalar('train/'+key, val.item(), global_step)
                except:
                    tensorboard.add_scalar('train/'+key, val, global_step)
                print("{}:\t {:.3f}".format(key, val))



