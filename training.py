"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch
import numpy as np
from utils.logger import statistics_log
from evaluation import prepare_task_input, evaluate_embedding
import time

def training(train_loader, learner, args):
    print('\n={}/{}=Iterations/Batches'.format(args.max_iter, len(train_loader)))
    t0 = time.time()
    learner.model.train()
    for i in np.arange(args.max_iter+1):
        try:
            batch = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)
        
        feats, _ = prepare_task_input(learner.model, batch, args, is_contrastive=True)

        losses = learner.forward(feats, use_perturbation=args.use_perturbation)
        
        if (args.print_freq>0) and ((i%args.print_freq==0) or (i==args.max_iter)):
            statistics_log(args.tensorboard, losses=losses, global_step=i)
            evaluate_embedding(learner.model, args, i)
            learner.model.train()
        ## STOPPING CRITERION (due to some license issue, we still need some time to release the data)
        # you need to implement your own stopping criterion, the one we typically use is 
        # diff (cluster_assignment_at_previous_step - cluster_assignment_at_previous_step) / all_data_samples <= criterion
    return None   



             