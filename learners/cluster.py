"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .cluster_utils import target_distribution
from .contrastive_utils import SupConLoss
from .criterion import KCL

class ClusterLearner(nn.Module):
	def __init__(self, model, optimizer, temperature, base_temperature):
		super(ClusterLearner, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.contrast_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
		self.cluster_loss = nn.KLDivLoss(size_average=False)
		self.kcl = KCL()

	def forward(self, inputs, use_perturbation=False):
		embd0 = self.model.get_embeddings(inputs[0], pooling="mean")
		embd1 = self.model.get_embeddings(inputs[1], pooling="mean")
		embd2 = self.model.get_embeddings(inputs[2], pooling="mean")

		# Instance-CL loss
		feat1 = F.normalize(self.model.head(embd1), dim=1)
		feat2 = F.normalize(self.model.head(embd2), dim=1)
		features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
		contrastive_loss = self.contrast_loss(features)
		loss = contrastive_loss

        # clustering loss
		output = self.model.get_cluster_prob(embd0)
		target = target_distribution(output).detach()
		cluster_loss = self.cluster_loss((output+1e-08).log(),target)/output.shape[0]
		loss += cluster_loss

		# consistency loss (this loss is used in the experiments of our NAACL paper, we included it here just in case it might be helpful for your specific applications)
		local_consloss_val = 0
		if use_perturbation:
			local_consloss = self.model.local_consistency(embd0, embd1, embd2, self.kcl)
			loss += local_consloss
			local_consloss_val = local_consloss.item()
				
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return {"Instance-CL_loss":contrastive_loss.detach(), "clustering_loss":cluster_loss.detach(), "local_consistency_loss":local_consloss_val}
