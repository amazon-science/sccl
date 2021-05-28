"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from transformers import AutoModel, AutoTokenizer

class SCCLBert(nn.Module):
    def __init__(self, bert_model, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()
        print(bert_model[0].tokenizer)  

        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.emb_size = self.sentbert.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def get_embeddings(self, features, pooling="mean"):
        bert_output = self.sentbert.forward(**features)
        attention_mask = features['attention_mask'].unsqueeze(-1)
        all_output = bert_output[0]
        mean_output = torch.sum(all_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2

    







