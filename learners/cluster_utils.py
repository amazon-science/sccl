"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()