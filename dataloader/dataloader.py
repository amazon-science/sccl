"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class TextClustering(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

class AugmentPairSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'label': self.train_y[idx]}


def augment_loader(args):
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    train_text = train_data['text'].fillna('.').values
    train_text1 = train_data['text1'].fillna('.').values
    train_text2 = train_data['text2'].fillna('.').values
    train_label = train_data['label'].astype(int).values

    train_dataset = AugmentPairSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

def train_unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    train_text = train_data['text'].fillna('.').values
    train_label = train_data['label'].astype(int).values

    train_dataset = TextClustering(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

