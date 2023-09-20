"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans


def get_mean_embeddings(bert, input_ids, attention_mask):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):

        text = batch['text']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)

    # Perform KMeans clustering
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)

    print("all_embeddings shape:", all_embeddings.shape)
    print("Iterations:", clustering_model.n_iter_)
    print("Centers shape:", clustering_model.cluster_centers_.shape)
    
    return clustering_model.cluster_centers_



