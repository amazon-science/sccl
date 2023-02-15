import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import csv

from models.Transformers import SCCLBert
from utils.optimizer import get_bert


def main(
    data_path: str,
    weights_path: str,
    results_path: str,
    batch_size: int,
    model_max_length: int,
    args: argparse.Namespace,
) -> None:
    dataset = pd.read_csv(data_path)
    bert, tokenizer = get_bert(args)
    model = SCCLBert(bert, tokenizer, cluster_centers=np.zeros((64, 768)), alpha=args.alpha)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.to(torch.device("cuda"))
    model = nn.DataParallel(model)
    batches = prepare_batches(dataset=dataset, batch_size=batch_size)
    embeddings = execute_inference(model=model, batches=batches, model_max_length=model_max_length)
    results = prepare_results(dataset=dataset, embeddings=embeddings)
    results.to_csv(results_path, quoting=csv.QUOTE_NONNUMERIC, index=False)


def prepare_batches(
    dataset: pd.DataFrame,
    batch_size: int,
) -> List[Dict[str, List[str]]]:
    print(len(dataset))
    batches, current_batch = [], []
    for datapoint in dataset["text"].tolist():
        if len(current_batch) == batch_size:
            batches.append({"text": current_batch})
            current_batch = []
        current_batch.append(datapoint)
    if len(current_batch) > 0:
        batches.append({"text": current_batch})
    return batches


@torch.no_grad()
def execute_inference(
    model: SCCLBert,
    batches: List[Dict[str, List[str]]],
    model_max_length: int
) -> np.ndarray:
    model.eval()
    all_embeddings = []
    for batch in tqdm(batches, desc="Inference..."):
        batch_tokens = model.module.tokenizer.batch_encode_plus(
            batch["text"],
            max_length=model_max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        batch_embeddings = model(batch_tokens['input_ids'].cuda(), batch_tokens['attention_mask'].cuda(), task_type="evaluate")
        all_embeddings.append(batch_embeddings.detach().cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def prepare_results(dataset: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    embeddings_columns = [f"feature_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(data=embeddings, columns=embeddings_columns)
    return pd.concat([dataset, embeddings_df], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--use_pretrain', type=str, default='SBERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=400)
    parser.add_argument('--model_max_length', type=int, required=False, default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        weights_path=args.weights_path,
        results_path=args.results_path,
        batch_size=args.batch_size,
        model_max_length=args.model_max_length,
        args=args,
    )
