import csv
import math
import os
import random
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, List, Any

import numpy as np
import pandas as pd

import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stackoverflow')
parser.add_argument('--augtype', type=str, default='ctxt_insertbertroberta')
parser.add_argument('--aug_min', type=int, default=1) 
parser.add_argument('--aug_p', type=float, default=0.2)
parser.add_argument('--aug_max', type=int, default=10)
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)
args = parser.parse_args()


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def contextual_augment(
    data_source, data_target, rank, world_size,
    textcol="text", aug_p=0.2, device1="cuda", device2="cuda",
):
    ### contextual augmentation 
    print(f"\n-----transformer_augment-----\n")
    augmenter1 = naw.ContextualWordEmbsAug(
        model_path='roberta-base', action="substitute", aug_min=1, aug_p=aug_p, device=device1, verbose=1, silence=False)
    
    augmenter2 = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="substitute", aug_min=1, aug_p=aug_p, device=device2, verbose=1, silence=False)
        
    train_data = pd.read_csv(data_source)
    min_idx = math.floor((rank / world_size) * len(train_data))
    max_idx = math.floor(((rank + 1) / world_size) * len(train_data))
    train_data = train_data.iloc[min_idx:max_idx]
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    auglist1, auglist2 = [], []
    for txt in tqdm(train_text, desc="Augmenting..."):
        atxt1 = augmenter1.augment(txt)
        atxt2 = augmenter2.augment(txt)
        if atxt1 is None or atxt2 is None:
            print("Alert. None detected")
        auglist1.append(str(atxt1))
        auglist2.append(str(atxt2))
        
    train_data[textcol+"1"] = auglist1
    train_data[textcol+"2"] = auglist2
    print(train_data)
    train_data.to_csv(data_target, quoting=csv.QUOTE_NONNUMERIC, index=False)
    
    for o, a1, a2 in zip(train_text[:5], auglist1[:5], auglist2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)


def word_deletion(data_source, data_target, textcol="text", aug_p=0.2):
    ### wordnet based data augmentation
    print(f"\n-----word_deletion-----\n")
    aug = naw.RandomWordAug(aug_min=1, aug_p=aug_p)
    
    train_data = pd.read_csv(data_source)
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in train_text:
        atxt = aug.augment(txt, n=2, num_thread=1)
        augtxts1.append(str(atxt[0]))
        augtxts2.append(str(atxt[1]))
        
    train_data[textcol+"1"] = pd.Series(augtxts1)
    train_data[textcol+"2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)
    
    for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)
        
        
def randomchar_augment(data_source, data_target, textcol="text", aug_p=0.2, augstage="post"):
    ### wordnet based data augmentation
    print(f"\n*****random char aug: rate--{aug_p}, stage: {augstage}*****\n")
    aug = nac.RandomCharAug(action="swap", aug_char_p=aug_p, aug_word_p=aug_p)
    train_data = pd.read_csv(data_source)
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))
    run_pre_aug_closure = partial(run_pre_aug, aug=aug)
    result = _run_with_executor(
        run_pre_aug_closure,
        train_text,
        max_workers=32,
        desc="Pre aug...",
        chunk_size=1024
    )
    augtxts1, augtxts2 = [], []
    for atxt in result:
        augtxts1.append(str(atxt[0]))
        augtxts2.append(str(atxt[1]))

    train_data[textcol+"1"] = pd.Series(augtxts1)
    train_data[textcol+"2"] = pd.Series(augtxts2)
    for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)
    train_text1 = train_data[textcol+"1"].fillna('.').astype(str).values
    train_text2 = train_data[textcol+"2"].fillna('.').astype(str).values

    augtxts1, augtxts2 = [], []
    run_post_aug_closure = partial(run_post_aug, aug=aug)
    result = _run_with_executor(
        run_post_aug_closure,
        list(zip(train_text1, train_text2)),
        max_workers=32,
        desc="Post aug...",
        chunk_size=1024
    )
    for atxt1, atxt2 in result:
        augtxts1.append(str(atxt1))
        augtxts2.append(str(atxt2))

    train_data[textcol+"1"] = pd.Series(augtxts1)
    train_data[textcol+"2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)

    for o1, a1, o2, a2 in zip(train_text1[:2], augtxts1[:2], train_text2[:2], augtxts2[:2]):
        print("-----Original Text1: \n", o1)
        print("-----Augmentation1: \n", a1)
        print("-----Original Text2: \n", o2)
        print("-----Augmentation2: \n", a2)


def _run_with_executor(
    fun,
    iterable,
    max_workers: int,
    desc: Optional[str] = None,
    chunk_size: int = 1,
    progress_bar: bool = True,
) -> List[Any]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor_map = executor.map(fun, iterable, chunksize=chunk_size)
        if progress_bar:
            results = list(
                tqdm(
                    executor_map,
                    total=len(iterable),
                    desc=desc,
                )
            )
        else:
            results = list(executor_map)
        return results


def run_pre_aug(txt, aug):
    atxt = aug.augment(txt, n=2, num_thread=1)
    return str(atxt[0]), str(atxt[1])


def run_post_aug(texts, aug):
    txt1, txt2 = texts
    atxt1 = aug.augment(txt1, n=1, num_thread=1)
    atxt2 = aug.augment(txt2, n=1, num_thread=1)
    return atxt1, atxt2
        

def augment_files(datadir="./", targetdir="./", dataset="wiki1m_unique", aug_p=0.1, augtype="trans_subst"):
    set_global_random_seed(0)
    device1 = f"cuda:{args.gpuid}"
    device2 = f"cuda:{args.gpuid+1}"
    
    DataSource = os.path.join(datadir, dataset + ".csv")
    DataTarget = os.path.join(targetdir, '{}_{}_{}.csv'.format(dataset, augtype, int(aug_p*100)))

    if augtype == "word_deletion":
        augseq = word_deletion(DataSource, DataTarget, textcol="text", aug_p=aug_p)
    elif augtype == "trans_subst":
        DataTarget = os.path.join(targetdir, '{}_{}_{}_rank_{}_world_size_{}.csv'.format(dataset, augtype, int(aug_p * 100), args.rank, args.world_size))
        augseq = contextual_augment(
            DataSource, DataTarget,
            rank=args.rank,
            world_size=args.world_size,
            textcol="text",
            aug_p=aug_p,
            device1=device1,
            device2=device2,
        )
    elif augtype == "charswap":
        augseq = randomchar_augment(DataSource, DataTarget, textcol="text", aug_p=aug_p, augstage="post")
    else:
        print("Please specify AugType!!")
        
        
if __name__ == '__main__':
    
    datadir = os.path.join(os.path.dirname(__file__), "..", "data", "source")
    targetdir = os.path.join(os.path.dirname(__file__), "..", "data", "target")
    
    augment_files(datadir=datadir, targetdir=targetdir, dataset=args.dataset, aug_p=0.2, augtype=args.augtype)






