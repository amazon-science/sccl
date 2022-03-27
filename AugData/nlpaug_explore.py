import os
import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stackoverflow')
parser.add_argument('--augtype', type=str, default='ctxt_insertbertroberta')
parser.add_argument('--aug_min', type=int, default=1) 
parser.add_argument('--aug_p', type=float, default=0.2)
parser.add_argument('--aug_max', type=int, default=10)
parser.add_argument('--gpu_id', type=str, default='1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def contextual_augment(data_source, data_target, textcol="text", aug_p=0.2, device1="cuda", device2="cuda"):
    # contextual augmentation
    print(f"\n-----transformer_augment-----\n")
    print(device1, device2)
    augmenter1 = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", aug_min=1, aug_p=aug_p, device=device1)
    
    augmenter2 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", aug_min=1, aug_p=aug_p, device=device2)
        
    train_data = pd.read_csv(data_source, sep='\t', engine="python")
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    aug_list1, aug_list2 = [], []
    start = time.time()
    for i, txt in enumerate(train_text):
        if i % 100 == 0:
            print(i, time.time() - start)
        atxt1 = augmenter1.augment(txt)
        atxt2 = augmenter2.augment(txt)
        aug_list1.append(str(atxt1))
        aug_list2.append(str(atxt2))
        
    train_data[textcol+"1"] = pd.Series(aug_list1)
    train_data[textcol+"2"] = pd.Series(aug_list2)
    train_data.to_csv(data_target, index=False)
    
    for o, a1, a2 in zip(train_text[:5], aug_list1[:5], aug_list2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmented Text1: \n", a1)
        print("-----Augmented Text2: \n", a2)


def word_deletion(data_source, data_target, textcol="text", aug_p=0.2):
    # wordnet based data augmentation
    print(f"\n-----word_deletion-----\n")
    aug = naw.RandomWordAug(aug_min=1, aug_p=aug_p)
    
    train_data = pd.read_csv(data_source, sep='\t', engine="python")
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    aug_txt1, aug_txt2 = [], []
    for txt in train_text:
        atxt = aug.augment(txt, n=2, num_thread=1)
        aug_txt1.append(str(atxt[0]))
        aug_txt2.append(str(atxt[1]))
        
    train_data[textcol+"1"] = pd.Series(aug_txt1)
    train_data[textcol+"2"] = pd.Series(aug_txt2)
    train_data.to_csv(data_target, index=False)
    
    for o, a1, a2 in zip(train_text[:5], aug_txt1[:5], aug_txt2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)
        
        
def randomchar_augment(data_source, data_target, textcol="text", aug_p=0.2, augstage="post"):
    # wordnet based data augmentation
    print(f"\n*****random char aug: rate--{aug_p}, stage: {augstage}*****\n")
    aug = nac.RandomCharAug(action="swap", aug_char_p=aug_p, aug_word_p=aug_p)
    
    train_data = pd.read_csv(data_source, sep='\t', engine="python")
    if augstage == "init":
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
    else:
        train_text1 = train_data[textcol+"1"].fillna('.').astype(str).values
        train_text2 = train_data[textcol+"2"].fillna('.').astype(str).values
        
        augtxts1, augtxts2 = [], []
        for txt1, txt2 in zip(train_text1, train_text2):
            atxt1 = aug.augment(txt1, n=1, num_thread=1)
            atxt2 = aug.augment(txt2, n=1, num_thread=1)
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
        
        
def augment_files(data_path="./", target_path="./", data_name="wiki1m_unique", aug_p=0.1, augtype="trans_subst"):
    set_global_random_seed(0)
    # device1=torch.cuda.set_device(0)
    # device1 = torch.device('cuda:0')
    # device2 = torch.device('cuda:0')
    # device2=torch.cuda.set_device(1)
    device1 = 'cuda:0'
    device2 = 'cuda:0'

    data_source = os.path.join(data_path, data_name + ".csv")
    data_target = os.path.join(target_path, '{}_{}_{}.csv'.format(data_name, augtype, int(aug_p * 100)))

    if augtype == "word_deletion":
        word_deletion(data_source, data_target, textcol="text", aug_p=aug_p)
    elif augtype == "trans_subst":
        contextual_augment(data_source, data_target, textcol="text", aug_p=aug_p, device1=device1, device2=device2)
    elif augtype == "charswap":
        randomchar_augment(data_source, data_target, textcol="text", aug_p=aug_p, augstage="post")
    else:
        print("Please specify AugType!!")
        
        
if __name__ == '__main__':
    data_dir = 'pre_data'
    target_dir = 'processed_data'
    
    datasets = ["agnews", "search_snippets", "stackoverflow", "biomedical", "googlenews_TS", "googlenews_T", "googlenews_S", 'tweet']
    dataset = args.dataset
    assert dataset in datasets
    
    # for dataset in datasets:
    augment_files(data_path=data_dir, target_path=target_dir, data_name=dataset, aug_p=0.1, augtype="trans_subst")
    augment_files(data_path=data_dir, target_path=target_dir, data_name=dataset, aug_p=0.2, augtype="trans_subst")

    # augment_files(data_path=data_dir, target_path=target_dir, data_name=dataset, aug_p=0.1, augtype="word_deletion")
    # augment_files(data_path=data_dir, target_path=target_dir, data_name=dataset, aug_p=0.2, augtype="word_deletion")

    # datasets = ["agnews_trans_subst_20", "searchsnippets_trans_subst_20", "stackoverflow_trans_subst_20",
    # "biomedical_trans_subst_20", "googlenews_TS_trans_subst_20", "googlenews_T_trans_subst_20", "googlenews_S_trans_subst_20"]
    #
    # for dataset in datasets:
    #     augment_files(datadir=datadir, targetdir=targetdir, dataset=dataset, aug_p=0.2, augtype="charswap")
