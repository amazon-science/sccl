"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import time
import numpy as np
import random
from sklearn import cluster
import scipy.spatial
import sklearn.metrics

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as util_data
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss
from models.Transformers import SCCLMatrix

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)


            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
        
    def train_step_virtual(self, input_ids, attention_mask):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += 0.5*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()


    


        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train_step_explicit(self, input_ids, attention_mask):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        for i in np.arange(self.args.max_iter+1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                self.evaluate_embedding(i)
                self.model.train()

        return None   

    
    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        
        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embeddings = all_embeddings.cpu().numpy()
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
        
        # clustering accuracy 
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc":acc, "acc_model":acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
            
        np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
        np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
        np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
        # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
        # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
        # np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
        # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

        print('[Representation] Clustering scores:',confusion.clusterscores()) 
        print('[Representation] ACC: {:.3f}'.format(acc)) 
        print('[Model] Clustering scores:',confusion_model.clusterscores()) 
        print('[Model] ACC: {:.3f}'.format(acc_model))
        return None



             


class MatrixDECTrainer(nn.Module):
    def __init__(self, model, optimizer, train_loader, dataset, labels, args,
                 include_contrastive_loss=False,
                 device="cpu",
                 patience=6,
                 canonicalization_test_function=None,
                 canonicalization_side_information=None):
        super(MatrixDECTrainer, self).__init__()
        self.model = model
        assert self.model.__class__.__name__ == "SCCLMatrix"

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        self.include_contrastive_loss = include_contrastive_loss

        self.device = device
        self.dataset = dataset.astype("float32")
        self.labels = labels
        self.patience = patience
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

        self.canonicalization_test_function = canonicalization_test_function
        self.canonicalization_side_information = canonicalization_side_information
    
    def find_empty_clusters(self, iter=0):
        cluster_centers = None
        for name, param in self.model.named_parameters():
            if name == "cluster_centers":
                cluster_centers = param.detach().cpu().numpy()
                break

        dist_matrix = scipy.spatial.distance_matrix(self.dataset, cluster_centers)

        labels = np.full(self.dataset.shape[0], fill_value=-1)
        min_cluster_distances = []

        index = list(range(self.dataset.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            cluster_distances = dist_matrix[x_i]
            min_cluster_distances.append(min(cluster_distances))
            labels[x_i] = np.argmin(cluster_distances)

        n_samples_in_cluster = np.bincount(labels, minlength=cluster_centers.shape[0])
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            print(f"Empty clusters: {empty_clusters}")
            points_by_min_cluster_distance = np.argsort(-np.array(min_cluster_distances))
            i = 0
            empty_cluster_idxs = list(empty_clusters)
            random.shuffle(empty_cluster_idxs)
            for cluster_idx in empty_cluster_idxs:
                while n_samples_in_cluster[labels[points_by_min_cluster_distance[i]]] == 1:
                    i += 1
                new_cluster_center_point_idx = points_by_min_cluster_distance[i]
                # Set empty cluster with furthest point from new cluster
                cluster_centers[cluster_idx] = self.dataset[new_cluster_center_point_idx]
                labels[new_cluster_center_point_idx] = cluster_idx
                n_samples_in_cluster[labels[points_by_min_cluster_distance[i]]] -= 1
                n_samples_in_cluster[cluster_idx] += 1
                i += 1

            n_samples_in_cluster = np.bincount(labels, minlength=cluster_centers.shape[0])
            empty_clusters = np.where(n_samples_in_cluster == 0)[0]
            if len(empty_clusters) > 0:
                print("Empty cluster found!!!")

            param.data.copy_(torch.tensor(cluster_centers, requires_grad=True, device=param.device))



    def train_step(self, points, i=0, new_batch=False):
        
        embd1, embd2 = self.model(points, task_type="virtual")
        # Figure out how corresponding points are distribute inside the batch

        # Instance-CL loss
        if self.include_contrastive_loss:
            feat1, feat2 = self.model.contrast_logits(embd1, embd2)
            losses = self.contrast_loss(feat1, feat2)
            loss = self.eta * losses["loss"]
        else:
            losses = {}

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            if self.include_contrastive_loss:
                loss += cluster_loss
            else:
                loss = cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        '''
        linear_matrix_1 = None
        linear_matrix_0 = None
        # Identity regularization loss
        for (name, param) in self.model.named_parameters():
            if name == "linear_matrix.1.weight":
                linear_matrix_1 = param
            elif name == "linear_matrix.0.weight":
                linear_matrix_0 = param

        param = torch.matmul(linear_matrix_1, linear_matrix_0)
        regularization_loss = 5e-3  * torch.norm(param - torch.eye(param.shape[0], device=param.device))
        loss += regularization_loss
        losses["regularization_loss"] = loss.item()
        '''

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if new_batch:
            self.find_empty_clusters(iter=i)

        return losses

    def evaluate_embedding(self, step):
        examples = []
        for x, y in zip(self.dataset, self.labels):
            examples.append({"x": x, "y": y})
 
        dataloader = util_data.DataLoader(examples, batch_size=self.args.batch_size, shuffle=False, num_workers=1)  
        print('---- {} evaluation batches ----'.format(len(dataloader)))

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['x'], batch['y'] 
                embeddings = self.model(batch['x'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)

        all_pred = torch.argmax(all_prob, dim=1).tolist()
        rand = sklearn.metrics.adjusted_rand_score(all_labels.tolist(), all_pred)
        print(f"Rand Score: {rand}")







        return all_pred, rand
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        all_pred, rand_score = self.evaluate_embedding(0)
        if self.canonicalization_side_information is not None and self.canonicalization_test_function is not None:
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = self.canonicalization_test_function(self.canonicalization_side_information.p, self.canonicalization_side_information.side_info, all_pred, self.canonicalization_side_information.true_ent2clust, self.canonicalization_side_information.true_clust2ent)
            print(f"Pre-training performance:\nrand_score:\t{rand_score}\nmacro_f1:\t{macro_f1}\nmicro_f1:\t{micro_f1}\npairwise_f1:\t{pairwise_f1}")

        self.model.train()

        prev_labels = None
        patience_counter = 0

        for i in np.arange(self.args.max_iter+1):
            try:
                batch = next(train_loader_iter)
                new_batch = False
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)
                new_batch = True

            batch = batch.to(self.device)
            losses = self.train_step(batch, i=i, new_batch= i > 0 and (i % 50 == 0 or i == self.args.max_iter - 2))

            # TODO(Vijay): add test metrics to losses given to tensorboard


            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                all_pred, rand_score = self.evaluate_embedding(i)
                losses["rand_score"] = rand_score

                if self.canonicalization_side_information is not None and self.canonicalization_test_function is not None:
                    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = self.canonicalization_test_function(self.canonicalization_side_information.p, self.canonicalization_side_information.side_info, all_pred, self.canonicalization_side_information.true_ent2clust, self.canonicalization_side_information.true_clust2ent)
                    losses["macro_f1"] = macro_f1
                    losses["micro_f1"] = micro_f1
                    losses["pairwise_f1"] = pairwise_f1

                if prev_labels is None:
                    prev_labels = all_pred
                else:
                    cur_labels = all_pred
                    agreement_with_previous = sklearn.metrics.adjusted_rand_score(cur_labels, prev_labels)
                    if agreement_with_previous == 1.0:
                        patience_counter += 1
                        if patience_counter == self.patience:
                            print(f"Ran out of patience after {i} iterations")
                            break
                    else:
                        patience_counter = 0
                    prev_labels = cur_labels


                statistics_log(self.args.tensorboard, losses=losses, global_step=i)


                self.model.train()

        return all_pred
