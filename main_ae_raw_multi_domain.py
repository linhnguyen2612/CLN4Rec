import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

from model.VAE_raw import VAE
from parsers.parser_vae import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_VAE import DataLoaderVAE
from torch.nn import Linear
from mask_optimization_for_vae import *
from scipy import sparse
import bottleneck as bn
import random
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def ndcg(X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))

        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                         for n in heldout_batch.getnnz(axis=1)])

        return DCG / IDCG


def recall(X_pred, heldout_batch, k=100):
        batch_users = X_pred.shape[0]
        idx = bn.argpartition(-X_pred, k, axis=1)
        X_pred_binary = np.zeros_like(X_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

        X_true_binary = (heldout_batch > 0).toarray()
        tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
            np.float32)
        recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
        return recall

from copy import deepcopy

args = parse_vae_args()
data = DataLoaderVAE(args)
if True:
    task_ids = data.task_ids
    train_data = data.real_train_data
    test_data = data.real_test_data

device = torch.device("cuda:0")
print("get data success")

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]
        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def evaluate(model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()
    
    for m in metrics:
        m['score'] = []
    
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
            
        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
        
    return [x['score'] for x in metrics]

def run(model, opts, train_data_in, train_data_out, batch_size, n_epochs, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data_in, data_out = train_data_out, shuffle=True):
            ratings_in = batch.get_ratings_to_dev()
            for optimizer in opts:
                optimizer.zero_grad()   
            _, loss = model(ratings_in, dropout_rate=dropout_rate, index = None, user_ratings_out = None)   
            loss.backward()
            for optimizer in opts:
                optimizer.step()


model_kwargs = {
    'args':args,
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': data.n_items
}
metrics = [{'metric': ndcg, 'k': 100}]

best_ndcg = -np.inf
train_scores, test_scores = [], []

model = VAE(**model_kwargs).to(device)
model_best = VAE(**model_kwargs).to(device)
optimizer = optim.Adam(model.parameters(), lr = args.lr)

#decoder_params = set(model.decoder.parameters())
#encoder_params = set(model.encoder.parameters())

#optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
#optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)

print("start training")
test_metrics = [{'metric': ndcg, 'k': 20}, {'metric': ndcg, 'k': 50}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]
start_training_time = time()
if True:    
    for epoch in range(args.n_epochs): 
        if True:
            run(model, [optimizer], train_data, None, args.batch_size, 1, dropout_rate = 0.05)
        test_score = []
        train_score = []
        for j in range(len(task_ids)):
            train_data_j = test_data[j][0]
            test_data_j = test_data[j][1]
            test_score.append(
                evaluate(model, train_data_j, test_data_j, test_metrics, 1)[-1]
            )
            train_score.append(
                evaluate(model, train_data_j, train_data_j, test_metrics, 1)[-1]
            )
            #print(f'epoch {epoch} | train score task {task_ids[j]}: recall@50: {train_score[-1]:.4f}')
            print(f'epoch {epoch} | test score task {task_ids[j]}: recall@50: {test_score[-1]:.4f}')
        test_scores.append(sum(test_score)/len(test_score))
        train_scores.append(sum(train_score)/len(train_score))
        #print(f'epoch {epoch} | train score all task: recall@50: {train_scores[-1]:.4f}')
        print(f'epoch {epoch} | test score all task: recall@50: {test_scores[-1]:.4f}')
    print('-----------------------------------------------------')
print(f'Training time is {time() - start_training_time}')