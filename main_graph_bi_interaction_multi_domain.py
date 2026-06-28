import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.Graph_BI_interactions import Graph_BI_interactions
from parsers.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT_MultiTask_full_2
from torch.nn import Linear
from mask_optimization_double_hard import *

sys.path.append("/content/drive/MyDrive/KGAT-pytorch-master")
def evaluate(model, dataloader, Ks, device, test_cold, masking_train = 1, items_out_task = None):
    
    train_user_dict = dataloader.train_user_dict
    if test_cold == 0:
        test_batch_size = dataloader.test_batch_size
        test_user_dict = dataloader.test_user_dict
    else:
        test_batch_size = dataloader.test_cold_batch_size
        test_user_dict = dataloader.test_cold_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            if masking_train == 1:
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, items_out_task = items_out_task)
            else:
                batch_metrics = calc_metrics_at_k_without_masking(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, dataloader.train_data_masking_test_user_dict, items_out_task = items_out_task)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args, index, task_n_epochs, pretrain_model = None, pre_data = None):
    # seed
    task_ids = [int(elem) for elem in args.task_ids.split(',')]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    # load data
    if True:
        data = DataLoaderKGAT_MultiTask_full_2(args, logging)
    data.load_cf_basic_full(data.n_entities)
    #data.device = device
    if True:
        if args.use_pretrain == 1:
            user_pre_embed = torch.tensor(data.user_pre_embed)
            item_pre_embed = torch.tensor(data.item_pre_embed)
        else:
            user_pre_embed, item_pre_embed = None, None
        print(data.max_id)
        # construct model & optimizer
        n_items = data.n_items
        item_ids = torch.arange(n_items, dtype=torch.long).to(device)
        model = Graph_BI_interactions(args, data.n_users, data.n_entities, data.n_relations,
                                      data.A_in, user_pre_embed, item_pre_embed)
        if args.use_pretrain == 2:
            model = load_model(model, args.pretrain_model_path)
        model.to(device)
        cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logging.info(model)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = args.Ks.replace("'", "")
    Ks = Ks.replace("[", "")
    Ks = Ks.replace("]", "")
    Ks = [int(elem) for elem in Ks.split(',')]
    k_min = min(Ks)
    k_max = max(Ks)
    n_epochs = task_n_epochs
    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    model.check_2 = False
    # train model
    new_lr = args.lr
    model.hard_or_random = args.hard_negative_sampling 
    for key in data.test_user_dict:
        print(key)
        print(data.test_user_dict[key])
        break
    print("---------------------------------SEP----------------------------------------------")
    old_epochs = -1
    if n_epochs == 0:
        #return model, data
        n_epochs += 1
        old_epochs = 0
    start_training_time = time()
    for epoch in range(1, n_epochs + 1):
        data.check_multi_task = False
        data.load_cf_basic_full(data.n_entities)
        time0 = time()
        model.check = args.dropout_net
        model.check_2 = False
        model.train()
        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            if old_epochs == 0:
                break
            if not args.train_cf:
                break
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')
            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()
            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0 and args.train_cf and old_epochs != 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))
        logging.info('CF Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            if True:
                for j in range(len(task_ids)):
                    print('Task id {}'.format(task_ids[j]))
                    data.check_multi_task = True
                    data.load_full_cf(task_ids[j], data.n_entities)
                    time6 = time()
                    _, metrics_dict = evaluate(model, data, Ks, device, 0, args.masking_train, items_out_task = None)
                    logging.info('Task id {} CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                        task_ids[j], epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

                    epoch_list.append(epoch)
                    if j == index:
                        for k in Ks:
                            for m in ['precision', 'recall', 'ndcg']:
                                metrics_list[k][m].append(metrics_dict[k][m])
                        best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)
                        if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                            save_model(model, args.save_dir, epoch, best_epoch)
                            logging.info('Save model on epoch {:04d}!'.format(epoch))
                            best_epoch = epoch
                    if args.one_or_two == 2:
                        model.check_2 = True
                    time6 = time()
                    try:
                      _, metrics_dict = evaluate(model, data, Ks, device, 1, items_out_task = None)
                      logging.info('Task id {} CF Evaluation (Cold - start): Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                          task_ids[j], epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
                    except:
                      print('No cold start')
                    print("---------------------------------SEP----------------------------------------------")
                    
                    
    print(f'Training time is {time() - start_training_time}')
                
                
    if old_epochs == 0:
        return model, data
    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    #best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    #logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        #int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))
    
    return model, data


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))



if __name__ == '__main__':
    model = None
    data = None
    args = parse_kgat_args()
    tasks_n_epochs = [int(elem) for elem in args.task_n_epochs.split(',')]
    model, data = train(args, 0, args.n_epoch, model, data)
    # predict(args)


