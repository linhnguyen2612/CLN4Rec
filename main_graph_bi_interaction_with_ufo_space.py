import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

from model.Graph_Bi_interaction_with_UFO_SPACE import Graph_Bi_interaction_with_UFO_SPACE
from parsers.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT_MultiTask_full_2
from torch.nn import Linear
from mask_optimization_ufo_space_7 import *

sys.path.append("/content/drive/MyDrive/KGAT-pytorch-master")
def evaluate(model, dataloader, Ks, device, test_cold, masking_train = 1, items_out_task = None, return_one = False, user_ids_used = None):
    
    train_user_dict = dataloader.train_user_dict
    if test_cold == 0:
        test_batch_size = dataloader.test_batch_size
        test_user_dict = dataloader.test_user_dict
    elif test_cold == 1:
        test_batch_size = dataloader.test_cold_batch_size
        test_user_dict = dataloader.test_cold_user_dict

    model.eval()
    user_ids = list(test_user_dict.keys())
    if user_ids_used is not None:
        user_ids = list(user_ids_used.reshape(-1))
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
    if pre_data is None:
        data = DataLoaderKGAT_MultiTask_full_2(args, logging)
    else:
        data = pre_data
    data.load_full_cf(task_ids[index], data.n_entities)
    #data.device = device
    if pretrain_model is None:
        if args.use_pretrain == 1:
            user_pre_embed = torch.tensor(data.user_pre_embed)
            item_pre_embed = torch.tensor(data.item_pre_embed)
        else:
            user_pre_embed, item_pre_embed = None, None
        print(data.max_id)
        # construct model & optimizer
        n_items = data.n_items
        item_ids = torch.arange(n_items, dtype=torch.long).to(device)
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        model =  Graph_Bi_interaction_with_UFO_SPACE(args, data.n_users, data.n_entities, data.n_relations,
                                                     device,
                                                     item_ids,
                                                     data.A_in, user_pre_embed, item_pre_embed,
                                                     task_ids,
                                                     data.n_cf_train / args.weight_n_cf_train,
                                                     users_group = data.users_group,
                                                     items_group = data.items_group,
                                                     n_clusters_users = data.n_clusters_users,
                                                     n_clusters_items = data.n_clusters_items)
        if args.use_pretrain == 2:
            model = load_model(model, args.pretrain_model_path)
        model.task_ids_index = index
        model.load_task_id(device)
        model.to(device)
        cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = pretrain_model
        model.just_tunning_embedding = True
        if args.use_task_mask and args.use_task_mask_for_gradient_protecting:
            cf_optimizer = MaskedAdamUFO_SPACE_7(params = model.parameters(), lr=args.lr, 
                                                 index = index,
                                                 task_ids = task_ids, 
                                                 device = device,
                                                 limits = args.limits,
                                                 n_users = data.n_users,
                                                 n_entities = data.n_entities,
                                                 n_items = data.n_items)
        else:
            cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model.task_ids_index = index
        model.load_task_id(device)
        #model.task_id = args.downstream_task_id
    logging.info(model)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    user_ids_used_for_new_mask = None
    check_first_time_enable_multi_task_masks_mode = True
    check_increase_n_true_task_masks = True
    if n_epochs == 0:
        #return model, data
        n_epochs += 1
        old_epochs = 0
    count = 0
    start_training_time = time()
    for epoch in range(1, n_epochs + 1):
        model.check_test = False
        #if epoch % args.use_two_loss_checkpoint == 0:
            #model.use_two_loss = True
        if args.use_task_mask:
            if epoch > args.epoch_not_binary_mask or old_epochs == 0:
                model.model_epoch_binary_for_multi_tasks[index] = True
        model.use_two_loss = False
        time0 = time()
        model.train()
        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        if args.data_name == 'douban': n_cf_batch = 10
        #if args.data_name == 'ml_1m': n_cf_batch = 100
        full_total_loss = 0
        task_mask_similarity_total_loss = 0
        if epoch % args.one_task_mask_epoch_frame:
            model.just_one_task_mask_for_users = 1
            model.just_one_task_mask_for_items = 1
        else:
            model.just_one_task_mask_for_users = 0
            model.just_one_task_mask_for_items = 0
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
            cf_batch_loss, task_mask_similarity_loss, full_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_full_loss')
            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()
            if epoch != n_epochs and not (epoch >= 33 and index > 0):
                full_batch_loss.backward()
            else:
                cf_batch_loss.backward()
            try:
                cf_optimizer.step()
            except:
                cf_optimizer.step(model)
            cf_optimizer.zero_grad()
            #if index > 0 and args.hard_optimizer:
              #model.entity_user_embed.weight = nn.Parameter(torch.from_numpy(cf_optimizer.old_embedding_weight).to(torch.float32).to(device))
            cf_total_loss += cf_batch_loss.item()
            full_total_loss += full_batch_loss.item()
            try:
                task_mask_similarity_total_loss += task_mask_similarity_loss.item()
            except:
                task_mask_similarity_total_loss += task_mask_similarity_loss
            try:
                task_mask_similarity_loss_copy = task_mask_similarity_loss.item()
            except:
                task_mask_similarity_loss_copy = task_mask_similarity_loss
            if (iter % args.cf_print_every) == 0 and args.train_cf and old_epochs != 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter CF Loss {:.4f} | Iter task mask similarity Loss {:.4f} |Iter full Loss {:.4f} |Iter Mean CF Loss {:.4f} |Iter Mean task mask similarity Loss {:.4f} |Iter Mean full Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), task_mask_similarity_loss_copy, full_batch_loss.item(), cf_total_loss / iter, task_mask_similarity_total_loss / iter, full_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean CF Loss {:.4f} | Iter Mean task mask similarity Loss {:.4f} | Iter Mean full Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch, task_mask_similarity_total_loss / n_cf_batch, full_total_loss / n_cf_batch))
        if epoch % args.use_two_loss_checkpoint == 0:
            model.use_two_loss = False
            
        logging.info('CF Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
        #print(model.check)
        # evaluate cf
        model.check_test = True
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            if index == 0:
                time6 = time()
                _, metrics_dict = evaluate(model, data, Ks, device, 0, args.masking_train, items_out_task = None)
                logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                    epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

                epoch_list.append(epoch)
                for k in Ks:
                    for m in ['precision', 'recall', 'ndcg']:
                        print('loz')
                        print(metrics_dict[k][m])
                        metrics_list[k][m].append(metrics_dict[k][m])
                best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)

                #if should_stop:
                    #break

                if metrics_list[k_max]['recall'].index(best_recall) == len(epoch_list) - 1:
                    old_model_state_file = save_model(model, args.save_dir, epoch, best_epoch, task_ids[index])
                    logging.info('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch
                
            else:
                old_epoch_binary = model.epoch_binary
                for j in range(index + 1):
                    print('Task id {}'.format(task_ids[j]))
                    data.load_full_cf(task_ids[j], data.n_entities)
                    model.task_ids_index = j
                    #print()
                    model.load_task_id(device) 
                    if j < index:
                        model.epoch_binary = True
                        with torch.no_grad():
                          h_list = data.h_list.to(device)
                          t_list = data.t_list.to(device)
                          r_list = data.r_list.to(device)
                          relations = list(data.laplacian_dict.keys())
                          #model(h_list, t_list, r_list, relations, mode='update_att')
                        #with torch.no_grad():
                            #print(model.get_task_embedding_binary())
                    else:
                        model.epoch_binary = old_epoch_binary
                        h_list = data.h_list.to(device)
                        t_list = data.t_list.to(device)
                        r_list = data.r_list.to(device)
                        relations = list(data.laplacian_dict.keys())
                        #model(h_list, t_list, r_list, relations, mode='update_att')
                    time6 = time()
                    _, metrics_dict = evaluate(model, data, Ks, device, 0, args.masking_train, items_out_task = None)
                    logging.info('Task id {} CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                        task_ids[j], epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

                    
                    if True:
                        for k in Ks:
                            for m in ['precision', 'recall', 'ndcg']:
                                if j == 0:
                                    metrics_list[k][m].append([])
                                metrics_list[k][m][-1].append(metrics_dict[k][m])
                epoch_list.append(epoch)
                if True:
                    if True:
                        for k in Ks:
                            for m in ['precision', 'recall', 'ndcg']:
                                metrics_list[k][m][-1] = sum(metrics_list[k][m][-1])/len(metrics_list[k][m][-1])
                        best_recall, should_stop = early_stopping(metrics_list[k_max]['recall'], args.stopping_steps)
                        if True:
                            if metrics_list[k_max]['recall'].index(best_recall) == len(epoch_list) - 1:
                                old_model_state_file = save_model(model, args.save_dir, epoch, best_epoch, task_ids[index])
                                logging.info('Save model on epoch {:04d}!'.format(epoch))
                                best_epoch = epoch
    print(f'Training time is {time() - start_training_time}')
                    
            #if should_stop:
                #break
    if True:
        model = load_model(model, old_model_state_file)
        print(old_model_state_file)
    else:
        print('No epochs')      
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
    
    #model.old_embedding = model.entity_user_embed.weight.detach().cpu().numpy()
    return model, data



if __name__ == '__main__':
    model = None
    data = None
    args = parse_kgat_args()
    tasks_n_epochs = [int(elem) for elem in args.task_n_epochs.split(',')]
    for index, n_epochs in enumerate(tasks_n_epochs):
        model, data = train(args, index, n_epochs, model, data)
    # predict(args)


