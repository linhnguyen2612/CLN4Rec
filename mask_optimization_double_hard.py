import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

from collections import defaultdict
from copy import deepcopy
from itertools import chain


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


# ----------------- copy from pytorch-bert-pretrained package ---------------------------------


class MaskedAdamDoubleHard(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, index = None, task_ids = None, embedding_task_layer = None, embedding_task_layer_attention_score = None, device = None, limit = None, n_users = None, n_entities = None, n_items = None, items_out_task = None):
        if lr is not required and lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(MaskedAdamDoubleHard, self).__init__(params, defaults)
        with torch.no_grad():
            self.task_id = torch.tensor([elem for elem in task_ids[:index]], dtype = torch.long).view(1, -1).to(device)
            print(self.task_id.shape)
            self.task_mask_embedding = torch.sigmoid(embedding_task_layer(self.task_id)).detach().cpu()
            self.task_mask_embedding, _ = torch.max(self.task_mask_embedding, dim = 1)
            print(self.task_mask_embedding.shape)
            self.task_mask_embedding[self.task_mask_embedding < limit] = 0.
            self.task_mask_embedding[self.task_mask_embedding >= limit] = 1.
            self.task_mask_embedding_attention_score = torch.sigmoid(embedding_task_layer_attention_score(self.task_id)).detach().cpu()
            self.task_mask_embedding_attention_score, _ = torch.max(self.task_mask_embedding_attention_score, dim = 1)
            self.task_mask_embedding_attention_score[self.task_mask_embedding_attention_score < limit] = 0.
            self.task_mask_embedding_attention_score[self.task_mask_embedding_attention_score >= limit] = 1.
            self.task_mask_embedding = torch.concat([self.task_mask_embedding, self.task_mask_embedding_attention_score], dim = 0)
            self.task_mask_embedding, _ = torch.max(self.task_mask_embedding, dim = 0)
            self.task_mask_embedding = self.task_mask_embedding.to(device).view(1, -1)
            self.task_mask_embedding = 1 - self.task_mask_embedding
            print('Percentage of nonzero values in mask  is {}'.format(self.task_mask_embedding[0][self.task_mask_embedding[0] == 0].shape[0]/self.task_mask_embedding[0].shape[0]))
            self.n_users = n_users
            self.n_entities = n_entities
            self.n_items = n_items
            self.items_out_task = torch.tensor(items_out_task, dtype = torch.long).to(device)
            print(self.n_items)
            print(self.n_entities)
            print(self.n_users)
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None, t=None, list_embedding_masks = None, task_id = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p_id,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')


                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])



                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD. 
                if p.grad.data.shape[0] == self.n_users + self.n_entities and p.grad.data.shape[1] == self.task_mask_embedding.shape[1]:
                    #print(update)
                    update[self.n_entities : ] *= self.task_mask_embedding
                    update[ : self.n_items] *= self.task_mask_embedding
                    #print(update)
                    #print(self.task_mask_embedding)
                    #print(update.shape)
                else:
                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data
                    

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']




        return 