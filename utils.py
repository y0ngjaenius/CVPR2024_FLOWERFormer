import os
import json
import math
import random
from typing import List

import torch
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from scipy.stats import kendalltau, spearmanr, rankdata, stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def seed_everything(seed: int = 29):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def evaluation(pred, true):
    loss = mean_squared_error(pred, true)
    mae_loss = mean_absolute_error(pred, true)

    r2 = r2_score(pred, true)

    pred_rank = rankdata(pred)
    true_rank = rankdata(true)
    tau, _ = kendalltau(pred_rank, true_rank)
    coeff, _ = spearmanr(pred_rank, true_rank)

    metric = {"mse": loss, "mae": mae_loss, "r2": r2, "kt": tau, "sp": coeff}

    return metric


def collate_edge_masks(edge_masks_list, total_num_edges):
    """collates list of edge mask tensors from many dags. Since the dags vary
    in depth, edge mask tensors are padded to the maximum depth.
    """
    max_depth = max(edge_masks.shape[0] for edge_masks in edge_masks_list)

    # output tensor that will contain all the edge masks
    edge_masks_collated = torch.zeros((max_depth, total_num_edges), dtype=bool)

    i = 0
    for edge_masks in edge_masks_list:
        # copy these masks into the output tensor
        depth, num_edges = edge_masks.shape
        if depth > 0:
            edge_masks_collated[:depth, i : (i + num_edges)] = edge_masks
        i += num_edges

    return edge_masks_collated


def collate_fn(data_list):
    data_batch = Batch.from_data_list(data_list, exclude_keys=["edge_masks"])
    edge_masks_list = [data.edge_masks for data in data_list]
    data_batch.edge_masks = collate_edge_masks(edge_masks_list, data_batch.num_edges)
    return data_batch


def get_data(data, perm, batch_size, train_ratio, num_train):
    permed_data = [data[i] for i in perm]
    train_data, test_data = permed_data[:num_train], permed_data[num_train:]
    val_data, test_data = test_data[:40], test_data[40:]
    train_data = train_data[: int(num_train * train_ratio)]

    train_loader = DataLoader(
        train_data, collate_fn=collate_fn, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_data, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader


def get_perms(seeds, N, other_domain=None):
    perms = []
    perms_type = ["ori", "2021", "202121"]
    for seed in seeds:
        np.random.seed(seed)
        if seed == 21 and not other_domain:
            perms.append([i for i in range(N)])
        else:
            perms.append(np.random.permutation(N).tolist())
    return perms, perms_type


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, paired_dataset):
        datasetA = [d[0] for d in paired_dataset]
        datasetB = [d[1] for d in paired_dataset]
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx]

    def __len__(self):
        return len(self.datasetA)


def collate_2cell(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list], exclude_keys=["edge_masks"])
    batchB = Batch.from_data_list([data[1] for data in data_list], exclude_keys=["edge_masks"])
    A_edge_masks_list = [data[0].edge_masks for data in data_list]
    B_edge_masks_list = [data[1].edge_masks for data in data_list]
    batchA.edge_masks = collate_edge_masks(A_edge_masks_list, batchA.num_edges)
    batchB.edge_masks = collate_edge_masks(B_edge_masks_list, batchB.num_edges)
    return batchA, batchB


def get_data_2cell(data, perm, batch_size, train_ratio, num_train=5896):
    permed_data = [data[i] for i in perm]
    train_data, test_data = permed_data[:num_train], permed_data[num_train:]
    train_ratio = train_ratio
    _num = len(train_data)
    train_data = train_data[: int(_num * train_ratio)]
    val_data, test_data = test_data[:40], test_data[40:]

    train_dataset = PairDataset(train_data)
    val_dataset = PairDataset(val_data)
    test_dataset = PairDataset(test_data)

    train_loader = DataLoader(
        train_dataset, collate_fn=collate_2cell, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=collate_2cell, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_dataset, collate_fn=collate_2cell, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader


def test_xk(
    true_scores: List[float],
    predict_scores: List[float],
    ratios: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0],
) -> List:
    """
    Calculate p@topK.

    Args:
        true_scores (List[float]): Architectures' actual performances.
        predict_scores (List[float]): Predicted scores of the architectures.
        ratios (List[float]): Top ratios to calculate. Default: [0.01, 0.05, 0.1, 0.5, 1.0]
    """
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    ratios = [r for r in ratios if int(num_archs * r) > 0]
    patks = {}
    for ratio in ratios:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks[ratio] = (
            k,
            ratio,
            len(arch_inds),
            p,
            stats.kendalltau(
                reorder_true_scores[arch_inds], reorder_predict_scores[arch_inds]
            ).correlation,
        )
    return patks


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_with_warmup_scheduler(optimizer, num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_epochs, num_training_steps=max_epoch
    )
    return scheduler


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Option(dict):
    def __init__(self, *args, **kwargs):
        args = [arg if isinstance(arg, dict) else json.loads(open(arg).read()) for arg in args]
        super(Option, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = Option(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = Option(v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Option, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Option, self).__delitem__(key)
        del self.__dict__[key]
