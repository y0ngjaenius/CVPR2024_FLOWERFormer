import sys
import json
import yaml
import warnings

warnings.filterwarnings("ignore")

import torch
from tqdm import trange

from utils import (
    cosine_with_warmup_scheduler,
    seed_everything,
    Option,
    get_data,
    get_data_2cell,
    get_perms,
)
from train_utils import train_dict
from one_cell.flowerformer import FLOWERFormer
from two_cell.flowerformer_2cell import FLOWERFormer2Cell


def get_model(cfg, device):
    model_dict = {
        "1-FLOWERFormer": FLOWERFormer,
        "2-FLOWERFormer": FLOWERFormer2Cell,
    }

    if cfg.dataset.name in ["nb101", "nb201", "nbgraph", "nbasr"]:
        num_cell = "1"
    else:
        num_cell = "2"
    model_name = cfg.model.type
    model_code = num_cell + "-" + model_name
    model = model_dict[model_code](cfg.dag.dim_in, 1, cfg)
    model = model.to(device)
    if num_cell == "1":
        train_func, eval_func = train_dict["1cell"]
    else:
        train_func, eval_func = train_dict["2cell"]
    get_data_func = get_data if num_cell == "1" else get_data_2cell
    num_train = {
        "nb101": 7290,
        "nb201": 7813,
        "nb301": 5896,
        "nbgraph": 13103,
        "nbasr": 4121,
    }
    return model, train_func, eval_func, get_data_func, num_train[cfg.dataset.name]


def main(cfg):
    seeds = [21, 2021, 202121]
    data = torch.load(cfg.dataset.dir)
    other_domain = cfg.dataset.name in ["nbgraph", "nbasr"]
    perms, _ = get_perms(seeds, len(data), other_domain)
    train_ratios = [0.5, 0.1, 0.05, 0.01]

    device = cfg.accelerator
    results = {train_ratio: {"kt": [], "sp": [], "pak": []} for train_ratio in train_ratios}
    for train_ratio in train_ratios:
        for seed in seeds:
            for perm in perms:
                seed_everything(seed)
                model, train_func, eval_func, get_data_func, num_train = get_model(cfg, device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay
                )
                scheduler = cosine_with_warmup_scheduler(
                    optimizer, cfg.optim.num_warmup_epochs, cfg.optim.max_epoch
                )
                train_loader, val_loader, test_loader = get_data_func(
                    data, perm, cfg.train.batch_size, train_ratio=train_ratio, num_train=num_train
                )
                best_val_kt = 0
                best_val_model = model.state_dict()
                for epoch in trange(cfg.optim.max_epoch):
                    train_loss = train_func(model, train_loader, optimizer, device, cfg)
                    val_metrics = eval_func(model, val_loader, device)
                    print(
                        "Epoch: {}, Train Loss: {}, Val KT: {}, Val SP: {}".format(
                            epoch + 1, train_loss, val_metrics["kt"], val_metrics["sp"]
                        )
                    )
                    if val_metrics["kt"] > best_val_kt:
                        best_val_kt = val_metrics["kt"]
                        best_val_model = model.state_dict()

                    scheduler.step()
                model.load_state_dict(best_val_model)

                test_metrics = eval_func(model, test_loader, device)
                results[train_ratio]["kt"].append(test_metrics["kt"])
                results[train_ratio]["sp"].append(test_metrics["sp"])
                results[train_ratio]["pak"].append(test_metrics["pak"])
                pak_unpack = [
                    test_metrics["pak"][ratio][3] for ratio in [0.01, 0.05, 0.1, 0.5, 1.0]
                ]
                print("Test KT: {}, Test P@k: {}".format(test_metrics["kt"], pak_unpack))
    return results


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        yaml_object = yaml.safe_load(f)
    cfg = Option(yaml_object)
    results = main(cfg)
    dataset, model = cfg.dataset.name, cfg.model.type
    with open("/output/{}_{}.json".format(dataset, model), "w") as f:
        json.dump(results, f)
