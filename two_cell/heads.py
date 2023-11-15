import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.graphgym.models.layer import MLP, new_layer_config


class GNNGraphHead(nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(
                dim_in, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg
            )
        )
        self.pooling_fun = global_mean_pool

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

    def embedding(self, batch):
        with torch.no_grad():
            graph_emb = self.pooling_fun(batch.x, batch.batch)
        return graph_emb


class GNNGraphHead2Cell(nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(
                dim_in * 2, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg
            )
        )
        self.pooling_fun = global_mean_pool

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        batch1, batch2 = batch
        graph1_emb, graph2_emb = self.pooling_fun(batch1.x, batch1.batch), self.pooling_fun(
            batch2.x, batch2.batch
        )
        graph_emb = self.layer_post_mp(torch.cat([graph1_emb, graph2_emb], dim=-1))
        batch1.graph_feature = batch2.graph_feature = graph_emb
        pred, label = self._apply_index(batch1)
        return pred, label

    def embedding(self, batch):
        batch1, batch2 = batch
        with torch.no_grad():
            graph1_emb, graph2_emb = self.pooling_fun(batch1.x, batch1.batch), self.pooling_fun(
                batch2.x, batch2.batch
            )
        return graph1_emb, graph2_emb


head_dict = {
    "graph": GNNGraphHead,
    "graph2cell": GNNGraphHead2Cell,
}
