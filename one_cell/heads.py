import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.graphgym.models.layer import MLP, new_layer_config


class GNNGraphHead(nn.Module):
    """
    Reference: https://github.com/rampasek/GraphGPS
    GNN prediction head for graph prediction tasks.
    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg)
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


head_dict = {"graph": GNNGraphHead}
