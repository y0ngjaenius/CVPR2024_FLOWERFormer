import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import norm
from performer_pytorch import SelfAttention
from torch_geometric.utils import to_dense_batch
import torch_geometric.graphgym.register as register
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from torch_geometric.graphgym.models.layer import (
    new_layer_config,
    BatchNorm1dNode,
    LayerConfig,
)


from async_mpnn import DAGLayer
from dag_attention import Attention
from heads import head_dict
from node_encoders import node_encoder_dict
from edge_encoders import edge_encoder_dict


class GPSLayer(nn.Module):
    """
    Minimal form of GPS layer (https://github.com/rampasek/GraphGPS) and modification for two-cell.
    """

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        bigbird_cfg=None,
        log_attn_weights=False,
        dag_cfg=None,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = nn.ReLU

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in [
            "Transformer",
            "BiasedTransformer",
        ]:
            raise NotImplementedError(
                f"Logging of attention weights is not supported " f"for '{global_model_type}' global attention model."
            )
        self.attn_after_mpnn = dag_cfg.attn_after_mpnn
        self.ff = dag_cfg.ff

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == "None":
            self.local_model = None

        elif local_gnn_type == "DAG":
            self.local_model = DAGLayer(
                dim_h, dim_h, dropout, dag_cfg.bidirectional, dag_cfg.conv_type, dag_cfg=dag_cfg
            )
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                act="relu",
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == "None":
            self.self_attn = None
        elif global_model_type == "DAG":
            self.self_attn = Attention(dim_h, num_heads, dropout=self.attn_dropout, bias=False)
        elif global_model_type == "Performer":
            self.self_attn = SelfAttention(dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False)
        else:
            raise ValueError(f"Unsupported global x-former model: " f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = norm.LayerNorm(dim_h)
            self.norm1_attn = norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        batch1, batch2 = batch
        h1 = batch1.x
        h2 = batch2.x
        h_in1 = h1
        h_in2 = h2
        h1_out_list = []
        h2_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            if self.local_gnn_type == "DAG":
                local_out1, local_out2 = self.local_model(batch1, batch2)
                h_local_1, h_local_2 = local_out1.x, local_out2.x
                h_local_1, h_local_2 = self.dropout_local(h_local_1), self.dropout_local(h_local_2)
                h_local_1, h_local_2 = h_in1 + h_local_1, h_in2 + h_local_2
            if self.layer_norm:
                h_local_1, h_local_2 = self.norm1_local(h_local_1, batch1.batch), self.norm1_local(
                    h_local_2, batch2.batch
                )
            if self.batch_norm:
                h_local_1, h_local_2 = self.norm1_local(h_local_1), self.norm1_local(h_local_2)
            if self.attn_after_mpnn:
                h1 = h_local_1
                h2 = h_local_2
            h1_out_list.append(h_local_1)
            h2_out_list.append(h_local_2)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type == "DAG":
                h_attn_1, _ = self.self_attn(
                    h1,
                    dag_rr_edge_index=batch1.dag_rr_edge_index,
                )
                h_attn_2, _ = self.self_attn(
                    h2,
                    dag_rr_edge_index=batch2.dag_rr_edge_index,
                )
            elif self.global_model_type == "Performer":
                h_dense, mask = to_dense_batch(h1, batch1.batch)
                h_attn_1 = self.self_attn(h_dense, mask=mask)[mask]
                h_dense, mask = to_dense_batch(h2, batch2.batch)
                h_attn_2 = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn_1, h_attn_2 = self.dropout_attn(h_attn_1), self.dropout_attn(h_attn_2)
            h_attn_1, h_attn_2 = h_in1 + h_attn_1, h_in2 + h_attn_2
            if self.layer_norm:
                h_attn_1, h_attn_2 = self.norm1_attn(h_attn_1, batch1.batch), self.norm1_attn(h_attn_2, batch2.batch)
            if self.batch_norm:
                h_attn_1, h_attn_2 = self.norm1_attn(h_attn_1), self.norm1_attn(h_attn_2)
            h1_out_list.append(h_attn_1)
            h2_out_list.append(h_attn_2)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        if self.self_attn is not None and self.attn_after_mpnn:
            h1, h2 = h_attn_1, h_attn_2
        else:
            h1, h2 = sum(h1_out_list), sum(h2_out_list)

        # Feed Forward block.
        h1, h2 = h1 + self._ff_block(h1), h2 + self._ff_block(h2)
        if self.layer_norm:
            h1, h2 = self.norm2(h1, batch1.batch), self.norm2(h2, batch2.batch)
        if self.batch_norm:
            h1, h2 = self.norm2(h1), self.norm2(h2)
        batch1.x = h1
        batch2.x = h2
        return batch1, batch2

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in, cfg):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, cfg)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False, has_bias=False, cfg=cfg)
                )
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, cfg)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg)
                )

    def forward(self, batch):
        batch1, batch2 = batch
        for module in self.children():
            batch1 = module(batch1)
            batch2 = module(batch2)
        return batch1, batch2


class GeneralLayer(nn.Module):
    """
    General wrapper for layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """

    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps, momentum=layer_config.bn_mom)
            )
        if layer_config.dropout > 0:
            layer_wrapper.append(nn.Dropout(p=layer_config.dropout, inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    """
    General wrapper for a stack of multiple layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        num_layers (int): Number of layers in the stack
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): The dimension for the inner layers
        final_act (bool): Whether has activation after the layer stack
        **kwargs (optional): Additional args
    """

    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_out if layer_config.dim_inner is None else layer_config.dim_inner
        for i in range(layer_config.num_layers):
            d_in = layer_config.dim_in if i == 0 else dim_inner
            d_out = layer_config.dim_out if i == layer_config.num_layers - 1 else dim_inner
            has_act = layer_config.final_act if i == layer_config.num_layers - 1 else True
            inter_layer_config = copy.deepcopy(layer_config)
            inter_layer_config.dim_in = d_in
            inter_layer_config.dim_out = d_out
            inter_layer_config.has_act = has_act
            layer = GeneralLayer(name, inter_layer_config, **kwargs)
            self.add_module("Layer_{}".format(i), layer)

    def forward(self, batch):
        batch1, batch2 = batch
        for layer in self.children():
            batch1 = layer(batch1)
            batch2 = layer(batch2)
        return batch1, batch2


def GNNPreMP(dim_in, dim_out, num_layers, cfg):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer(
        "linear",
        layer_config=new_layer_config(dim_in, dim_out, num_layers, has_act=False, has_bias=False, cfg=cfg),
    )


class FLOWERFormer2Cell(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, cfg)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp, cfg)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split("+")
        except Exception:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                GPSLayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    num_heads=cfg.gt.n_heads,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    bigbird_cfg=cfg.gt.bigbird,
                    log_attn_weights=cfg.train.mode == "log-attn-weights",
                    dag_cfg=cfg.dag,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out, cfg=cfg)

    def forward(self, batch1, batch2):
        for module in self.children():
            batch1, batch2 = module((batch1, batch2))
        return batch1, batch2

    def embedding(self, batch1, batch2):
        with torch.no_grad():
            for module in list(self.children())[:-1]:
                batch1, batch2 = module((batch1, batch2))
            emb1, emb2 = self.post_mp.embedding(batch1, batch2)
        return emb1, emb2
