import torch
import torch.nn as nn
from torch_geometric.data import Batch
from performer_pytorch import SelfAttention
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import norm
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode

from .heads import head_dict
from .dag_attention import Attention
from .node_encoders import node_encoder_dict
from .edge_encoders import edge_encoder_dict
from .async_mpnn import DAGLayer


class GPSLayer(nn.Module):
    """
    Minimal form of GPS layer (https://github.com/rampasek/GraphGPS).
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
                dim_h,
                dim_h,
                dropout,
                dag_cfg.bidirectional,
                dag_cfg.conv_type,
                dag_cfg=dag_cfg,
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
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            if self.local_gnn_type == "DAG":
                local_out = self.local_model(batch)
                h_local = local_out.x
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.
            elif self.local_gnn_type == "CustomGatedGCN":
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                )
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            if self.attn_after_mpnn:
                h = h_local
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type == "DAG":
                h_attn, _ = self.self_attn(
                    h,
                    dag_rr_edge_index=batch.dag_rr_edge_index,
                )
            elif self.global_model_type == "Performer":
                h_dense, mask = to_dense_batch(h, batch.batch)
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            if self.attn_after_mpnn:
                h_attn = h_local + h_attn
                h_attn = self.norm1_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        if self.self_attn is not None and self.attn_after_mpnn:
            h = h_attn
        else:
            h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

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
        for module in self.children():
            batch = module(batch)
        return batch


class FLOWERFormer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, cfg)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
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

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

    def embedding(self, batch):
        with torch.no_grad():
            for module in list(self.children())[:-1]:
                batch = module(batch)
            batch = self.post_mp.embedding(batch)
        return batch
