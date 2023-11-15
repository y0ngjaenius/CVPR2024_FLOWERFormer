import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing


class Attention(MessagePassing):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, bias=False, symmetric=False, **kwargs):
        super().__init__(node_dim=0, aggr="add")
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.0)
            nn.init.constant_(self.to_v.bias, 0.0)

    def forward(self, x, dag_rr_edge_index, return_attn=False):
        # Compute value matrix
        v = self.to_v(x)

        x_struct = x

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute self-attention
        attn = None
        if dag_rr_edge_index is not None:
            out = self.propagate(
                dag_rr_edge_index,
                v=v,
                qk=qk,
                edge_attr=None,
                size=None,
                return_attn=return_attn,
            )
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = (
                    torch.sparse_coo_tensor(
                        dag_rr_edge_index,
                        attn,
                    )
                    .to_dense()
                    .transpose(0, 1)
                )
            out = rearrange(out, "n h d -> n (h d)")
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention based on MPNN"""
        qk_i = rearrange(qk_i, "n (h d) -> n h d", h=self.num_heads)
        qk_j = rearrange(qk_j, "n (h d) -> n h d", h=self.num_heads)
        v_j = rearrange(v_j, "n (h d) -> n h d", h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)
