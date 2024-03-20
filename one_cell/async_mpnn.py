import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, index_to_mask


class AttnConv(MessagePassing):
    """
    Reference: https://github.com/LUOyk1999/DAGformer
    """

    def __init__(self, emb_dim, num_edge_attr=2, reverse_flow=False, dag_cfg=None):
        flow = "target_to_source" if reverse_flow else "source_to_target"
        super().__init__(aggr="add", flow=flow)
        self.edge_lin = nn.Linear(num_edge_attr, emb_dim)
        self.attn_lin = nn.Linear(2 * emb_dim, 1)

    def forward(self, h, h_prev, edge_index, edge_attr):
        edge_emb = self.edge_lin(edge_attr)
        return self.propagate(edge_index, h=h, h_prev=h_prev, edge_emb=edge_emb)

    def message(self, h_j, h_prev_i, edge_emb, index, size_i):
        q, k, v = h_prev_i, h_j + edge_emb, h_j
        alpha_j = self.attn_lin(torch.cat([q, k], -1))
        alpha_j = softmax(alpha_j, index=index, num_nodes=size_i)
        return alpha_j * v


class DAGLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        emb_dim,
        p_dropout,
        bidirectional=True,
        flow_type="fnb",
        dag_cfg=None,
    ):
        """
        Args:
            in_dim: input dimension
            emb_dim: embedding dimension
            p_dropout: dropout probability
            bidirectional: whether to use bidirectional aggregation
            conv_type: type of convolutional layer
                fnb: forward & backward
                fbc: forward/backward separately & concat
                of: only forward
        """
        super().__init__()
        assert bidirectional == (flow_type != "of")
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(p=p_dropout, inplace=True)

        self.agg_forward = AttnConv(in_dim, emb_dim, False, dag_cfg)
        self.combine_forward = nn.GRUCell(emb_dim, emb_dim)
        if bidirectional:
            self.agg_backward = AttnConv(emb_dim, emb_dim, True, dag_cfg)
            self.combine_backward = nn.GRUCell(emb_dim, emb_dim)

        self.type = flow_type
        if flow_type == "fbc":
            self.concat = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, batch):
        h_prev, edge_index, edge_attr, edge_masks = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_masks,
        )
        num_nodes = batch.num_nodes
        root_mask = ~index_to_mask(edge_index[1], num_nodes)

        h_prev = self.dropout(h_prev)
        h_forward = torch.zeros_like(h_prev)
        h_forward[root_mask] = self.combine_forward(h_prev[root_mask])
        edge_masks_it = iter(edge_masks)

        for edge_mask in edge_masks_it:
            # only include edges that connect the previous topo. gen.
            # to the current one
            edge_index_masked = edge_index[:, edge_mask]
            edge_attr_masked = edge_attr[edge_mask]

            msg = self.agg_forward(h_forward, h_prev, edge_index_masked, edge_attr_masked)

            # embed only the current topological generation of nodes
            node_mask = index_to_mask(edge_index_masked[1], num_nodes)
            h_forward[node_mask] = self.combine_forward(h_prev[node_mask], msg[node_mask])

        if self.bidirectional:
            root_mask = ~index_to_mask(edge_index[0], num_nodes)
            edge_masks_it = iter(reversed(edge_masks))
            if self.type == "fnb":
                h_prev = h_forward
            else:
                h_prev = batch.x
            h_prev = self.dropout(h_prev)
            h_backward = torch.zeros_like(h_prev)
            h_backward[root_mask] = self.combine_backward(h_prev[root_mask])
            for edge_mask in edge_masks_it:
                edge_index_masked = edge_index[:, edge_mask]
                edge_attr_masked = edge_attr[edge_mask]

                msg = self.agg_backward(h_backward, h_prev, edge_index_masked, edge_attr_masked)

                node_mask = index_to_mask(edge_index_masked[0], num_nodes)
                h_backward[node_mask] = self.combine_backward(h_prev[node_mask], msg[node_mask])

            if self.type == "fnb":
                batch.x = h_backward

            else:
                h = torch.cat([h_forward, h_backward], dim=1)
                h = self.concat(h)
                batch.x = h

        else:
            batch.x = h_forward
        batch.x = batch.x.relu()
        return batch
