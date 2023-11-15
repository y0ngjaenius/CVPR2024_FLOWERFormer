import math

import torch
from torch import nn


class DAGNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, cfg):
        super().__init__()

        dim_in = cfg.dag.dim_in
        self.pe_use = cfg.dag.pe

        self.pe_encoder = nn.Embedding(num_embeddings=4, embedding_dim=emb_dim)
        self.encoder = nn.Linear(dim_in, emb_dim)
        self.position = torch.arange(500).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))

        self.pe = torch.zeros(500, emb_dim)
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)

    def forward(self, batch):
        device = batch.x.device
        batch.x = self.encoder(batch.x)
        if self.pe_use:
            dagpe = batch.abs_pe
            self.pe = self.pe.to(device)
            dagpe = self.pe[: dagpe.shape[0]][dagpe]
            batch.x = batch.x + dagpe
        return batch


node_encoder_dict = {"DAGNode": DAGNodeEncoder}
