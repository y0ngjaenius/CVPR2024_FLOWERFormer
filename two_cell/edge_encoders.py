import torch


class DummyEdgeEncoder(torch.nn.Module):
    """Reference: https://github.com/rampasek/GraphGPS"""

    def __init__(self, emb_dim, cfg):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        return batch


edge_encoder_dict = {"DummyEdge": DummyEdgeEncoder}
