import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_max_pool, global_mean_pool


def make_edge_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class GCNWithEdgeWeights(nn.Module):
    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 graph_feat_dim: int,
                 embedding_size: int):
        super().__init__()
        # convolutional layers
        self.conv0 = NNConv(
            node_feat_dim,
            embedding_size,
            make_edge_mlp(edge_feat_dim, 8, node_feat_dim * embedding_size),
            aggr="max",
        )
        self.conv1 = NNConv(
            embedding_size,
            embedding_size,
            make_edge_mlp(edge_feat_dim, 8, embedding_size * embedding_size),
            aggr="max",
        )
        self.conv2 = NNConv(
            embedding_size,
            embedding_size,
            make_edge_mlp(edge_feat_dim, 8, embedding_size * embedding_size),
            aggr="max",
        )
        # fully connected classifier
        self.fc1 = nn.Linear(embedding_size * 2 + graph_feat_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, edge_index, edge_attr, batch, graph_features):
        h = self.conv0(x, edge_index, edge_attr)
        h = F.relu6(h)
        h = self.conv1(h, edge_index, edge_attr)
        h = F.relu6(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu6(h)
        # global pooling
        pooled = torch.cat([
            global_max_pool(h, batch),
            global_mean_pool(h, batch)
        ], dim=1)
        # concatenate with external graph features
        combined = torch.cat([pooled, graph_features], dim=1)
        out_h = F.relu(self.fc1(combined))
        logits = self.fc2(out_h)
        return logits.view(-1), combined


class WrappedModel(nn.Module):
    """Wrap GCN to return only logits for explainers."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x, edge_index, edge_attr, batch, graph_features):
        out, _ = self.base(x, edge_index, edge_attr, batch, graph_features)
        return out