from abc import abstractmethod
import torch

from model.modules import *


class Encoder(nn.Module):
    def __init__(self, args, factor=True):
        super(Encoder, self).__init__()
        self.args = args
        self.factor = factor

    def node2edge_temporal(self, inputs, rel_matrix):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.

        # [num_sims, num_atoms, num_timesteps * num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        edges = torch.matmul(rel_matrix, x)
        edges = edges.view(
            inputs.size(0) * edges.size(1), inputs.size(2), inputs.size(3)
        )
        edges = edges.transpose(2, 1)

        # [num_sims * num_edges, num_dims, num_timesteps]
        return edges

    def edge2node(self, x, rel_matrix):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_matrix):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        edges = torch.matmul(rel_matrix, x)
        return edges

    @abstractmethod
    def forward(self, inputs, rel_matrix, mask_idx=None):
        pass
