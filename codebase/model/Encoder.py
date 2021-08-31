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

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(
            inputs.size(0) * receivers.size(1), inputs.size(2), inputs.size(3)
        )
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(
            inputs.size(0) * senders.size(1), inputs.size(2), inputs.size(3)
        )
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_matrix):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_matrix):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    @abstractmethod
    def forward(self, inputs, rel_rec, rel_send, mask_idx=None):
        pass
