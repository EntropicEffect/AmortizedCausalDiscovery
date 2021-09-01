from abc import abstractmethod
import torch

from model.modules import *


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

    def node2edge_temporal(self, inputs):
        # NOTE: Assumes that we have the same graph across all samples.
        # [num_sims, num_atoms, num_dims, num_timesteps]
        num_atoms = inputs.size(1)

        return edges

    def node2edge(self, x):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        # NOTE: Assumes that we have the same graph across all samples.
        edges = 0
        return edges

        @ abstractmethod
        def forward(self, inputs, mask_idx=None):
            pass
