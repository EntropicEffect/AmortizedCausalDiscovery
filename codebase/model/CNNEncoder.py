from model.modules import *
from model.Encoder import Encoder


class CNNEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self, args, n_in, n_hid, n_out, do_prob=0.0
    ):
        super().__init__(args)

        self.cnn = CNN(n_in, n_hid, n_hid, do_prob)

        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)

        self.fc_out = nn.Linear(n_hid, n_out)

    def forward(self, inputs):

        x = self.cnn(inputs)
        print("x shape: ", x.shape)
        x = x.view(inputs.size(0), inputs.size(1) * inputs.size(1), -1)
        x = self.mlp1(x)
        return self.fc_out(x)
