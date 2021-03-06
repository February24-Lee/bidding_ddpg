import torch
import torch.nn as nn
from .Attention import MultiHeadAttention
from .BetweenSubLayer import InBetweenSubLayer
from .PointwiseFeedForward import PointWiseFeedForward
from .PreprocessInput import PreprocessInput

class EncoderSubLayer(nn.Module):
    def __init__(self, model_dim, num_head, drop_prob, pointWise_dim):
        super().__init__()

        self.multi_headed = MultiHeadAttention(model_dim, num_head)
        self.in_between = InBetweenSubLayer(model_dim, drop_prob)
        self.point_wise = PointWiseFeedForward(model_dim, pointWise_dim)
        self.in_between2 = InBetweenSubLayer(model_dim, drop_prob)

    def forward(self, inp):
        after_attention = self.in_between(
            self.multi_headed(inp, inp, inp),
            inp
        )

        output_sub = self.in_between2(
            self.point_wise(after_attention),
            after_attention
        )

        return output_sub

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer, is_cuda):
        super().__init__()
        self.preprocess = PreprocessInput(vocab_size, model_dim, drop_prob, is_cuda)
        self.all_sublayers = nn.Sequential(*[
            EncoderSubLayer(model_dim, num_head, drop_prob, pointWise_dim)
            for _ in range(num_sublayer)
        ])

    def forward(self, inp):
        inp = self.preprocess(inp)
        inp = self.all_sublayers(inp)
        return inp
