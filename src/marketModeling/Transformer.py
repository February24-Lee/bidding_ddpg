import torch
from torch import nn
import torch.nn.functional as F

from .TransformerModels import Encoder, Decoder, PositionalEncoding, EncoderSubLayer

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        reshaped_input = input.view(input.size(0), -1)
        return reshaped_input

class NeuralNet(nn.Module):
    def __init__(self, vocal_size, num_features, num_outputs, h1, h2,
                 emb_dropout, lin_dropout):
        super(NeuralNet, self).__init__()
        embedding_dim = int(vocal_size ** 0.25)
        self.layers = nn.Sequential(
            nn.Embedding(vocal_size, embedding_dim),
            Flatten(),
            nn.Dropout(emb_dropout),
            nn.Linear(embedding_dim * num_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(True),
            nn.Dropout(lin_dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(True),
            nn.Dropout(lin_dropout),
            nn.Linear(h2, num_outputs),
            nn.Sigmoid())
        self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out

class Transformer(nn.Module):
    def __init__(self, vocal_size, num_features, num_outputs,
                        model_dim, drop_prob, point_wise_dim, num_sublayer, num_head, is_cuda):
        super().__init__()
        self.embedding_dim = int(vocal_size ** 0.25)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.model_dim = model_dim

        # output torch.Size([128, 301])
        self.embedding = nn.Embedding(vocal_size, self.embedding_dim)

        # Middle layer size is model_dim - 1
        self.middle_layer = nn.Linear(self.embedding_dim * num_features, model_dim)

        self.encoder_sublayers = nn.Sequential(
            *[EncoderSubLayer(model_dim, num_head, drop_prob, point_wise_dim) for _ in range(num_sublayer)])
        self.encode_pos_encoder = PositionalEncoding(model_dim, drop_prob, is_cuda)

        self.linear = nn.Linear(self.model_dim, 1)
        

    def forward(self, x):
        batch_size = x.size(0)
        emb_x = self.embedding(x)
        input = emb_x.view(batch_size, self.num_features * self.embedding_dim)
        middle_layer = F.relu(self.middle_layer(input))

        # Not using DRSA encoding and it repeat inside element so there no need for mapping.
        # (~60% confidence)
        input_x = middle_layer.unsqueeze(1).repeat(1, self.num_outputs, 1)
        encoded_inp_pos = self.encode_pos_encoder(input_x)
        encoded_input = self.encoder_sublayers(encoded_inp_pos)
        outputs = encoded_input

        new_output = outputs.view(self.num_outputs * batch_size, self.model_dim)
        preds = torch.transpose(torch.sigmoid(self.linear(new_output)), 0, 1)[0]
        return preds.view(batch_size, self.num_outputs)
