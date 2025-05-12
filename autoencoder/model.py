import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                self.encoder_layers.append(
                    nn.Linear(input_dim, encoder_hidden_dims[i]))
            else:
                self.encoder_layers.append(
                    nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                self.encoder_layers.append(nn.ReLU())
                self.encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                self.decoder_layers.append(
                    nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(
                    nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x/x.norm(dim=-1, keepdim=True)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        x = x/x.norm(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x/x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        x = x/x.norm(dim=-1, keepdim=True)
        return x
