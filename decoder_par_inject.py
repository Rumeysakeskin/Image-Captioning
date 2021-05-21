import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, feature_size, output_size, embed_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = embed_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size + embed_size , embed_size, num_layers=num_layers)

        self.out = nn.Linear(embed_size, output_size)

        self.map = nn.Linear(feature_size, embed_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden, feature):

        feature = self.map(feature)

        output = self.embedding(input)

        output = F.relu(output)

        output = torch.cat((feature, output), dim=2)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        #output = output.squeeze(0)

        return output, hidden


def get_decoder(feature_size=2048, output_size=10000, embed_size=128, num_layers=1):
    return Decoder(feature_size=feature_size, output_size=output_size, embed_size=embed_size, num_layers=num_layers)