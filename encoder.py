import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, x):

        x = self.encoder(x)

        return x[:, :, 0, 0]


def get_encoder():
    model = models.resnet50(pretrained=True)
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    modules = list(model.children())[:-1]
    encoder = Encoder(encoder=nn.Sequential(*modules))

    return encoder