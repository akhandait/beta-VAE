import torch.nn as nn

def weightsInit(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_normal_(layer.weight.data)

    # if classname.find('BatchNorm') != -1:
    #     layer.weight.data.normal_(1.0, 0.02)
    #     layer.bias.data.fill_(0)
