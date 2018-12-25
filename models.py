import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, inChannels=3, latentSize=256, nfE=64, resBlocks=5):
        nn.Module.__init__(self)

        # TODO: Try BatchNorm, Residual blocks.
        model = [nn.Conv2d(inChannels, nfE, 4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True)]

        for _ in range(resBlocks):
            model += [ResidualBlock(nfE)]

        model += [nn.Conv2d(nfE, nfE * 2, 4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 2, nfE * 4, 4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 4, nfE * 8, 4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True)]

        model += [Reshape(-1, nfE * 8 * 4 * 4),
                  nn.Linear(nfE * 8 * 4 * 4, latentSize * 2)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Reparametrization(nn.Module):
    def __init__(self, latentSize=256):
        nn.Module.__init__(self)
        self.latentSize = latentSize

    def forward(self, x):
        self.mean = x[..., : self.latentSize].type(x.type())
        self.stdDeviation = nn.Softplus()(x[..., self.latentSize : ]).type(x.type())

        return self.mean + self.stdDeviation * \
            torch.randn(self.mean.shape[0], self.latentSize).type(x.type())

    def klDivergence(self, mean=True):
        if not hasattr(self, 'mean'):
            raise RuntimeError('Cannot evaluate KL Divergence without a forward pass ' + \
                'before it.')

        loss = -0.5 * torch.sum(2 * torch.log(self.stdDeviation) - \
            torch.pow(self.stdDeviation, 2) - torch.pow(self.mean, 2) + 1)

        if mean:
            return loss / self.mean.shape[0]
        return loss

class Decoder(nn.Module):
    def __init__(self, outChannels=3, latentSize=256, nfD=512, resBlocks=5):
        nn.Module.__init__(self)

        model = [nn.Linear(latentSize, nfD * 4 * 4),
                 Reshape(-1, nfD, 4, 4)]

        model += [nn.ConvTranspose2d(nfD, nfD // 2, 4, stride=2, padding=1),
                  nn.LeakyReLU(inplace=True),
                  nn.ConvTranspose2d(nfD // 2, nfD // 4, 4, stride=2, padding=1),
                  nn.LeakyReLU(inplace=True),
                  nn.ConvTranspose2d(nfD // 4, nfD // 8, 4, stride=2, padding=1),
                  nn.LeakyReLU(inplace=True)]

        for _ in range(resBlocks):
            model += [ResidualBlock(nfD // 8)]

        model += [nn.ConvTranspose2d(nfD // 8, outChannels, 4, stride=2, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Create a layer to reshape within Sequential layers, for convenience.
class Reshape(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ResidualBlock(nn.Module):
    def __init__(self, inFeatures, LeakyReLU=True):
        nn.Module.__init__(self)

        convBlock = [nn.ZeroPad2d(1),
                     nn.Conv2d(inFeatures, inFeatures, 3, bias=False),
                     # nn.InstanceNorm2d(inFeatures)]
                     # nn.BatchNorm2d(inFeatures)]
                     ]

        if LeakyReLU:
            convBlock += [nn.LeakyReLU(inplace=True)]
        else:
            convBlock += [nn.ReLU(inplace=True)]

        convBlock += [nn.ZeroPad2d(1),
                     nn.Conv2d(inFeatures, inFeatures, 3, bias=False),
                     # nn.InstanceNorm2d(inFeatures)]
                     # nn.BatchNorm2d(inFeatures)]
                     ]

        if LeakyReLU:
            convBlock += [nn.LeakyReLU(inplace=True)]
        else:
            convBlock += [nn.ReLU(inplace=True)]

        self.convBlock = nn.Sequential(*convBlock)

    def forward(self, x):
        return x + self.convBlock(x)
