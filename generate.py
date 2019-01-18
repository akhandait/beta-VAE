import argparse
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils

from models import Encoder, Reparametrization, Decoder
from dataset import CelebA

parser = argparse.ArgumentParser()

parser.add_argument('--netE', type=str, default='',
    help="path to the weights of the encoder network")
parser.add_argument('--netD', type=str, default='',
    help="path to the weights of the decoder network")
parser.add_argument('--latentSize', type=int, default=512, help='batch size')
parser.add_argument('--nfE', type=int, default=64)
parser.add_argument('--nfD', type=int, default=512)
parser.add_argument('--resBlocks', type=int, default=5,
    help='number of residual blocks in both encoder and decoder')
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output',
    help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass
try:
    os.makedirs(opt.outDir + '/samples')
except OSError:
    pass

useCuda = torch.cuda.is_available() and not opt.disableCuda
device = torch.device("cuda:0" if useCuda else "cpu")
Tensor = torch.cuda.FloatTensor if useCuda else torch.Tensor

# Networks.
E = Encoder(latentSize=opt.latentSize, nfE=opt.nfE, resBlocks=opt.resBlocks).to(device)
R = Reparametrization(latentSize=opt.latentSize).to(device)
D = Decoder(latentSize=opt.latentSize, nfD=opt.nfD, resBlocks=opt.resBlocks).to(device)

E.load_state_dict(torch.load(opt.netE))
D.load_state_dict(torch.load(opt.netD))

E.eval()
R.eval()
D.eval()

distance = nn.MSELoss(reduction='sum')

# gaussianSamples = np.random.normal(size=(1, opt.latentSize))
# gaussianSamples = np.repeat(gaussianSamples, 15, axis=0)
# gaussianSamples = torch.from_numpy(gaussianSamples).float().to(device)



# l = []
# for j in range(opt.latentSize):
# # for i in range(15):
# #     gaussianSamples[i][511] = -2 + i * 4 / 14
#     # temp1 = gaussianSamples[0][j]
#     # temp2 = gaussianSamples[14][j]
#     copy = gaussianSamples.clone()
#     copy[0][j] = -2
#     copy[14][j] = 2
#     samples = D(copy)
#     # gaussianSamples[0][j] = temp1
#     # gaussianSamples[14][j] = temp2
#     print(distance(samples[0], samples[14]))
#     if distance(samples[0], samples[14]).item() > 1000:
#         l.append(j)
# for i in range(15):
#     gaussianSamples[i][119] = -2 + i * 4 / 14

# print(l)

# gaussianSamples = torch.from_numpy(gaussianSamples).float().to(device)

dataloader = DataLoader(CelebA('../Datasets/CelebA/img_align_celeba/'), \
        batch_size=64, shuffle=True, num_workers=6)
testBatch = next(iter(dataloader)).to(device)

samples = E(testBatch)
samples = R(samples)
samples = D(samples)
torchvision.utils.save_image(testBatch, opt.outDir + '/samples/' + 'original.png',
    nrow=8, normalize=True)
torchvision.utils.save_image(samples, opt.outDir + '/samples/' + 'sample_posterior.png',
    nrow=8, normalize=True)

# for i in range(1, 15):
#     print(distance(samples[0], samples[i]))
