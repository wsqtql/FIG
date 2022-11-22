import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images_ourmnist", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,4,4,2,1)
        self.conv1_bn = nn.BatchNorm2d(4)
        # 14
        self.conv2 = nn.Conv2d(4,8,4,2,1)
        self.conv2_bn = nn.BatchNorm2d(8)
        # 7
        self.conv3 = nn.Conv2d(8,16,3,2,1)
        self.conv3_bn = nn.BatchNorm2d(16)
        # 4
        self.conv4 = nn.Conv2d(16, 32, 4, 1, 0)
        self.conv4_bn = nn.BatchNorm2d(32)
    def forward(self, img):
        c1 = F.leaky_relu(self.conv1_bn(self.conv1(img)))
        c2 = F.leaky_relu(self.conv2_bn(self.conv2(c1)))
        c3 = F.leaky_relu(self.conv3_bn(self.conv3(c2)))
        c4 = F.leaky_relu(self.conv4_bn(self.conv4(c3)))
        return c4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 1
        self.deconv0 = nn.ConvTranspose2d(32, 16, 4, 1, 0)
        self.deconv0_bn = nn.BatchNorm2d(16)
        # 28
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(8)
        # 56
        self.deconv2 = nn.ConvTranspose2d(8, 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(4)
        # 112
        self.deconv3 = nn.ConvTranspose2d(4, 1, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(1)

    def forward(self, z):
        x = F.relu(self.deconv0_bn(self.deconv0(z)))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.tanh(self.deconv3_bn(self.deconv3(x)))
        return x

criterion_self = torch.nn.MSELoss()
criterion_other = torch.nn.MSELoss()
kl_loss = nn.KLDivLoss()
sfm = nn.Softmax(dim=1)
sim = nn.CosineSimilarity()

encoder = Encoder()
decoder = Decoder()

if torch.cuda.is_available():
    device = torch.device('cuda')
    encoder.to(device)
    decoder.to(device)
    criterion_self.to(device)
    criterion_other.to(device)
    kl_loss.to(device)
    sfm.to(device)
    sim.to(device)


import itertools
# The target minority class
train_data = ImageFolder('/home/S2_WSQ/PyTorch-GAN-master/less_mnist', transform=transforms.Compose([transforms.ToTensor(),transforms.Grayscale(1), transforms.Normalize([0.5], [0.5])]))
train_loader = DataLoader(train_data, opt.batch_size, True)
# The whole training dataset
train_data2 = ImageFolder('/home/S2_WSQ/PyTorch-GAN-master/less/2', transform=transforms.Compose([transforms.ToTensor(),transforms.Grayscale(1),  transforms.Normalize([0.5], [0.5])]))
train_loader2 = DataLoader(train_data2, opt.batch_size, True)

optimizer = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer2 = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr2, betas=(opt.b1, opt.b2)
)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
from cos import Cos
import random
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(zip(train_loader2, train_loader)):

        real_imgs = Variable(batch[0][0].type(Tensor))
        real_imgs2 = Variable(batch[1][0].type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer2.zero_grad()
        encoded_imgs2 = encoder(real_imgs2)
        decoded_imgs2 = decoder(encoded_imgs2)

        # Loss measures generator's ability to fool the discriminator
        g_loss2 = criterion_self(decoded_imgs2, real_imgs2)
        loss2 = g_loss2
        loss2.backward()
        optimizer2.step()

        optimizer.zero_grad()
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        encoded_imgs3 = encoder(real_imgs2)
        decoded_imgs3 = decoder(encoded_imgs3)
        flip_imgs = torch.flip(encoded_imgs,dims=[0])
        rand = random.random()
        rand = random.random()
        ex_imgs = rand * encoded_imgs + (1 - rand) * flip_imgs
        final_img = torch.cat((ex_imgs,encoded_imgs),0)
        final_img2 = torch.cat((final_img,encoded_imgs3),0)

        decoder_ex_imgs = decoder(final_img2)
        encoder_ex_imgs = encoder(decoder_ex_imgs)

        latent_KL = Cos().cos(final_img2.detach().cpu().numpy())
        real_KL  = Cos().cos(decoder_ex_imgs.detach().cpu().numpy())
        sim_loss = 1000 * kl_loss(torch.log(torch.from_numpy(latent_KL)), torch.from_numpy(real_KL))
        # Loss measures generator's ability to fool the discriminator
        g_loss =  criterion_self(decoded_imgs, real_imgs) + criterion_other(encoder_ex_imgs,final_img2)
        loss = g_loss + sim_loss
        loss.backward()
        optimizer.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
            % (epoch, opt.n_epochs, i, len(train_loader2), loss2.item())
        )

        batches_done = epoch * len(train_loader2) + i
        if batches_done % 200 == 0:
            save_image(decoder_ex_imgs.data[:10], "images_ourmnist/%d.png" % batches_done, nrow=10, normalize=True)

torch.save(encoder.state_dict(), 'encoder.pkl')
torch.save(decoder.state_dict(), 'decoder.pkl')