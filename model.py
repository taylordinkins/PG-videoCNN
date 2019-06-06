import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.init as weight_init
from torch.autograd import Variable

channels = 1


class Encoder(nn.Module):
	def __init__(self, latent_size=2048, time_latent_size=64, hidden_latent_size=1024):
		super(Encoder, self).__init__()
		self.latent_size = latent_size
		self.leaky = nn.LeakyReLU(0.2, inplace=True)

		self.conv0 = nn.Conv2d(channels, 32, 5, stride=1, padding=(2, 2))
		self.batch0 = nn.BatchNorm2d(32)
		self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
		# 32 x 64 x 64
		# # 32 x 32 x 32

		self.conv1 = nn.Conv2d(32, 64, 5, stride=1, padding=(2, 2))
		self.batch1 = nn.BatchNorm2d(64)
		self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
		# 64 x 32 x 32
		## 64 x 16 x 16

		self.conv2 = nn.Conv2d(64, 128, 1, stride=1)
		self.batch2 = nn.BatchNorm2d(128)
		self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
		# 128 x 16 x 16
		# #128 x 8 x 8

		#self.hidden_units = 128*16*16
		self.hidden_units = 128*8*8

		self.linear0 = nn.Linear(self.hidden_units, hidden_latent_size)
		self.linear1 = nn.Linear(hidden_latent_size, latent_size)

		self.tlinear0 = nn.Linear(1, time_latent_size)
		self.tlinear1 = nn.Linear(time_latent_size, time_latent_size)
		self.tlinear2 = nn.Linear(time_latent_size, time_latent_size)


	def forward(self, x, t):
		x = self.leaky(self.conv0(x))
		x = self.batch0(x)
		x, mpi0 = self.maxpool0(x)
		#print(x.shape)

		x = self.leaky(self.conv1(x))
		x = self.batch1(x)
		x, mpi1 = self.maxpool1(x)
		#print(x.shape)

		x = self.leaky(self.conv2(x))
		x = self.batch2(x)
		x, mpi2 = self.maxpool2(x)
		#print(x.shape)

		x = x.view(-1, self.hidden_units)
		x = self.leaky(self.linear0(x))
		x = self.linear1(x)
		#print(x.shape)

		t = t.view(-1, 1)
		t = self.leaky(self.tlinear0(t))
		t = self.leaky(self.tlinear1(t))
		t = self.tlinear2(t)

		#print(x.shape, t.shape)
		out = torch.cat((x, t), 1)
		return out, [mpi0, mpi1, mpi2]


class Decoder(nn.Module):
	def __init__(self, latent_size=2048, time_latent_size=64, hidden_latent_size=1024):
		super(Decoder, self).__init__()
		self.fc_size = latent_size + time_latent_size

		self.linear3 = nn.Linear(self.fc_size, hidden_latent_size)
		#self.linear4 = nn.Linear(hidden_latent_size, 128*16*16)
		self.linear4 = nn.Linear(hidden_latent_size, 128*8*8)

		self.unpool0 = nn.MaxUnpool2d(2)
		self.deconv0 = nn.ConvTranspose2d(128, 64, 1, stride=1)
		self.batch0 = nn.BatchNorm2d(64)

		self.unpool1 = nn.MaxUnpool2d(2)
		self.deconv1 = nn.ConvTranspose2d(64, 32, 5, stride=1, padding=(2, 2))
		self.batch1 = nn.BatchNorm2d(32)

		self.unpool2 = nn.MaxUnpool2d(2)
		self.deconv2 = nn.ConvTranspose2d(32, channels, 5, stride=1, padding=(2, 2))


	def forward(self, x, mpis):
		x = F.relu(self.linear3(x))
		x = F.relu(self.linear4(x))

		#x = x.view(-1, 128, 16, 16)
		x = x.view(-1, 128, 8, 8)
		x = self.unpool0(x, mpis[2])
		x = F.relu(self.deconv0(x))
		x = self.batch0(x)

		x = self.unpool1(x, mpis[1])
		x = F.relu(self.deconv1(x))
		x = self.batch1(x)

		x = self.unpool2(x, mpis[0])
		x = self.deconv2(x)

		return x



class EncoderDecoder(nn.Module):
	def __init__(self, args):
		super(EncoderDecoder, self).__init__()
		latent_size = args.latent_size
		time_latent_size = args.time_latent
		hidden_latent = args.hidden_latent
		self.encoder = Encoder(latent_size, time_latent_size, hidden_latent)
		self.decoder = Decoder(latent_size, time_latent_size, hidden_latent)

	def forward(self, x, t):
		x, mpis = self.encoder(x, t)
		x = self.decoder(x, mpis)

		return x
