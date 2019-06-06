import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.init as weight_init
from torch.autograd import Variable
import math

channels = 10

level_channels_mapping = { 2: 512, 3: 256, 4: 128, 5: 64, 6: 32, 7: 32, 8:32}

class Encoder(nn.Module):
	def __init__(self, resolution=128, latent_size=2048, time_latent_size=64, hidden_latent_size=1024):
		super(Encoder, self).__init__()
		self.latent_size = latent_size
		self.leaky = nn.LeakyReLU(0.2, inplace=True)
		self.max_resolution = resolution
		self.max_level = int(math.log2(resolution))

		self.first_conv = nn.Conv2d(channels, 32, 1, stride=1)  # initial 1x1 convolution to 32 channels
		self.first_batch = nn.BatchNorm2d(32)
		self.first_conv_two = nn.Conv2d(32, 32, 3, stride=1, padding=(1,1))  # initial 3x3 convolution to 32 channels
		self.first_batch_two = nn.BatchNorm2d(32)

		self.level = 2
		self.current_resolution = 2**self.level
		self.stable = True
		self.alpha = 1.0
		# 32x128x128

		# define for first 4x4 layer
		# difference is no output downsample
		self.level_downsample_layers = [nn.AvgPool2d(2)]
		self.next_level_downsample_layers = [None]
		self.level_channel_layers_img = [nn.Conv2d(32, 512, 1, stride=1)]
		self.level_img_bn = [nn.BatchNorm2d(512)]
		self.level_conv_layers = [nn.Conv2d(512, 512, 3, stride=1, padding=(1, 1))]
		self.level_bn_layers = [nn.BatchNorm2d(512)]

		# start here with size 8x8
		for curr_level in range(3, self.max_level+1):
			curr_channels = level_channels_mapping[curr_level]
			higher_channels = level_channels_mapping[curr_level+1]
			curr_resolution = 2**curr_level
			# faded out
			self.level_downsample_layers.append(nn.AvgPool2d(2))  # run this before doing the level channel conversion (left side, faded out)
			self.level_channel_layers_img.append(nn.Conv2d(32, curr_channels, 1, stride=1))  # from RGB style channel conversion, left side
			self.level_img_bn.append(nn.BatchNorm2d(curr_channels))

			# utilized always except downsample on 4x4
			self.level_conv_layers.append(nn.Conv2d(curr_channels, curr_channels*2, 3, stride=1, padding=(1, 1)))  # do more channels - right side
			self.level_bn_layers.append(nn.BatchNorm2d(curr_channels*2))  # right side pass
			self.next_level_downsample_layers.append(nn.AvgPool2d(2))  # right side pass
		
		# apply convolution, then linear layers
		self.last_conv = nn.Conv2d(512, 512, 1)
		self.last_bn = nn.BatchNorm2d(512)
		self.hidden_units = 512*4*4

		self.linear0 = nn.Linear(self.hidden_units, hidden_latent_size)
		self.linear1 = nn.Linear(hidden_latent_size, latent_size)

		self.tlinear0 = nn.Linear(1, time_latent_size)
		self.tlinear1 = nn.Linear(time_latent_size, time_latent_size)
		self.tlinear2 = nn.Linear(time_latent_size, time_latent_size)


	def forward(self, x, t):
		# always called
		x = self.leaky(self.first_conv(x))
		x = self.first_batch(x)
		x = self.leaky(self.first_conv_two(x))
		x = self.first_batch_two(x)

		train_level = self.level - 2
		# called for whichever level is currently training
		y = self.leaky(self.level_channel_layers_img[train_level](x))
		y = self.level_img_bn[train_level](y)
		# don't need downsampling for current level 
		y = self.leaky(self.level_conv_layers[train_level](y))
		y = self.level_bn_layers[train_level](y)

		#print(y.shape)
		while train_level > 0:
			# downsample and pass to next layer
			y = self.next_level_downsample_layers[train_level](y)

			# update train_level
			train_level = train_level - 1

			# faded out
			if not self.stable and train_level == self.level - 3:
				# next layer gets downsampled img input
				z = self.level_downsample_layers[train_level](x)
				z = self.level_channel_layers_img[train_level](z)
				z = self.level_img_bn[train_level](z)

				y = torch.add((1-self.alpha*z), self.alpha*y)

			# get downsampled from previous layer
			y = self.leaky(self.level_conv_layers[train_level](y))
			y = self.level_bn_layers[train_level](y)
			#print(y.shape)

		y = self.leaky(self.last_conv(y))
		y = self.last_bn(self.last_bn(y))
		y = y.view(-1, self.hidden_units)
		y = self.leaky(self.linear0(y))
		y = self.linear1(y)

		t = t.view(-1, 1)
		t = self.leaky(self.tlinear0(t))
		t = self.leaky(self.tlinear1(t))
		t = self.tlinear2(t)

		out = torch.cat((y, t), 1)
		return y

	def update_level(self):
		self.level += 1
		self.current_resolution = 2**self.level

	def update_alpha(self, new_alpha):
		self.alpha = new_alpha


enc = Encoder()
x = torch.randn(32, 10, 4, 4)
t = torch.tensor(40).float().repeat(32)
print(enc(x, t).shape)

class Decoder(nn.Module):
	def __init__(self, resolution=128, latent_size=2048, time_latent_size=64, hidden_latent_size=1024):
		super(Decoder, self).__init__()
		self.leaky = nn.LeakyReLU(0.2, inplace=True)
		self.max_resolution = resolution
		self.max_level = int(math.log2(resolution))
		self.level = 2
		self.current_resolution = 2**self.level
		self.stable = True
		self.alpha = 1.0

		# apply linear layers, then deconv
		self.fc_size = latent_size + time_latent_size
		self.linear3 = nn.Linear(self.fc_size, hidden_latent_size)
		self.linear4 = nn.Linear(hidden_latent_size, 512*4*4)
		self.first_deconv = self.ConvTranspose2d(512, 512, 1, stride=1)
		self.first_bn = self.BatchNorm2d(512)

		# define first for 4x4 layer
		# each one gets an upsample for its own input, except first layer
		self.level_upsample_layers = [None]
		self.level_channel_layers_img = [nn.ConvTranspose2d(512, 32, 1, stride=1)]
		self.level_img_bn = [nn.BatchNorm2d(32)]
		self.level_conv_layers = [nn.ConvTranspose2d(512, 512, 3, stride=1, padding=(1, 1))]
		self.level_bn_layers = [nn.BatchNorm2d(512/2)]

		# start here with size 8x8
		for curr_level in range(3, self.max_level+1):
			curr_channels = level_channels_mapping[curr_level]
			higher_channels = level_channels_mapping[curr_level+1]
			curr_resolution = 2**curr_level
			# faded out, except upsample
			# level channel layers applied always on the last layer (128x128), before feeding to last_deconv
			self.level_upsample_layers.append(nn.Upsample(scale_factor=2))  # run this before doing the to RGB (left side, faded out)
			self.level_channel_layers_img.append(nn.ConvTranspose2d(curr_channels, 32, 1, stride=1))  # to RGB style channel conversion, left side
			self.level_img_bn.append(nn.BatchNorm2d(32))

			self.level_conv_layers.append(nn.ConvTranspose2d(curr_channels*2, curr_channels, 3, stride=1, padding=(1, 1)))  # do less channels - right side
			self.level_bn_layers.append(nn.BatchNorm2d(curr_channels/2))  # right side pass

		

		self.last_deconv = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=(1, 1))  # 3x3 convolution to 32 channels
		self.last_batch = nn.BatchNorm2d(32)
		self.last_deconv_two = nn.ConvTranspose2d(32, 1, 1, stride=1)  # last 1x1 convolution to 1 channel - image


	def forward(self, x, mpis):
		x = self.leaky(self.linear3(x))
		x = self.leaky(self.linear4(x))

		x = x.view(-1, 512, 4, 4)
		x = self.leaky(first_deconv(x))
		x = self.first_bn(x)

		train_level = self.level - 2

		# always apply the 4x4 conv
		x = self.leaky(self.level_conv_layers[train_level](x))
		x = self.level_bn_layers[train_level](x)

		for curr_level in range(1, train_level+1):
			# upsample if not output from 4x4 conv
			y = self.level_upsample_layers[curr_level](x)

			# only fading the second last layer
			# left side
			if not self.stable and curr_level == train_level - 1:
				z = self.leaky(self.level_channel_layers_img[curr_level](y))
				z = self.level_img_bn(z)



		return x

	def update_level(self):
		self.level += 1
		self.current_resolution = 2**self.level

	def update_alpha(self, new_alpha):
		self.alpha = new_alpha



# class EncoderDecoder(nn.Module):
# 	def __init__(self, args):
# 		super(EncoderDecoder, self).__init__()
# 		latent_size = args.latent_size
# 		time_latent_size = args.time_latent
# 		hidden_latent = args.hidden_latent
# 		self.encoder = Encoder(latent_size, time_latent_size, hidden_latent)
# 		self.decoder = Decoder(latent_size, time_latent_size, hidden_latent)

# 	def forward(self, x, t):
# 		x, mpis = self.encoder(x, t)
# 		x = self.decoder(x, mpis)

# 		return x
