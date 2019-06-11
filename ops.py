import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd



def grad_penalty_3dim(args, netD, data, fake):
	alpha = torch.randn(args.batch_size, 1, requires_grad=True).cuda()
	alpha = alpha.expand(args.batch_size, data.nelement()//args.batch_size)
	alpha = alpha.contiguous().view(args.batch_size, 3, 64, 64)
	interpolates = alpha * data + ((1 - alpha) * fake).cuda()
	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
			grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
			create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.l
	return gradient_penalty


def consistency_term(args, netD, data, Mtag=0):
	d1, d_1 = netD(data, xn=True)
	d2, d_2 = netD(data, xn=True)

	consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1) - Mtag
	return consistency_term.mean()