# 1 frame every 40ms

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.init as weight_init
from torch.autograd import Variable
import argparse
import sys
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import BlockFrameDataset
from model import EncoderDecoder
from scipy.misc import imresize

KTH_PATH = '/scratch/eecs-share/dinkinst/kth/data/'

def load_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--hidden_latent', default=2048, type=int)
    parser.add_argument('--time_latent', default=64, type=int)
    parser.add_argument('--latent_size', default=2048, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output', default=4096, type=int)
    parser.add_argument('--dataset', default='kth', type=str)
    parser.add_argument('--shape', default=None, type=int)

    args = parser.parse_args()

    return args


# data from loaders is in tensors dictionary
# two keys: instance, label
# don't care about labels, just the instance batch
# shape: (batch, channels, frames, h, w)
def fetch_kth_data(args):
    print('Fetching train...\n')
    train_set = BlockFrameDataset(KTH_PATH, dataset='train', shape=args.shape)
    print('Fetching dev...\n')
    dev_set = BlockFrameDataset(KTH_PATH, dataset='dev', shape=args.shape)
    print('Fetching test...\n')
    test_set = BlockFrameDataset(KTH_PATH, dataset='test', shape=args.shape)

    # range 0-1 scaling
    train_set.normalize()
    dev_set.normalize()
    test_set.normalize()
    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True, 
                     drop_last=True)
    dev_loader = torch.utils.data.DataLoader(
                    dataset=dev_set,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False, 
                    drop_last=True)

    return train_loader, dev_loader, test_loader


def eval_model(network, dev_loader, epoch):
    network.eval()
    num_frames = 15
    ms_per_frame = 40

    criterion = nn.MSELoss()
    dev_loss = 0
    with torch.no_grad():
        batch_num = 0
        for item in dev_loader:
            item = item['instance'].cuda()

            frames_processed = 0
            batch_loss = 0

            # fit a whole batch for all the different milliseconds
            for i in range(num_frames-1):
                for j in range(i+1, num_frames):
                    frame_diff = j - i
                    time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()

                    seq = item[:, :, i, :, :]
                    seq_targ = item[:, :, j, :, :]
                    
                    # downsample
                    # seq = F.interpolate(seq, size=(64, 64))
                    # seq_targ = F.interpolate(seq_targ, size=(64, 64))


                    outputs = network(seq, time_delta)
                    error = criterion(outputs, seq_targ)
                    batch_loss += error.cpu().item()
                    frames_processed += 1
                    if i == 0:
                        save_image(outputs, '/scratch/eecs-share/dinkinst/kth/img/dev_output_{}_epoch_{}.png'.format(j, epoch))

            batch_num += 1
            print('Dev Batch #{} Total Error {}'.format(batch_num, batch_loss))
            dev_loss += batch_loss
        print('\nDev Total {} Scaled {}'.format(dev_loss, dev_loss/frames_processed))
    return dev_loss



# 40ms is diff between one frame
def main(args):
    num_frames = 15
    ms_per_frame = 40

    network = EncoderDecoder(args).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = nn.MSELoss()

    train_loader, dev_loader, test_loader = fetch_kth_data(args)

    # test_tens = next(iter(train_loader))['instance'][0, :, :, :, :].transpose(0, 1)
    # print(test_tens.shape)
    # save_image(test_tens, './img/test_tens.png')
    # print(next(iter(train_loader))['instance'][0, :, 0, :, :].shape)
    train_loss = []
    dev_loss = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_num = 0
        for item in train_loader:
            #label = item['label']
            item = item['instance'].cuda()

            frames_processed = 0
            batch_loss = 0

            # fit a whole batch for all the different milliseconds
            for i in range(num_frames-1):
                for j in range(i+1, num_frames):
                    network.zero_grad()
                    frame_diff = j - i
                    time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()
                    time_delta.requires_grad = True

                    seq = item[:, :, i, :, :]
                    #print(seq.shape)

                    # downsample
                    #seq = F.interpolate(seq, size=(64, 64))
                    #print(seq.shape)

                    seq.requires_grad = True

                    seq_targ = item[:, :, j, :, :]

                    # downsample
                    #seq_targ = F.interpolate(seq_targ, size=(64, 64))

                    seq_targ.requires_grad = False

                    assert seq.requires_grad and time_delta.requires_grad, 'No Gradients'

                    outputs = network(seq, time_delta)
                    error = criterion(outputs, seq_targ)
                    error.backward()
                    optimizer.step()

                    batch_loss += error.cpu().item()
                    frames_processed += 1

                    if i == 0:
                        save_image(outputs, '/scratch/eecs-share/dinkinst/kth/img/train_output_{}_epoch_{}.png'.format(j, epoch))

            batch_num += 1
            epoch_loss += batch_loss
            print('Epoch {} Batch #{} Total Error {}'.format(epoch, batch_num, batch_loss))
        print('\nEpoch {} Total Loss {} Scaled Loss {}\n'.format(epoch, epoch_loss, epoch_loss/frames_processed))
        train_loss.append(epoch_loss)
        if epoch % 10 == 0:
            torch.save(network.state_dict(), KTH_PATH+str('/model_new_{}.pth'.format(epoch)))
            torch.save(optimizer.state_dict(), KTH_PATH+str('/optim_new_{}.pth'.format(epoch)))

        dev_loss.append(eval_model(network, dev_loader, epoch))
        network.train()

    plt.plot(range(args.epochs), train_loss)
    plt.grid()
    plt.savefig('/scratch/eecs-share/dinkinst/kth/img/loss_train.png', dpi=64)
    plt.close('all')
    plt.plot(range(args.epochs), dev_loss)
    plt.grid()
    plt.savefig('/scratch/eecs-share/dinkinst/kth/img/loss_dev.png', dpi=64)
    plt.close('all')



if __name__ == '__main__':
    args = load_args()
    main(args)

