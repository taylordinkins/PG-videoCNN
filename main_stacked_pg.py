# 1 frame every 40ms

# right now just doing 1 frame prediction
# easy to modify to do up to 5 horizon (from 10 frames)



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
from model_stacked_pg import EncoderDecoder
from scipy.misc import imresize

KTH_PATH = '/scratch/eecs-share/dinkinst/kth/data/'
IMG_PATH = '/nfs/stak/users/dinkinst/Homework/videoCNN/img_stacked/'

def load_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--hidden_latent', default=4096, type=int)
    parser.add_argument('--time_latent', default=64, type=int)
    parser.add_argument('--latent_size', default=4096, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--output', default=4096, type=int)
    parser.add_argument('--dataset', default='kth', type=str)
    parser.add_argument('--steps', default=20000, type=int)
    parser.add_argument('--start_resolution', default=4, type=int)
    parser.add_argument('--max_resolution', default=128, type=int)

    args = parser.parse_args()

    return args


# data from loaders is in tensors dictionary
# two keys: instance, label
# don't care about labels, just the instance batch
# shape: (batch, channels, frames, h, w)
def fetch_kth_data(args, shape=None):
    print('Fetching train...\n')
    train_set = BlockFrameDataset(KTH_PATH, dataset='train', shape=shape)
    print('Fetching dev...\n')
    dev_set = BlockFrameDataset(KTH_PATH, dataset='dev', shape=shape)
    print('Fetching test...\n')
    test_set = BlockFrameDataset(KTH_PATH, dataset='test', shape=shape)

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


def eval_model(network, dev_loader, resolution, epoch):
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
            for i in range(10, num_frames-1):
                for j in range(i+1, num_frames):
                    start_frame = i - 10
                    frame_diff = j - i
                    time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()

                    seq = item[:, :, start_frame:i, :, :]
                    seq = seq.squeeze()
                    seq_targ = item[:, :, j, :, :]
                    


                    outputs = network(seq, time_delta)
                    error = criterion(outputs, seq_targ)
                    batch_loss += error.cpu().item()
                    frames_processed += 1
                    if i == 10 and batch_num % 10 == 0 and epoch % 5 == 0:
                        save_image(outputs, IMG_PATH+'dev_output_res_{}_batch_{}_epoch_{}.png'.format(resolution, batch_num, epoch))

            batch_num += 1
            print('Dev: Resolution {}, Epoch {}, Batch #{}, Total Error {}'.format(resolution, epoch, batch_num, batch_loss))
            dev_loss += batch_loss
        print('\nDev: Resolution {}, Epoch {}, Total {}, Scaled {}\n'.format(resolution, epoch, dev_loss, dev_loss/frames_processed))
    return dev_loss



# 40ms is diff between one frame
def main(args):
    num_frames = 15
    ms_per_frame = 40

    network = EncoderDecoder(args).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.0, 0.9))
    criterion = nn.MSELoss()

    start_resolution = args.start_resolution
    max_resolution = args.max_resolution
    train_steps = int(args.steps)
    train_loader, dev_loader, test_loader = fetch_kth_data(args, shape=start_resolution)

    # test_tens = next(iter(train_loader))['instance'][0, :, :, :, :].transpose(0, 1)
    # print(test_tens.shape)
    # save_image(test_tens, './img/test_tens.png')
    # print(next(iter(train_loader))['instance'][0, :, 0, :, :].shape)
    resolution_loss = []
    dev_loss = []
    current_resolution = start_resolution
    step_scaling_resolution = 1
    fade_alpha = args.batch_size/args.steps

    # loop until resolution hits max and trains on it
    while current_resolution <= max_resolution:
        print('Current resolution {}'.format(current_resolution))
        stable, _ = network.get_stability()
        epoch = 0
        steps = 0
        curr_res_loss = 0
        curr_res_dev_loss = 0
        while steps < train_steps:
            epoch_loss = 0
            batch_num = 0
            for item in train_loader:
                #label = item['label']
                item = item['instance'].cuda()

                frames_processed = 0
                batch_loss = 0

                # fit a whole batch for all the different milliseconds
                for i in range(10, num_frames-1):
                    for j in range(i+1, num_frames):
                        start_frame = i - 10
                        network.zero_grad()
                        frame_diff = j - i
                        time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()
                        time_delta.requires_grad = True

                        seq = item[:, :, start_frame:i, :, :]
                        seq = seq.squeeze()
                        seq.requires_grad = True

                        seq_targ = item[:, :, j, :, :]

                        seq_targ.requires_grad = False

                        assert seq.requires_grad and time_delta.requires_grad, 'No Gradients'

                        outputs = network(seq, time_delta)
                        #print(outputs.shape)
                        error = criterion(outputs, seq_targ)
                        error.backward()
                        optimizer.step()

                        batch_loss += error.cpu().item()
                        frames_processed += 1
                        #stable, _ = network.get_stability()
                        if i == 10 and batch_num % 20 == 0 and epoch % 5 == 0:
                            save_image(outputs, IMG_PATH+'train_output_res_{}_batch_{}_epoch_{}_stable_{}.png'.format(current_resolution, batch_num, epoch, str(stable)))

                batch_num += 1
                epoch_loss += batch_loss
                #stable, _ = network.get_stability()
                print('Training: Resolution {}, Stable {}, Epoch {}, Batch #{}, Total Error {}'.format(current_resolution, str(stable), epoch, batch_num, batch_loss))
                steps += args.batch_size
                if steps >= train_steps:
                    break
                if not stable:
                    net_alpha, _ = network.get_alpha()
                    new_alpha = min(net_alpha + fade_alpha, 1.0)
                    print('Alpha update: old {}, new {}'.format(net_alpha, new_alpha))
                    network.update_alpha(new_alpha)
            #stable, _ = network.get_stability()
            print('\nTrain Epoch End: Resolution {}, Stable {} \n\tEpoch {}, Total Loss {}, Scaled Loss {}\n'.format(current_resolution, str(stable), epoch, epoch_loss, epoch_loss/frames_processed))
            print('Steps {}, Out of {}\n'.format(steps, int(train_steps)))
            if stable:
                curr_res_dev_loss += eval_model(network, dev_loader, current_resolution, epoch)
                curr_res_loss += epoch_loss
            network.train()
            epoch += 1

        #stable, _ = network.get_stability()
        print('\nSaving models...\n')
        torch.save(network.state_dict(), KTH_PATH+str('/model_pg_{}_{}.pth'.format(current_resolution, str(stable))))
        torch.save(optimizer.state_dict(), KTH_PATH+str('/optim_pg_{}_{}.pth'.format(current_resolution, str(stable))))
        if stable:
            resolution_loss.append(curr_res_loss)
            dev_loss.append(curr_res_dev_loss)
            if current_resolution >= max_resolution:
                break
            print('Fading in next layer...\n')
            network.update_stability()
            network.update_alpha(fade_alpha)
            network.update_level()
            current_resolution, _ = network.get_resolution()
            train_loader, dev_loader, test_loader = fetch_kth_data(args, shape=current_resolution)
            step_scaling_resolution += 1
            train_steps = args.steps*step_scaling_resolution
            fade_alpha = args.batch_size/train_steps
        else:
            print('Fading completed...\n')
            network.update_stability()
            network.update_alpha(1.0)          
            
            



if __name__ == '__main__':
    args = load_args()
    main(args)

