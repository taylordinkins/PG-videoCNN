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
import random

KTH_PATH = '/scratch/eecs-share/dinkinst/kth/data/'
IMG_PATH = '/nfs/stak/users/dinkinst/Homework/videoCNN/img_stacked/'

def load_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--hidden_latent', default=2048, type=int)
    parser.add_argument('--latent_size', default=2048, type=int)
    parser.add_argument('--time_latent', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--output', default=4096, type=int)
    parser.add_argument('--dataset', default='kth', type=str)
    parser.add_argument('--steps', default=600000, type=int)
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


def eval_model(network, dev_loader, resolution, percent_steps, epoch):
    network.eval()
    num_frames = 15
    ms_per_frame = 40

    criterion = nn.MSELoss()
    dev_loss = 0
    with torch.no_grad():
        batch_num = 0
        for item in dev_loader:
            item = item['instance'].cuda()
            batch_loss = 0

            # fit a whole batch for all the different milliseconds
            
            start_frame = 0
            j = start_frame+10
            frame_diff = 1
            time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()

            seq = item[:, :, start_frame:start_frame+10, :, :]
            seq = seq.squeeze()
            seq_targ = item[:, :, j, :, :]
            


            outputs = network(seq, time_delta)
            error = criterion(outputs, seq_targ)
            batch_loss += error.cpu().item()
            img_print = torch.cat((seq[:8, :, :, :], outputs[:8, :, :, :]), dim=1)
            if batch_num % 10 == 0 and epoch % 10 == 0:
                
                for indx in range(img_print.shape[0]):
                    save_image(img_print[indx].unsqueeze(1), IMG_PATH+'dev_output_res_{}_batch_{}_steps_{}_{}.png'.format(resolution, batch_num, percent_steps, indx))  
                            
            batch_num += 1
            dev_loss += batch_loss
        print('Dev: Resolution {}, Steps {}, Total {}\n'.format(resolution, percent_steps, dev_loss))
    return dev_loss



# 40ms is diff between one frame
def main(args):
    num_frames = 15
    ms_per_frame = 40

    network = EncoderDecoder(args).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.0, 0.99))
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
        steps = 0
        epoch = 0
        curr_res_loss = 0
        curr_res_dev_loss = 0
        while steps < train_steps:
            epoch_loss = 0
            batch_num = 0
            for start_frame in range(5):
                for item in train_loader:
                    #label = item['label']
                    item = item['instance'].cuda()
                    batch_loss = 0

                    # fit a whole batch for all the different milliseconds
                    network.zero_grad()
                    frame_diff = 1
                    time_delta = torch.tensor(frame_diff * ms_per_frame).float().repeat(args.batch_size).cuda()

                    seq = item[:, :, start_frame:start_frame+10, :, :]
                    seq = seq.squeeze()

                    j = start_frame+10
                    seq_targ = item[:, :, j, :, :]

                    outputs = network(seq, time_delta)
                    #print(seq.shape)
                    #print(outputs.shape)
                    error = criterion(outputs, seq_targ)
                    error.backward()
                    optimizer.step()

                    batch_loss += error.cpu().item()
                    #stable, _ = network.get_stability()
                    img_print = torch.cat((seq[:8, :, :, :], outputs[:8, :, :, :]), dim=1)
                    #print(seq[:8, :, :, :].shape, outputs[:8, :, :, :].shape, img_print.shape)
                    percent_steps = steps/train_steps
                    if batch_num % 50 == 0 and epoch % 10 == 0:
                        for indx in range(img_print.shape[0]):  
                            save_image(img_print[indx].unsqueeze(1), IMG_PATH+'train_output_res_{}_batch_{}_steps_{}_stable_{}_{}.png'.format(current_resolution, batch_num, percent_steps, str(stable), indx))

                    batch_num += 1
                    epoch_loss += batch_loss
                    #stable, _ = network.get_stability()
                    steps += args.batch_size
                    if steps >= train_steps:
                        break
                    if not stable:
                        net_alpha, _ = network.get_alpha()
                        new_alpha = min(net_alpha + fade_alpha, 1.0)
                        #print('Alpha update: old {}, new {}'.format(net_alpha, new_alpha))
                        network.update_alpha(new_alpha)
                #stable, _ = network.get_stability()
            print('\nTrain Epoch {} End: Resolution {}, Stable {} \n\tSteps {}, Total Train Loss {}'.format(epoch, current_resolution, str(stable), percent_steps, epoch_loss))
            #print('Steps {}, Out of {}'.format(steps, int(train_steps)))
            if stable:
                curr_res_dev_loss += eval_model(network, dev_loader, current_resolution, percent_steps, epoch)
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
            #step_scaling_resolution += 1
            train_steps = args.steps*step_scaling_resolution
            fade_alpha = args.batch_size/train_steps
        else:
            print('Fading completed...\n')
            network.update_stability()
            network.update_alpha(1.0)          
            
            



if __name__ == '__main__':
    args = load_args()
    main(args)

