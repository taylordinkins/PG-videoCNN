# 1 frame every 40ms

# right now just doing 1 frame prediction
# easy to modify to do up to 5 horizon (from 10 frames)



import numpy as np
import torch, time, os
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.init as weight_init
from torch.autograd import Variable
import argparse
import sys
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import BlockFrameDataset, MovingMNIST
from model_stacked_pg import EncoderDecoder
from scipy.misc import imsave
from scipy.misc import imresize
import random

KTH_PATH = '/scratch/eecs-share/dinkinst/kth/data/'
MMN_PATH = '/scratch/eecs-share/MovingMNIST'

def load_args():

    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--hidden_latent', default=1024, type=int)
    parser.add_argument('--latent_size', default=512, type=int)
    parser.add_argument('--time_latent', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--output', default=4096, type=int)
    parser.add_argument('--dataset', default='kth', type=str)
    parser.add_argument('--steps', default=1200000, type=int)
    parser.add_argument('--start_resolution', default=4, type=int)
    parser.add_argument('--max_resolution', default=128, type=int)
    parser.add_argument('--grow', default=1, type=int)

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


def fetch_mmn_data(args, shape = None):
    print('Fetching train...\n')
    train_set = MovingMNIST(root=MMN_PATH, train=True, download=True, shape=shape)
    print('Fetching test...\n')
    test_set = MovingMNIST(root=MMN_PATH, train=False, download=True, shape=shape)


    batch_size = args.batch_size
    train_set.normalize()
    test_set.normalize()

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True,
                     drop_last=True)
    dev_loader = test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True)

    return train_loader, dev_loader, test_loader

def save_image(path, np_img):
    img_seq = []
    vert_images = []
    for b in range(8):
        for f in range(np_img.shape[1]):
            img_seq.append(np_img[b,f])
        vert_images.append(np.hstack(img_seq))
        img_seq = []

    img = np.vstack(vert_images)
    imsave(path, img)

def eval_model(network, dev_loader, image_dir, resolution, percent_steps, epoch):
    network.eval()
    ms_per_frame = 40

    input_frames = 10
    start_frame = 0

    criterion = nn.MSELoss()
    dev_loss = 0
    with torch.no_grad():
        batch_num = 0
        for item in dev_loader:
            item = item['instance'].cuda()
            batch_loss = 0

            # fit a whole batch for all the different milliseconds
            network.zero_grad()

            seq = item[:, :, start_frame:start_frame+input_frames, :, :]
            seq = seq.squeeze()

            seq_targ = item[:, :, start_frame+input_frames, :, :]


            outputs = network(seq)
            error = criterion(outputs, seq_targ)
            batch_loss += error.cpu().item()
            if (batch_num % 10 == 0 and epoch % 50 == 0) or percent_steps >= 1.0:
                img_print = torch.cat((seq[:8, :, :, :], outputs[:8, :, :, :]), dim=1).data.cpu().numpy()#.permute(0,2,3,1).data.cpu().numpy()
                path = image_dir+'dev_output_res_{}_batch_{}_steps_{}.png'.format(resolution, batch_num, percent_steps)
                save_image(path, img_print)

            batch_num += 1
            dev_loss += batch_loss
        print('Dev: Resolution {}, Steps {}, Total {}\n'.format(resolution, percent_steps, dev_loss))
    return dev_loss


def get_data_loaders(args, current_resolution):
    if args.dataset == 'kth':
        return fetch_kth_data(args, shape=current_resolution)
    else:
        return fetch_mmn_data(args, shape=current_resolution)

def save_graph_data(save_dir, start_time, cur_loss, image_count_shown):

    elapsed_mins = round((time.time() - start_time)) / 60
    #save current_time & loss to a file
    save_string = "{:0.2f},{}".format(elapsed_mins,cur_loss)
    f=open(save_dir+'loss_log.txt','a')
    f.write(save_string+'\n') ; 
    f.close()
    
    #save current time & image count shown 
    save_string = "{:0.2f},{}".format(elapsed_mins,image_count_shown)
    f=open(save_dir+'images_show_log.txt','a')
    f.write(save_string+'\n') ; 
    f.close()

# 40ms is diff between one frame
def main(args):

    start_resolution = args.start_resolution
    max_resolution = args.max_resolution


    SAVE_PATH = '/nfs/guille/wong/wonglab2/XAI/matt/PG-videoCNN/'
    ms_per_frame = 40
    input_frames = 10

    start_time = time.time()

    network = EncoderDecoder(args).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.90, 0.99))
    criterion = nn.MSELoss()

    train_steps = int(args.steps)


    if args.dataset == 'kth':
        dataset_frames = 15
        save_dir = SAVE_PATH + 'kth_grow{}/'.format(str(args.grow))
        max_resolution=128
    else:
        dataset_frames = 20
        max_resolution=64
        save_dir = SAVE_PATH + 'mmnist_grow{}/'.format(str(args.grow))


    IMG_PATH = save_dir + 'img_stacked/'
    os.makedirs(save_dir , exist_ok=True)
    os.makedirs(IMG_PATH , exist_ok=True)

    #clear the log files
    open(save_dir+'images_show_log.txt', 'w').close()
    open(save_dir+'loss_log.txt', 'w').close()

    total_steps_possible = ((train_steps * 2) * ((np.log2(max_resolution) - np.log2(start_resolution)))) + train_steps
    total_steps = 0

    resolution_loss = []
    dev_loss = []
    current_resolution = start_resolution
    step_scaling_resolution = 1
    fade_alpha = args.batch_size/args.steps

    if args.grow == 0:
        train_steps = total_steps_possible
        current_resolution = max_resolution
        network.set_level(int(np.log2(current_resolution)))

    train_loader, dev_loader, test_loader = get_data_loaders(args, current_resolution)
        

    print("Models and datasets loaded")
    start_time = time.time()

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
            for start_frame in range(dataset_frames-input_frames):
                for item in train_loader:
                    #label = item['label']
                    item = item['instance'].cuda()
                    batch_loss = 0

                    # fit a whole batch for all the different milliseconds
                    network.zero_grad()
                    frame_diff = 1

                    seq = item[:, :, start_frame:start_frame+input_frames, :, :]
                    seq = seq.squeeze()

                    seq_targ = item[:, :, start_frame+input_frames, :, :]

                    outputs = network(seq)
                    error = criterion(outputs, seq_targ)
                    error.backward()
                    optimizer.step()

                    loss = error.cpu().item()

                    batch_loss += loss
                    percent_steps = steps/train_steps
                    total_steps += args.batch_size

                    batch_num += 1
                    epoch_loss += batch_loss
                    steps += args.batch_size

                    if batch_num % 50 == 0:
                        print("{} RES {}, cur% {:0.2f}, total% {:0.1f}, loss {:0.4f} ".format("Stable" if stable else "Fading", current_resolution, percent_steps * 100, (total_steps/total_steps_possible) *100, loss))
                    if batch_num % 200 == 0:    
                        save_graph_data(save_dir, start_time, loss, total_steps)

                    if batch_num % 50 == 0 and epoch % 50 == 0 and stable and current_resolution > 16 and steps > 0.5:
                        img_print = torch.cat((seq[:8, :, :, :], outputs[:8, :, :, :]), dim=1).data.cpu().numpy()#.permute(0,2,3,1).data.cpu().numpy()
                        path = IMG_PATH+'train_output_res_{}_batch_{}_steps_{}_stable_{}.png'.format(current_resolution, batch_num, percent_steps, str(stable))
                        save_image(path, img_print)


                    
                    if percent_steps >= 0.985:
                        img_print = torch.cat((seq[:8, :, :, :], outputs[:8, :, :, :]), dim=1).data.cpu().numpy()#.permute(0,2,3,1).data.cpu().numpy()
                        path = IMG_PATH+'train_output_res_{}_batch_{}_steps_{}_stable_{}.png'.format(current_resolution, batch_num, percent_steps, str(stable))
                        save_image(path, img_print)
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
                curr_res_dev_loss += eval_model(network, dev_loader, IMG_PATH, current_resolution, percent_steps, epoch)
                curr_res_loss += epoch_loss
            network.train()
            epoch += 1

        #stable, _ = network.get_stability()
        print('\nSaving models...\n')
        torch.save(network.state_dict(), save_dir+str('model_pg.pth'))
        torch.save(optimizer.state_dict(), save_dir+str('optim_pg.pth'))
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
            train_loader, dev_loader, test_loader = get_data_loaders(args, current_resolution)
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

