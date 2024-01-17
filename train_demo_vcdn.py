import argparse
import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from data_generator.data_loader_vcdn import PictureLoader
from data_generator.utils import set_seed

from data_generator.utils import read_config

cfg = read_config()
args = cfg['balls2D']
args = argparse.Namespace(**args)

set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PictureLoader(args, phase=phase, trans_to_tensor=trans_to_tensor)

    datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    # n_his: 10
    # self.n_rollout * (args.time_step - args.n_his - args.n_roll + 1)
    data_n_batches[phase] = len(dataloaders[phase])

args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


def main():
    # print(args)
    show_data_shape = True
    # 开始训练
    for epoch in range(0, args.n_epoch):
        phases = ['train', 'valid']

        for phase in phases:

            bar = ProgressBar(maxval=data_n_batches[phase])
            loader = dataloaders[phase]

            for i, data in bar(enumerate(loader)):
                # data: [kps_preload, kps_gt, graph_gt, actions]

                if use_gpu:
                    if isinstance(data, list):
                        # nested transform
                        data = [[d.cuda() for d in dd] if isinstance(dd, list) else dd.cuda() for dd in data]
                    else:
                        data = data.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    if args.stage == 'dy':
                        '''
                        hyperparameter on the length of data
                        '''
                        n_samples = args.n_identify + args.n_his + args.n_roll

                        '''
                        load data
                        '''
                        # if detect keypoints during runtime
                        imgs, kps_gt, graph_gt = data[:3]
                        B, _, H, W = imgs.size()
                        imgs = imgs.view(B, n_samples, 3, H, W)

                        actions = data[-1]
                        B = kps_gt.size(0)

                        if show_data_shape:

                            print("=== Shapes of vcdn Data===")
                            print("image\t\t [B, chosen_frames, channel, W, H]:\t", imgs.shape)
                            print("ball_position\t [B, chosen_frames, n_ball, x&y]:\t", kps_gt.shape)

                            print("=== Shapes of vcdn Casual Graph===")

                            """
                            edge_type: Pairwise constraint type between balls
                            no relation, Spring or SlideJoint
                            n_type == 3
                            """
                            print("edge_type\t [B, n_ball, n_ball, n_type]:\t\t", graph_gt[0].shape)

                            """
                            edge_attr: The initial distance between the balls
                            """
                            print("edge_attr\t [B, n_ball, n_ball, n_attr]:\t\t", graph_gt[1].shape)

                            """
                            actions: The force (acceleration) applied to the ball per frame
                            """
                            print("actions\t\t [B, chosen_frame, n_obj, Fx&Fy]:\t", actions.shape)
                            show_data_shape = False


if __name__ == '__main__':
    main()
