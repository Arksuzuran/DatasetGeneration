import multiprocessing as mp
import os
import time

from PIL import Image

import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset

from data_generator.utils import rand_int, load_data, resize, crop


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


def get_crop_params(phase, img, crop_size):
    w, h = img.size

    if w < h:
        tw = crop_size
        th = int(crop_size * h / w)
    else:
        th = crop_size
        tw = int(crop_size * w / h)

    if phase == 'train':
        if w == tw and h == th:
            return 0, 0, h, w
        assert False
        i = rand_int(0, h - th)
        j = rand_int(0, w - tw)

    else:
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

    return i, j, th, tw


def resize_and_crop(phase, src, scale_size, crop_size):
    """
    缩放并裁剪
    """
    # resize the images
    src = resize(src, scale_size)

    # crop the images
    crop_params = get_crop_params(phase, src, crop_size)
    src = crop(src, crop_params[0], crop_params[1], crop_params[2], crop_params[3])

    return src


def default_loader(path):
    return pil_loader(path)


class PictureLoader(Dataset):

    def __init__(self, args, phase, trans_to_tensor=None, loader=default_loader):
        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        self.data_dir = os.path.join(self.args.data_dir, phase)

        self.stat_path = os.path.join(self.args.data_dir, 'stat.h5')
        self.stat = None

        self.data_names = ['attrs', 'states', 'actions', 'rels']

        # n_rollout 一共生成的视频的数量
        ratio = self.args.train_valid_ratio
        if phase in {'train'}:
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase in {'valid'}:
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.time_step
        self.scale_size = args.scale_size
        self.crop_size = args.crop_size

    def load_data(self):
        """
        从stat.h5文件中读取全体数据的stat
        """
        self.stat = load_data(self.data_names, self.stat_path)

    def __len__(self):
        args = self.args
        if args.stage == 'kp':
            length = self.n_rollout * args.time_step
        elif args.stage in 'dy':
            # 每个视频内，并不是全部元素都可以作为滑动窗口的第一个元素
            # 滑动窗口内，要先有n_his个元素用来作为历史信息，随后的一个元素作为预测值
            # 一共要预测n_roll次
            length = self.n_rollout * (args.time_step - args.n_his - args.n_roll + 1)
        return length

    # 进行for…in…迭代时：
    # 此时idx是滑动窗口的第一个元素
    def __getitem__(self, idx):
        args = self.args
        suffix = '.png'

        if args.stage == 'kp':
            src_rollout = idx // args.time_step
            src_timestep = idx % args.time_step
        elif args.stage in 'dy':
            # time_step: 500
            # n_his: 10 可以使用多少个历史帧来进行预测
            # n_roll 20
            # 一个视频内，允许作为窗口第一个元素的下标最大值
            offset = args.time_step - args.n_his - args.n_roll + 1
            # 当前在第几个视频
            src_rollout = idx // offset
            # 当前窗口起点在哪个元素
            src_timestep = idx % offset

        '''
        used for keypoint detection
        关键点检测采样
        src：根据idx来依序加载的图片
        des：随机加载的图片
        图片会经由处理管道进行处理
        '''
        if args.stage == 'kp':
            src_path = os.path.join(self.data_dir, str(src_rollout), 'fig_%d%s' % (src_timestep, suffix))

            # use the same rollout if in Cloth
            # des_rollout = rand_int(0, self.n_rollout) if args.env in ['Ball'] else src_rollout
            des_rollout = rand_int(0, self.n_rollout)
            des_timestep = rand_int(0, args.time_step)
            des_path = os.path.join(self.data_dir, str(des_rollout), 'fig_%d%s' % (des_timestep, suffix))

            src = self.loader(src_path)
            des = self.loader(des_path)

            src = resize_and_crop(self.phase, src, self.scale_size, self.crop_size)
            des = resize_and_crop(self.phase, des, self.scale_size, self.crop_size)

            src = self.trans_to_tensor(src)
            des = self.trans_to_tensor(des)

            return src, des

        '''
        used for dynamics modeling
        因果建模和推理
        
        '''
        if args.stage in 'dy':
            imgs = []
            kp_preload = None

            # load images for graph inference
            # n_identify: 100
            # 随机选取元素，进行图结构推理
            infer_st_idx = rand_int(0, args.time_step - args.n_identify + 1)

            # if detect keypoints during runtime
            for i in range(infer_st_idx, infer_st_idx + args.n_identify):
                path = os.path.join(self.data_dir, str(src_rollout), 'fig_%d%s' % (i, suffix))
                img = self.loader(path)
                img = resize_and_crop(self.phase, img, self.scale_size, self.crop_size)
                img = self.trans_to_tensor(img)
                imgs.append(img)

            # load images for dynamics prediction
            for i in range(args.n_his + args.n_roll):
                path = os.path.join(self.data_dir, str(src_rollout), 'fig_%d%s' % (src_timestep + i, suffix))
                img = self.loader(path)
                img = resize_and_crop(self.phase, img, self.scale_size, self.crop_size)
                img = self.trans_to_tensor(img)
                imgs.append(img)

            imgs = torch.cat(imgs, 0)
            assert imgs.size(0) == (args.n_identify + args.n_his + args.n_roll) * 3

            # 1
            # 读取边的ground truth
            # get ground truth edge type
            data_path = os.path.join(self.data_dir, str(src_rollout) + '.h5')
            # metadata: [attrs_all, states_all, actions_all, rel_attrs_all]
            metadata = load_data(self.data_names, data_path)

            edge_type = metadata[3][0, :, 0].astype(np.int32)
            edge_attr = metadata[3][0, :, 1:]

            # gt: ground truth
            # n_ball x n_ball x 3
            edge_type_gt = np.zeros((args.n_ball, args.n_ball, args.edge_type_num))
            # n_ball x n_ball x 1
            edge_attr_gt = np.zeros((args.n_ball, args.n_ball, edge_attr.shape[1]))

            cnt = 0
            for x in range(args.n_ball):
                for y in range(x):
                    edge_type_gt[x, y, edge_type[cnt]] = 1.
                    edge_type_gt[y, x, edge_type[cnt]] = 1.
                    edge_attr_gt[x, y] = edge_attr[cnt]
                    edge_attr_gt[y, x] = edge_attr[cnt]
                    cnt += 1

            edge_type_gt = torch.FloatTensor(edge_type_gt)
            edge_attr_gt = torch.FloatTensor(edge_attr_gt)

            graph_gt = edge_type_gt, edge_attr_gt

            # 2
            # 读取状态(坐标)的ground truth
            # get ground truth keypoint position
            # 划到[-1,1]: 80是物理引擎中场景的边界lim
            states = metadata[1] / 80.
            kps_gt_id = states[infer_st_idx:infer_st_idx + args.n_identify, :, :2]
            kps_gt_dy = states[src_timestep:src_timestep + args.n_his + args.n_roll, :, :2]
            kps_gt = np.concatenate([kps_gt_id, kps_gt_dy], 0)
            kps_gt[:, :, 1] *= -1  # y反转？
            kps_gt = torch.FloatTensor(kps_gt)

            # 3
            # 读取力的ground truth
            # get ground truth actions
            # 600: 一个估计的上界？
            actions = metadata[2] / 600.
            actions_id = actions[infer_st_idx:infer_st_idx + args.n_identify]
            actions_dy = actions[src_timestep:src_timestep + args.n_his + args.n_roll]
            actions = np.concatenate([actions_id, actions_dy], 0)
            actions = torch.FloatTensor(actions)
            # actions: (n_identify + n_his + n_roll) x n_ball x action_dim
            # print('actions size', actions.size())

            # if detecting keypoints during runtime
            return imgs, kps_gt, graph_gt, actions


