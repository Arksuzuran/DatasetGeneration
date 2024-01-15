import multiprocessing as mp
import os
import time

from PIL import Image

import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset

from physics_engine import BallEngine

from utils import rand_int
from utils import init_stat, combine_stat, load_data, store_data
from utils import resize, crop


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


# 生成小球的数据集
def gen_Ball(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']
    n_ball = info['n_ball']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim    # radius
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = 2              # ddx, ddy

    # 对于该进程负责的所有rollout里，所有时间步的statistics
    # mean, std, count
    # stats: [attr, state, action]
    # attr: [ [attr1_mean, attr1_std, attr1_count], [attr2...], ...]
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = BallEngine(dt, state_dim, action_dim=2)

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        engine.init(n_ball)

        # 小球的数量
        n_obj = engine.num_obj

        # 时间步 * 物体 * 属性维数
        attrs_all = np.zeros((time_step, n_obj, attr_dim))
        # 时间步 * 物体 * 状态维数
        states_all = np.zeros((time_step, n_obj, state_dim))
        # 时间步 * 物体 * 动作维数
        actions_all = np.zeros((time_step, n_obj, action_dim))
        # 时间步 * {param_dim = n_ball * (n_ball - 1)} * 2
        # param_dim即两两之间的关系：[relation_type, coefficient]
        rel_attrs_all = np.zeros((time_step, engine.param_dim, 2))

        act = np.zeros((n_obj, 2))
        # 对每个时间步进行迭代
        for j in range(time_step):
            # 当前状态
            state = engine.get_state()
            # 整数除法
            vel_dim = state_dim // 2
            # 前vel_dim维：位置
            pos = state[:, :vel_dim]
            # 后面的维数：速度（velocity）
            vel = state[:, vel_dim:]

            # 通过两次的位置信息，计算此刻的速度
            if j > 0:
                vel = (pos - states_all[j - 1, :, :vel_dim]) / dt

            # 计算各属性，状态信息，关系
            attrs = np.zeros((n_obj, attr_dim))
            attrs[:] = engine.radius

            attrs_all[j] = attrs
            states_all[j, :, :vel_dim] = pos
            states_all[j, :, vel_dim:] = vel
            rel_attrs_all[j] = engine.param

            # 带有随机噪声的力 用于下一步迭代
            act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
            act = np.clip(act, -1000, 1000)
            engine.step(act)

            actions_all[j] = act.copy()

        datas = [attrs_all, states_all, actions_all, rel_attrs_all]
        # stat.h5 存储上述数据data：[attrs_all, states_all, actions_all, rel_attrs_all]
        store_data(data_names, datas, rollout_dir + '.h5')

        # 渲染，导出至图片帧或视频
        engine.render(states_all, actions_all, engine.get_param(), video=False, image=True,
                      path=rollout_dir, draw_edge=False, verbose=False)

        # 转换为float64
        datas = [datas[i].astype(np.float64) for i in range(len(datas))]

        # 对物体属性，物体状态，action，三者分别求解均值，标准差，元素数量
        # stats: [ 物体属性, 物体状态, action]
        # 物体属性：[ 属性1, 属性2, ...]
        # 属性1: [mean, std, 时间步的数量(即这是多少个状态在求平均)]
        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            # 所有时间步下，所有物体的统计量
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


class PhysicsDataset(Dataset):

    def __init__(self, args, phase, trans_to_tensor=None, loader=default_loader):
        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        self.data_dir = os.path.join(self.args.dataf, phase)

        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        if args.env in ['Ball']:
            self.data_names = ['attrs', 'states', 'actions', 'rels']
        else:
            raise AssertionError("Unknown env")

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

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        # 多进程，此处num_workers值一般为10
        # 根据传入参数来确定 等下要传给具体gen方法的参数
        infos = []
        # 为每个进程设置不同参数
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw}

            if self.args.env in ['Ball']:
                info['env'] = 'Ball'
                info['n_ball'] = self.args.n_ball

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        # 调用上面的gen_{env}方法，生成数据data
        if env in ['Ball']:
            data = pool.map(gen_Ball, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        #
        # 已经获得生成数据，接下来求全体的统计信息
        if self.phase == 'train':
            if env in ['Ball']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            # 即依次融合，求整个的stat
            # 这里的data：
            # [视频1[mean, std, cnt]，视频2[mean, std, cnt]，...]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            # 保存stat
            store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

        else:
            print("Loading stat from %s ..." % self.stat_path)
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
        suffix = '.png' if args.env in ['Ball'] else '.jpg'

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
            src_path = os.path.join(args.dataf, self.phase, str(src_rollout), 'fig_%d%s' % (src_timestep, suffix))

            # use the same rollout if in Cloth
            # des_rollout = rand_int(0, self.n_rollout) if args.env in ['Ball'] else src_rollout
            des_rollout = rand_int(0, self.n_rollout)
            des_timestep = rand_int(0, args.time_step)
            des_path = os.path.join(args.dataf, self.phase, str(des_rollout), 'fig_%d%s' % (des_timestep, suffix))

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

            # if using detected keypoints
            # 1
            if args.preload_kp == 1:
                # if using preload keypoints
                # 读取关键点数据
                path = os.path.join(args.dataf + '_nKp_%d' % args.n_kp, self.phase, str(src_rollout) + '.h5')
                # frame_offset: 1
                # 按照 frame_offset 的步长进行采样
                kps_pred = load_data(['keypoints'], path)[0][::args.frame_offset]

                # kps: B x (n_identify + n_his + n_roll) x n_kp x 2
                kps_preload = np.concatenate([
                    kps_pred[infer_st_idx : infer_st_idx + args.n_identify],
                    kps_pred[src_timestep : src_timestep + args.n_his + args.n_roll]], 0)
                kps_preload = torch.FloatTensor(kps_preload)

            # 需要直接读取图像
            else:
                # if detect keypoints during runtime
                for i in range(infer_st_idx, infer_st_idx + args.n_identify):
                    path = os.path.join(args.dataf, self.phase, str(src_rollout), 'fig_%d%s' % (i, suffix))
                    img = self.loader(path)
                    img = resize_and_crop(self.phase, img, self.scale_size, self.crop_size)
                    img = self.trans_to_tensor(img)
                    imgs.append(img)

                # load images for dynamics prediction
                for i in range(args.n_his + args.n_roll):
                    path = os.path.join(args.dataf, self.phase, str(src_rollout), 'fig_%d%s' % (src_timestep + i, suffix))
                    img = self.loader(path)
                    img = resize_and_crop(self.phase, img, self.scale_size, self.crop_size)
                    img = self.trans_to_tensor(img)
                    imgs.append(img)

                imgs = torch.cat(imgs, 0)
                assert imgs.size(0) == (args.n_identify + args.n_his + args.n_roll) * 3


            if args.env in ['Ball']:
                # 1
                # 读取边的ground truth
                # get ground truth edge type
                data_path = os.path.join(args.dataf, self.phase, str(src_rollout) + '.h5')
                # metadata: [attrs_all, states_all, actions_all, rel_attrs_all]
                metadata = load_data(self.data_names, data_path)

                edge_type = metadata[3][0, :, 0].astype(np.int)
                edge_attr = metadata[3][0, :, 1:]

                # gt: ground truth
                # n_kp x n_kp x 3
                edge_type_gt = np.zeros((args.n_kp, args.n_kp, args.edge_type_num))
                # n_kp x n_kp x 1
                edge_attr_gt = np.zeros((args.n_kp, args.n_kp, edge_attr.shape[1]))

                cnt = 0
                for x in range(args.n_kp):
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
                kps_gt[:, :, 1] *= -1   # y反转？
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
                # actions: (n_identify + n_his + n_roll) x n_kp x action_dim
                # print('actions size', actions.size())

                # if using detected keypoints
                # 1
                if args.preload_kp == 1:
                    # if using preloaded keypoints
                    return kps_preload, kps_gt, graph_gt, actions
                else:
                    # if detecting keypoints during runtime
                    return imgs, kps_gt, graph_gt, actions

            else:
                raise AssertionError("Unknown env %s" % args.env)


