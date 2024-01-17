from torch.utils.data import Dataset
import numpy as np
import os
from random import randint
import torchvision
import pybullet as pb
import cv2
import torch
from data_generator.utils import read_config


class VideoLoader(Dataset):
    def __init__(self, phase="train", resolution=112, load_cd=True, sampling_mode="rand",
                 load_ab=False, load_state=True, load_confounders=True):
        """
        Dataloader for BallsCF
        :param phase: 'train', 'test' or 'val' split
        :param resolution: Image resolution, default is 112x112
        :param load_cd: if False, the dataloader does not read CD video
        :param sampling_mode: 'rand' for random selection in the video, 'fix' for sampling at fixed timestamps and
        'full' for loading the entire video. This is useful for training De-rendering, where only two images are
        needed instead of the entire video
        :param load_ab: if False, the dataloader does not read AB video
        :param load_state: True to load the 2D projection of the ground truth state
        """
        super(VideoLoader, self).__init__()
        assert sampling_mode in ['rand', 'fix', "full"]

        self.phase = phase
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.load_confounders = load_confounders

        self.cfg = {}
        self.load_config()

        self.n_rollout = 0
        self.data_path = ''
        self.video_length = 0
        self.seed = 1
        self.scene = ""

    def load_config(self):
        self.cfg = read_config()

    def load_scene_config(self):
        config = self.cfg[self.scene]

        self.data_path = self.cfg['root_data_path'] + self.scene
        assert os.path.isdir(self.data_path)

        self.n_rollout = config['n_rollout']
        self.video_length = config['video_length']
        self.seed = config['seed']

    def __len__(self):
        return self.n_rollout

    def get_projection_matrix(self):
        """
        不同子类的摄像机参数不同，由子类实现具体的参数
        """
        raise NotImplementedError("Please Implement this method")

    def __getitem__(self, item):
        out = {}
        dir_name = str(self.seed) + "_" + str(item + 1)
        if self.load_ab:
            ab = os.path.join(self.data_path, dir_name, "ab", 'rgb.mp4')
            rgb_ab, r_ab = get_rgb(ab, self.sampling_mode, self.video_length)
            out['rgb_ab'] = rgb_ab

        if self.load_cd:
            cd = os.path.join(self.data_path, dir_name, "cd", 'rgb.mp4')
            rgb_cd, r_cd = get_rgb(cd, self.sampling_mode, self.video_length)
            out['rgb_cd'] = rgb_cd

        if self.load_state:

            states = np.load(os.path.join(self.data_path, dir_name, 'cd', 'states.npy'))
            out["states_cd"] = states

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112))
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]


        if self.load_confounders:
            confounders = np.load(os.path.join(self.data_path, dir_name, 'confounders.npy'))
            out["confounders"] = confounders
            print(confounders)

        return out


class BallsLoader(VideoLoader):
    def __init__(self, **kwargs):
        super(BallsLoader, self).__init__(**kwargs)

        self.scene = "balls"
        self.load_scene_config()

    def __len__(self):
        return self.n_rollout

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 1, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix


class BlocktowerLoader(VideoLoader):
    def __init__(self, **kwargs):
        super(BlocktowerLoader, self).__init__(**kwargs)

        self.scene = "blocktower"
        self.load_scene_config()

    def __len__(self):
        return self.n_rollout

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix


class CollisionLoader(VideoLoader):
    def __init__(self, **kwargs):
        super(CollisionLoader, self).__init__(**kwargs)

        self.scene = "collision"
        self.load_scene_config()

    def __len__(self):
        return self.n_rollout

    def get_projection_matrix(self):
        viewMatrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
            (4, 4)).transpose()
        projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
        return viewMatrix, projectionMatrix


def convert_to_2d(pose, view, projection, resolution):
    center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
    center_pose = view @ center_pose
    center_pose = projection @ center_pose
    center_pose = center_pose[:3] / center_pose[-1]
    center_pose = (center_pose + 1) / 2 * resolution
    center_pose[1] = resolution - center_pose[1]
    return center_pose[:2].astype(int).flatten()


def get_rgb(filedir, sampling_mode, video_length):
    if sampling_mode == "full":
        rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        rgb = 2 * (rgb / 255) - 1
        rgb = rgb.permute(0, 3, 1, 2)
        r = list(range(video_length))
    else:
        t = randint(0, int(0.15 * video_length)) if sampling_mode == "rand" else int(0.15 * video_length)

        r = [t, t + int(0.15 * video_length)]
        capture = cv2.VideoCapture(filedir)
        list_rgb = []
        for i in r:
            capture.set(1, i)
            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_rgb.append(frame)
        rgb = np.stack(list_rgb, 0)
        rgb = 2 * (rgb / 255) - 1
        rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
        rgb = torch.FloatTensor(rgb)
    return rgb, r
