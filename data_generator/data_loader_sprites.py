import argparse
import time
import numpy as np
from torch.utils.data import Dataset
from data_generator.utils import read_config

cfg = read_config()
ARGS = cfg['sprites']
ARGS = argparse.Namespace(**ARGS)


def sprites_act(seed=0, return_labels=True):
    # directions = ['front', 'left', 'right']
    # actions = ['walk', 'spellcard', 'slash']
    directions = ['front']
    actions = ['walk']
    start = time.time()
    path = ARGS.npy_save_dir
    X_train = []
    X_test = []
    if return_labels:
        A_train = []
        A_test = []
        D_train = []
        D_test = []
    for act in range(len(actions)):
        for i in range(len(directions)):
            label = 3 * act + i
            print(actions[act], directions[i], act, i, label)
            x = np.load(path + '%s_%s_frames_train.npy' % (actions[act], directions[i]))
            X_train.append(x)
            y = np.load(path + '%s_%s_frames_test.npy' % (actions[act], directions[i]))
            X_test.append(y)
            if return_labels:
                a = np.load(path + '%s_%s_attributes_train.npy' % (actions[act], directions[i]))
                A_train.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1
                D_train.append(d)

                a = np.load(path + '%s_%s_attributes_test.npy' % (actions[act], directions[i]))
                A_test.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1
                D_test.append(d)

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    np.random.seed(seed)
    ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[ind]
    if return_labels:
        A_train = np.concatenate(A_train, axis=0)
        D_train = np.concatenate(D_train, axis=0)
        A_train = A_train[ind]
        D_train = D_train[ind]
    ind = np.random.permutation(X_test.shape[0])
    X_test = X_test[ind]
    if return_labels:
        A_test = np.concatenate(A_test, axis=0)
        D_test = np.concatenate(D_test, axis=0)
        A_test = A_test[ind]
        D_test = D_test[ind]
        print(A_test.shape, D_test.shape, X_test.shape, 'shapes')
    print(X_train.shape, X_test.min(), X_test.max())
    end = time.time()
    print('data loaded in %.2f seconds...' % (end - start))

    if return_labels:
        return X_train, X_test, A_train, A_test, D_train, D_test
    else:
        return X_train, X_test


class SpriteLoader(Dataset):
    def __init__(self, phase='train', return_labels=True):
        self.phase = phase
        self.return_labels = return_labels

        self.images = []
        self.A_label = []
        self.D_label = []

        self.load_data()
        self.N = len(self.images)

    def load_data(self):
        if self.phase == 'train':
            if self.return_labels:
                self.images, _, self.A_label, _, self.D_label, _ = sprites_act(return_labels=self.return_labels)
            else:
                self.images, _ = sprites_act(return_labels=self.return_labels)
        elif self.phase == 'test':
            if self.return_labels:
                _, self.images, _, self.A_label, _, self.D_label = sprites_act(return_labels=self.return_labels)
            else:
                _, self.images = sprites_act(return_labels=self.return_labels)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        images = self.images[index]
        out = {"images": images, "index": index}
        if self.return_labels:
            A_label = self.A_label[index]
            D_label = self.D_label[index]
            out["A_label"] = A_label
            out["D_label"] = D_label

        return out
