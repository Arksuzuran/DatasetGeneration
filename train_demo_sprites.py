import argparse
import torch
from progressbar import ProgressBar
from torch.utils.data import Dataset, DataLoader
from data_generator.data_loader_sprites import SpriteLoader

from data_generator.utils import read_config

cfg = read_config()
args = cfg['sprites']
args = argparse.Namespace(**args)

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'test']:
    datasets[phase] = SpriteLoader(phase=phase, return_labels=True)

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=1,
        shuffle=True if phase == 'train' else False,
        num_workers=10)

    data_n_batches[phase] = len(dataloaders[phase])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def main():
    show_data_shape = True
    # 开始训练
    for epoch in range(0, 1):
        print("=== EPOCH ", epoch + 1, " ===")
        phases = ['train', 'test']

        for phase in phases:

            bar = ProgressBar(maxval=data_n_batches[phase])
            loader = dataloaders[phase]

            for i, data in bar(enumerate(loader)):

                with torch.set_grad_enabled(phase == 'train'):
                    images = data['images'].to(device)
                    A_label = data['A_label'].to(device)
                    D_label = data['D_label'].to(device)

                    if show_data_shape:
                        print("=== Shapes of Sprite Data ===")

                        print("images\t\t\t [B, n_frames, W, H, channel]:\t", images.shape)

                        """
                        attr: ['body', 'bottomwear', 'topwear', 'hair']
                        type: each attr has 6 different types. 
                        """
                        print("attribute labels\t [B, n_frames, n_attr, type]:\t", A_label.shape)

                        """
                        action: ['spellcard', 'walk', 'slash']
                        direction: ['front', 'left', 'right']
                        label = 3 * act_label + dir_label
                        For example, if the character is walking left, then the label is 4.
                        """
                        print("action labels\t\t [B, n_frames, n_action*n_direction]:\t\t", D_label.shape)
                        show_data_shape = False


if __name__ == '__main__':
    main()
