import os

import torch
import argparse
from models.Derendering import Derendering
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import random
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

from data_generator.data_loaders_cophy import BlocktowerLoader, BallsLoader, CollisionLoader

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int, help="Number of Epoch for training. Can be set to 0 for evaluation")
parser.add_argument('--n_keypoints', default=4, type=int, help="Number of keypoints to use")
parser.add_argument('--dataset', default="balls", type=str, help="Datasets, should be one of 'blocktower', 'balls' or 'collision")
parser.add_argument('--lr', default=0.0001, type=float, help="Learning Rate")
parser.add_argument('--n_coefficients', default=4, type=int, help="Number of coefficients to use")
parser.add_argument('--mode', default="fixed", type=str, help="'fixed': use fixed dilatation filter bank. 'learned': learn the filters via gradient descent")
parser.add_argument('--seed', default=1, type=int, help="Random seed")
parser.add_argument('--name', default='derendering', type=str, help="Name for weight saving")
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
BATCHSIZE = 1

os.makedirs("./trained_models/unsupervised_derendering", exist_ok=True)
SAVE_PATH = f"./trained_models/unsupervised_derendering/{args.name}.nn"
datasets = {"blocktower": BlocktowerLoader, "balls": BallsLoader, "collision": CollisionLoader}


def evaluate():
    print(args)
    dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112, sampling_mode="fix", load_state=True, load_cd=True), batch_size=1, shuffle=False)

    # Path to the saved weights
    state_dict = torch.load(SAVE_PATH, map_location=device)

    model = Derendering(n_keypoints=args.n_keypoints,
                        mode=args.mode,
                        n_coefficients=args.n_coefficients,
                        device=device).to(device)

    model.load_state_dict(state_dict)
    validate(model, dataloader, viz=False)


def vizu(source, target, target_hat, target_keypoints):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    source = source.detach().cpu().numpy()[0].transpose((1, 2, 0))
    target = target.detach().cpu().numpy()[0].transpose((1, 2, 0))
    pred = target_hat.detach().cpu().numpy()[0].transpose((1, 2, 0))

    source = (source + 1) / 2
    target = (target + 1) / 2
    pred = (pred + 1) / 2

    ax1.imshow(source)
    ax2.imshow(target)
    ax3.imshow(target)
    ax4.imshow(pred)

    keypoints_target = target_keypoints.detach().cpu().numpy()[0]
    X, Y = (keypoints_target[:, 1] + 1) / 2 * 112, (keypoints_target[:, 0] + 1) / 2 * 112
    ax3.scatter(X, Y, c="black", s=2)

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    ax1.set_title("Source")
    ax2.set_title("Target")
    ax3.set_title("Keypoints")
    ax4.set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False):
    model.eval()
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            rgb = x['rgb_cd'].to(device)
            source, target = rgb[:, 0], rgb[:, 1]
            out = model(source, target)

            costs = get_loss(target, out["target"])

            cd = target.cpu().numpy().transpose(0, 2, 3, 1)
            predictions = out["target"].cpu().numpy().transpose(0, 2, 3, 1)
            costs.append(
                np.mean([peak_signal_noise_ratio(cd[b], predictions[b], data_range=2) for b in range(cd.shape[0])]))
            list_loss.append(costs[0].cpu().detach().numpy())

            if viz:
                vizu(rgb[:, 0], target, out["target"], out["target_keypoints"])

    print("Val Loss : ", np.mean(list_loss))


def compute_grad(rgb):
    rgb = rgb.view(-1, 3, 112, 112)
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).expand(1, 3, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).expand(1, 3, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)

    return sobel


def get_loss(target, target_hat):
    loss_rgb = MSE(target_hat, target)
    loss_grad = MSE(compute_grad(target), compute_grad(target_hat))

    loss = 10_000 * loss_rgb + loss_grad * 0.1

    return [loss, loss_rgb, loss_grad]


def main():
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dataloader = DataLoader(
        datasets[args.dataset](phase='train', resolution=112, sampling_mode="rand"),
        batch_size=BATCHSIZE, num_workers=1, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(
        datasets[args.dataset](phase='val', resolution=112, sampling_mode='fix'),
        batch_size=BATCHSIZE, num_workers=1, pin_memory=True)

    model = Derendering(n_keypoints=args.n_keypoints,
                        mode=args.mode,
                        n_coefficients=args.n_coefficients,
                        device=device).to(device)

    trained_params = model.parameters()
    optim = torch.optim.Adam(trained_params, lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            rgb = x['rgb_cd'].to(device)
            states = x["states_cd"].to(device)
            confounders = x["confounders"].to(device)

            print("=== Shapes of Cophy Data ===")

            """
            chosen frames from video cd
            which frame is extracted depends on the dataloader
            """
            print("images\t\t [B, chosen_frames, channel, W, H]:\t", rgb.shape)

            """
            motion trail of video cd
            13 = 3 + 4 + 4 + 3
            (Position coordinates, direction of motion, velocity, angular velocity)
            """
            print("states\t\t [B, all_frames, n_obj, motion_attr]:\t", states.shape)

            """
            mass of objects
            """
            print("confounders\t [B, n_obj]:\t\t\t\t", confounders.shape)

    #         source, target = rgb[:, 0], rgb[:, 1]
    #         out = model(source, target)
    #         cost = get_loss(target, out["target"])
    #
    #         optim.zero_grad()
    #         cost[0].backward()
    #         optim.step()
    #
    #     if epoch == 100:
    #         optim = torch.optim.Adam(trained_params, lr=args.lr)
    #
    #     validate(model, val_dataloader)
    #     torch.save(model.state_dict(), SAVE_PATH)
    # validate(model, val_dataloader)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
