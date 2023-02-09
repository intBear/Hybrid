import torch
import math
import torch.nn as nn
import logging
import torchvision.datasets as datasets
import random
import imageio.v2 as imageio
from torchvision import transforms
from torchvision.utils import save_image
from decoder import Decoder
from utils import *
import argparse
from train_hybrid import Trainer
import wandb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--logdir",
    help="Path to save logs",
    default=f"./logs"
)
parser.add_argument(
    "--epoch",
    help="Number of iterations to train for",
    type=int,
    default=50000,
)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument(
    "--lr0",
    help="Learning rate of decoder",
    type=float,
    default=2e-4,
)
parser.add_argument(
    "--lr1",
    help="Learning rate of latent code weight",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--dim_hidden",
    help="width of hidden layers",
    type=int,
    default=24,
)
parser.add_argument(
    "--num_layers",
    help="number of hidden layers",
    type=int,
    default=2,
)
parser.add_argument(
    "--L",
    help="the Level of transmitting features",
    type=int,
    default=7,
)
parser.add_argument(
    "--model_path",
    help="save codec models.",
    type=str,
    default='models/train_codec/',
)
parser.add_argument(
    "-fd",
    "--full_dataset",
    help="Whether to use full dataset",
    default=False,
    action='store_true',
)
parser.add_argument(
    "-iid",
    "--image_id",
    help="Image ID to train on, if not the full dataset",
    type=int,
    default=15,
)
parser.add_argument(
    "--resume",
    type=int,
    default=0,
)
parser.add_argument(
    "--lmbda",
    help="The parameter of Rate-Distortion.",
    type=float,
    default=0.018,
)
parser.add_argument(
    "--use_positional_info",
    type=int,
    default=1,
)

parser.add_argument(
    "--positional_freq",
    type=int,
    default=10,
)
# Wandb arguments
parser.add_argument(
    "--use_wandb",
    type=int,
    default=1,
)

parser.add_argument(
    "--wandb_project_name",
    type=str,
    default="Hybrid",
)

parser.add_argument(
    "--wandb_entity",
    type=str,
    default="intbear",
)

parser.add_argument(
    "--wandb_job_type",
    help="Wandb job type. This is useful for grouping runs together.",
    type=str,
    default=None,
)

args = parser.parse_args()
dtype = torch.float32
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.use_wandb:
    # Initialize wandb experiment
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        job_type=args.wandb_job_type,
        config=args,
    )
if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id


L = args.L
results = {'psnr': [], 'loss': []}
# load dataset Kodak 768*512
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    # Load Image
    img = imageio.imread(f"/media/Dataset/kodak/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)
    C, H, W = img.shape
    # Set up model
    if args.use_positional_info:
        _, out_dim = get_positional_embedder(args.positional_freq, True)
    else:
        out_dim = 0
    decoder = Decoder(args, device, out_dim).to(device)
    # varied learning rate
    code_length = int((H * W) * (4 / 3) * (1 - (1 / 4 ** L)))
    lat_layer = torch.nn.Embedding(1, code_length, max_norm=1)
    lat_layer = lat_layer.to(device)
    # weight initial
    torch.nn.init.normal_(
        lat_layer.weight.data,
        0.0,
        1.0 / math.sqrt(H * W),
    )
    rd_losses = RateDistortionLoss(args.lmbda)
    trainer = Trainer(
        device,
        img,
        decoder,
        rd_losses,
        lat_layer,
        args,
        print_freq=1,
        )
    trainer.train(args.epoch)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    print(f'Best training loss: {trainer.best_vals["loss"]:.2f}')
    results['psnr'].append(trainer.best_vals['psnr'])
    results['loss'].append(trainer.best_vals['loss'])
    # Save best model

