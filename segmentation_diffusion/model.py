import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion_utils.diffusion_multinomial import MultinomialDiffusion
from layers.layers import SegmentationUnet


def add_model_args(parser):
    # Model params
    parser.add_argument('--loss_type', type=str, default='vb_stochastic')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=64)
    parser.add_argument('--dp_rate', type=float, default=0.)


def get_model_id(args):
    return 'multinomial_diffusion'


def get_model(args, data_shape):
    data_shape = torch.Size(data_shape)

    current_shape = data_shape

    if (data_shape[-1] // 4) % 2 == 0:
        dim_mults = (1, 2, 4, 8)
    else:
        dim_mults = (1, 4, 8)

    dynamics = SegmentationUnet(
        num_classes=args.num_classes,
        dim=args.diffusion_dim,
        num_steps=args.diffusion_steps,
        dim_mults=dim_mults,
        dropout=args.dp_rate
    )

    base_dist = MultinomialDiffusion(
        args.num_classes, current_shape, dynamics, timesteps=args.diffusion_steps,
        loss_type=args.loss_type)

    return base_dist
