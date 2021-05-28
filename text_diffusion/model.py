import torch
import torch.nn as nn
from diffusion_utils.diffusion_multinomial import MultinomialDiffusion

from layers.transformer import LinearAttentionTransformerEmbedding


class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x



def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--actnorm', type=eval, default=False)
    parser.add_argument('--perm_channel', type=str, default='none', choices={'conv', 'shuffle', 'none'})
    parser.add_argument('--perm_length', type=str, default='reverse', choices={'reverse', 'none'})

    parser.add_argument('--input_dp_rate', type=float, default=0.0)

    # Transformer params.
    parser.add_argument('--transformer_dim', type=int, default=512)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_depth', type=int, default=2)
    parser.add_argument('--transformer_blocks', type=int, default=1)
    parser.add_argument('--transformer_dropout', type=float, default=0.0)
    parser.add_argument('--transformer_reversible', type=eval, default=False)

    parser.add_argument('--transformer_local_heads', type=int, default=4)
    parser.add_argument('--transformer_local_size', type=int, default=128)

    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--diffusion_sharing', type=eval, default=True)
    parser.add_argument('--diffusion_loss', type=str, default='vb_stochastic')
    parser.add_argument('--diffusion_parametrization', type=str, default='x0')

    parser.add_argument('--diffusion_binary_scale', type=float, default=.5)


def get_model_id(args):
    return 'multinomial_diffusion_v2'


def get_model(args, data_shape, num_classes):
    data_shape = data_shape
    num_classes = num_classes
    input_dp_rate = args.input_dp_rate
    transformer_dim = args.transformer_dim
    transformer_heads = args.transformer_heads
    transformer_depth = args.transformer_depth
    transformer_blocks = args.transformer_blocks
    transformer_local_heads = args.transformer_local_heads
    transformer_local_size = args.transformer_local_size
    transformer_reversible = args.transformer_reversible
    diffusion_steps = args.diffusion_steps
    diffusion_loss = args.diffusion_loss
    diffusion_parametrization = args.diffusion_parametrization

    C, L = 1, data_shape[0]

    current_shape = (L,)

    class DynamicsTransformer(nn.Module):
        def __init__(self):
            super(DynamicsTransformer, self).__init__()
            self.transformer = LinearAttentionTransformerEmbedding(
                input_dim=num_classes,
                output_dim=num_classes,
                dim=transformer_dim,
                heads=transformer_heads,
                depth=transformer_depth,
                n_blocks=transformer_blocks,
                max_seq_len=L,
                num_timesteps=diffusion_steps,
                causal=False,  # auto-regressive or not
                ff_dropout=0,  # dropout for feedforward
                attn_layer_dropout=input_dp_rate,
                # dropout right after self-attention layer
                attn_dropout=0,  # dropout post-attention
                n_local_attn_heads=transformer_local_heads,
                # number of local attention heads for (qk)v attention.
                # this can be a tuple specifying the exact number of local
                # attention heads at that depth
                local_attn_window_size=transformer_local_size,
                # receptive field of the local attention
                reversible=transformer_reversible,
                # use reversible nets, from Reformer paper
            )

            self.rezero = Rezero()

        def forward(self, t, x):
            x = self.transformer(x, t)
            x = x.permute(0, 2, 1)
            x = self.rezero(x)
            return x

    dynamics = DynamicsTransformer()

    base_dist = MultinomialDiffusion(
        num_classes, current_shape, dynamics,
        timesteps=diffusion_steps,
        loss_type=diffusion_loss,
        parametrization=diffusion_parametrization)
    return base_dist
