from vit_pytorch.vit import Transformer, pair
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
import numpy as np

# normalize (0, 1) to (-1, 1)
def preprocess(t):
    return t * 2 - 1

# denormalize from (-1, 1) to (0, 1)
def postprocess(t):
    return torch.clamp((t + 1) / 2, min=0, max=1)


class My_ViT(nn.Module):
    # def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64):
        super(My_ViT, self).__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # print('number of patches', num_patches)
        # patch_dim = channels * patch_height * patch_width
        # print('patch_dim', patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(channels, dim)
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # relative positional embedding
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # print('positional embedding', self.pos_embedding.shape)

        spatial_dim = 2
        sigma = 10
        self.pos_embedding = nn.Parameter(torch.randn((spatial_dim, dim)) * sigma)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        print('cls token', self.cls_token.shape)
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=0)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        b, c, h, w = img.shape
        rel_pos = uniform_coordinates(h, w, flatten=False)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


class InputMapping:
# class InputMapping(nn.Module):
    def __init__(self, mapping_size=256, dim=2, B='gaussian', sigma=10):
        # super(InputMapping, self).__init__()
        if B == 'gaussian':
            self.B = torch.randn((dim, mapping_size)) * sigma
        elif B == 'uniform':
            self.B = torch.rand((dim, mapping_size)) * sigma
        else:
            raise ValueError('wrong B type. Got {}'.format(B))
        # self.B = nn.Parameter(self.B)

    def map(self, x):
    # def forward(self, x):
        x_proj = torch.mm((2 * np.pi * x), self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def uniform_coordinates(h, w, _range=(-1, 1), flatten=True):
    _from, _to = _range
    coords = [torch.linspace(_from, _to, steps=h), torch.linspace(_from, _to, steps=w)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

    return mgrid.detach()


def uniform_coordinates_3d(h, w, t, _range=(-1, 1), flatten=True):
    _from, _to = _range
    coords = [torch.linspace(_from, _to, steps=h),
              torch.linspace(_from, _to, steps=w),
              torch.linspace(_from, _to, steps=t)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w t c -> (h w t) c')

    return mgrid.detach()


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False):
        super(Siren, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        self.init_wb(c=c, w0=w0)
        self.w0 = w0
        self.c = c
        self.is_first = is_first

    def init_wb(self, c, w0):
        self.fc = nn.Linear(self.dim_in, self.dim_out)

        w_std = 1 / self.dim_in if self.is_first else np.sqrt(c / self.dim_in) / w0

        self.fc.weight.data.uniform_(-w_std, w_std)
        self.fc.bias.data.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.fc(x)
        if self.is_first:
            out = self.w0 * out
        out = torch.sin(out)
        return out


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, w0=1., w0_initial=30., c=6.):
        super(SirenNet, self).__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim_hidden

        layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:  # if final layer
                final_layer = nn.Linear(dim_hidden, dim_out)
                w_std = np.sqrt(c / dim_hidden) / w0
                final_layer.weight.data.uniform_(-w_std, w_std)
                layers.append(final_layer)
                break

            if i == 0:  # if first layer
                layer_w0 = w0_initial
                layer_dim_in = dim_in
                is_first = True
            else:
                layer_w0 = w0
                layer_dim_in = dim_hidden
                is_first = False

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                c=c,
                is_first=is_first,
            ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, out_size=None):
        out = self.layers(x)
        if out_size is not None:
            if len(out_size) == 2:
                h, w = out_size
                out = rearrange(out, '(h w) c -> () c h w', h=h, w=w)
            elif len(out_size) == 3:
                h, w, t = out_size
                out = rearrange(out, '(h w t) c -> t c h w', h=h, w=w, t=t)
            else:
                raise ValueError('wrong output size')

        return postprocess(out)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, use_fourier=True, fourier_dim=256):
        super(MLP, self).__init__()
        self.num_layers = n_layers
        self.use_fourier = use_fourier
        if self.use_fourier:
            dim_in = fourier_dim * 2
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(dim_in, dim_hidden))
                layers.append(nn.ReLU())
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(dim_hidden, dim_out))
            else:
                layers.append(nn.Linear(dim_hidden, dim_hidden))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x, out_size=None):
        out = self.layers(x)

        if out_size is not None:
            if len(out_size) == 2:
                h, w = out_size
                out = rearrange(out, '(h w) c -> () c h w', h=h, w=w)
            elif len(out_size) == 3:
                h, w, t = out_size
                out = rearrange(out, '(h w t) c -> t c h w', h=h, w=w, t=t)
            else:
                raise ValueError('wrong output size')

        return postprocess(out)


def flat_to_2d(_in, _size):
    h, w = _size
    out = rearrange(_in, '(h w) c -> () c h w', h=h, w=w)
    return out


def flat_to_3d(_in, _size):
    h, w, t = _size
    out = rearrange(_in, '(h w t) c -> t c h w', h=h, w=w, t=t)
    return out

