from vit_pytorch.vit import Transformer
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import interpolate, pad

# normalize (0, 1) to (-1, 1)
def preprocess(t):
    return t * 2 - 1

# denormalize from (-1, 1) to (0, 1)
def postprocess(t):
    return torch.clamp((t + 1) / 2, min=0, max=1)


# class Orig_ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#         x = self.transformer(x)
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)


class My_ViT(nn.Module):
    def __init__(self, latent_dim, hidden_dim, ff_dim, depth, heads, mlp_dim, patch_size=4, channels=3, dim_head=64, fill_mismatch='pad'):
        super(My_ViT, self).__init__()
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((channels + ff_dim) * patch_size * patch_size, hidden_dim)
        )

        spatial_dim = 2
        sigma = 10
        assert ff_dim % 2 == 0, 'Fourier features should be divided by 2.'
        self.pos_embedding = nn.Parameter(torch.randn((spatial_dim, ff_dim // 2)) * sigma)

        self.latent_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout=0)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.fill_mismatch = fill_mismatch

    def map_ff(self, x):
        x_proj = torch.matmul((2 * np.pi * x), self.pos_embedding)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b, c, h, w = img.shape

        if (not h % self.patch_size == 0) or (not w % self.patch_size == 0):
            if self.fill_mismatch == 'pad':
                h_pad, w_pad = (h // self.patch_size + 1) * self.patch_size - h, (w // self.patch_size + 1) * self.patch_size - w
                zero_pad_LRUD = (0, w_pad, 0, h_pad)  # pad size order: Left Right Up Down
                img = pad(img, pad=zero_pad_LRUD)
                b, c, h, w = img.shape
            else:
                refined_hw = round(h / self.patch_size) * self.patch_size, round(w / self.patch_size) * self.patch_size
                img = interpolate(img, size=refined_hw, mode='bilinear', align_corners=False)
                b, c, h, w = img.shape

        # rel_pos = uniform_coordinates(h, w, flatten=False).permute(2, 0, 1).unsqueeze(0).to(device)
        # rel_pos = repeat(rel_pos, '() c h w -> b c h w', b=b)
        rel_pos = uniform_coordinates(h, w, flatten=False).unsqueeze(0).to(device)
        rel_pos = repeat(rel_pos, '() h w c -> b h w c', b=b)
        ff = self.map_ff(rel_pos)
        bchw_ff = rearrange(ff, 'b h w c -> b c h w')
        x = torch.cat([img, bchw_ff], dim=1)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        latent_tokens = repeat(self.latent_token, '() n d -> b n d', b=b)
        x = torch.cat((latent_tokens, x), dim=1)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.mlp_head(x)
        return x, ff


class MLP_Norm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_Norm, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU()
        )
        self.gamma_predictor = nn.Linear(out_dim, out_dim)
        self.beta_predictor = nn.Linear(out_dim, out_dim)

    def forward(self, x, latent):
        shared_f = self.shared(latent)
        # shared_f = latent
        gamma = self.gamma_predictor(shared_f)
        beta = self.beta_predictor(shared_f)

        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)

        x = x * gamma + beta

        return x


class MLP_Generator(nn.Module):
    def __init__(self, num_layers, latent_dim, ff_dim, hidden_dim, out_dim=3, channel_first=True):
        super(MLP_Generator, self).__init__()
        self.num_layers = num_layers
        layers = []
        norm_layers = []
        self.act = nn.GELU()
        self.channel_first = channel_first
        # self.norm_shared = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.GELU()
        # )
        for i in range(self.num_layers):
            if i == 0:  # first layer
                layers.append(nn.Linear(ff_dim, hidden_dim))
                norm_layers.append(MLP_Norm(latent_dim, hidden_dim))

            elif i == self.num_layers - 1:  # final layer
                layers.append(nn.Linear(hidden_dim, out_dim))

            else:  # middle layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                norm_layers.append(MLP_Norm(latent_dim, hidden_dim))

        self.layers = nn.Sequential(*layers)
        self.norm_layers = nn.Sequential(*norm_layers)

    # w/o skip connections
    # def forward(self, latent, ff):
    #     x = ff
    #     for i in range(self.num_layers - 1):
    #         x = self.layers[i](x)
    #         x = self.norm_layers[i](x, latent)
    #         x = self.act(x)
    #     x = self.layers[self.num_layers](x)
    #
    #     if self.channel_first:
    #         x = rearrange(x, 'b h w c -> b c h w')
    #
    #     return postprocess(x)

    def forward(self, latent, ff):
        x = self.layers[0](ff)
        # latent_shared = self.norm_shared(latent)
        for i in range(self.num_layers - 1):
            res = x
            x = self.norm_layers[i](x, latent)
            # x = self.norm_layers[i](x, latent_shared)
            x = self.act(x)
            x = self.layers[i + 1](x)
            if i < self.num_layers - 2:
                x = res + x

        if self.channel_first:
            x = rearrange(x, 'b h w c -> b c h w')

        return postprocess(x)


class Wrapper(nn.Module):
    def __init__(self, encoder, decoder, ff_dim=None):
        super(Wrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.own_ff = False
        if ff_dim is not None:
            self.own_ff = True
            spatial_dim = 2
            ff_dim = ff_dim
            sigma = 10
            self.pos_embedding = nn.Parameter(torch.randn((spatial_dim, ff_dim // 2)) * sigma)

    def map_ff(self, x):
        x_proj = torch.matmul((2 * np.pi * x), self.pos_embedding)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x):
        if not self.own_ff:
            latent, ff = self.encoder(x)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            latent = self.encoder(x)
            b, c, h, w = x.shape
            rel_pos = uniform_coordinates(h, w, flatten=False).unsqueeze(0).to(device)
            rel_pos = repeat(rel_pos, '() h w c -> b h w c', b=b)
            ff = self.map_ff(rel_pos)
        return self.decoder(latent, ff)


def uniform_coordinates(h, w, _range=(-1, 1), flatten=True):
    _from, _to = _range
    coords = [torch.linspace(_from, _to, steps=h), torch.linspace(_from, _to, steps=w)]
    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)
    if flatten:
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

    return mgrid.detach()

