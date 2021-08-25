# from vit_pytorch.vit import Transformer, Attention, FeedForward
from torch import einsum
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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                Attention(dim, heads=heads, dim_head=dim_head),
                # PreNorm(dim, FeedForward(dim, mlp_dim))
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        mid_feats = []  # custom
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            mid_feats.append(x)  # custom
        x = torch.cat(mid_feats, dim=2)  # custom
        return x


class My_ViT(nn.Module):
    def __init__(self, latent_dim, hidden_dim, ff_dim, depth, heads, mlp_dim, patch_size=8, channels=3, dim_head=64, fill_mismatch='pad'):
        super(My_ViT, self).__init__()
        self.patch_size = patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear((channels + ff_dim) * patch_size * patch_size, hidden_dim)
        # )

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear((channels + ff_dim) * patch_size * patch_size, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size),
            nn.Conv2d(channels + ff_dim, ff_dim, 3, 1),  # 8 x 8 -> 6 x 6
            nn.GELU(),
            nn.Conv2d(ff_dim, ff_dim, 3, 1),  # 6 x 6 -> 4 x 4
            nn.GELU(),
            nn.Conv2d(ff_dim, ff_dim, 3, 1),  # 4 x 4 -> 2 x 2
            nn.GELU(),
            nn.Conv2d(ff_dim, ff_dim, 2, 1),
            nn.GELU(),
            nn.Conv2d(ff_dim, hidden_dim, 1, 1),
        )

        spatial_dim = 2
        sigma = 10
        assert ff_dim % 2 == 0, 'Fourier features should be divided by 2.'
        self.pos_embedding = nn.Parameter(torch.randn((spatial_dim, ff_dim // 2)) * sigma)

        self.latent_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            # nn.LayerNorm(hidden_dim * depth),
            # nn.Linear(hidden_dim, latent_dim)
            nn.Linear(hidden_dim * depth, latent_dim),
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
        x = rearrange(x, '(b n) c s1 s2 -> b n c (s1 s2)', b=b)[:, :, :, 0]  # when using conv embedding
        b, n, _ = x.shape

        latent_tokens = repeat(self.latent_token, '() n d -> b n d', b=b)
        x = torch.cat((latent_tokens, x), dim=1)

        x = self.transformer(x)  # batch, n_patches, latents

        x = x[:, 0]  # pick latent of 'cls'(latent) token

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


# MLP generator, but gets weight from the encoder.
class MLP_G_dummy(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super(MLP_G_dummy, self).__init__()
        self.w_mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim * latent_dim),
        )

        self.out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_channels)
        )
        self.act = nn.GELU()

    def forward(self, x, ff):
        b, h, w, d = ff.shape
        y = rearrange(ff, 'b h w d -> b (h w) d')

        b, md = x.shape
        x = rearrange(x, 'b (m d) -> b m d', d=d)
        m = int(md / d)  # number of layers
        layer_weights = self.w_mapping(x)  # b m (d d)

        for i in range(m):
            i_weight = layer_weights[:, i]
            i_weight = rearrange(i_weight, 'b (c1 c2) -> b c1 c2', c1=d, c2=d)
            y = torch.bmm(y, i_weight)
            y = self.act(y)

        y = self.out(y)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)

        return y
