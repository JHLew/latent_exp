from models import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import _Dataset
import torch.optim as optim
import shutil
import os
from torchvision.transforms.functional import to_pil_image


def train(save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    n_epochs = 500
    batch_size = 8
    lr = 1e-4
    batch_res = 256

    # shared params
    latent_dim = 256
    ff_dim = 128

    # ViT params
    vit_layers = 6
    vit_heads = 8
    vit_mlp_dim = 64

    # INR params
    inr_layers = 5
    inr_hidden_dim = 256

    vit = My_ViT(latent_dim=latent_dim, dim=ff_dim, depth=vit_layers, heads=vit_heads, mlp_dim=vit_mlp_dim)
    INR_G = MLP_Generator(num_layers=inr_layers, latent_dim=latent_dim, ff_dim=ff_dim, hidden_dim=inr_hidden_dim)
    model = Wrapper(vit, INR_G)
    model = nn.DataParallel(model).to(device)

    # paths
    train_paths = ['/dataset/DIV2K/train_HR', '/dataset/Flickr2K/Flickr2K_HR']
    valid_paths = ['/dataset/DIV2K/valid_HR']
    valid_path = './valid'

    train_data = _Dataset(train_paths, resolution=batch_res, is_train=True)
    valid_data = _Dataset(valid_paths, resolution=256, is_train=False)

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

    # loss function
    loss = nn.L1Loss()
    # loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, data in enumerate(tqdm(train_loader)):
            gt, _ = data
            gt = gt.to(device)

            recon = model(gt)

            recon_loss = loss(recon, gt)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

        validate(model, valid_loader, valid_path, epoch)
        torch.save(model.module.state_dict(), save_path)


def validate(model, loader, valid_path, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss()
    n = len(loader.dataset)

    cur_valid_path = os.path.join(valid_path, f'{ep}')
    os.makedirs(cur_valid_path, exist_ok=True)

    with torch.no_grad():
        total_loss = 0
        for i, data in enumerate(loader):
            gt, name = data
            gt = gt.to(device)
            recon = model(gt)
            if not recon.shape == gt.shape:
                b, c, h, w = gt.shape
                recon = recon[:b, :c, :h, :w]
            recon_loss = loss(recon, gt)
            total_loss += recon_loss.item()

            # save reconstructed image
            to_pil_image(recon[0].cpu()).save(os.path.join(cur_valid_path, f'{name[0]}'))

        avg_loss = total_loss / n
    print(f'Validation loss at Epoch {ep}: {avg_loss}.')


if __name__ == '__main__':
    save_path = './ViTINR.pth'
    train(save_path)
