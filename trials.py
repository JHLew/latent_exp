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
    vit = My_ViT(latent_dim=256, dim=128, depth=4, heads=4, mlp_dim=64)
    INR_G = MLP_Generator(num_layers=5, latent_dim=256, ff_dim=128, hidden_dim=64)
    model = Wrapper(vit, INR_G)
    model = nn.DataParallel(model).to(device)

    # hyper parameters
    n_epochs = 500
    batch_size = 8
    lr = 1e-4
    batch_res = 256

    # paths
    train_paths = ['/dataset/DIV2K/train_HR', '/dataset/Flickr2k/Flick2K_HR']
    valid_paths = ['/dataset/DIV2K/valid_HR']
    valid_path = './valid'

    train_data = _Dataset(train_paths, resolution=batch_res, is_train=True)
    valid_data = _Dataset(valid_paths, is_train=False)

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
    total_loss = 0
    n = len(loader.dataset)

    os.makedirs(valid_path, exist_ok=True)

    for i, data in enumerate(loader):
        gt, name = data
        gt = gt.to(device)
        with torch.no_grad():
            recon = model(gt)
            recon_loss = loss(recon, gt)
            total_loss += recon_loss.item()

        # save reconstructed image
        to_pil_image(recon[0].cpu()).save(os.path.join(valid_path, f'{ep}/{name[0]}'))

    avg_loss = total_loss / n
    print(f'Validation loss at Epoch {ep}: {avg_loss}.')


if __name__ == '__main__':
    save_path = './ViTINR.pth'
    train(save_path)
