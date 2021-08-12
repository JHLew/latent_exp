from models import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import _Dataset
import torch.optim as optim


def train(save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = My_ViT(latent_dim=256, dim=128, depth=4, heads=4, mlp_dim=64)
    INR_G = MLP_Generator(num_layers=5, latent_dim=256, ff_dim=128, hidden_dim=64)
    model = Wrapper(vit, INR_G)
    model = nn.DataParallel(model).to(device)

    n_epochs = 500
    batch_size = 8

    train_data = _Dataset('/DF2K', resolution=256, is_train=True)
    valid_data = _Dataset('/DF2K', is_train=False)

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=10)

    # loss function
    loss = nn.L1Loss()
    # loss = nn.MSELoss()

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, data in enumerate(tqdm(train_loader)):
            gt = data
            gt = gt.to(device)

            recon = model(gt)

            recon_loss = loss(recon, gt)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

        validate(model, valid_loader, epoch)
        torch.save(model.module.state_dict(), save_path)


def validate(model, loader, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss()
    total_loss = 0
    n = len(loader.dataset)

    for i, data in enumerate(loader):
        gt = data
        gt = gt.to(device)
        with torch.no_grad():
            recon = model(gt)
            recon_loss = loss(recon, gt)
            total_loss += recon_loss.item()

    avg_loss = total_loss / n

    print(f'Validation loss at Epoch {ep}: {avg_loss}.')


if __name__ == '__main__':
    save_path = './ViTINR.pth'
    train(save_path)
