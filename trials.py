from models import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import _Dataset
import torch.optim as optim
import shutil
import os
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter
import shutil
from UNet import *
from vit_pytorch import ViT
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet18


def train(save_path):
    exp_name = 'p8_ff64_ld4096_h20_inr5128'
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    resume = 0  # 0 for fresh training, else resume from pretraining.
    n_epochs = 5000
    batch_size = 32
    lr = 1e-5
    batch_res = 256
    valid_every = 20
    loss_type = 'MSE'

    # shared params
    latent_dim = 4096  # 1216  # orig: 1024, current: 8x8x(3+16)
    ff_dim = 64  # 256

    # ViT params: ViT-Base / 16
    vit_layers = 6  # 12
    vit_patch_size = 8
    vit_heads = 20  # 12

    # vit_hidden_dim = 768  # 768
    vit_mlp_dim = 2048  # 3072
    vit_hidden_dim = (vit_patch_size ** 2) * 7
    print('Main dimension:', vit_hidden_dim)

    # INR params: SIREN with activations as GELU instead of sine.
    inr_layers = 5
    inr_hidden_dim = 128

    # models
    # vit = ViT(image_size=256, patch_size=16, num_classes=latent_dim, dim=vit_hidden_dim, depth=vit_layers,
    #           heads=vit_heads, mlp_dim=vit_mlp_dim)
    vit = My_ViT(latent_dim=latent_dim, hidden_dim=vit_hidden_dim, ff_dim=ff_dim,
                 depth=vit_layers, heads=vit_heads,
                 mlp_dim=vit_mlp_dim, patch_size=vit_patch_size)
    # vgg = vgg16(num_classes=latent_dim).eval()
    # model = UNet(3, 3, skip=False)
    # model = UpNet(vit, latent_dim, 3)
    # vgg = Hier_Encoder(latent_dim)
    # model = modulated_UPNet(vgg, latent_dim, 3)

    INR_G = MLP_Generator(num_layers=inr_layers, latent_dim=latent_dim, ff_dim=ff_dim, hidden_dim=inr_hidden_dim)
    # INR_G = MLP_G_dummy(latent_dim=latent_dim)
    model = Wrapper(vit, INR_G)

    # use multi-gpu
    model = nn.DataParallel(model).to(device)

    # paths
    train_paths = ['/dataset/DIV2K/train_HR', '/dataset/Flickr2K/Flickr2K_HR']
    valid_paths = ['/dataset/DIV2K/valid_HR']
    valid_path = f'./valid/{exp_name}'
    logs = f'./logs/{exp_name}'

    # dataset & dataloader
    train_data = _Dataset(train_paths, resolution=batch_res, is_train=True)
    valid_data = _Dataset(valid_paths, resolution=256, is_train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)
    ipe = len(train_data) // batch_size + 1  # iterations per epoch

    # loss function
    loss_fn = None
    if loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'MSE':
        loss_fn = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not resume == 0:  # if resume training
        ckpt = torch.load(save_path)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
    else:
        if os.path.exists(logs):
            shutil.rmtree(logs)

    # recording & tracking
    os.makedirs(logs, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    writer = SummaryWriter(logs)
    best = 100  # to keep record of best performance in validation

    for epoch in range(resume, n_epochs):
        # validate(model, valid_loader, loss_fn, valid_path, epoch)  # for debugging
        # quit()
        epoch_train_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            gt, _ = data
            gt = gt.to(device)

            # forwarding
            recon = model(gt)

            # loss computation
            recon_loss = loss_fn(recon, gt)

            # backprop & update
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            epoch_train_loss += recon_loss.item()
            to_pil_image(recon[0].cpu()).save(os.path.join(valid_path, f'train_{i}_recon.png'))
            to_pil_image(gt[0].cpu()).save(os.path.join(valid_path, f'train_{i}_gt.png'))

        epoch_train_loss /= ipe  # average
        print(f'Training loss at Epoch {epoch}: {epoch_train_loss}.')

        if (epoch + 1) % valid_every == 0:
            valid_loss = validate(model, valid_loader, loss_fn, valid_path, epoch)
            writer.add_scalars(f'{loss_type}', {'Train': epoch_train_loss, 'Valid': valid_loss}, epoch)
            if valid_loss < best:
                best = valid_loss
                ckpt = {'opt': optimizer.state_dict(), 'model': model.module.state_dict()}
                torch.save(ckpt, save_path)
        else:
            writer.add_scalars(f'{loss_type}', {'Train': epoch_train_loss}, epoch)


def validate(model, loader, loss_fn, valid_path, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(loader.dataset)
    ep = ep + 1  # add 1 to make it intuitive

    cur_valid_path = os.path.join(valid_path, f'{ep}')
    os.makedirs(cur_valid_path, exist_ok=True)

    with torch.no_grad():
        total_loss = 0
        for i, data in enumerate(loader):
            gt, name = data
            gt = gt.to(device)
            gt = interpolate(gt, size=(256, 256), mode='bicubic', align_corners=False)
            recon = model(gt)
            if not recon.shape == gt.shape:
                b, c, h, w = gt.shape
                recon = recon[:b, :c, :h, :w]

            recon_loss = loss_fn(recon, gt)
            total_loss += recon_loss.item()

            # save reconstructed image
            to_pil_image(recon[0].cpu()).save(os.path.join(cur_valid_path, f'{name[0]}'))

        avg_loss = total_loss / n
    print(f'Validation loss at Epoch {ep}: {avg_loss}.')
    return avg_loss


if __name__ == '__main__':
    save_path = './ViTINR.pth'
    train(save_path)
