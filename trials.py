from models import *
from vit_pytorch import ViT
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch


if __name__ == '__main__':
    img = Image.open('./examples/0002x4.png')
    img = img.resize((256, 256))
    img = preprocess(to_tensor(img)).cuda().unsqueeze(0)

    v = My_ViT(
        image_size=(256, 256),
        patch_size=32,
        num_classes=1024,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    v = v.cuda().eval()

    p = v(img)
    print(p.shape)
    torch.save(v.state_dict(), './vit.pth')

