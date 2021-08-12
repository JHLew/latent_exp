from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor

class _Dataset(Dataset):
    def __init__(self, path, resolution=None, is_train=True):
        super(_Dataset, self).__init__()
        self.path = path
        self.resolution = resolution
        self.is_train = is_train
        if self.resolution is None:
            self.is_train = False

        self.img_list = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = Image.open(img)

        # data augmentation

        img = to_tensor(img)

        return img


