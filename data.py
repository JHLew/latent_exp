from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor
from random import randint


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
        if self.is_train:
            w, h = img.size
            # cropping
            # choose size
            w_size, h_size = randint(self.resolution, w), randint(self.resolution, h)

            # choose region
            w_range, h_range = w - w_size, h - h_size
            w_start, h_start = randint(0, w_range), randint(0, h_range)
            w_end, h_end = w_start + w_range, h_start + h_range

            # crop
            augmented_img = img.crop((w_start, w_end, h_start, h_end))

            # resize
            augmented_img = augmented_img.resize((self.resolution, self.resolution))

            # flip
            flip_flag = randint(0, 1)
            if flip_flag == 1:
                augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)

            # rotate
            rotate_flag = randint(0, 3)
            if rotate_flag == 1:
                augmented_img = augmented_img.rotate(Image.ROTATE_90)
            elif rotate_flag == 2:
                augmented_img = augmented_img.rotate(Image.ROTATE_180)
            elif rotate_flag == 3:
                augmented_img = augmented_img.rotate(Image.ROTATE_270)

            img = augmented_img

        img = to_tensor(img)

        return img