import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from utils.image_utils import load_img

#import torch.nn.functional as F
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(rgb_dir))
        self.inp_filenames = [os.path.join(rgb_dir,  x) for x in inp_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        inp_img = Image.open(inp_path).convert('RGB')

        w, h = inp_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)

        hh, ww = inp_img.shape[1], inp_img.shape[2]
        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        aug = random.randint(0, 8)
        if aug == 1:
            inp_img = inp_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(rgb_dir))

        self.inp_filenames = [os.path.join(rgb_dir, x) for x in inp_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target, altered to base on input files

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return inp_img, filename


class DataLoaderVal_(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal_, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir)))

        self.inp_filenames = [os.path.join(rgb_dir, x) for x in inp_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target
        self.mul = 16

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        w, h = inp_img.size
        #h, w = inp_img.shape[2], inp_img.shape[3]
        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename


class DataLoaderTest_(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderTest_, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))

        self.clean_filenames = [os.path.join(rgb_dir, 'low', x) for x in clean_files if is_image_file(x)]
        self.inp_size = len(self.clean_filenames)

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[inp_index])))

        clean_filename = os.path.split(self.clean_filenames[inp_index])[-1]

        clean = clean.permute(2, 0, 1)

        return clean, clean_filename
