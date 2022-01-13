import random
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import PIL
from PIL import Image
import random
import numpy as np


class AlzheimersData(data.Dataset):

    def __init__(self, imageFolderDataset, should_invert=False, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.should_invert = should_invert
        self.transform = transform

    def __getitem__(self, idx):
        # tuple of the form (Image_PATH,label)
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # Whether image 2 should be in same class or not
        same_class = random.choice([True, False])

        if(same_class):
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if(img0_tuple[1] == img1_tuple[1]):
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if(self.should_invert):
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1] != img1_tuple[1])]))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
