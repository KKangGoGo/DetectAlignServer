import numpy as np
import PIL.ImageOps
import requests
import torch

from PIL import Image
from torch.utils.data import Dataset

class ConvertImageData(Dataset):
    def __init__(self, image_tuple, compare_tuple, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.image_tuple = image_tuple
        self.compare_tuple = compare_tuple

    def __getitem__(self, index):
        img0_tuple = self.image_tuple
        img1_tuple = self.compare_tuple

        # img1_tuple[0]는 url 정보임
        img0 = Image.open(requests.get(img0_tuple[0], stream=True).raw)
        img1 = Image.open(requests.get(img1_tuple[0], stream=True).raw)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)