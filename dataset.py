"""Custom Dataset for Agnes Martin images"""

from torch.utils.data import Dataset
from os import listdir
from os.path import isfile
from PIL import Image


class AMDataset(Dataset):

    def __init__(self, data_path: str, transform = None):
        self.images = [data_path + f for f in listdir(data_path) if isfile(data_path + f)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_to_load = self.images[item]
        im = Image.open(image_to_load).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im
