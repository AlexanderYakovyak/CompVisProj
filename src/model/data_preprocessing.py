import os
import torch
from carslocalization.file_name_reader import read_images_coordinates_filenames

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors


class CarLocalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.images, self.coords = read_images_coordinates_filenames(root)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.images[idx])
        coords_path = os.path.join(self.root, self.coords[idx])

        img = read_image(img_path)
        coords = []

        with open(coords_path, 'r') as file:
            for line in file:
                x, y = map(int, line.split())
                coords.append((x, y))

        num_objs = len(coords)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        isgroup = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)
        img = img/255.0

        target = {"coords": coords, "labels": labels, "image_id": image_id, "isgroup": isgroup}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)