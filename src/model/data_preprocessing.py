import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class CarLocalDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, root, transforms):
        self.metadata = metadata
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.metadata['image'][idx])
        img = read_image(img_path)

        box = np.array([
            self.metadata['xmin'][idx],
            self.metadata['ymin'][idx],
            self.metadata['xmax'][idx],
            self.metadata['ymax'][idx],
        ], dtype=np.float32)

        num_objs = 1

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        area = (box[2]-box[0])*(box[3]-box[1])
        isgroup = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)
        img = img/255.0

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(box, format="XYXY", canvas_size=F.get_size(img))
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['isgroup'] = isgroup

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.metadata['image'])
