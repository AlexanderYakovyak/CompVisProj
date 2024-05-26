import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional
from torchvision.transforms import v2 as T


def crop_coordinates(coords, params):
    i, j, h, w = params

    adjusted_coords = []
    for cord in coords:
        x, y = cord
        adjusted_x = x - j
        adjusted_y = y - i

        if 0 <= adjusted_x < w and 0 <= adjusted_y < h:
            adjusted_coords.append((adjusted_x, adjusted_y))

    return adjusted_coords


class CustomRandomResizedCrop(torch.nn.Module):

    def forward(self, img, target):
        crop = T.RandomResizedCrop(224)
        params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        img_crop = functional.crop(img, *params)
        coords_crop = crop_coordinates(target["coords"], params)

        target["coords"] = coords_crop

        return img_crop, target


class CustomRandomHorizontalFlip(torch.nn.Module):

    def forward(self, img, target):

        w = img.shape[-1]
        p = np.random.rand()

        if p < 0.5:
            img_flipped = functional.hflip(img)
            coords = target["coords"]
            coords_flipped = []
            for cord in coords:
                x, y = cord
                adjusted_x = w - x
                coords_flipped.append((adjusted_x,y))

            target["coords"] = coords_flipped

            return img_flipped, target

        return img, target


def transform_images(images, coordinates, mean, std):
    transformed_images = []
    transformed_coordinates = []

    for image_index in range(len(images)):
        img = Image.fromarray(images[image_index])
        coords = coordinates[image_index]

        transforms = T.Compose([
            T.ToImage(),
            CustomRandomResizedCrop(),
            CustomRandomHorizontalFlip(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])

        transformed_img, transformed_coords = transforms(img, coords)
        transformed_img = transformed_img.numpy().transpose((1, 2, 0))

        transformed_images.append(transformed_img)
        transformed_coordinates.append(transformed_coords)

    return transformed_images, transformed_coordinates


"""def get_transform(train):
    transforms = []
    if train:
        transforms.append(CustomRandomResizedCrop())
        transforms.append(CustomRandomHorizontalFlip())
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)"""


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
