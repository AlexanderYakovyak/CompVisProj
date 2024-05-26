import pandas as pd

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torchvision



from utilities import utils
from model.yolo_testing import test_yolo
from model.data_preprocessing import CarLocalDataset
from image_processing.images_transformation import get_transform

train_root = '../resources/Kaggle_Cars_Dataset/training_images/'
train_csv_dir = '../resources/Kaggle_Cars_Dataset/train_solution_bounding_boxes.csv'


def test_kaggle():
    metadata = pd.read_csv(train_csv_dir)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    train_dataset = CarLocalDataset(metadata=metadata, root=train_root, transforms=get_transform(train=True))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    images, targets = next(iter(train_data_loader))
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections




if __name__ == '__main__':
    #test_yolo()
    test_kaggle()
