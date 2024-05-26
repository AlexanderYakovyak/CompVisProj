import torch

from src.model.data_preprocessing import CarLocalDataset

from src.image_processing.images_transformation import get_transform
from src.utilities.engine import train_one_epoch, evaluate
from src.utilities.utils import collate_fn

test_root = '../resources/Kaggle_Cars_Dataset/testing_images/'

train_root = '../resources/Kaggle_Cars_Dataset/training_images/'
train_csv_dir = '../resources/Kaggle_Cars_Dataset/train_solution_bounding_boxes.csv'

def train_evaluate_model():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1

    dataset = CarLocalDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = CarLocalDataset('data/PennFudanPed', get_transform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )


    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)